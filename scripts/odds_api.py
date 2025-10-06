# scripts/odds_api.py
from __future__ import annotations

import os
import re
import time
import math
import json
from datetime import datetime, timedelta, timezone
from typing import Iterable, Dict, Any, List

import pandas as pd
import requests


ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "americanfootball_nfl"

# Env guards (prevent credit burn)
EVENT_WINDOW_HOURS = int(os.environ.get("ODDS_API_EVENT_WINDOW_HOURS", "168"))  # next 7 days by default
MAX_EVENTS = int(os.environ.get("ODDS_API_MAX_EVENTS", "40"))                   # cap per run


def _norm(s: str) -> str:
    """Simple normalizer for fuzzy team matches."""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    # drop city tokens that often cause mismatches but keep nickname
    drop = {
        "kansas", "city", "new", "york", "new", "england", "los", "angeles", "san",
        "francisco", "bay", "tampa", "bay", "green", "bay", "washington",
        "jacksonville", "carolina", "atlanta", "houston", "indianapolis",
        "detroit", "chicago", "cleveland", "cincinnati", "miami", "buffalo",
        "philadelphia", "pittsburgh", "seattle", "denver", "dallas", "minnesota",
        "tennessee", "arizona", "las", "vegas", "baltimore", "orleans",
    }
    toks = [t for t in s.split() if t not in drop]
    out = "".join(toks) or s.replace(" ", "")
    return out


def _within_window(commence_iso: str, hours: int) -> bool:
    try:
        dt = datetime.fromisoformat(commence_iso.replace("Z", "+00:00"))
    except Exception:
        return True  # be permissive
    now = datetime.now(timezone.utc)
    return now <= dt <= now + timedelta(hours=hours)


def _req(url: str, params: Dict[str, Any], *, timeout: int = 20) -> requests.Response:
    r = requests.get(url, params=params, timeout=timeout)
    if r.status_code == 401 or r.status_code == 403:
        raise RuntimeError(f"Odds API auth/permission error {r.status_code}: {r.text}")
    if r.status_code == 429:
        raise RuntimeError("Odds API rate limited (429)")
    if r.status_code >= 400:
        raise RuntimeError(f"Odds API error {r.status_code}: {r.text}")
    return r


def _discover_events(api_key: str, markets: Iterable[str], window_hours: int) -> List[Dict[str, Any]]:
    """
    Hit the 'odds' endpoint once with h2h to get event list, then
    we'll query props per-event below. (Single discovery call = fewer credits.)
    """
    url = f"{ODDS_API_BASE}/sports/{SPORT_KEY}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
    }
    r = _req(url, params)
    events = r.json()

    # Window filter
    events = [ev for ev in events if _within_window(ev.get("commence_time", ""), window_hours)]
    # Deduplicate by id
    seen = set()
    uniq = []
    for ev in events:
        eid = ev.get("id")
        if not eid or eid in seen:
            continue
        seen.add(eid)
        uniq.append(ev)
    return uniq


def _apply_selection_filters(
    events: List[Dict[str, Any]],
    teams: List[str] | None,
    wanted_event_ids: List[str] | None,
) -> List[Dict[str, Any]]:
    """Filter by team nicknames (fuzzy) or explicit event IDs."""
    if not events:
        return []

    # If explicit event ids provided, prefer that
    if wanted_event_ids:
        wanted = set(wanted_event_ids)
        kept = [ev for ev in events if ev.get("id") in wanted]
        return kept

    # Team nickname fuzzy match
    if teams:
        want = {_norm(t) for t in teams if t.strip()}
        kept = []
        for ev in events:
            h = _norm(ev.get("home_team", ""))
            a = _norm(ev.get("away_team", ""))
            if (h in want) or (a in want):
                kept.append(ev)
        return kept

    # No filters -> keep all
    return list(events)


def _fetch_props_for_event(api_key: str, event_id: str, markets: Iterable[str]) -> List[Dict[str, Any]]:
    """
    Pull player prop markets for a single event id.
    """
    url = f"{ODDS_API_BASE}/sports/{SPORT_KEY}/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": ",".join(markets),
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    r = _req(url, params)
    return r.json()


def fetch_props(
    date: str = "today",                 # accepted but not strictly used by the Odds API
    season: int | None = None,
    teams: List[str] | None = None,
    events: List[str] | None = None,
    markets: List[str] | None = None,
    provider_order: str | None = None,
) -> pd.DataFrame:
    """
    High-level:
      1) Discover events once (h2h) within ODDS_API_EVENT_WINDOW_HOURS.
      2) Apply selection filters (teams / events) with fuzzy matching.
      3) If filters keep 0 events, fallback to full slate (avoid empty runs).
      4) Fetch player prop markets per event_id.
      5) Flatten into a tidy DataFrame of (event_id, home_team, away_team, market, player, line, price, bookmaker, ...).
    """
    api_key = os.environ.get("ODDS_API_KEY", "").strip()
    if not api_key:
        print("[odds_api] No ODDS_API_KEY set; returning empty frame.")
        return pd.DataFrame()

    # reasonable default markets if none provided
    default_markets = [
        "player_receptions",
        "player_receiving_yds",
        "player_rush_yds",
        "player_rush_attempts",
        "player_pass_yds",
        "player_pass_tds",
        "player_anytime_td",
    ]
    markets = markets or default_markets
    order = (provider_order or "odds,dk").strip()

    print(f"[odds_api] window={EVENT_WINDOW_HOURS}h cap={MAX_EVENTS} markets={','.join(markets)} order={order}")
    if teams:
        print(f"[odds_api] team filter: {teams}")
    if events:
        print(f"[odds_api] event-id filter: {events}")

    # 1) Discover events (one call)
    try:
        evs = _discover_events(api_key, markets=["h2h"], window_hours=EVENT_WINDOW_HOURS)
    except Exception as e:
        print(f"[odds_api] discovery error: {e}")
        return pd.DataFrame()

    # 2) Apply filters
    filtered = _apply_selection_filters(evs, teams, events)
    print(f"[odds_api] selection filter -> {len(filtered)} events kept")

    # 3) Fallback if we killed everything
    if not filtered and evs:
        print("[odds_api] filters yielded 0 events; falling back to full discovered slate.")
        filtered = evs

    # 4) Cap to MAX_EVENTS
    filtered = filtered[:MAX_EVENTS]
    if not filtered:
        print("[odds_api] 0 events after filtering")
        return pd.DataFrame()

    # For visibility list what we are about to fetch
    print("[odds_api] fetching props for events:")
    for ev in filtered:
        print("   -", ev.get("id"), "|", ev.get("away_team"), "@", ev.get("home_team"), "|", ev.get("commence_time"))

    # 5) Fetch per event and flatten
    rows: List[Dict[str, Any]] = []
    for ev in filtered:
        eid = ev.get("id")
        if not eid:
            continue
        try:
            payload = _fetch_props_for_event(api_key, eid, markets=markets)
        except Exception as e:
            print(f"[odds_api] event {eid} props failed: {e}")
            continue

        # payload: list of bookmakers each with markets -> outcomes -> player/point/price
        for book in (payload or []):
            bookmaker = book.get("bookmaker_key") or book.get("key") or book.get("title")
            for m in book.get("markets", []):
                market_key = m.get("key")
                for outc in m.get("outcomes", []):
                    rows.append({
                        "event_id": eid,
                        "home_team": ev.get("home_team"),
                        "away_team": ev.get("away_team"),
                        "commence_time": ev.get("commence_time"),
                        "bookmaker": bookmaker,
                        "market": market_key,
                        "label": outc.get("description") or outc.get("name") or outc.get("participant"),
                        "line": outc.get("point"),
                        "price_american": outc.get("price"),
                        "side": outc.get("side"),  # 'over'/'under' sometimes
                    })

        # be a good API citizen
        time.sleep(0.15)

    if not rows:
        print("[odds_api] no player-prop rows returned across all events.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Basic cleanup
    df = df.drop_duplicates()
    # normalize player label
    if "label" in df.columns:
        df["player"] = df["label"].fillna("").astype(str).str.strip()
    else:
        df["player"] = ""
    return df

