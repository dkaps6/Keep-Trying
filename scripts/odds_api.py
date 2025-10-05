# scripts/odds_api.py
from __future__ import annotations

import os
import time
import json
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests


API_BASE = "https://api.the-odds-api.com/v4"
SPORT   = "americanfootball_nfl"

# Player markets we care about (as returned by The Odds API)
PLAYER_MARKETS = [
    "player_receptions",
    "player_reception_yds",       # receiving yards
    "player_rush_yds",
    "player_rush_attempts",
    "player_pass_yds",
    "player_pass_tds",
    "player_rush_reception_yds",  # combo yards
    "player_anytime_td",
]


# ----------------- small utils -----------------

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _key() -> str:
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key:
        print("[odds_api] ⚠️  no ODDS_API_KEY in env; skipping API calls")
    return key

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default

def _env_list(name: str, default: str = "") -> List[str]:
    raw = os.environ.get(name, default)
    return [x.strip() for x in raw.split(",") if x.strip()]

def _book_allowlist() -> List[str]:
    raw = os.environ.get("ODDS_API_BOOKMAKERS", "draftkings,fanduel,betmgm,caesars")
    return [b.strip() for b in raw.split(",") if b.strip()]


# ----------------- HTTP helpers -----------------

@dataclass
class HttpResp:
    status: int
    json: Any
    headers: Dict[str, Any]
    url: str


def _get(path: str, params: Dict[str, Any]) -> HttpResp:
    url = f"{API_BASE}{path}"
    try:
        r = requests.get(url, params=params, timeout=25)
    except requests.exceptions.RequestException as e:
        return HttpResp(status=599, json={"error": str(e)}, headers={}, url=url)

    js = None
    try:
        js = r.json()
    except Exception:
        js = {"raw": r.text}

    return HttpResp(status=r.status_code, json=js, headers=dict(r.headers), url=r.url)


def _print_quota(h: Dict[str, Any], prefix: str = "") -> None:
    rem = h.get("x-requests-remaining")
    used = h.get("x-requests-used")
    lim  = h.get("x-requests-limit")
    if rem or used or lim:
        print(f"[odds_api] {prefix}credits remaining={rem} used={used} limit={lim}")


# ----------------- event discovery -----------------

def _list_events(api_key: str) -> List[Dict[str, Any]]:
    """
    Use the 'odds' endpoint with markets=h2h to get the slate + H2H prices.
    We also write game_lines.csv here.
    """
    params = dict(apiKey=api_key, regions="us", markets="h2h", oddsFormat="american", dateFormat="iso")
    resp = _get(f"/sports/{SPORT}/odds", params)
    if resp.status != 200:
        print(f"[odds_api] events list failed: HTTP {resp.status} :: {resp.url}")
        print(f"[odds_api] details: {resp.json}")
        return []

    _print_quota(resp.headers, prefix="")
    events = resp.json if isinstance(resp.json, list) else []

    # Flatten to game_lines.csv (implied win prob from 1st bookmaker present)
    rows = []
    for e in events:
        # Some books include draws; we only use the two team outcomes.
        meta = {
            "event_id": e.get("id"),
            "home_team": e.get("home_team"),
            "away_team": e.get("away_team"),
            "commence_time": e.get("commence_time"),
        }
        wp_home, wp_away = _implied_winprobs_from_h2h(e.get("bookmakers", []), meta["home_team"], meta["away_team"])
        meta.update({"home_wp": wp_home, "away_wp": wp_away})
        rows.append(meta)

    gl = pd.DataFrame(rows)
    Path("outputs").mkdir(parents=True, exist_ok=True)
    gl.to_csv("outputs/game_lines.csv", index=False)

    return events


def _implied_winprobs_from_h2h(bookmakers: List[Dict[str, Any]], home: str, away: str) -> Tuple[float, float]:
    """
    Use the first bookmaker in our allow-list that has both teams priced.
    """
    allow = _book_allowlist()
    def am_to_p(odds: float) -> float:
        if odds is None:
            return float("nan")
        o = float(odds)
        return (-o)/((-o)+100.0) if o < 0 else 100.0/(o+100.0)

    for b in bookmakers or []:
        if b.get("key") not in allow:  # skip
            continue
        for m in b.get("markets", []):
            if m.get("key") != "h2h":
                continue
            outs = m.get("outcomes", [])
            # outcomes have 'name' set to team name
            ph = pa = float("nan")
            for o in outs:
                if o.get("name") == home:
                    ph = am_to_p(o.get("price"))
                elif o.get("name") == away:
                    pa = am_to_p(o.get("price"))
            if pd.isna(ph) or pd.isna(pa):
                continue
            s = ph + pa
            if s <= 0:
                continue
            return ph/s, pa/s
    return float("nan"), float("nan")


# ----------------- selection / windows -----------------

def _include_filters_from_env() -> Tuple[List[str], set[str]]:
    teams  = [t.strip().lower() for t in os.environ.get("ODDS_API_INCLUDE_TEAMS", "").split(",") if t.strip()]
    events = {e.strip() for e in os.environ.get("ODDS_API_INCLUDE_EVENTS", "").split(",") if e.strip()}
    return teams, events

def _apply_selection_filters(events: List[Dict[str, Any]], teams: List[str], events_set: set[str]) -> List[Dict[str, Any]]:
    if not teams and not events_set:
        return events
    out = []
    for ev in events:
        if events_set and ev.get("id") in events_set:
            out.append(ev)
            continue
        if teams:
            h = (ev.get("home_team") or "").lower()
            a = (ev.get("away_team") or "").lower()
            if any(t in h or t in a for t in teams):
                out.append(ev)
    return out

def _filter_window(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    hours = _env_int("ODDS_API_EVENT_WINDOW_HOURS", 168)  # default 7d
    cap   = _env_int("ODDS_API_MAX_EVENTS", 40)
    if hours <= 0:
        return events[:cap]
    end = _now_utc() + timedelta(hours=hours)
    keep = []
    for e in events:
        try:
            ct = datetime.fromisoformat(e.get("commence_time").replace("Z", "+00:00"))
        except Exception:
            ct = None
        if (ct is None) or (ct <= end):
            keep.append(e)
    return keep[:cap]


# ----------------- props fetching -----------------

def _flatten_event_props(ev: Dict[str, Any], allow_books: List[str], markets_set: set[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    meta = {
        "event_id": ev.get("id"),
        "home_team": ev.get("home_team"),
        "away_team": ev.get("away_team"),
        "commence_time": ev.get("commence_time"),
    }
    for bk in ev.get("bookmakers", []):
        bkey = bk.get("key")
        if allow_books and bkey not in allow_books:
            continue
        for m in bk.get("markets", []):
            mkey = m.get("key")
            if mkey not in markets_set:
                continue
            outs = m.get("outcomes", []) or []
            # group by (player, line)
            by: Dict[Tuple[str, float], Dict[str, float]] = {}
            for o in outs:
                player = o.get("description") or o.get("name")  # TD has desc=player, name=Yes/No
                point  = o.get("point")
                side   = o.get("name")  # Over/Under OR Yes/No
                price  = o.get("price")
                if player is None:
                    continue
                k = (player, point)
                d = by.setdefault(k, {})
                d[side] = price
            for (player, point), d in by.items():
                price_over  = d.get("Over", d.get("Yes"))
                price_under = d.get("Under", d.get("No"))
                if price_over is None and price_under is None:
                    continue
                row = dict(meta)
                row.update({
                    "book": bkey,
                    "market_key": mkey,
                    "player": player,
                    "line": point,
                    "price_over": price_over,
                    "price_under": price_under,
                })
                rows.append(row)
    return pd.DataFrame(rows)


def fetch_props(date: str = "today",
                season: str = "2025",
                teams: List[str] | None = None,
                events: List[str] | None = None) -> pd.DataFrame:
    """
    Main entry. Returns a flattened DataFrame of player props for the selected slate.
    Also writes outputs/game_lines.csv from H2H odds.
    """
    key = _key()
    if not key:
        return pd.DataFrame()

    # 1) list events + write game_lines.csv
    events_all = _list_events(key)

    # 2) apply time window + selection filters
    evs = _filter_window(events_all)

    env_teams, env_events = _include_filters_from_env()
    teams       = [t.lower() for t in (teams or [])] or env_teams
    events_set  = set(events or []) or set(env_events)
    if teams or events_set:
        evs = _apply_selection_filters(evs, teams, events_set)
        print(f"[odds_api] selection filter -> {len(evs)} events kept")

    if not evs:
        print("[odds_api] 0 events after filtering")
        return pd.DataFrame()

    # 3) iterate events -> fetch markets for each event
    allow_books = _book_allowlist()
    markets_set = set(PLAYER_MARKETS)
    DBG = Path("outputs/debug"); DBG.mkdir(parents=True, exist_ok=True)

    frames: List[pd.DataFrame] = []
    for i, ev in enumerate(evs, 1):
        ev_id = ev.get("id")
        # Per-event query
        params = dict(
            apiKey=key,
            regions="us",
            markets=",".join(PLAYER_MARKETS),
            oddsFormat="american",
            dateFormat="iso",
            eventIds=ev_id,  # limit to this event
        )
        resp = _get(f"/sports/{SPORT}/odds", params)
        if resp.status != 200:
            print(f"[odds_api] event {i}/{len(evs)} {ev_id} props failed: HTTP {resp.status} :: {resp.url}")
            print(f"[odds_api] details: {resp.json}")
            # stop if quota done
            if resp.status == 401:
                _print_quota(resp.headers, prefix="")
                break
            continue

        _print_quota(resp.headers, prefix="")
        ev_payloads = resp.json if isinstance(resp.json, list) else []
        if ev_payloads:
            ev_full = ev_payloads[0]
        else:
            ev_full = ev  # fallback to discovery meta (no props)
        # Write debug files for transparency
        (DBG / f"event_{ev_id}.json").write_text(json.dumps(ev_full, indent=2))
        (DBG / f"event_{ev_id}_discovery.json").write_text(json.dumps({
            k: ev.get(k) for k in ("id", "sport_key", "sport_title", "commence_time", "home_team", "away_team")
            if k in ev
        }, indent=2))

        df_ev = _flatten_event_props(ev_full, allow_books, markets_set)
        frames.append(df_ev)

        # tiny politeness backoff to avoid bursty credit use
        time.sleep(0.15)

    if not frames:
        print("[odds_api] 0 player-prop rows returned across all events.")
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    return out
