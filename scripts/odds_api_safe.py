# scripts/odds_api_safe.py
from __future__ import annotations

import os
import time
import json
import math
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd


"""
Drop-in Odds API v4 fetcher for player props.

WHY THIS FILE:
- Replaces earlier attempts that hit /players/props (404/422). The correct v4 endpoint is /odds
- Mirrors the approach used by the recent "player-props-scraper" repo (Odds API centric)
- Returns a normalized DataFrame your engine/pricing step expects:
    [event_id, event_time, home_team, away_team, book, market, player, point, over_odds, under_odds]

REQUIRED ENV:
- THE_ODDS_API_KEY

OPTIONAL ENV:
- THE_ODDS_API_URL (default: https://api.the-odds-api.com/v4/sports)
- ODDS_REGIONS (default: us)
- ODDS_ODDS_FORMAT (default: american)
- ODDS_DATE_FORMAT (default: iso)

ENGINE CALL:
- Your engine imports: from scripts.odds_api_safe import get_props
- Signature: get_props(date, season, hours, books, markets, order, selection=None, teams=None, events=None)
- Only date/hours/books/markets are used by this fetcher; selection/teams/events are handled upstream if needed.
"""

DEFAULT_API_URL = os.getenv("THE_ODDS_API_URL", "https://api.the-odds-api.com/v4/sports")
DEFAULT_REGIONS = os.getenv("ODDS_REGIONS", "us")
DEFAULT_FORMAT  = os.getenv("ODDS_ODDS_FORMAT", "american")
DEFAULT_DF      = os.getenv("ODDS_DATE_FORMAT", "iso")

NFL_KEY = "americanfootball_nfl"

# OddsAPI market keys we’ll support out of the box (extend as you like)
DEFAULT_MARKETS = [
    "player_pass_tds",
    "player_pass_yds",
    "player_pass_attempts",
    "player_pass_completions",
    "player_pass_interceptions",
    "player_rush_yds",
    "player_rush_attempts",
    "player_receptions",
    "player_receiving_yds",
    "player_anytime_td",
]

# Books to include if none are provided (matches what you typically use)
DEFAULT_BOOKS = ["draftkings", "fanduel", "betmgm", "caesars"]

UTC = timezone.utc


# ------------------------------
# HTTP helpers (with retries)
# ------------------------------
def _http_get(url: str, query: Dict[str, str], retries: int = 3, sleep: float = 0.8) -> Any:
    q = urllib.parse.urlencode(query)
    full = f"{url}?{q}"
    last_err = None
    for _ in range(max(1, retries)):
        try:
            req = urllib.request.Request(full, headers={
                "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                               "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
                "Accept": "application/json, text/plain, */*",
            })
            with urllib.request.urlopen(req, timeout=35) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except Exception as e:
            last_err = e
            time.sleep(sleep)
    raise RuntimeError(f"GET failed after retries: {full}\nDetail: {repr(last_err)}")


def _parse_iso(ts: str) -> datetime:
    # Odds API returns ISO e.g. "2025-10-06T17:25:00Z"
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(UTC)


# ------------------------------
# Core normalization
# ------------------------------
def _rows_from_event(ev: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Flatten the Odds API event structure into rows at granularity:
      event_id, event_time, home_team, away_team, book, market, player, point, over_odds, under_odds
    """
    out: List[Dict[str, Any]] = []
    event_id = ev.get("id")
    home     = ev.get("home_team")
    away     = ev.get("away_team")
    start    = ev.get("commence_time")
    evt_ts   = _parse_iso(start) if start else None

    for bk in ev.get("bookmakers", []):
        book_key = bk.get("key")
        for mkt in bk.get("markets", []):
            market_key = mkt.get("key")
            # For player props, outcomes carry player name + description "Over"/"Under" with a shared "point"
            # Structure: outcomes: [{name: "Player Name", description: "Over", price: -115, point: 67.5}, {... Under ...}]
            # We need to group by (player, point).
            cache: Dict[Tuple[str, float], Dict[str, Any]] = {}
            for oc in mkt.get("outcomes", []):
                player_name = oc.get("name") or ""
                desc        = str(oc.get("description") or "").lower()  # "over" / "under" (or sometimes blank for TD Yes/No)
                price       = oc.get("price")
                point       = oc.get("point")
                try:
                    pt = float(point) if point is not None else math.nan
                except Exception:
                    pt = math.nan

                key = (player_name, pt)
                row = cache.get(key)
                if row is None:
                    row = {
                        "event_id": event_id,
                        "event_time": evt_ts.isoformat() if evt_ts else None,
                        "home_team": home,
                        "away_team": away,
                        "book": book_key,
                        "market": market_key,
                        "player": player_name,
                        "point": pt,
                        "over_odds": None,
                        "under_odds": None,
                    }
                    cache[key] = row

                if desc == "over":
                    row["over_odds"] = price
                elif desc == "under":
                    row["under_odds"] = price
                else:
                    # Some yes/no markets (anytime TD on some books) may not use Over/Under wording.
                    # Put the single price into "over_odds" and leave under None.
                    if row.get("over_odds") is None:
                        row["over_odds"] = price
                    else:
                        row["under_odds"] = price

            out.extend(cache.values())
    return out


def _filter_events_by_time(
    events: List[Dict[str, Any]],
    start_dt: Optional[datetime],
    end_dt: Optional[datetime],
) -> List[Dict[str, Any]]:
    if start_dt is None and end_dt is None:
        return events

    kept = []
    for ev in events:
        ct = ev.get("commence_time")
        if not ct:
            continue
        evt = _parse_iso(ct)
        if (start_dt is None or evt >= start_dt) and (end_dt is None or evt <= end_dt):
            kept.append(ev)
    return kept


# ------------------------------
# Public entry
# ------------------------------
def get_props(
    date: Optional[str] = None,
    season: Optional[str] = None,
    hours: Optional[int] = None,
    books: Optional[List[str]] = None,
    markets: Optional[List[str]] = None,
    order: str = "odds",          # unused here
    selection: Optional[str] = None,  # upstream selection filter if desired
    teams: Optional[List[str]] = None, # upstream team filter if desired
    events: Optional[List[str]] = None # upstream event-id filter if desired
) -> pd.DataFrame:
    """
    Main fetcher your engine calls. Returns a pandas DataFrame with columns:
      event_id, event_time, home_team, away_team, book, market, player, point, over_odds, under_odds
    """
    api_key = os.getenv("THE_ODDS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing THE_ODDS_API_KEY in environment.")

    books = books or DEFAULT_BOOKS
    markets = markets or DEFAULT_MARKETS

    # Build a time window to mimic your UI (date OR hours window)
    # - If 'date' looks like 'YYYY-MM-DD', we keep events from 00:00 to 23:59 UTC that day
    # - Else if 'hours' is provided, keep events within now..now+hours
    start_dt: Optional[datetime] = None
    end_dt: Optional[datetime] = None

    if date and date.lower() != "today":
        try:
            d = datetime.fromisoformat(date).date()
            start_dt = datetime(d.year, d.month, d.day, 0, 0, tzinfo=UTC)
            end_dt   = start_dt + timedelta(days=1)
        except Exception:
            # Fallback to hours if date parse fails
            pass

    if start_dt is None and hours is not None and hours > 0:
        start_dt = datetime.now(tz=UTC)
        end_dt   = start_dt + timedelta(hours=hours)

    # Query /odds with all needed markets & books
    base_url = f"{DEFAULT_API_URL}/{NFL_KEY}/odds"
    params   = {
        "apiKey": api_key,
        "regions": DEFAULT_REGIONS,
        "markets": ",".join(markets),
        "oddsFormat": DEFAULT_FORMAT,
        "dateFormat": DEFAULT_DF,
        "bookmakers": ",".join(books),
    }

    payload = _http_get(base_url, params, retries=3, sleep=0.8)
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected Odds API response: {type(payload)}")

    # Filter by time window (if any)
    filtered_events = _filter_events_by_time(payload, start_dt, end_dt)

    # Flatten to rows
    rows: List[Dict[str, Any]] = []
    for ev in filtered_events:
        rows.extend(_rows_from_event(ev))

    df = pd.DataFrame(rows)
    # Apply optional simple filters upstream-style (teams/events/selection)
    # Your engine also does this, but light redundancy is fine.
    if teams:
        teams_lower = [t.lower() for t in teams]
        keep = []
        for _, r in df.iterrows():
            bucket = f"{str(r.get('home_team') or '').lower()} {str(r.get('away_team') or '').lower()}"
            if any(t in bucket for t in teams_lower):
                keep.append(True)
            else:
                keep.append(False)
        df = df[keep]

    if events:
        evset = set(events)
        df = df[df["event_id"].isin(evset)]

    if selection:
        s = str(selection).strip().lower()
        if s:
            df = df[df["player"].astype(str).str.lower().str.contains(s, na=False)]

    df = df.reset_index(drop=True)
    return df
