# scripts/props_hybrid.py
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
Hybrid props fetcher (OddsAPI-first, DK fallback stubs).

This file is loaded by your engine via:
    from scripts.props_hybrid import get_props as fn

It fetches NFL player props from The Odds API v4 `/odds` endpoint and
returns a normalized DataFrame with columns:

    event_id, event_time, home_team, away_team,
    book, market, player, point, over_odds, under_odds

You can later wire a direct DraftKings fallback by implementing
`_fetch_from_dk_fallback(...)` to return the *same rows* dicts.
"""

# ------------------------------
# Configuration (via env vars)
# ------------------------------
DEFAULT_API_URL = os.getenv("THE_ODDS_API_URL", "https://api.the-odds-api.com/v4/sports")
DEFAULT_REGIONS = os.getenv("ODDS_REGIONS", "us")
DEFAULT_FORMAT  = os.getenv("ODDS_ODDS_FORMAT", "american")
DEFAULT_DF      = os.getenv("ODDS_DATE_FORMAT", "iso")
NFL_KEY         = "americanfootball_nfl"

# Markets you want by default. Extend anytime.
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

# Books to include by default (can be overridden by engine args)
DEFAULT_BOOKS = ["draftkings", "fanduel", "betmgm", "caesars"]

UTC = timezone.utc


# ------------------------------
# Logging utility
# ------------------------------
def _log(msg: str) -> None:
    print(f"[props_hybrid] {msg}", flush=True)


# ------------------------------
# HTTP helpers (with retries)
# ------------------------------
def _http_get(url: str, query: Dict[str, str], retries: int = 3, sleep: float = 0.8) -> Any:
    q = urllib.parse.urlencode(query)
    full = f"{url}?{q}"
    last_err = None
    for attempt in range(max(1, retries)):
        try:
            req = urllib.request.Request(
                full,
                headers={
                    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                                   "Chrome/124.0.0.0 Safari/537.36"),
                    "Accept": "application/json, text/plain, */*",
                },
            )
            with urllib.request.urlopen(req, timeout=35) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except Exception as e:
            last_err = e
            # Very basic backoff (helps with rate limits)
            time.sleep(sleep * (1 + attempt * 0.5))
    raise RuntimeError(f"GET failed after retries: {full}\nDetail: {repr(last_err)}")


def _parse_iso(ts: str) -> datetime:
    # OddsAPI returns ISO like "2025-10-06T17:25:00Z"
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(UTC)


# ------------------------------
# Normalization
# ------------------------------
def _rows_from_oddsapi_event(ev: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Flatten a single OddsAPI event into rows at granularity:
    (event_id, event_time, home_team, away_team, book, market, player, point, over_odds, under_odds)
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

            # Group outcomes by (player, point). Outcomes usually have:
            #  - name: "Player Name"
            #  - description: "Over"/"Under" (sometimes missing for yes/no style)
            #  - price: American odds
            #  - point: float (line), can be null for yes/no
            cache: Dict[Tuple[str, float], Dict[str, Any]] = {}

            for oc in mkt.get("outcomes", []):
                player_name = oc.get("name") or ""
                desc        = str(oc.get("description") or "").lower()
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
                    # Sometimes yes/no style markets don't label Over/Under.
                    # Put first price in over_odds, second in under_odds.
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
# Primary source: OddsAPI
# ------------------------------
def _fetch_from_oddsapi(
    date: Optional[str],
    hours: Optional[int],
    books: List[str],
    markets: List[str],
) -> List[Dict[str, Any]]:
    api_key = os.getenv("THE_ODDS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing THE_ODDS_API_KEY in environment.")

    # Build optional time window
    start_dt: Optional[datetime] = None
    end_dt: Optional[datetime] = None

    if date and date.lower() != "today":
        try:
            d = datetime.fromisoformat(date).date()
            start_dt = datetime(d.year, d.month, d.day, 0, 0, tzinfo=UTC)
            end_dt   = start_dt + timedelta(days=1)
        except Exception:
            pass

    if start_dt is None and hours is not None and hours > 0:
        start_dt = datetime.now(tz=UTC)
        end_dt   = start_dt + timedelta(hours=hours)

    base_url = f"{DEFAULT_API_URL}/{NFL_KEY}/odds"
    params   = {
        "apiKey": api_key,
        "regions": DEFAULT_REGIONS,
        "markets": ",".join(markets),
        "oddsFormat": DEFAULT_FORMAT,
        "dateFormat": DEFAULT_DF,
        "bookmakers": ",".join(books),
    }

    _log(f"OddsAPI GET {base_url} [books={','.join(books)} markets={','.join(markets)}]")
    payload = _http_get(base_url, params, retries=3, sleep=0.8)
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected OddsAPI response type: {type(payload)}")

    filtered = _filter_events_by_time(payload, start_dt, end_dt)
    _log(f"OddsAPI events kept after time filter: {len(filtered)}")
    rows: List[Dict[str, Any]] = []
    for ev in filtered:
        rows.extend(_rows_from_oddsapi_event(ev))
    return rows


# ------------------------------
# (Optional) Secondary source: DK fallback stub
# ------------------------------
def _fetch_from_dk_fallback(
    date: Optional[str],
    hours: Optional[int],
    markets: List[str],
) -> List[Dict[str, Any]]:
    """
    Stub for future direct-DK fallback (e.g., via requests/BeautifulSoup).
    Must return the *same* row dicts as OddsAPI flattening.
    For now, returns an empty list (no-op).
    """
    # TODO: implement DK-only scrape and map into the same row schema.
    return []


# ------------------------------
# Public entry point used by engine
# ------------------------------
def get_props(
    date: Optional[str] = None,
    season: Optional[str] = None,
    hours: Optional[int] = None,
    books: Optional[List[str]] = None,
    markets: Optional[List[str]] = None,
    order: str = "odds",
    selection: Optional[str] = None,
    teams: Optional[List[str]] = None,
    events: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Return props as a DataFrame with required columns:
        event_id, event_time, home_team, away_team, book, market, player, point, over_odds, under_odds
    """
    books = books or DEFAULT_BOOKS
    markets = markets or DEFAULT_MARKETS

    # 1) Primary: OddsAPI
    odds_rows = _fetch_from_oddsapi(date=date, hours=hours, books=books, markets=markets)

    # 2) Secondary (optional): DK fallback
    dk_rows = _fetch_from_dk_fallback(date=date, hours=hours, markets=markets)

    all_rows = odds_rows + dk_rows
    df = pd.DataFrame(all_rows)

    if df.empty:
        _log("No props found (empty DataFrame).")
        return df

    # Optional light filters (engine may also apply)
    if teams:
        tl = [t.lower() for t in teams]
        keep = []
        for _, r in df.iterrows():
            bucket = f"{str(r.get('home_team') or '').lower()} {str(r.get('away_team') or '').lower()}"
            keep.append(any(t in bucket for t in tl))
        df = df[keep]

    if events:
        evset = set(events)
        if "event_id" in df.columns:
            df = df[df["event_id"].isin(evset)]

    if selection:
        s = str(selection).strip().lower()
        if s and "player" in df.columns:
            df = df[df["player"].astype(str).str.lower().str.contains(s, na=False)]

    df = df.reset_index(drop=True)
    _log(f"returning {len(df)} rows to engine")
    return df
