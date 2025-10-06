"""
props_hybrid.py
Fetch NFL player props from The Odds API v4 using the per-event endpoint.

- Uses correct (short) market keys, e.g. player_pass_yds (NOT player_passing_yards)
- Optionally probes each event to discover which markets are actually offered
  to avoid 422 INVALID_MARKET responses and save credits.
- Logs x-requests-* credit headers after every HTTP call.
"""

from __future__ import annotations

import os
import time
import json
import math
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

# ---------- Config ----------
NFL_SPORT = "americanfootball_nfl"
BASE = "https://api.the-odds-api.com/v4"

DEFAULT_BOOKS = ["draftkings", "fanduel"]
DEFAULT_REGIONS = "us"
DEFAULT_ODDS_FMT = "american"
DEFAULT_DATE_FMT = "iso"

# ✅ canonical NFL player-prop market keys (matches your screenshots + docs)
VALID_MARKETS_NFL: List[str] = [
    # Passing
    "player_pass_yds", "player_pass_tds", "player_pass_attempts",
    "player_pass_completions", "player_pass_interceptions",
    "player_pass_longest_completion",
    # Rushing
    "player_rush_yds", "player_rush_attempts", "player_rush_longest",
    # Receiving
    "player_reception_yds", "player_receptions", "player_reception_longest",
    "player_reception_tds",
    # Combo / Specials
    "player_anytime_td",
    "player_pass_rush_reception_tds",
    "player_pass_rush_reception_yds",
    # Defensive / Kicking (offered by fewer books; keep them optional)
    "player_sacks", "player_solo_tackles", "player_tackles_assists",
    "player_field_goals", "player_pats",
]

# Legacy → Canonical mapper (accepts old names but converts to valid ones)
LEGACY_MARKET_MAP = {
    "player_passing_yards": "player_pass_yds",
    "player_passing_tds": "player_pass_tds",
    "player_passing_attempts": "player_pass_attempts",
    "player_passing_completions": "player_pass_completions",
    "player_passing_interceptions": "player_pass_interceptions",
    "player_passing_longest_completion": "player_pass_longest_completion",
    "player_rushing_yards": "player_rush_yds",
    "player_rushing_attempts": "player_rush_attempts",
    "player_rushing_longest": "player_rush_longest",
    "player_receiving_yards": "player_reception_yds",
    "player_receptions_total": "player_receptions",
    "player_receiving_longest": "player_reception_longest",
    "player_receiving_tds": "player_reception_tds",
    "player_anytime_touchdown": "player_anytime_td",
}

# ---------------- Utilities ----------------

def _env_api_key(api_key: Optional[str]) -> str:
    """Resolve API key from arg -> env -> fail."""
    k = api_key or os.getenv("THE_ODDS_API_KEY") or os.getenv("ODDS_API_KEY")
    if not k:
        raise RuntimeError("Missing THE_ODDS_API_KEY (or pass api_key=...).")
    return k

def _canon_books(books: Optional[str | Iterable[str]]) -> str:
    if books is None or (isinstance(books, str) and not books.strip()):
        books_list = DEFAULT_BOOKS
    elif isinstance(books, str):
        books_list = [b.strip() for b in books.split(",") if b.strip()]
    else:
        books_list = list(books)
    return ",".join(books_list)

def _canon_markets(markets: Optional[Iterable[str] | str]) -> List[str]:
    """Normalize incoming markets → canonical keys; default to a safe bundle."""
    if markets is None or (isinstance(markets, str) and not markets.strip()):
        return [
            "player_pass_yds", "player_rush_yds",
            "player_reception_yds", "player_receptions",
            "player_anytime_td",
        ]

    if isinstance(markets, str):
        raw = [m.strip() for m in markets.split(",") if m.strip()]
    else:
        raw = [str(m).strip() for m in markets if str(m).strip()]

    out: List[str] = []
    for m in raw:
        m2 = LEGACY_MARKET_MAP.get(m, m)
        if m2 in VALID_MARKETS_NFL:
            out.append(m2)
        else:
            logging.warning("[props_hybrid] ignoring unknown market key: %s", m)
    # Dedup but keep order
    seen = set()
    norm = []
    for m in out:
        if m not in seen:
            norm.append(m)
            seen.add(m)
    return norm

def _hdr_credits(r: requests.Response) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    used = r.headers.get("x-requests-used")
    remaining = r.headers.get("x-requests-remaining")
    last = r.headers.get("x-requests-last")
    def _i(x): 
        try: return int(x) if x is not None else None
        except: return None
    return _i(used), _i(remaining), _i(last)

def _get(url: str, params: Dict[str, Any], timeout: int = 15) -> requests.Response:
    r = requests.get(url, params=params, timeout=timeout)
    # log credits on every call
    u, rem, last = _hdr_credits(r)
    logging.info("[props_hybrid] credits | used=%s remaining=%s reset=%s", u, rem, last)
    if r.status_code >= 400:
        # Let caller decide how to handle 422 etc; include body for diagnostics
        msg = f"GET {r.url} -> {r.status_code} | body={r.text[:500]}"
        raise requests.HTTPError(msg, response=r)
    return r

# ---------------- Fetchers ----------------

def _list_events(api_key: str, regions: str, books: str) -> List[Dict[str, Any]]:
    """List NFL events (no quota cost)."""
    url = f"{BASE}/sports/{NFL_SPORT}/events"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "bookmakers": books,   # narrows to events priced by these books
        "dateFormat": DEFAULT_DATE_FMT,
    }
    r = _get(url, params)
    return r.json() or []

def _probe_event_markets(api_key: str, event_id: str, regions: str, books: str) -> List[str]:
    """
    Ask which markets exist for this event across our target books.
    This is cheap compared to wasting per-market odds calls that 422.
    """
    url = f"{BASE}/sports/{NFL_SPORT}/events/{event_id}/markets"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "bookmakers": books,
    }
    try:
        r = _get(url, params)
        data = r.json() or []
        # The response is a list of markets: [{"key":"player_pass_yds", ...}, ...]
        keys = []
        for m in data:
            k = m.get("key")
            if k: keys.append(k)
        return keys
    except requests.HTTPError:
        # Some providers don’t support this endpoint consistently; fall back.
        logging.info("[props_hybrid] market probe failed for event=%s; assuming unknown", event_id)
        return []

def _fetch_event_props(
    api_key: str,
    event_id: str,
    markets: List[str],
    regions: str,
    books: str,
    sleep: float,
    timeout: int,
    use_probe: bool,
) -> List[Dict[str, Any]]:
    """
    Fetch desired markets for a single event via /events/{id}/odds.
    Returns raw rows (bookmaker, market, outcomes...) for later flattening.
    """
    if not markets:
        return []

    desired = markets
    if use_probe:
        offered = set(_probe_event_markets(api_key, event_id, regions, books))
        if offered:
            desired = [m for m in desired if m in offered]
            if not desired:
                return []

    # The endpoint supports comma-separated markets; keep chunks modest (<= 6)
    rows: List[Dict[str, Any]] = []
    for i in range(0, len(desired), 6):
        chunk = desired[i : i + 6]
        url = f"{BASE}/sports/{NFL_SPORT}/events/{event_id}/odds"
        params = {
            "apiKey": api_key,
            "regions": regions,
            "bookmakers": books,
            "markets": ",".join(chunk),
            "oddsFormat": DEFAULT_ODDS_FMT,
            "dateFormat": DEFAULT_DATE_FMT,
        }
        try:
            r = _get(url, params, timeout=timeout)
        except requests.HTTPError as e:
            # If INVALID_MARKET for 1+ items in chunk, try per-market to salvage others
            body = str(getattr(e, "response", None).text if getattr(e, "response", None) else "")
            if "INVALID_MARKET" in body or "422" in str(e):
                for m in chunk:
                    try:
                        rr = _get(url, {**params, "markets": m}, timeout=timeout)
                        rows.extend(rr.json() or [])
                    except requests.HTTPError:
                        logging.info("[props_hybrid] skip event=%s market=%s (invalid)", event_id, m)
                continue
            raise

        rows.extend(r.json() or [])
        if sleep:
            time.sleep(sleep)
    return rows

# ---------------- Public: get_props ----------------

def get_props(
    *,
    api_key: Optional[str] = None,
    date: Optional[str] = None,
    season: Optional[int] = None,
    window: Optional[int | str] = None,
    cap: int = 0,
    markets: Optional[Iterable[str] | str] = None,
    books: Optional[Iterable[str] | str] = None,
    order: str = "odds",
    team_filter: Optional[List[str]] = None,
    selection: Optional[str] = None,
    event_ids: Optional[List[str]] = None,
    regions: str = DEFAULT_REGIONS,
    use_probe: bool = True,
    sleep: float = 0.15,
    timeout: int = 15,
) -> pd.DataFrame:
    """
    Main entry from engine.
    Returns a raw (unnormalized) DataFrame combining all requested events/markets.
    """
    logging.info("[props_hybrid] Calling get_props() with API key and cap=%s", cap)

    key = _env_api_key(api_key)
    books_str = _canon_books(books)
    mkts = _canon_markets(markets)

    # Get events
    events = event_ids or [e["id"] for e in _list_events(key, regions, books_str)]
    if cap and cap > 0:
        events = events[:cap]

    all_rows: List[Dict[str, Any]] = []
    for eid in events:
        try:
            ev_rows = _fetch_event_props(
                api_key=key,
                event_id=eid,
                markets=mkts,
                regions=regions,
                books=books_str,
                sleep=sleep,
                timeout=timeout,
                use_probe=use_probe,
            )
            # The per-event response is a list with a single event object (if any)
            for ev in ev_rows:
                ev_id = ev.get("id") or eid
                commence_time = ev.get("commence_time")
                home = ev.get("home_team"); away = ev.get("away_team")
                for bk in (ev.get("bookmakers") or []):
                    bkey = bk.get("key")
                    for mk in (bk.get("markets") or []):
                        mkey = mk.get("key")
                        for out in (mk.get("outcomes") or []):
                            # Outcomes vary by market; props typically include "description" (player)
                            row = {
                                "event_id": ev_id,
                                "commence_time": commence_time,
                                "home_team": home,
                                "away_team": away,
                                "bookmaker": bkey,
                                "market_key": mkey,
                                "name": out.get("name"),
                                "description": out.get("description"),
                                "price": out.get("price"),
                                "point": out.get("point"),
                                "team": out.get("team"),
                                "player": out.get("description"),  # convenience alias
                            }
                            all_rows.append(row)
        except Exception as e:
            logging.info("[props_hybrid] event=%s fetch error: %s", eid, e)

    df = pd.DataFrame(all_rows)
    logging.info("[props_hybrid] flattened %d rows", len(df))
    return df
