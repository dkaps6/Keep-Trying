# scripts/props_hybrid.py
from __future__ import annotations

import os
import re
import time
import json
import math
import typing as T
import itertools
import datetime as dt

import pandas as pd
import requests

from .market_keys import VALID_MARKETS, ALIASES, NON_NFL_GLOBAL

NFL_SPORT = "americanfootball_nfl"
BASE = "https://api.the-odds-api.com/v4"

DEFAULT_BOOKS = ["draftkings", "fanduel"]
DEFAULT_REGIONS = "us"
DEFAULT_ODDS_FORMAT = "american"
DEFAULT_DATEFORMAT = "iso"

HEADERS_OF_INTEREST = (
    "x-requests-remaining",
    "x-requests-used",
    "x-requests-reset",
)

def _log(msg: str) -> None:
    print(f"[props_hybrid] {msg}")

def _env_api_key() -> str:
    key = os.getenv("THE_ODDS_API_KEY") or os.getenv("ODDS_API_KEY") or ""
    return key.strip()

def _coerce_books(books: T.Optional[T.Union[str, T.List[str]]]) -> str:
    if not books:
        return ",".join(DEFAULT_BOOKS)
    if isinstance(books, str):
        raw = [b.strip().lower() for b in books.split(",") if b.strip()]
    else:
        raw = [str(b).strip().lower() for b in books if str(b).strip()]
    # keep distinct, preserve order, and only pass commas (no list brackets)
    dedup = []
    for b in raw:
        if b not in dedup:
            dedup.append(b)
    return ",".join(dedup)

def _alias_and_filter_markets(markets: T.Optional[T.Union[str, T.List[str]]]) -> T.List[str]:
    """
    Apply alias fixes (e.g., player_passing_yards -> player_pass_yds),
    drop non-NFL/global items, and keep only VALID_MARKETS.
    """
    if not markets:
        return []  # caller may choose sane defaults

    if isinstance(markets, str):
        mlist = [m.strip().lower() for m in markets.split(",") if m.strip()]
    else:
        mlist = [str(m).strip().lower() for m in markets if str(m).strip()]

    fixed = []
    for m in mlist:
        m = ALIASES.get(m, m)
        if m in NON_NFL_GLOBAL:
            continue
        if m in VALID_MARKETS:
            fixed.append(m)

    # dedupe, preserve order
    out = []
    for m in fixed:
        if m not in out:
            out.append(m)
    return out

def _http_get(url: str, params: dict) -> T.Tuple[dict, dict]:
    """
    GET with tiny retry/backoff + return (json, response_headers).
    Raises RuntimeError on final failure with helpful message.
    """
    last = None
    for i in range(3):
        try:
            r = requests.get(url, params=params, timeout=30)
            # store interesting headers for credit counter
            hdrs = {k.lower(): v for k, v in r.headers.items()}
            if r.status_code == 200:
                try:
                    return r.json(), hdrs
                except Exception:
                    raise RuntimeError("Response was not valid JSON.")
            # bubble API error body if present
            try:
                body = r.json()
            except Exception:
                body = r.text
            last = f"HTTP {r.status_code}: {body}"
        except Exception as e:
            last = repr(e)
        time.sleep(0.8 * (i + 1))
    raise RuntimeError(f"GET failed after retries: {url}\nDetail: {last}")

def _print_credits(hdrs: dict) -> None:
    left = hdrs.get("x-requests-remaining")
    used = hdrs.get("x-requests-used")
    reset = hdrs.get("x-requests-reset")
    if any([left, used, reset]):
        _log(f"credits | used={used} remaining={left} reset={reset}")

def _list_event_ids(api_key: str, date: str, books_csv: str) -> T.List[str]:
    """
    We fetch a light market (e.g., h2h) to obtain current event ids for NFL.
    """
    url = f"{BASE}/sports/{NFL_SPORT}/odds"
    params = dict(
        apiKey=api_key,
        regions=DEFAULT_REGIONS,
        markets="h2h",
        oddsFormat=DEFAULT_ODDS_FORMAT,
        dateFormat=DEFAULT_DATEFORMAT,
        bookmakers=books_csv,
    )
    data, hdrs = _http_get(url, params)
    _print_credits(hdrs)

    ev_ids = []
    if isinstance(data, list):
        for ev in data:
            eid = str(ev.get("id") or ev.get("eventId") or ev.get("key") or "")
            if eid:
                ev_ids.append(eid)
    # dedupe, preserve order
    out = []
    for e in ev_ids:
        if e not in out:
            out.append(e)
    _log(f"event shells fetched: {len(out)}")
    return out

def _fetch_event_props(api_key: str, event_id: str, books_csv: str, markets: T.List[str]) -> T.List[dict]:
    """
    Player props must be pulled per-event:
    GET /sports/{sport}/events/{eventId}/odds?markets=...
    """
    if not markets:
        return []

    url = f"{BASE}/sports/{NFL_SPORT}/events/{event_id}/odds"
    params = dict(
        apiKey=api_key,
        regions=DEFAULT_REGIONS,
        markets=",".join(markets),
        oddsFormat=DEFAULT_ODDS_FORMAT,
        dateFormat=DEFAULT_DATEFORMAT,
        bookmakers=books_csv,
    )
    data, hdrs = _http_get(url, params)
    _print_credits(hdrs)

    # Return raw for now; normalization happens in normalize_props.
    if isinstance(data, dict) and "bookmakers" in data:
        # single-event payload (some accounts return dict)
        return [data]
    if isinstance(data, list):
        return data
    return []

def _to_df(rows: T.List[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    try:
        return pd.json_normalize(rows)
    except Exception:
        return pd.DataFrame(rows)

# --------------------------------------------------------------------
# Public entry point expected by engine.py
# --------------------------------------------------------------------
def get_props(
    *,
    api_key: T.Optional[str] = None,
    date: str = "today",
    season: T.Optional[int] = None,
    cap: int = 0,
    markets: T.Optional[T.Union[str, T.List[str]]] = None,
    books: T.Optional[T.Union[str, T.List[str]]] = None,
    order: str = "odds",
    team_filter: T.Optional[T.List[str]] = None,
    selection: T.Optional[str] = None,
    event_ids: T.Optional[T.List[str]] = None,
    window: T.Optional[T.Union[int, str]] = None,
    hours: T.Optional[int] = None,
    **_,
) -> pd.DataFrame:
    """
    Fetches NFL props using Odds API v4 with the corrected market keys.
    - Reads API key from THE_ODDS_API_KEY if not provided.
    - Applies aliasing + validation to markets before calling the API.
    - Only passes a clean, comma-separated bookmakers string.
    - Prints live credits after each network call.
    """
    api_key = (api_key or _env_api_key()).strip()
    if not api_key:
        raise RuntimeError("Missing THE_ODDS_API_KEY in environment.")

    # sanitize books & markets
    books_csv = _coerce_books(books)
    clean_markets = _alias_and_filter_markets(markets)

    if not clean_markets:
        # choose a lean, safe default set
        clean_markets = [
            "player_pass_yds",
            "player_rush_yds",
            "player_reception_yds",
            "player_receptions",
            "player_anytime_td",
        ]
        _log(f"no markets passed; using defaults: {','.join(clean_markets)}")
    else:
        _log(f"markets => {','.join(clean_markets)}")

    # 1) discover event ids (unless the engine supplied specific ones)
    if not event_ids:
        event_ids = _list_event_ids(api_key, date=date, books_csv=books_csv)
        if cap and cap > 0:
            event_ids = event_ids[:cap]
    else:
        _log(f"using supplied event_ids={len(event_ids)} (cap={cap})")
        if cap and cap > 0:
            event_ids = event_ids[:cap]

    if not event_ids:
        _log("no events available with current filters; returning empty.")
        return pd.DataFrame()

    # 2) fetch per-event props
    all_rows: T.List[dict] = []
    for eid in event_ids:
        rows = _fetch_event_props(api_key, eid, books_csv, clean_markets)
        if rows:
            # include eventId at top level if missing for traceability
            for r in rows:
                r.setdefault("eventId", eid)
            all_rows.extend(rows)

    if not all_rows:
        _log("no props returned from Odds API.")
        return pd.DataFrame()

    df = _to_df(all_rows)
    _log(f"raw rows (pre-normalize): {len(df)}")
    return df
