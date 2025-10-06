# scripts/props_hybrid.py
from __future__ import annotations

import os
import sys
import time
import json
import math
import re
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

# --------------------------------------------------------------------------------------
# Logging (quiet by default; engine prints key lines)
# --------------------------------------------------------------------------------------
LOG = logging.getLogger(__name__)
if not LOG.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[props_hybrid] %(message)s"))
    LOG.addHandler(h)
LOG.setLevel(logging.INFO)

# --------------------------------------------------------------------------------------
# Config / constants
# --------------------------------------------------------------------------------------

ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY") or os.getenv("ODDS_API_KEY") or ""

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"  # you can parametrize later if you add NBA/NHL, etc.
REGIONS = "us"
ODDS_FORMAT = "american"
DATE_FORMAT = "iso"

# market-name mapping: your sheet/engine -> Odds API player prop key
MARKET_MAP = {
    "player_receptions": "player_receptions",
    "player_receiving_yds": "player_receiving_yards",
    "player_rush_yds": "player_rushing_yards",
    "player_rush_attempts": "player_rushing_attempts",
    "player_pass_yds": "player_passing_yards",
    "player_pass_tds": "player_passing_tds",
    "player_anytime_td": "player_touchdown_anytime",
    # allow commonly-typed alternatives
    "player_receiving_yards": "player_receiving_yards",
    "player_rushing_yards": "player_rushing_yards",
    "player_passing_yards": "player_passing_yards",
    "player_passing_tds": "player_passing_tds",
    "player_touchdown_anytime": "player_touchdown_anytime",
}

# default set if engine passes "default" or None
DEFAULT_MARKETS = [
    "player_receptions",
    "player_receiving_yards",
    "player_rushing_yards",
    "player_rushing_attempts",
    "player_passing_yards",
    "player_passing_tds",
    "player_touchdown_anytime",
]

# Normalize some common book name spellings -> Odds API bookmaker keys
BOOK_MAP = {
    "dk": "draftkings",
    "draftkings": "draftkings",
    "fanduel": "fanduel",
    "fd": "fanduel",
    "betmgm": "betmgm",
    "mgm": "betmgm",
    "caesars": "caesars",
    "czr": "caesars",
}

REQUESTS_RETRIES = 2
REQUESTS_SLEEP_S = 0.8

# Will be filled with credit headers on each request
_last_credits: Dict[str, Any] = {}


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_date(date_str: str) -> datetime:
    """
    Accepts either 'today' or an ISO date (YYYY-MM-DD). Returns UTC midnight.
    """
    if str(date_str).lower() == "today":
        dt = _now_utc().date()
        return datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)
    try:
        y, m, d = map(int, date_str.split("-"))
        return datetime(y, m, d, tzinfo=timezone.utc)
    except Exception:
        # fallback to today if weird format
        dt = _now_utc().date()
        return datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)


def _headers() -> Dict[str, str]:
    return {"Accept": "application/json"}


def _record_credits(resp: requests.Response) -> None:
    global _last_credits
    try:
        remaining = resp.headers.get("x-requests-remaining")
        used = resp.headers.get("x-requests-used")
        if remaining is not None or used is not None:
            _last_credits = {
                "requests_remaining": int(remaining) if remaining is not None else None,
                "requests_used": int(used) if used is not None else None,
                "captured_at": _now_utc().isoformat(),
            }
    except Exception:
        # Don't let headers parsing break flow
        pass


def _http_get(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    GET with simple retry; raises with body detail on failure.
    Also captures credits headers.
    """
    last_err_text = None
    for i in range(REQUESTS_RETRIES + 1):
        r = requests.get(url, params=params, headers=_headers(), timeout=30)
        _record_credits(r)
        if r.ok:
            try:
                return r.json()
            except Exception:
                # Non-JSON means nothing we can do here
                raise RuntimeError(f"Non-JSON from {url}")
        last_err_text = r.text
        time.sleep(REQUESTS_SLEEP_S)
    raise RuntimeError(
        f"GET failed after retries: {r.url}\nDetail: {r.status_code} {r.reason}\nBody: {last_err_text}"
    )


def _normalize_books(books: Optional[List[str]]) -> List[str]:
    if not books:
        # your engine often passes ['draftkings', 'fanduel', ...]; keep that
        return ["draftkings", "fanduel", "betmgm", "caesars"]
    out: List[str] = []
    for b in books:
        k = BOOK_MAP.get(str(b).lower().strip(), b)
        if k not in out:
            out.append(k)
    return out


def _normalize_markets(markets: Optional[List[str]]) -> List[str]:
    """
    Accept engine/default names and return Odds API keys.
    """
    if not markets or str(markets).lower() == "default":
        return DEFAULT_MARKETS[:]
    out: List[str] = []
    for m in markets:
        key = MARKET_MAP.get(m, m)
        if key not in out:
            out.append(key)
    return out


def _apply_selection_filters(
    name: str,
    home: str,
    away: str,
    teams: Optional[List[str]],
    selection_regex: Optional[re.Pattern]
) -> bool:
    """
    Keep an event if it passes team and selection filters.
    - teams: list of substrings e.g., ['Jets', 'Cowboys']
    - selection_regex: if provided, must match the event title
    """
    if teams:
        bucket = f"{home} {away} {name}".lower()
        ok = any(t.lower() in bucket for t in teams)
        if not ok:
            return False
    if selection_regex:
        if not selection_regex.search(name or ""):
            return False
    return True


def _as_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return [s.strip() for s in x.split(",") if s.strip()]
    return []


# --------------------------------------------------------------------------------------
# Core Odds API flow
# --------------------------------------------------------------------------------------

def _fetch_event_shells(
    date: str,
    hours: int,
    books: List[str],
    teams: Optional[List[str]],
    selection: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Get the slate shells from the H2H odds endpoint — it includes event ids,
    start time, teams. This is cheap and works reliably.
    """
    start = _parse_date(date)
    # we use 'hours' window forward from the midnight of 'date'
    end = start + timedelta(hours=hours if hours and hours > 0 else 168)

    params = {
        "regions": REGIONS,
        "markets": "h2h",  # cheap market; we only want ids/teams
        "oddsFormat": ODDS_FORMAT,
        "dateFormat": DATE_FORMAT,
        "bookmakers": ",".join(books),
        "apiKey": ODDS_API_KEY,
    }
    url = f"{BASE_URL}/sports/{SPORT}/odds"

    data = _http_get(url, params)
    shells: List[Dict[str, Any]] = []

    # optional selection filter
    sel_rx = None
    if selection:
        try:
            sel_rx = re.compile(selection, re.IGNORECASE)
        except Exception:
            sel_rx = re.compile(re.escape(selection), re.IGNORECASE)

    for ev in data:
        try:
            event_id = ev.get("id")
            commence = ev.get("commence_time")  # ISO
            if not event_id or not commence:
                continue
            dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
            if dt < start or dt > end:
                continue
            home = ev.get("home_team", "")
            away = ev.get("away_team", "")
            name = ev.get("sport_title", "") or f"{away} @ {home}"

            if not _apply_selection_filters(name, home, away, teams, sel_rx):
                continue

            shells.append(
                {
                    "id": event_id,
                    "commence_time": dt,
                    "home_team": home,
                    "away_team": away,
                    "name": name,
                }
            )
        except Exception:
            # ignore weird entries
            continue

    LOG.info(f"event shells kept: {len(shells)} in window")
    return shells


def _fetch_event_props(
    event_id: str,
    books: List[str],
    markets: List[str],
) -> Dict[str, Any]:
    """
    Fetch player props for a single event (per Odds API docs).
    """
    params = {
        "regions": REGIONS,
        "markets": ",".join(markets),
        "oddsFormat": ODDS_FORMAT,
        "dateFormat": DATE_FORMAT,
        "bookmakers": ",".join(books),
        "apiKey": ODDS_API_KEY,
    }
    url = f"{BASE_URL}/sports/{SPORT}/events/{event_id}/odds"
    return _http_get(url, params)


def _flat_props_rows(
    event_shell: Dict[str, Any],
    payload: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Flatten the Odds API per-event props payload to rows.
    """
    rows: List[Dict[str, Any]] = []
    if not isinstance(payload, dict):
        return rows

    bookmakers = payload.get("bookmakers") or []
    for bk in bookmakers:
        book_key = (bk.get("key") or "").lower()
        markets = bk.get("markets") or []
        for mk in markets:
            market_key = mk.get("key")  # e.g., 'player_receiving_yards'
            outcomes = mk.get("outcomes") or []
            for oc in outcomes:
                # outcomes shape differs a little by market; be defensive
                outcome_name = oc.get("name")
                participant = oc.get("participant") or oc.get("description")
                line = oc.get("point")
                price = oc.get("price")
                last_update = oc.get("last_update") or mk.get("last_update")

                rows.append(
                    {
                        "event_id": event_shell["id"],
                        "commence_time": event_shell["commence_time"],
                        "home_team": event_shell["home_team"],
                        "away_team": event_shell["away_team"],
                        "book": book_key,
                        "market": market_key,
                        "player": participant or outcome_name,  # for ATD, 'name' is the player
                        "outcome": outcome_name,               # e.g., Over/Under/Yes/No
                        "line": line,
                        "price": price,                        # American odds
                        "last_update": last_update,
                    }
                )
    return rows


def _build_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=[
            "event_id","commence_time","home_team","away_team","book","market",
            "player","outcome","line","price","last_update"
        ])
    df = pd.DataFrame(rows)
    # sort a bit for readability
    if "commence_time" in df.columns:
        df = df.sort_values(["commence_time","event_id","book","market","player","outcome"]).reset_index(drop=True)
    return df


def _print_and_write_credits(write_dir: str) -> None:
    if not _last_credits:
        LOG.info("credits: (not reported by API headers)")
        return
    remaining = _last_credits.get("requests_remaining")
    used = _last_credits.get("requests_used")
    LOG.info(f"credits used={used} remaining={remaining}")
    try:
        os.makedirs(write_dir, exist_ok=True)
        out = os.path.join(write_dir, "oddsapi_credits.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(_last_credits, f, indent=2)
    except Exception:
        pass


# --------------------------------------------------------------------------------------
# Public entry point used by engine
# --------------------------------------------------------------------------------------

def get_props(
    date: str = "today",
    hours: int = 168,
    books: Optional[List[str]] = None,
    markets: Optional[List[str]] = None,
    *,
    selection: Optional[str] = None,
    teams: Optional[List[str]] = None,
    write_dir: str = "outputs",
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Main entry called by engine.

    Parameters expected by your pipeline (flexible):
      - date: 'today' or 'YYYY-MM-DD'
      - hours: forward window
      - books: list of bookmakers (keys: draftkings,fanduel,betmgm,caesars)
      - markets: list of your names or Odds API keys
      - selection: optional regex on event title (e.g., 'Chiefs|Jaguars')
      - teams: optional list of team substrings
      - write_dir: where to write 'oddsapi_credits.json'
    """
    if not ODDS_API_KEY:
        raise RuntimeError("Missing THE_ODDS_API_KEY in environment.")

    books_norm = _normalize_books(books)
    markets_norm = _normalize_markets(markets)

    LOG.info(f"books={books_norm}")
    LOG.info(f"markets={markets_norm}")

    # 1) Find slate events (ids) using cheap H2H call
    event_shells = _fetch_event_shells(
        date=date,
        hours=hours,
        books=books_norm,
        teams=_as_list(teams),
        selection=selection,
    )
    if not event_shells:
        LOG.info("No events in window after filters.")
        _print_and_write_credits(write_dir)
        return _build_dataframe([])

    # 2) For each event id, fetch props for requested markets
    all_rows: List[Dict[str, Any]] = []
    for sh in event_shells:
        try:
            payload = _fetch_event_props(
                event_id=sh["id"],
                books=books_norm,
                markets=markets_norm,
            )
            rows = _flat_props_rows(sh, payload)
            all_rows.extend(rows)
        except Exception as e:
            # keep going if a single game fails (rare)
            LOG.info(f"event {sh['id']} props fetch failed: {e}")

    # 3) Build DF and print/write credits
    df = _build_dataframe(all_rows)
    LOG.info(f"props rows: {len(df)}")
    _print_and_write_credits(write_dir)
    return df
