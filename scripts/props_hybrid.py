# scripts/props_hybrid.py
from __future__ import annotations

import os
import time
import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests


# -----------------------------
# Config / helpers
# -----------------------------

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "americanfootball_nfl"
PLAYER_PROPS_ENDPOINT = f"{ODDS_API_BASE}/sports/{SPORT_KEY}/players/props"

DEFAULT_BOOKS = ["draftkings", "fanduel", "betmgm", "caesars"]

# You can request many markets; these are commonly supported:
# (This list is intentionally conservative to avoid INVALID_MARKET)
DEFAULT_MARKETS = [
    "player_receptions",
    "player_receiving_yards",
    "player_rush_yards",
    "player_rush_attempts",
    "player_pass_yards",
    "player_pass_tds",
    "player_touchdown_anytime",
    # If you want more, add carefully and test:
    # "player_pass_attempts",
    # "player_pass_completions",
    # "player_pass_interceptions",
]

OUTPUTS_DIR = "outputs"
CREDITS_FILE = os.path.join(OUTPUTS_DIR, "oddsapi_credits.json")


def _ensure_outputs_dir() -> None:
    try:
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
    except Exception:
        pass


def _log(msg: str) -> None:
    print(f"[props_hybrid] {msg}")


def _odds_api_key() -> str:
    key = os.getenv("THE_ODDS_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Missing THE_ODDS_API_KEY in environment.")
    return key


def _http_get(url: str, params: Dict[str, Any], retries: int = 2, sleep: float = 0.8) -> Tuple[Dict, Dict[str, str]]:
    """
    GET with small retry. Returns (json, headers)
    Raises RuntimeError with response text on failure.
    """
    last_err = None
    for i in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                try:
                    return r.json(), {k.lower(): v for k, v in r.headers.items()}
                except Exception as e:
                    raise RuntimeError(f"Failed to parse JSON: {e}. Body[:500]={r.text[:500]}")
            else:
                last_err = f"{r.status_code} {r.text[:500]}"
        except Exception as e:
            last_err = str(e)
        time.sleep(sleep)
    raise RuntimeError(f"GET failed after retries: {url}\nDetail: {last_err}")


def _read_credits_from_headers(headers: Dict[str, str]) -> Dict[str, Any]:
    """
    The Odds API returns usage headers:
      X-Requests-Used, X-Requests-Remaining, X-Requests-Limit (names are case-insensitive)
    We lower-case headers earlier, so read lower keys.
    """
    used = headers.get("x-requests-used")
    remaining = headers.get("x-requests-remaining")
    limit_ = headers.get("x-requests-limit")
    return {
        "requests_used": int(used) if (used and used.isdigit()) else used,
        "requests_remaining": int(remaining) if (remaining and remaining.isdigit()) else remaining,
        "requests_limit": int(limit_) if (limit_ and limit_.isdigit()) else limit_,
    }


# -----------------------------
# Odds API: player props fetch
# -----------------------------

def _fetch_from_oddsapi(
    date: Optional[str] = None,
    hours: Optional[int] = None,
    books: Optional[Iterable[str]] = None,
    markets: Optional[Iterable[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Hit the The Odds API PLAYER PROPS endpoint and return a list of props (raw json rows)
    plus a small 'meta' dict that includes usage/credits.

    Notes:
    - `date` and `hours` are accepted for compatibility with the engine signature,
      but The Odds API player-props endpoint does not filter by those directly.
      (You can post-filter by commence_time if you want; we leave that to the engine.)
    """
    api_key = _odds_api_key()

    books_list = list(books) if books else DEFAULT_BOOKS
    mkts_list = list(markets) if markets else DEFAULT_MARKETS

    params = {
        "apiKey": api_key,
        "regions": "us",
        "bookmakers": ",".join(books_list),
        "markets": ",".join(mkts_list),
        "oddsFormat": "american",
        "dateFormat": "iso",
    }

    _log(f"OddsAPI GET {PLAYER_PROPS_ENDPOINT} books={books_list} markets={mkts_list}")
    payload, headers = _http_get(PLAYER_PROPS_ENDPOINT, params=params)

    # Credits/usage info
    credits = _read_credits_from_headers(headers)
    _ensure_outputs_dir()
    try:
        with open(CREDITS_FILE, "w", encoding="utf-8") as f:
            json.dump(credits, f, indent=2)
    except Exception:
        pass

    # Also print a friendly line
    _log(f"Credits: used={credits.get('requests_used')} "
         f"remaining={credits.get('requests_remaining')} "
         f"limit={credits.get('requests_limit')}")

    # Payload is a list of games or props grouped by event. We just return it; we’ll
    # flatten to a DataFrame in `get_props`.
    return payload, credits


# -----------------------------
# Normalization
# -----------------------------

def _flatten_props_to_df(payload: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert Odds API /players/props payload into a tidy DataFrame.

    Output columns (stable, safe for further joins):
      - event_id
      - commence_time
      - home_team
      - away_team
      - bookmaker
      - market
      - player
      - outcome_name        (e.g., 'Over', 'Under', 'Yes')
      - point               (line)
      - price               (American odds)
    """
    rows: List[Dict[str, Any]] = []

    for ev in payload or []:
        event_id = ev.get("id") or ev.get("event_id")
        commence_time = ev.get("commence_time")
        home = ev.get("home_team")
        away = ev.get("away_team")

        for book in (ev.get("bookmakers") or []):
            bookmaker = book.get("key") or book.get("title") or ""
            for mk in (book.get("markets") or []):
                market_key = mk.get("key") or mk.get("name") or ""
                for outcome in (mk.get("outcomes") or []):
                    row = {
                        "event_id": event_id,
                        "commence_time": commence_time,
                        "home_team": home,
                        "away_team": away,
                        "bookmaker": bookmaker,
                        "market": market_key,
                        "player": outcome.get("description") or outcome.get("name") or outcome.get("player") or "",
                        "outcome_name": outcome.get("name") or outcome.get("label") or "",
                        "point": outcome.get("point"),
                        "price": outcome.get("price"),
                    }
                    rows.append(row)

    df = pd.DataFrame(rows)
    # Basic hygiene
    if not df.empty:
        # normalize strings
        for c in ["bookmaker", "market", "player", "outcome_name", "home_team", "away_team"]:
            if c in df.columns:
                df[c] = df[c].astype(str)
    return df


# -----------------------------
# Public entry point
# -----------------------------

def get_props(
    *,
    date: Optional[str] = None,
    season: Optional[str] = None,
    window: Optional[int] = None,
    books: Optional[Iterable[str]] = None,
    markets: Optional[Iterable[str]] = None,
    order: Optional[str] = None,
    teams: Optional[Iterable[str]] = None,
    events: Optional[Iterable[str]] = None,
    selection: Optional[str] = None,
) -> pd.DataFrame:
    """
    Engine calls this with kwargs. We only use what applies to the Odds API fetch.

    Returns a pandas DataFrame of raw player props,
    and writes credits info to outputs/oddsapi_credits.json.
    """
    # Fetch raw payload + credits
    payload, credits = _fetch_from_oddsapi(
        date=date,
        hours=window,
        books=books,
        markets=markets,
    )

    df = _flatten_props_to_df(payload)

    # Optional: post-filter by team substrings (if provided)
    if teams:
        terms = [str(t).strip().lower() for t in (teams if isinstance(teams, (list, tuple, set)) else [teams])]
        def _keep_team(row: pd.Series) -> bool:
            bucket = f"{row.get('home_team','')}".lower() + " " + f"{row.get('away_team','')}".lower()
            return any(t in bucket for t in terms)
        if not df.empty:
            df = df[df.apply(_keep_team, axis=1)]

    # Optional: selection regex substring on player/market/outcome (simple contains)
    if selection:
        s = str(selection).lower()
        if not df.empty:
            df = df[
                df["player"].str.lower().str.contains(s, na=False)
                | df["market"].str.lower().str.contains(s, na=False)
                | df["outcome_name"].str.lower().str.contains(s, na=False)
            ]

    _log(f"normalized props rows: {len(df)}")
    return df
