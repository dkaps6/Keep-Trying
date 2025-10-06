# scripts/props_hybrid.py
from __future__ import annotations

import os
import re
import time
import json
import typing as T
import urllib.parse
import urllib.request
from dataclasses import dataclass

# -----------------------------
# Config / constants
# -----------------------------

ODDS_BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
DEFAULT_REGION = "us"
DEFAULT_ODDS_FORMAT = "american"
DEFAULT_DATE_FORMAT = "iso"

# Accept the aliases your model uses and map them to OddsAPI v4 keys
MARKET_MAP = {
    # receiving / receptions
    "player_receiving_yds": "player_receiving_yards",
    "receiving_yards": "player_receiving_yards",
    "rec_yards": "player_receiving_yards",
    "player_receptions": "player_receptions",
    "receptions": "player_receptions",

    # rushing
    "player_rush_yds": "player_rushing_yards",
    "rushing_yards": "player_rushing_yards",
    "rush_yards": "player_rushing_yards",
    "player_rush_attempts": "player_rushing_attempts",
    "rushing_attempts": "player_rushing_attempts",
    "rush_attempts": "player_rushing_attempts",

    # passing
    "player_pass_yds": "player_passing_yards",
    "passing_yards": "player_passing_yards",
    "pass_yards": "player_passing_yards",

    "player_pass_tds": "player_passing_tds",
    "passing_tds": "player_passing_tds",
    "pass_tds": "player_passing_tds",

    "player_pass_attempts": "player_passing_attempts",
    "pass_attempts": "player_passing_attempts",

    "player_pass_completions": "player_passing_completions",
    "pass_completions": "player_passing_completions",

    "player_pass_interceptions": "player_passing_interceptions",
    "pass_interceptions": "player_passing_interceptions",

    # anytime TD
    "player_anytime_td": "player_touchdown_anytime",
    "anytime_td": "player_touchdown_anytime",
    "player_touchdown_anytime": "player_touchdown_anytime",
}

# a compact default set covering your main markets
DEFAULT_MARKETS = [
    "player_receptions",
    "player_receiving_yards",
    "player_rushing_yards",
    "player_rushing_attempts",
    "player_passing_yards",
    "player_passing_tds",
    "player_touchdown_anytime",
]


# -----------------------------
# Helpers
# -----------------------------

def _log(msg: str) -> None:
    print(f"[props_hybrid] {msg}", flush=True)


def _split_csv_maybe_list(x: T.Optional[T.Union[str, T.List[str]]]) -> T.List[str]:
    """Accept list[str] or comma-separated str; return list[str] cleaned."""
    if x is None:
        return []
    if isinstance(x, list):
        # Make sure there are no inner bracket/quote artifacts
        return [i.strip() for i in x if isinstance(i, str) and i.strip()]
    s = str(x).strip()
    if not s:
        return []
    # If someone passed a stringified list "['dk','fd']" -> strip brackets/quotes first
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    parts = [p.strip().strip("'").strip('"') for p in s.split(",")]
    return [p for p in parts if p]


def _normalize_books(books: T.Optional[T.Union[str, T.List[str]]]) -> T.List[str]:
    # turn into list; allow common bookmaker keys
    items = _split_csv_maybe_list(books)
    # sometimes people pass "draftkings, fanduel" etc – just lower-case
    items = [i.lower() for i in items]
    # remove accidental single letters (symptom of string join on a repr)
    items = [i for i in items if len(i) > 2]
    return items


def _normalize_markets(markets: T.Optional[T.Union[str, T.List[str]]]) -> T.List[str]:
    raw = _split_csv_maybe_list(markets)
    if not raw:
        return DEFAULT_MARKETS[:]

    normalized: T.List[str] = []
    for m in raw:
        key = m.strip().lower()
        # strip any accidental punctuation
        key = re.sub(r"[^a-z0-9_]", "", key)
        # map alias -> OddsAPI key if we know it
        mapped = MARKET_MAP.get(key, key)
        normalized.append(mapped)

    # keep only unique, preserve order
    seen: set[str] = set()
    out: T.List[str] = []
    for m in normalized:
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _http_get(url: str, retries: int = 2, sleep: float = 0.8) -> str:
    last_err: T.Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return r.read().decode("utf-8")
        except Exception as e:
            last_err = e
            time.sleep(sleep)
    raise RuntimeError(f"GET failed after retries: {url}\nDetail: {last_err}")


# -----------------------------
# Core fetcher
# -----------------------------

@dataclass
class OddsRow:
    raw: dict


def _fetch_from_oddsapi(
    *,
    date: str,
    season: str,
    hours: int,
    books: T.Optional[T.Union[str, T.List[str]]] = None,
    markets: T.Optional[T.Union[str, T.List[str]]] = None,
    order: T.Optional[str] = None,
) -> T.List[OddsRow]:
    api_key = os.getenv("THE_ODDS_API_KEY") or os.getenv("ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("Missing THE_ODDS_API_KEY (or ODDS_API_KEY) in environment.")

    books_list = _normalize_books(books)
    markets_list = _normalize_markets(markets)

    # build query parameters
    params = {
        "apiKey": api_key,
        "regions": DEFAULT_REGION,
        "oddsFormat": DEFAULT_ODDS_FORMAT,
        "dateFormat": DEFAULT_DATE_FORMAT,
        "markets": ",".join(markets_list),
    }
    if books_list:
        params["bookmakers"] = ",".join(books_list)

    # Optional ordering (OddsAPI may ignore; harmless)
    if order:
        params["sort"] = order

    base_url = ODDS_BASE + "?" + urllib.parse.urlencode(params, safe=",")
    _log(f"OddsAPI GET {base_url} [books={books_list} markets={','.join(markets_list)}]")

    payload = _http_get(base_url, retries=2, sleep=0.8)
    try:
        data = json.loads(payload)
    except Exception:
        _log(f"Non-JSON response:\n{payload[:500]}")
        raise

    # Some OddsAPI responses are lists of events with nested markets
    if not isinstance(data, list):
        # OddsAPI returns {'message': '...', 'error_code': ...} on errors
        raise RuntimeError(f"Unexpected OddsAPI payload: {data}")

    rows: T.List[OddsRow] = [OddsRow(raw=item) for item in data]
    return rows


# -----------------------------
# Public surface for engine
# -----------------------------

def get_props(
    *,
    date: str,
    season: str,
    hours: int = 36,
    cap: int = 0,
    books: T.Optional[T.Union[str, T.List[str]]] = None,
    markets: T.Optional[T.Union[str, T.List[str]]] = None,
    order: T.Optional[str] = None,
    **kwargs,
) -> list:
    """
    Return a list-like structure the rest of your pipeline expects.
    This function is called by engine._import_odds_fetcher() wrapper.
    """
    odds_rows = _fetch_from_oddsapi(
        date=date,
        season=season,
        hours=hours,
        books=books,
        markets=markets,
        order=order,
    )

    # If your downstream expects a DataFrame, you can convert here.
    # For now we just pass raw rows; your engine already handles None/empty.
    # Convert to list[dict] for easy merging in the engine.
    return [r.raw for r in odds_rows]
