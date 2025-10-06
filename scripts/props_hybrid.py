# scripts/props_hybrid.py
from __future__ import annotations

import os
import re
import time
import json
import typing as T
import urllib.parse
import urllib.request
import urllib.error
from dataclasses import dataclass

ODDS_BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
DEFAULT_REGION = "us"
DEFAULT_ODDS_FORMAT = "american"
DEFAULT_DATE_FORMAT = "iso"

# batch size (small keeps OddsAPI happy)
ODDS_API_MARKETS_MAX = int(os.getenv("ODDS_API_MARKETS_MAX", "3"))

# map our aliases -> OddsAPI v4 market names
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

# sensible default markets
DEFAULT_MARKETS = [
    "player_receptions",
    "player_receiving_yards",
    "player_rushing_yards",
    "player_rushing_attempts",
    "player_passing_yards",
    "player_passing_tds",
    "player_touchdown_anytime",
]

def _log(msg: str) -> None:
    print(f"[props_hybrid] {msg}", flush=True)

def _split_csv_maybe_list(x: T.Optional[T.Union[str, T.List[str]]]) -> T.List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [i.strip() for i in x if isinstance(i, str) and i.strip()]
    s = str(x).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    parts = [p.strip().strip("'").strip('"') for p in s.split(",")]
    return [p for p in parts if p]

def _normalize_books(books: T.Optional[T.Union[str, T.List[str]]]) -> T.List[str]:
    items = _split_csv_maybe_list(books)
    items = [i.lower() for i in items]
    # drop 1–2 letter garbage
    return [i for i in items if len(i) > 2]

def _normalize_markets(markets: T.Optional[T.Union[str, T.List[str]]]) -> T.List[str]:
    raw = _split_csv_maybe_list(markets)
    if not raw:
        return DEFAULT_MARKETS[:]
    out: T.List[str] = []
    seen: set[str] = set()
    for m in raw:
        key = re.sub(r"[^a-z0-9_]", "", m.lower())
        mapped = MARKET_MAP.get(key, key)
        if mapped and mapped not in seen:
            seen.add(mapped)
            out.append(mapped)
    return out

def _http_get(url: str, retries: int = 2, sleep: float = 0.8) -> str:
    last_err: T.Optional[Exception] = None
    for _ in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return r.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8")
            except Exception:
                pass
            last_err = RuntimeError(f"HTTPError {e.code}: {e.reason} – body: {body[:800]}")
            time.sleep(sleep)
        except Exception as e:
            last_err = e
            time.sleep(sleep)
    raise RuntimeError(f"GET failed after retries: {url}\nDetail: {last_err}")

@dataclass
class OddsRow:
    raw: dict

def _fetch_from_oddsapi_one_batch(*, api_key: str, books: T.List[str], markets: T.List[str]) -> T.List[dict]:
    params = {
        "apiKey": api_key,
        "regions": DEFAULT_REGION,
        "oddsFormat": DEFAULT_ODDS_FORMAT,
        "dateFormat": DEFAULT_DATE_FORMAT,
        "markets": ",".join(markets),
    }
    if books:
        params["bookmakers"] = ",".join(books)
    url = ODDS_BASE + "?" + urllib.parse.urlencode(params, safe=",")
    _log(f"OddsAPI GET {url} [books={books} markets={','.join(markets)}]")
    payload = _http_get(url, retries=2, sleep=0.8)
    data = json.loads(payload)
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected OddsAPI payload: {data}")
    return data

def _chunk(lst: T.List[str], n: int) -> T.List[T.List[str]]:
    return [lst[i:i+n] for i in range(0, len(lst), n)]

def _parse_invalid_markets_from_error_text(err_text: str) -> T.List[str]:
    # looks for  'Invalid markets: a, b, c'
    m = re.search(r"Invalid markets:\s*([a-z0-9_,\s]+)", err_text, flags=re.I)
    if not m:
        return []
    found = m.group(1)
    parts = [p.strip().lower() for p in found.split(",")]
    parts = [MARKET_MAP.get(p, p) for p in parts]  # normalize aliases if present
    return [p for p in parts if p]

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

    if not markets_list:
        _log("No markets requested after normalization.")
        return []

    batches = _chunk(markets_list, max(1, ODDS_API_MARKETS_MAX))
    merged: dict[str, dict] = {}

    for batch in batches:
        try:
            data = _fetch_from_oddsapi_one_batch(api_key=api_key, books=books_list, markets=batch)
        except RuntimeError as e:
            s = str(e)
            invalid = _parse_invalid_markets_from_error_text(s)
            if invalid:
                # remove invalids from this batch and retry once
                keep = [m for m in batch if m.lower() not in set(invalid)]
                _log(f"OddsAPI reported invalid markets {invalid}; dropping and retrying batch -> {keep}")
                if not keep:
                    _log("Batch became empty after pruning invalid markets; skipping.")
                    continue
                data = _fetch_from_oddsapi_one_batch(api_key=api_key, books=books_list, markets=keep)
            else:
                # some other error – bubble up
                raise

        for ev in data:
            key = str(ev.get("id") or ev.get("eventId") or ev.get("commence_time") or json.dumps(ev, sort_keys=True))
            if key not in merged:
                merged[key] = ev
            else:
                a = merged[key]
                b = ev
                if isinstance(a.get("bookmakers"), list) and isinstance(b.get("bookmakers"), list):
                    a["bookmakers"].extend(b["bookmakers"])
                else:
                    merged[key] = ev

    return [OddsRow(raw=v) for v in merged.values()]

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
    odds_rows = _fetch_from_oddsapi(
        date=date,
        season=season,
        hours=hours,
        books=books,
        markets=markets,
        order=order,
    )
    return [r.raw for r in odds_rows]
