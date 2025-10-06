# scripts/props_hybrid.py
from __future__ import annotations
import os, time, typing as t
import requests
import pandas as pd

from scripts.market_keys import NFL_DEFAULT_MARKETS, MARKET_SYNONYMS, VALID_MARKETS

BASE = "https://api.the-odds-api.com/v4"
NFL_SPORT = "americanfootball_nfl"

DEFAULT_REGIONS = "us"              # your account allows US books
DEFAULT_ODDS_FORMAT = "american"
DEFAULT_DATEFORMAT = "iso"

# ------------ utilities ------------
def _log(msg: str) -> None:
    print(f"[props_hybrid] {msg}")

def _env_api_key() -> str | None:
    return os.environ.get("THE_ODDS_API_KEY") or os.environ.get("ODDS_API_KEY")

def _parse_books(books: str | list[str] | None) -> str:
    if books is None:
        return ""
    if isinstance(books, list):
        return ",".join([b.strip() for b in books if b.strip()])
    s = str(books).strip()
    # remove brackets/quotes if someone passed a python-list string
    s = s.strip("[]").replace("'", "").replace('"', "")
    # collapse multiple commas/spaces
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return ",".join(parts)

def _to_list_csv(v: str | list[str] | None) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [x.strip() for x in v if str(x).strip()]
    return [x.strip() for x in str(v).split(",") if x.strip()]

def _print_credits(headers: requests.structures.CaseInsensitiveDict | dict | None) -> None:
    if not headers:
        return
    used = headers.get("x-requests-used")
    rem  = headers.get("x-requests-remaining")
    reset= headers.get("x-requests-reset")
    if used is not None and rem is not None:
        _log(f"credits | used={used} remaining={rem} reset={reset}")

def _http_get(url: str, params: dict, retries: int = 2, backoff: float = 0.8) -> tuple[dict | list | None, dict]:
    last = None
    for i in range(retries + 1):
        r = requests.get(url, params=params, timeout=30)
        _print_credits(r.headers)
        if r.status_code == 200:
            try:
                return r.json(), r.headers
            except Exception:
                return None, r.headers
        last = (r.status_code, r.text)
        time.sleep(backoff * (i + 1))
    _log(f"GET failed after retries: {url}?{params} | detail: {last}")
    return None, {}

def _has_outcomes(payload: dict) -> bool:
    for bk in (payload.get("bookmakers") or []):
        for mk in (bk.get("markets") or []):
            if mk.get("outcomes"):
                return True
    return False

# ------------ fetching ------------
def _fetch_event_ids(api_key: str, books_csv: str) -> list[dict]:
    """
    Use the sports odds endpoint (e.g., markets=h2h) just to enumerate event shells.
    """
    url = f"{BASE}/sports/{NFL_SPORT}/odds"
    params = dict(
        apiKey=api_key,
        regions=DEFAULT_REGIONS,
        oddsFormat=DEFAULT_ODDS_FORMAT,
        dateFormat=DEFAULT_DATEFORMAT,
        bookmakers=books_csv or "draftkings, fanduel",
        markets="h2h",
    )
    data, hdrs = _http_get(url, params)
    if not isinstance(data, list):
        return []
    _log(f"event shells fetched: {len(data)}")
    return data

def _fetch_event_props(api_key: str, event_id: str, books_csv: str, markets: list[str]) -> list[dict]:
    """
    Fetch props for one event. Try bulk first; if empty, try each market with synonyms.
    """
    if not markets:
        return []
    base_url = f"{BASE}/sports/{NFL_SPORT}/events/{event_id}/odds"
    common = dict(
        apiKey=api_key,
        regions=DEFAULT_REGIONS,
        oddsFormat=DEFAULT_ODDS_FORMAT,
        dateFormat=DEFAULT_DATEFORMAT,
        bookmakers=books_csv,
    )

    # 1) bulk try
    params = dict(common, markets=",".join(markets))
    bulk, hdrs = _http_get(base_url, params)
    if isinstance(bulk, dict):
        bulk.setdefault("eventId", event_id)
        if _has_outcomes(bulk):
            _log(f"event {event_id}: outcomes (bulk) ✓")
            return [bulk]
        _log(f"event {event_id}: no outcomes in bulk → retrying per-market")

    # 2) per-market with synonyms
    agg: list[dict] = []
    for canonical in markets:
        synonyms = MARKET_SYNONYMS.get(canonical, [canonical])
        hit = False
        for m in synonyms:
            p = dict(common, markets=m)
            d, h = _http_get(base_url, p)
            if isinstance(d, dict):
                d.setdefault("eventId", event_id)
                if _has_outcomes(d):
                    _log(f"event {event_id}: outcomes for '{m}' ✓")
                    agg.append(d)
                    hit = True
                    break
        if not hit:
            _log(f"event {event_id}: still no outcomes for '{canonical}'")
    if not agg:
        _log(f"event {event_id}: still no outcomes for requested markets")
    return agg

# ------------ flattening ------------
def _flatten(payloads: list[dict]) -> pd.DataFrame:
    """
    Flatten event -> bookmakers -> markets -> outcomes to rows.
    """
    rows: list[dict] = []
    for ev in payloads:
        event_id = ev.get("id") or ev.get("eventId")
        commence_time = ev.get("commence_time") or ev.get("commenceTime")
        home = ev.get("home_team")
        away = ev.get("away_team")
        for bk in (ev.get("bookmakers") or []):
            book = bk.get("key")
            for mk in (bk.get("markets") or []):
                market_key = mk.get("key")
                for oc in (mk.get("outcomes") or []):
                    rows.append(
                        dict(
                            event_id=event_id,
                            commence_time=commence_time,
                            home_team=home,
                            away_team=away,
                            book=book,
                            market=market_key,
                            name=oc.get("name"),
                            price=oc.get("price"),
                            point=oc.get("point"),
                        )
                    )
    return pd.DataFrame(rows)

# ------------ public API ------------
def get_props(
    *,
    api_key: str | None = None,
    date: str | None = None,
    season: int | None = None,
    window: int | str | None = None,
    cap: int = 0,
    markets: list[str] | str | None = None,
    books: list[str] | str | None = None,
    order: str = "odds",
    team_filter: list[str] | None = None,
    selection: str | None = None,
    event_ids: list[str] | None = None,
) -> pd.DataFrame:
    """
    Unified fetcher used by engine.py. Returns a DataFrame (can be empty).
    """
    key = api_key or _env_api_key()
    if not key:
        raise RuntimeError("Missing THE_ODDS_API_KEY in environment.")

    # Markets
    if markets is None:
        req_markets = list(NFL_DEFAULT_MARKETS)
        _log(f"no markets passed; using defaults: {','.join(req_markets)}")
    else:
        if isinstance(markets, str):
            req_markets = [m.strip() for m in markets.split(",") if m.strip()]
        else:
            req_markets = [m.strip() for m in markets if str(m).strip()]
        # filter only those we know (canonical or synonyms)
        req_markets = [m for m in req_markets if m in VALID_MARKETS or m in MARKET_SYNONYMS]
        if not req_markets:
            req_markets = list(NFL_DEFAULT_MARKETS)
            _log(f"requested markets invalid → using defaults: {','.join(req_markets)}")

    # Books
    books_csv = _parse_books(books) or "draftkings,fanduel"

    # Events: use supplied or discover from shells
    shells = _fetch_event_ids(key, books_csv) if not event_ids else []
    if event_ids:
        event_list = [{"id": e} for e in event_ids]
    else:
        # event shells from odds endpoint
        event_list = [{"id": x.get("id")} for x in shells if x.get("id")]
    _log(f"events queued: {len(event_list)}")

    # Cap number of events if requested
    if cap and cap > 0:
        event_list = event_list[:cap]

    # Fetch props per event
    payloads: list[dict] = []
    for ev in event_list:
        ev_id = ev["id"]
        ev_payloads = _fetch_event_props(key, ev_id, books_csv, req_markets)
        payloads.extend(ev_payloads)

    # Flatten
    df = _flatten(payloads)
    _log(f"flattened {len(df)} rows")
    return df

