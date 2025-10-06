# scripts/odds_api_v4.py
# OddsAPI v4 adapter (non-destructive): correct /odds endpoint, bookmaker mapping,
# window handling, one-pass filters, tidy DataFrame output.
#
# Requires: env ODDS_API_KEY
# Exposes:  fetch_props(), get_props(), get_props_df(), _fetch_events_from_provider(), events_to_dataframe()

from __future__ import annotations

import os
import re
import time
import json
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

# ----------------------- Config -----------------------

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()
SPORT_KEY = "americanfootball_nfl"                # OddsAPI sport key (underscore)
BASE_ODDS_URL = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds"

# Friendly → OddsAPI market keys (expand as you wish)
MARKET_MAP = {
    "player_receptions": "player_receptions",
    "player_receiving_yds": "player_receiving_yards",
    "player_rush_yds": "player_rushing_yards",
    "player_rush_attempts": "player_rushing_attempts",
    "player_pass_yds": "player_passing_yards",
    "player_pass_tds": "player_passing_tds",
    "player_anytime_td": "player_touchdown_anytime",
}
DEFAULT_MARKETS = list(MARKET_MAP.keys())

# Accept shorthand bookmaker names
BOOKMAP = {
    "dk": "draftkings", "draftkings": "draftkings",
    "fd": "fanduel", "fanduel": "fanduel",
    "mgm": "betmgm", "betmgm": "betmgm",
    "cz": "caesars", "czrs": "caesars", "caesars": "caesars",
    # add others here if you want them available
}

# ----------------------- Helpers -----------------------

def _to_hours(hours=None, window=None, lookahead=None, default: int = 24) -> int:
    for v in (hours, window, lookahead):
        if v is None:
            continue
        s = str(v).strip().lower()
        if s.endswith("h"):
            s = s[:-1]
        try:
            return int(float(s))
        except Exception:
            continue
    return int(default)

def _parse_date_anchor(date: Optional[str]) -> datetime:
    """‘today’ or None → today@UTC midnight; ‘YYYY-MM-DD’ → that date@UTC midnight."""
    if not date or str(date).strip().lower() == "today":
        dt = datetime.now(timezone.utc)
        return datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)
    try:
        y, m, d = [int(x) for x in str(date).split("-")]
        return datetime(y, m, d, tzinfo=timezone.utc)
    except Exception:
        dt = datetime.now(timezone.utc)
        return datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)

def _http_get(url: str, params: Dict[str, Any]) -> Any:
    """GET with retries + clear error text."""
    if not ODDS_API_KEY:
        raise RuntimeError("ODDS_API_KEY env variable is not set.")
    q = params.copy()
    q["apiKey"] = ODDS_API_KEY
    full = f"{url}?{urllib.parse.urlencode(q, doseq=True)}"

    backoff = 1.0
    last_err = ""
    for _ in range(4):
        try:
            with urllib.request.urlopen(full, timeout=30) as resp:
                data = resp.read()
                return json.loads(data.decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8", errors="ignore")
            except Exception:
                body = ""
            last_err = f"HTTP {e.code} {e.reason}: {body}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(backoff)
        backoff = min(8.0, backoff * 2.0)
    raise RuntimeError(f"GET failed after retries: {full}\nDetail: {last_err}")

# ----------------------- Filters -----------------------

def _apply_selection_filters(
    events_list: Iterable[Dict[str, Any]],
    teams: Optional[Iterable[str]] = None,
    events: Optional[Iterable[str]] = None,
    selection: Optional[str] = None,
):
    # selection
    sel_regex = None
    if selection is not None:
        s = str(selection).strip()
        if s != "":
            try:
                sel_regex = re.compile(s, re.IGNORECASE)
            except Exception:
                sel_regex = re.compile(re.escape(s), re.IGNORECASE)

    def _sel_ok(ev):
        if sel_regex is None:
            return True
        name = str(ev.get("name") or ev.get("title") or "")
        return bool(sel_regex.search(name))

    # teams
    team_terms = None
    if teams is not None:
        if isinstance(teams, str):
            teams = [t.strip() for t in teams.split(",") if t.strip()]
        if isinstance(teams, (list, tuple)) and len(teams) > 0 and str(teams).lower() not in ("all",):
            team_terms = [t.lower() for t in teams]

    def _team_ok(ev):
        if team_terms is None:
            return True
        home = str(ev.get("home_team", "")).lower()
        away = str(ev.get("away_team", "")).lower()
        name = str(ev.get("name") or ev.get("title") or "").lower()
        bucket = f"{home} {away} {name}"
        return any(term in bucket for term in team_terms)

    # ids
    event_ids = None
    if events is not None:
        if isinstance(events, str):
            events = [e.strip() for e in events.split(",") if e.strip()]
        if isinstance(events, (list, tuple)) and len(events) > 0 and str(events).lower() not in ("all",):
            event_ids = set(events)

    def _id_ok(ev):
        if event_ids is None:
            return True
        eid = ev.get("id") or ev.get("event_id") or ev.get("key")
        return (eid in event_ids)

    return [ev for ev in events_list if _sel_ok(ev) and _team_ok(ev) and _id_ok(ev)]

# ---------------- Provider (OddsAPI v4 /odds) ----------------

def _fetch_events_from_provider(
    date: Optional[str] = None,
    season: Optional[int] = None,
    hours: Optional[int] = None,
    books: str = "draftkings",
    markets: Optional[List[str] | str] = None,
    order: str = "odds",
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Call OddsAPI v4 /odds which returns games with bookmakers + markets.
    Returns a list of event dicts with props attached.
    """
    hrs = int(hours or 24)
    anchor = _parse_date_anchor(date)
    start = anchor
    end = anchor + timedelta(hours=hrs)

    # normalize books
    book_list: List[str] = []
    if books:
        if isinstance(books, str):
            book_list = [b.strip() for b in books.split(",") if b.strip()]
        elif isinstance(books, (list, tuple)):
            book_list = list(books)
    book_list = [BOOKMAP.get(b.lower(), b.lower()) for b in book_list if b]
    books_param = ",".join(book_list) if book_list else None

    # normalize markets
    if markets is None or (isinstance(markets, str) and markets.strip() == ""):
        wanted_markets = DEFAULT_MARKETS
    elif isinstance(markets, str):
        wanted_markets = [m.strip() for m in markets.split(",") if m.strip()]
    else:
        wanted_markets = list(markets)
    odds_markets = [MARKET_MAP.get(m, m) for m in wanted_markets]

    params = {
        "regions": "us",
        "oddsFormat": "american",
        "dateFormat": "iso",
        "markets": ",".join(odds_markets),
        # OddsAPI /odds does not take a direct time window filter; we’ll post-filter by commence_time.
    }
    if books_param:
        params["bookmakers"] = books_param

    data = _http_get(BASE_ODDS_URL, params)
    if not isinstance(data, list):
        data = []

    # Build events with attached props
    events: List[Dict[str, Any]] = []
    for g in data:
        try:
            # time filter (OddsAPI returns upcoming; we guard by window)
            ct = g.get("commence_time")
            # basic ISO parsing; if any failure, keep it
            keep = True
            try:
                if ct:
                    dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                    keep = (start <= dt <= end)
            except Exception:
                pass
            if not keep:
                continue

            ev = {
                "id": g.get("id"),
                "name": f"{g.get('away_team')} @ {g.get('home_team')}",
                "home_team": g.get("home_team"),
                "away_team": g.get("away_team"),
                "commence_time": g.get("commence_time"),
                "props": [],
            }
            # bookmakers → markets → outcomes
            for bk in g.get("bookmakers", []) or []:
                book_key = (bk.get("key") or "").lower()
                for mk in bk.get("markets", []) or []:
                    market_key = mk.get("key")
                    for oc in mk.get("outcomes", []) or []:
                        # For stat props, outcomes usually have name "Over"/"Under" with a point (line)
                        # Some player markets name outcome with player string; we capture generously.
                        ev["props"].append({
                            "event_id": ev["id"],
                            "player": oc.get("description") or oc.get("participant") or oc.get("name"),
                            "market": market_key,
                            "bookmaker": book_key,
                            "line": oc.get("point"),
                            # If an outcome is Over/Under, price is for that side; we keep single price
                            # Your normalizer can reshape to over/under rows as needed.
                            "price_name": oc.get("name"),
                            "price": oc.get("price"),
                        })
            events.append(ev)
        except Exception:
            continue

    return events

# ---------------- Events → DataFrame ----------------

def events_to_dataframe(events: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for ev in events:
        for p in ev.get("props", []):
            rows.append({
                "event_id": ev.get("id"),
                "event_name": ev.get("name"),
                "home_team": ev.get("home_team"),
                "away_team": ev.get("away_team"),
                "commence_time": ev.get("commence_time"),
                "player": p.get("player"),
                "market": p.get("market"),
                "book": p.get("bookmaker"),
                "price_name": p.get("price_name"),
                "vegas_line": p.get("line"),
                "vegas_price": p.get("price"),
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    inv = {v: k for k, v in MARKET_MAP.items()}
    df["market"] = df["market"].map(lambda m: inv.get(str(m), str(m)))
    return df

# ---------------- Public API ----------------

def fetch_props(
    date: Optional[str] = None,
    season: Optional[int] = None,
    window: Optional[str | int] = None,
    hours: Optional[int] = None,
    lookahead: Optional[int] = None,
    cap: int = 0,
    markets: Optional[List[str] | str] = None,
    order: str = "odds",
    books: str = "draftkings",
    team_filter: Optional[List[str] | str] = None,
    selection: Optional[str] = None,
    event_ids: Optional[List[str] | str] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    hrs = _to_hours(hours=hours, window=window, lookahead=lookahead, default=24)
    print(f"[odds_api] window={hrs}h cap={cap}")
    mstr = ",".join(markets) if isinstance(markets, (list, tuple)) else (markets or ",".join(DEFAULT_MARKETS))
    print(f"markets={mstr}\norder={order},{books}")

    events = _fetch_events_from_provider(
        date=date, season=season, hours=hrs, books=books, markets=markets, order=order, **kwargs
    ) or []

    # Apply filters once
    events = _apply_selection_filters(events_list=events, teams=team_filter, events=event_ids, selection=selection)
    if cap and cap > 0 and isinstance(events, list):
        events = events[:cap]

    df = events_to_dataframe(events)
    return df

# Aliases expected by engine
def get_props(*args, **kwargs) -> pd.DataFrame:
    return fetch_props(*args, **kwargs)
def get_props_df(*args, **kwargs) -> pd.DataFrame:
    return fetch_props(*args, **kwargs)
