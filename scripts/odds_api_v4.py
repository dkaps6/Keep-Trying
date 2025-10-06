# scripts/odds_api_v4.py
from __future__ import annotations

import os, re, time, json, urllib.parse, urllib.request
from typing import Any, Dict, Iterable, List, Optional
from datetime import datetime, timezone
import pandas as pd

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()
SPORT_KEY = "americanfootball_nfl"
BASE_ODDS_URL = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds"

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

BOOKMAP = {
    "dk": "draftkings", "draftkings": "draftkings",
    "fd": "fanduel", "fanduel": "fanduel",
    "mgm": "betmgm", "betmgm": "betmgm",
    "cz": "caesars", "czrs": "caesars", "caesars": "caesars",
}

def _http_get(url: str, params: Dict[str, Any]) -> Any:
    if not ODDS_API_KEY:
        raise RuntimeError("ODDS_API_KEY env variable is not set.")
    q = params.copy(); q["apiKey"] = ODDS_API_KEY
    full = f"{url}?{urllib.parse.urlencode(q, doseq=True)}"
    last_err = ""
    backoff = 1.0
    for _ in range(4):
        try:
            with urllib.request.urlopen(full, timeout=30) as resp:
                data = resp.read()
                return json.loads(data.decode("utf-8"))
        except urllib.error.HTTPError as e:
            try: body = e.read().decode("utf-8", errors="ignore")
            except Exception: body = ""
            last_err = f"HTTP {e.code} {e.reason}: {body}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(backoff); backoff = min(8.0, backoff * 2.0)
    raise RuntimeError(f"GET failed after retries: {full}\nDetail: {last_err}")

def _apply_selection_filters(events_list: Iterable[Dict[str, Any]],
                             teams: Optional[Iterable[str]] = None,
                             events: Optional[Iterable[str]] = None,
                             selection: Optional[str] = None):
    sel_regex = None
    if selection is not None and str(selection).strip() != "":
        try: sel_regex = re.compile(str(selection).strip(), re.IGNORECASE)
        except Exception: sel_regex = re.compile(re.escape(str(selection).strip()), re.IGNORECASE)

    def _sel_ok(ev):
        if sel_regex is None: return True
        name = str(ev.get("name") or ev.get("title") or "")
        return bool(sel_regex.search(name))

    team_terms = None
    if teams is not None:
        if isinstance(teams, str):
            teams = [t.strip() for t in teams.split(",") if t.strip()]
        if isinstance(teams, (list, tuple)) and len(teams) > 0 and str(teams).lower() not in ("all",):
            team_terms = [t.lower() for t in teams]

    def _team_ok(ev):
        if team_terms is None: return True
        home = str(ev.get("home_team", "")).lower()
        away = str(ev.get("away_team", "")).lower()
        name = str(ev.get("name") or ev.get("title") or "").lower()
        return any(t in f"{home} {away} {name}" for t in team_terms)

    event_ids = None
    if events is not None:
        if isinstance(events, str):
            events = [e.strip() for e in events.split(",") if e.strip()]
        if isinstance(events, (list, tuple)) and len(events) > 0 and str(events).lower() not in ("all",):
            event_ids = set(events)

    def _id_ok(ev):
        if event_ids is None: return True
        eid = ev.get("id") or ev.get("event_id") or ev.get("key")
        return eid in event_ids

    return [ev for ev in events_list if _sel_ok(ev) and _team_ok(ev) and _id_ok(ev)]

def _normalize_books(books: Optional[str | List[str]]) -> Optional[str]:
    if not books: return None
    if isinstance(books, str):
        bl = [b.strip() for b in books.split(",") if b.strip()]
    else:
        bl = list(books)
    bl = [BOOKMAP.get(b.lower(), b.lower()) for b in bl if b]
    return ",".join(bl) if bl else None

def _normalize_markets(markets: Optional[str | List[str]]) -> List[str]:
    if markets is None or (isinstance(markets, str) and markets.strip() == ""):
        wanted = DEFAULT_MARKETS
    elif isinstance(markets, str):
        wanted = [m.strip() for m in markets.split(",") if m.strip()]
    else:
        wanted = list(markets)
    return [MARKET_MAP.get(m, m) for m in wanted]

def _fetch_events_from_provider(date=None, season=None, hours=None,
                                books: str = "draftkings",
                                markets: Optional[List[str] | str] = None,
                                order: str = "odds", **kwargs) -> List[Dict[str, Any]]:
    # OddsAPI /odds returns upcoming games only; we won't hard-filter by time here.
    books_param = _normalize_books(books)
    odds_markets = _normalize_markets(markets)

    params = {
        "regions": "us",
        "oddsFormat": "american",
        "dateFormat": "iso",
        "markets": ",".join(odds_markets),
    }
    if books_param:
        params["bookmakers"] = books_param

    data = _http_get(BASE_ODDS_URL, params)
    if not isinstance(data, list):
        data = []

    events: List[Dict[str, Any]] = []
    for g in data:
        ev = {
            "id": g.get("id"),
            "name": f"{g.get('away_team')} @ {g.get('home_team')}",
            "home_team": g.get("home_team"),
            "away_team": g.get("away_team"),
            "commence_time": g.get("commence_time"),
            "props": [],
        }
        for bk in g.get("bookmakers", []) or []:
            book_key = (bk.get("key") or "").lower()
            for mk in bk.get("markets", []) or []:
                market_key = mk.get("key")
                for oc in mk.get("outcomes", []) or []:
                    ev["props"].append({
                        "event_id": ev["id"],
                        "player": oc.get("description") or oc.get("participant") or oc.get("name"),
                        "market": market_key,
                        "bookmaker": book_key,
                        "line": oc.get("point"),
                        "price_name": oc.get("name"),
                        "price": oc.get("price"),
                    })
        events.append(ev)
    return events

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
    df = pd.DataFrame(rows)
    inv = {v: k for k, v in MARKET_MAP.items()}
    if not df.empty:
        df["market"] = df["market"].map(lambda m: inv.get(str(m), str(m)))
    return df

def fetch_props(date=None, season=None, window=None, hours=None, lookahead=None,
                cap: int = 0, markets=None, order: str = "odds", books: str = "draftkings",
                team_filter=None, selection=None, event_ids=None, **kwargs) -> pd.DataFrame:
    # Note: we do not hard-filter by time here—engine can apply its own window.
    print(f"[odds_api] window={hours or window or 'n/a'}h cap={cap}")
    mstr = ",".join(markets) if isinstance(markets, (list, tuple)) else (markets or ",".join(DEFAULT_MARKETS))
    print(f"markets={mstr}\norder={order},{books}")

    events = _fetch_events_from_provider(date=date, season=season, hours=hours,
                                         books=books, markets=markets, order=order, **kwargs) or []
    print(f"[odds_api] fetched {len(events)} events from provider")

    # apply filters once
    events = _apply_selection_filters(events_list=events, teams=team_filter, events=event_ids, selection=selection)
    print(f"[odds_api] after team/selection/event filters -> {len(events)} events")

    if cap and cap > 0 and isinstance(events, list):
        events = events[:cap]

    df = events_to_dataframe(events)
    print(f"[odds_api] flattened to {len(df)} prop rows")
    return df

def get_props(*args, **kwargs) -> pd.DataFrame: return fetch_props(*args, **kwargs)
def get_props_df(*args, **kwargs) -> pd.DataFrame: return fetch_props(*args, **kwargs)
