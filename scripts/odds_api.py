# scripts/odds_api.py
# Complete Odds API adapter for NFL player props.
# - Requires env ODDS_API_KEY
# - date + window/hours/lookahead pick the slate
# - books + markets filters supported (with shorthand → full name mapping)
# - Applies selection/team/event-id filters once (blank == no filter)
# - Returns a tidy DataFrame of props

from __future__ import annotations

import os
import time
import re
import json
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "").strip()
SPORT_KEY = "americanfootball_nfl"   # must be underscore
BASE_URL = "https://api.the-odds-api.com/v4"

# Mapping from our “friendly” markets to Odds API player-prop market keys.
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

# Normalize bookmaker names: accept shorthand and common variants.
BOOKMAP = {
    "dk": "draftkings", "draftkings": "draftkings",
    "fd": "fanduel", "fanduel": "fanduel",
    "mgm": "betmgm", "betmgm": "betmgm",
    "cz": "caesars", "czrs": "caesars", "caesars": "caesars",
    "pointsbet": "pointsbet", "bet365": "bet365", "barstool": "barstool",
}


# --------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------

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
    """GET with retries and clearer error messages."""
    if not ODDS_API_KEY:
        raise RuntimeError("ODDS_API_KEY env variable is not set.")
    q = params.copy()
    q["apiKey"] = ODDS_API_KEY
    full = f"{url}?{urllib.parse.urlencode(q, doseq=True)}"
    backoff = 1.0
    last_err_text = ""
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
            last_err_text = f"HTTP {e.code} {e.reason}: {body}"
        except Exception as e:
            last_err_text = f"{type(e).__name__}: {e}"
        time.sleep(backoff)
        backoff = min(8.0, backoff * 2.0)
    raise RuntimeError(f"GET failed after retries: {full}\nDetail: {last_err_text}")


# --------------------------------------------------------------------
# Filtering (selection / team / ids)
# --------------------------------------------------------------------

def _apply_selection_filters(
    events_list: Iterable[Dict[str, Any]],
    teams: Optional[Iterable[str]] = None,
    events: Optional[Iterable[str]] = None,
    selection: Optional[str] = None,
):
    """Returns filtered list."""
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


# --------------------------------------------------------------------
# Provider fetch
# --------------------------------------------------------------------

def _fetch_events_from_provider(
    date: Optional[str] = None,
    season: Optional[int] = None,
    hours: Optional[int] = None,
    books: str = "dk",
    markets: Optional[List[str] | str] = None,
    order: str = "odds",
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Pull events and attach requested player props (organized by event).
    Returns list of event dicts with:
      - id, name, home_team, away_team, commence_time
      - props: list of {player, market, bookmaker, line, over_odds, under_odds}
    """
    hrs = int(hours or 24)
    anchor = _parse_date_anchor(date)
    start = anchor
    end = anchor + timedelta(hours=hrs)

    # 1) events (scheduled games)
    events_url = f"{BASE_URL}/sports/{SPORT_KEY}/events"
    events = _http_get(events_url, {
        "regions": "us",
        "oddsFormat": "american",
        "dateFormat": "iso",
        "commenceTimeFrom": start.isoformat().replace("+00:00", "Z"),
        "commenceTimeTo": end.isoformat().replace("+00:00", "Z"),
    })
    if not isinstance(events, list):
        events = []

    # Normalize book list -> API keys
    book_list: List[str] = []
    if books:
        if isinstance(books, str):
            book_list = [b.strip() for b in books.split(",") if b.strip()]
        elif isinstance(books, (list, tuple)):
            book_list = list(books)
    book_list = [BOOKMAP.get(b.lower(), b.lower()) for b in book_list if b]
    books_param = ",".join(book_list) if book_list else None
    regions = "us"

    # Normalize markets
    if markets is None or (isinstance(markets, str) and markets.strip() == ""):
        wanted_markets = DEFAULT_MARKETS
    elif isinstance(markets, str):
        wanted_markets = [m.strip() for m in markets.split(",") if m.strip()]
    else:
        wanted_markets = list(markets)
    odds_markets = [MARKET_MAP.get(m, m) for m in wanted_markets]

    # Build events dict
    by_id: Dict[str, Dict[str, Any]] = {}
    for ev in events:
        eid = ev.get("id") or ev.get("event_id")
        home = ev.get("home_team")
        away = ev.get("away_team")
        name = f"{away} @ {home}" if home and away else (ev.get("name") or "")
        by_id[eid] = {
            "id": eid,
            "name": name,
            "home_team": home,
            "away_team": away,
            "commence_time": ev.get("commence_time") or ev.get("commenceTime"),
            "props": [],
        }

    if not by_id:
        return []

    # 2) Player props endpoint per market (optionally with bookmakers)
    props_url = f"{BASE_URL}/sports/{SPORT_KEY}/players/props"
    for mkt in odds_markets:
        params = {
            "markets": mkt,
            "regions": regions,
            "oddsFormat": "american",
            "dateFormat": "iso",
        }
        if books_param:
            params["bookmakers"] = books_param

        data = _http_get(props_url, params)
        if not isinstance(data, list):
            continue

        for entry in data:
            try:
                eid = entry.get("event_id") or entry.get("eventId") or entry.get("event")
                event = by_id.get(eid)
                if not event:
                    continue

                player_name = entry.get("player_name") or entry.get("playerName") or entry.get("name")
                market_key = entry.get("market") or mkt
                bookmaker = (entry.get("bookmaker") or "").lower()
                # normalize bookmaker key back to your canonical label if desired
                # (we'll just keep raw API book key)
                line = entry.get("handicap") or entry.get("line")

                over_price = None
                under_price = None
                prices = entry.get("prices") or entry.get("outcomes") or []
                for o in prices:
                    oname = (o.get("name") or "").lower()
                    if "over" in oname:
                        over_price = o.get("price")
                        if line is None:
                            line = o.get("point")
                    elif "under" in oname:
                        under_price = o.get("price")
                        if line is None:
                            line = o.get("point")

                event["props"].append({
                    "event_id": eid,
                    "player": player_name,
                    "market": market_key,
                    "bookmaker": bookmaker,
                    "line": line,
                    "over_odds": over_price,
                    "under_odds": under_price,
                })
            except Exception:
                continue

        time.sleep(0.2)

    return [ev for ev in by_id.values() if ev.get("props")]


# --------------------------------------------------------------------
# Events -> DataFrame
# --------------------------------------------------------------------

def events_to_dataframe(events: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for ev in events:
        eid = ev.get("id")
        name = ev.get("name")
        ht = ev.get("home_team")
        at = ev.get("away_team")
        dt = ev.get("commence_time")
        for p in ev.get("props", []):
            rows.append({
                "event_id": eid,
                "event_name": name,
                "home_team": ht,
                "away_team": at,
                "commence_time": dt,
                "player": p.get("player"),
                "market": p.get("market"),
                "book": p.get("bookmaker"),
                "vegas_line": p.get("line"),
                "vegas_over_odds": p.get("over_odds"),
                "vegas_under_odds": p.get("under_odds"),
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    inv_map = {v: k for k, v in MARKET_MAP.items()}
    df["market"] = df["market"].map(lambda m: inv_map.get(str(m), str(m)))
    return df


# --------------------------------------------------------------------
# Public facade (used by engine)
# --------------------------------------------------------------------

def fetch_props(
    date: Optional[str] = None,
    season: Optional[int] = None,
    window: Optional[str | int] = None,
    hours: Optional[int] = None,
    lookahead: Optional[int] = None,
    cap: int = 0,
    markets: Optional[List[str] | str] = None,
    order: str = "odds",
    books: str = "dk",
    team_filter: Optional[List[str] | str] = None,
    selection: Optional[str] = None,
    event_ids: Optional[List[str] | str] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    hrs = _to_hours(hours=hours, window=window, lookahead=lookahead, default=24)

    # Pretty log (match previous style)
    print(f"[odds_api] window={hrs}h cap={cap}")
    mstr = ",".join(markets) if isinstance(markets, (list, tuple)) else (markets or ",".join(DEFAULT_MARKETS))
    print(f"markets={mstr}\norder={order},{books}")

    events = _fetch_events_from_provider(
        date=date,
        season=season,
        hours=hrs,
        books=books,
        markets=markets,
        order=order,
        **kwargs,
    ) or []

    events = _apply_selection_filters(
        events_list=events,
        teams=team_filter,
        events=event_ids,
        selection=selection,
    )
    if cap and cap > 0 and isinstance(events, list):
        events = events[:cap]

    df = events_to_dataframe(events)
    return df


# Convenience aliases
def get_props(*args, **kwargs) -> pd.DataFrame:
    return fetch_props(*args, **kwargs)

def get_props_df(*args, **kwargs) -> pd.DataFrame:
    return fetch_props(*args, **kwargs)
