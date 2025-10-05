# scripts/odds_api.py
from __future__ import annotations
import os, time, requests, pandas as pd
from typing import Any, Dict, List

BASE = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"

HEADERS = {"User-Agent": "keep-trying/props-pipeline (+github) python-requests"}

# ✅ Valid NFL player-prop market keys (docs list)
# See: https://the-odds-api.com/sports-odds-data/betting-markets.html
NFL_PROP_MARKETS = [
    "player_pass_yds",
    "player_pass_tds",
    "player_rush_yds",
    "player_rush_attempts",
    "player_reception_yds",
    "player_receptions",
    "player_rush_reception_yds",
    "player_anytime_td",
]

# Reasonable US books; adjust if you prefer regions="us" instead
DEFAULT_BOOKMAKERS = ["draftkings", "fanduel", "betmgm", "caesars", "pointsbetus"]

def _req(url: str, params: Dict[str, Any]) -> Any:
    r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    if r.status_code == 429:
        # gentle backoff to be polite; caller may retry
        time.sleep(1.0)
    r.raise_for_status()
    return r.json()

def _get_api_key() -> str:
    key = os.environ.get("ODDS_API_KEY", "")
    if not key:
        print("[odds_api] No ODDS_API_KEY set; returning empty props.")
    return key

def _list_events(api_key: str) -> List[Dict[str, Any]]:
    """
    Use the featured-odds endpoint with a single featured market (h2h)
    to obtain the list of event ids without pulling huge payloads.
    """
    url = f"{BASE}/sports/{SPORT}/odds"
    params = {
        "apiKey": api_key,
        "markets": "h2h",
        "regions": "us",
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    try:
        data = _req(url, params)
        if not isinstance(data, list):
            return []
        return data
    except requests.HTTPError as e:
        print(f"[odds_api] events list failed: {e}")
        return []

def _event_props(api_key: str, event_id: str, bookmakers: List[str]) -> List[Dict[str, Any]]:
    """
    Pull player props for a single event using the event-odds endpoint.
    This is REQUIRED for props; /odds only supports featured markets.
    """
    url = f"{BASE}/sports/{SPORT}/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",                # can also use bookmakers= to filter
        "bookmakers": ",".join(bookmakers),
        "markets": ",".join(NFL_PROP_MARKETS),
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    try:
        j = _req(url, params)
    except requests.HTTPError as e:
        # common cause of 422 here is a bad market key; our list uses doc keys
        print(f"[odds_api] event {event_id} props failed: {e}")
        return []
    rows: List[Dict[str, Any]] = []
    for bm in j.get("bookmakers", []):
        book = bm.get("key")
        for mkt in bm.get("markets", []):
            mkey = mkt.get("key")
            if mkey not in NFL_PROP_MARKETS:
                continue
            for oc in mkt.get("outcomes", []):
                player = oc.get("name")
                desc = (oc.get("description") or "").lower()  # "over" | "under" | "yes" | "no"
                line  = oc.get("point")
                price = oc.get("price")  # American odds
                if player is None or line is None or price is None:
                    continue
                # keep Over/Yes legs as our "Over" anchor (your model prices Over prob)
                if desc not in ("over", "yes", ""):
                    continue
                rows.append({
                    "event_id": j.get("id"),
                    "home_team": j.get("home_team"),
                    "away_team": j.get("away_team"),
                    "commence_time": j.get("commence_time"),
                    "book": book,
                    "market": mkey,        # keep doc key; engine maps/can display as-is
                    "player": player,
                    "line": float(line),
                    "price": float(price),
                })
    return rows

def fetch_props(date: str = "today", season: str = "2025",
                bookmakers: List[str] = DEFAULT_BOOKMAKERS) -> pd.DataFrame:
    """
    Returns a normalized DataFrame:
      [player, market, line, price, book, event_id, home_team, away_team, commence_time]
    """
    api_key = _get_api_key()
    if not api_key:
        return pd.DataFrame()

    events = _list_events(api_key)
    if not events:
        print("[odds_api] 0 events listed; returning empty props.")
        return pd.DataFrame()

    all_rows: List[Dict[str, Any]] = []
    for ev in events:
        eid = ev.get("id")
        if not eid:
            continue
        # Optional: basic date windowing could be added here by checking ev["commence_time"]
        rows = _event_props(api_key, eid, bookmakers)
        if rows:
            all_rows.extend(rows)
        time.sleep(0.15)  # small pause to be a good citizen

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("[odds_api] 0 player-prop rows returned across all events.")
        return df

    # Normalize to the columns your engine expects
    keep = ["player","market","line","price","book","event_id","home_team","away_team","commence_time"]
    for k in keep:
        if k not in df.columns:
            df[k] = None
    df = df[keep].copy()

    # Debug snapshot so you can inspect the raw pull
    try:
        Path("outputs").mkdir(parents=True, exist_ok=True)
        df.to_csv("outputs/props_raw.csv", index=False)
    except Exception:
        pass

    print(f"[odds_api] assembled {len(df)} player-prop rows from {len(events)} events")
    return df

# Backwards-compat names used by fetch_all
get_props = fetch_props
build_props_frame = fetch_props
load_props = fetch_props
