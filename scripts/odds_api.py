# scripts/odds_api.py
from __future__ import annotations
import os, time, requests, pandas as pd
from typing import Any, Dict, List, Optional

BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"

# Map The Odds API market keys -> your unified market names
MARKET_MAP = {
    "player_receiving_yards": "receiving_yards",
    "player_receptions": "receptions",
    "player_rushing_yards": "rushing_yards",
    "player_rushing_attempts": "rushing_attempts",
    "player_passing_yards": "passing_yards",
    "player_passing_tds": "passing_tds",
    "player_rush_receive_yards": "rush_rec_yards",
    "player_anytime_td": "anytime_td",
}

DEFAULT_MARKETS = list(MARKET_MAP.keys())
DEFAULT_BOOKS = ["draftkings","fanduel","betmgm","caesars","pointsbetus"]

def _req(url: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    r = requests.get(url, params=params, timeout=30)
    if r.status_code == 429:
        raise RuntimeError("Rate limited by The Odds API (429).")
    r.raise_for_status()
    return r.json()

def _extract_event_rows(ev: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    eid = ev.get("id")
    home, away = ev.get("home_team"), ev.get("away_team")
    commence = ev.get("commence_time")
    for bm in ev.get("bookmakers", []):
        book = bm.get("key")
        for mkt in bm.get("markets", []):
            mkey = mkt.get("key")
            u_market = MARKET_MAP.get(mkey)
            if not u_market:  # skip markets we don't support
                continue
            for oc in mkt.get("outcomes", []):
                # For player props, outcomes usually have: name (player), description (Over/Under/Yes/No), price (American), point (line)
                player = oc.get("name")
                side = (oc.get("description") or "").lower()
                line = oc.get("point")
                price = oc.get("price")
                if player is None or line is None or price is None:
                    continue
                # keep Over/Yes rows as anchors for pricing Over probabilities
                if side not in ("over", "yes", ""):
                    continue
                rows.append({
                    "event_id": eid,
                    "home_team": home,
                    "away_team": away,
                    "commence_time": commence,
                    "book": book,
                    "market_key": mkey,
                    "market": u_market,
                    "player": player,
                    "line": line,
                    "price": price,  # American odds
                })
    return rows

def fetch_props(*, date: str = "today", season: str = "2025",
                markets: List[str] = DEFAULT_MARKETS,
                books: List[str] = DEFAULT_BOOKS,
                regions: str = "us") -> pd.DataFrame:
    """
    Returns a normalized DataFrame:
      [player, market, line, price, book, event_id, home_team, away_team, commence_time]
    """
    api_key = os.environ.get("ODDS_API_KEY", "")
    if not api_key:
        print("[odds_api] No ODDS_API_KEY set; returning empty.")
        return pd.DataFrame()

    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": ",".join(markets),
        "oddsFormat": "american",
        "dateFormat": "iso",
        # Optional: "commenceTimeFrom": "...", "commenceTimeTo": "..."
    }

    try:
        data = _req(BASE, params)
    except Exception as e:
        print(f"[odds_api] request failed: {e}")
        return pd.DataFrame()

    all_rows: List[Dict[str, Any]] = []
    for ev in data:
        all_rows.extend(_extract_event_rows(ev))

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("[odds_api] 0 player-prop rows returned (check plan/markets/books/regions).")
        return df

    # Normalize columns for your engine
    keep = ["player","market","line","price","book","event_id","home_team","away_team","commence_time"]
    for k in keep:
        if k not in df.columns:
            df[k] = None
    df = df[keep].copy()
    return df

# Backwards-compat names the engine will look for
get_props = fetch_props
build_props_frame = fetch_props
load_props = fetch_props

