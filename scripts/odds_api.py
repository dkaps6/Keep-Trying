# scripts/odds_api.py
from __future__ import annotations
import os
import time
from typing import Any, Dict, List

import pandas as pd
import requests

BASE = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"
HEADERS = {"User-Agent": "keep-trying/props-pipeline python-requests"}

# Official NFL player-prop markets for the event-odds endpoint
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

def _key() -> str:
    k = os.environ.get("ODDS_API_KEY", "")
    if not k:
        print("[odds_api] ODDS_API_KEY missing")
    return k

def _get(url: str, params: Dict[str, Any]) -> Any:
    r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        # Make the failure obvious in logs without crashing the whole run
        print(f"[odds_api] {url} -> HTTP {r.status_code}: {r.text[:200]}")
        raise
    return r.json()

def _list_events(key: str) -> List[Dict[str, Any]]:
    """
    Enumerate events cheaply via a featured market (h2h).
    This endpoint is widely available across plans.
    """
    url = f"{BASE}/sports/{SPORT}/odds"
    params = dict(apiKey=key, markets="h2h", regions="us", oddsFormat="american", dateFormat="iso")
    try:
        data = _get(url, params)
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"[odds_api] events list failed: {e}")
        return []

def _event_props(key: str, event_id: str) -> List[Dict[str, Any]]:
    """
    Player props must be fetched from the per-event endpoint.
    IMPORTANT: we do NOT pass `bookmakers=` to avoid plan/entitlement 401s.
    """
    url = f"{BASE}/sports/{SPORT}/events/{event_id}/odds"
    params = dict(
        apiKey=key,
        regions="us",
        markets=",".join(NFL_PROP_MARKETS),
        oddsFormat="american",
        dateFormat="iso",
    )
    try:
        j = _get(url, params)
    except Exception as e:
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
                desc = (oc.get("description") or "").lower()  # "over"/"under"/"yes"/"no"
                # Keep Over/Yes rows as anchors for Over probability
                if desc not in ("over", "yes", ""):
                    continue
                player = oc.get("name")
                line   = oc.get("point")
                price  = oc.get("price")
                if player is None or line is None or price is None:
                    continue
                rows.append({
                    "event_id": j.get("id"),
                    "home_team": j.get("home_team"),
                    "away_team": j.get("away_team"),
                    "commence_time": j.get("commence_time"),
                    "book": book,
                    "market": mkey,
                    "player": player,
                    "line": float(line),
                    "price": float(price),
                })
    return rows

def fetch_props(date: str = "today", season: str = "2025") -> pd.DataFrame:
    """
    Returns a normalized DataFrame:
      [player, market, line, price, book, event_id, home_team, away_team, commence_time]
    Also writes outputs/props_raw.csv for debugging/inspection.
    """
    key = _key()
    if not key:
        return pd.DataFrame()

    events = _list_events(key)
    if not events:
        print("[odds_api] 0 events listed")
        return pd.DataFrame()

    all_rows: List[Dict[str, Any]] = []
    for ev in events:
        eid = ev.get("id")
        if not eid:
            continue
        all_rows.extend(_event_props(key, eid))
        time.sleep(0.12)  # polite rate spacing

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("[odds_api] 0 player-prop rows")
        return df

    keep = ["player","market","line","price","book","event_id","home_team","away_team","commence_time"]
    for k in keep:
        if k not in df.columns:
            df[k] = None
    df = df[keep].copy()

    try:
        os.makedirs("outputs", exist_ok=True)
        df.to_csv("outputs/props_raw.csv", index=False)
    except Exception:
        pass

    print(f"[odds_api] assembled {len(df)} rows")
    return df

# Backwards-compat function names your fetcher might look for
get_props = build_props_frame = load_props = fetch_props
