# scripts/providers/draftkings_free.py
from __future__ import annotations
import time
from typing import Any, Dict, Iterable, List, Optional
from pathlib import Path
import requests
import pandas as pd

# NFL event group on DraftKings
DK_EVENTGROUP_NFL = 88808
DK_EVENTGROUP_URL = f"https://sportsbook.draftkings.com/sites/US-SB/api/v5/eventgroups/{DK_EVENTGROUP_NFL}?format=json"

HEADERS = {
    "User-Agent": "keep-trying/props-pipeline (+github) python-requests",
    "Accept": "application/json",
}

# Map DK subcategory/market labels to our unified market names
MARKET_NAME_MAP = {
    "Player Receiving Yards": "receiving_yards",
    "Player Receptions": "receptions",
    "Player Rushing Yards": "rushing_yards",
    "Player Rushing Attempts": "rushing_attempts",
    "Player Passing Yards": "passing_yards",
    "Player Passing TDs": "passing_tds",
    "Player Rushing + Receiving Yards": "rush_rec_yards",
    "Anytime Touchdown Scorer": "anytime_td",
}

KEEP_MARKETS = set(MARKET_NAME_MAP.keys())

def _get_json(url: str) -> Dict[str, Any]:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()

def _american_int(s: Any) -> Optional[float]:
    if s is None: 
        return None
    try:
        return float(int(str(s)))
    except Exception:
        try:
            return float(s)
        except Exception:
            return None

def fetch_dk_props() -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      [player, market, line, price, book, event_id, home_team, away_team, commence_time]
    Only "Over/Yes" outcomes are kept (your model prices Over probability).
    """
    j = _get_json(DK_EVENTGROUP_URL)

    # Build event lookup: id -> meta
    events = j.get("eventGroup", {}).get("events", []) or []
    ev_map: Dict[str, Dict[str, Any]] = {}
    for ev in events:
        ev_map[str(ev.get("eventId"))] = {
            "event_id": str(ev.get("eventId")),
            "home_team": ev.get("homeTeam") or ev.get("teamOneName"),
            "away_team": ev.get("awayTeam") or ev.get("teamTwoName"),
            "commence_time": ev.get("startDate"),
        }

    # Walk all offer categories / subcategories to find player props
    rows: List[Dict[str, Any]] = []
    cats = j.get("eventGroup", {}).get("offerCategories", []) or []
    for cat in cats:
        for subdesc in cat.get("offerSubcategoryDescriptors", []) or []:
            sub = subdesc.get("offerSubcategory") or {}
            sub_name = sub.get("name")
            if sub_name not in KEEP_MARKETS:
                continue  # skip markets we don't model yet
            unified_market = MARKET_NAME_MAP[sub_name]

            for offer in sub.get("offers", []) or []:
                # Each "offer" is for a specific eventId, contains outcomes Over/Under
                # Sometimes offers is a list of lists
                if isinstance(offer, list):
                    offers_iter: Iterable = offer
                else:
                    offers_iter = [offer]

                for off in offers_iter:
                    event_id = str(off.get("eventId"))
                    if not event_id or event_id not in ev_map:
                        continue
                    for oc in off.get("outcomes", []) or []:
                        label = (oc.get("label") or "").lower()   # "over"|"under"|"yes"|"no"
                        participant = oc.get("participant") or oc.get("name") or oc.get("outcomeName")
                        price = _american_int(oc.get("oddsAmerican") or oc.get("oddsAmericanDisplay"))
                        line = oc.get("line")
                        # DK sometimes nests the numeric under 'line' or 'lineDisplay'; try both
                        if line is None:
                            ld = oc.get("lineDisplay")
                            try:
                                line = float(str(ld).replace("½", ".5")) if ld is not None else None
                            except Exception:
                                line = None
                        # Keep Over/Yes as our anchor
                        if label not in ("over", "yes", ""):
                            continue
                        if participant is None or price is None:
                            continue

                        meta = ev_map[event_id]
                        rows.append({
                            "player": participant,
                            "market": unified_market,
                            "line": float(line) if line is not None else None,
                            "price": float(price),
                            "book": "draftkings",
                            "event_id": meta["event_id"],
                            "home_team": meta["home_team"],
                            "away_team": meta["away_team"],
                            "commence_time": meta["commence_time"],
                        })
        # small pause between categories (be nice)
        time.sleep(0.05)

    df = pd.DataFrame(rows).dropna(subset=["player","market","price"])
    # DK may include some team specials; ensure we only keep the player props we mapped
    if not df.empty:
        df = df[df["market"].isin(MARKET_NAME_MAP.values())].copy()

    # Save a debug snapshot so you can inspect what we pulled
    try:
        Path("outputs").mkdir(parents=True, exist_ok=True)
        df.to_csv("outputs/props_raw.csv", index=False)
    except Exception:
        pass

    print(f"[draftkings_free] collected {len(df)} rows")
    return df
