# scripts/providers/draftkings_free.py
from __future__ import annotations
import time, random
from typing import Any, Dict, Iterable, List, Optional
from pathlib import Path
import requests
import pandas as pd

# NFL event group id
DK_EVENTGROUP_NFL = 88808

# Multiple regional hosts to dodge CI 403s
HOSTS = [
    "sportsbook.draftkings.com",
    "sportsbook-us-mi.draftkings.com",
    "sportsbook-us-nj.draftkings.com",
    "sportsbook-us-ny.draftkings.com",
    "sportsbook-us-pa.draftkings.com",
    "sportsbook-nash-usny.draftkings.com",
    "sportsbook-nash-usnj.draftkings.com",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Referer": "https://sportsbook.draftkings.com/leagues/football/nfl",
    "Connection": "keep-alive",
}

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
    last_err = None
    for host in random.sample(HOSTS, k=len(HOSTS)):
        u = url.replace("HOST", host)
        try:
            r = requests.get(u, headers=HEADERS, timeout=30)
            if r.status_code == 403:
                last_err = f"403 from {host}"
                time.sleep(0.25); continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = str(e)
            time.sleep(0.25)
    raise RuntimeError(last_err or "DraftKings fetch failed")

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
    url = "https://HOST/sites/US-SB/api/v5/eventgroups/{gid}?format=json".format(gid=DK_EVENTGROUP_NFL)
    j = _get_json(url)

    events = j.get("eventGroup", {}).get("events", []) or []
    ev_map: Dict[str, Dict[str, Any]] = {}
    for ev in events:
        ev_map[str(ev.get("eventId"))] = {
            "event_id": str(ev.get("eventId")),
            "home_team": ev.get("homeTeam") or ev.get("teamOneName"),
            "away_team": ev.get("awayTeam") or ev.get("teamTwoName"),
            "commence_time": ev.get("startDate"),
        }

    rows: List[Dict[str, Any]] = []
    cats = j.get("eventGroup", {}).get("offerCategories", []) or []
    for cat in cats:
        for subdesc in cat.get("offerSubcategoryDescriptors", []) or []:
            sub = subdesc.get("offerSubcategory") or {}
            sub_name = sub.get("name")
            if sub_name not in KEEP_MARKETS:
                continue
            unified = MARKET_NAME_MAP[sub_name]

            offers = sub.get("offers", []) or []
            for o in offers:
                of_list: Iterable = list(o) if isinstance(o, list) else [o]
                for off in of_list:
                    eid = str(off.get("eventId"))
                    if not eid or eid not in ev_map:
                        continue
                    for oc in off.get("outcomes", []) or []:
                        label = (oc.get("label") or "").lower()
                        if label not in ("over","yes",""):
                            continue
                        player = oc.get("participant") or oc.get("name") or oc.get("outcomeName")
                        price = _american_int(oc.get("oddsAmerican") or oc.get("oddsAmericanDisplay"))
                        line = oc.get("line")
                        if line is None:
                            ld = oc.get("lineDisplay")
                            try:
                                line = float(str(ld).replace("½", ".5")) if ld is not None else None
                            except Exception:
                                line = None
                        if player is None or price is None:
                            continue
                        meta = ev_map[eid]
                        rows.append({
                            "player": player,
                            "market": unified,
                            "line": float(line) if line is not None else None,
                            "price": float(price),
                            "book": "draftkings",
                            "event_id": meta["event_id"],
                            "home_team": meta["home_team"],
                            "away_team": meta["away_team"],
                            "commence_time": meta["commence_time"],
                        })
        time.sleep(0.05)

    df = pd.DataFrame(rows).dropna(subset=["player","market","price"])
    if not df.empty:
        try:
            Path("outputs").mkdir(parents=True, exist_ok=True)
            df.to_csv("outputs/props_raw.csv", index=False)
        except Exception:
            pass
    print(f"[draftkings_free] collected {len(df)} rows")
    return df

