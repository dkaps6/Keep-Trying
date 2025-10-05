# scripts/odds_api.py
from __future__ import annotations

import os, time, json
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Iterable, List, Tuple
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"
HEADERS = {"User-Agent": "keep-trying/props-pipeline python-requests"}

# window & caps (env override allowed)
EVENT_WINDOW_HOURS = int(os.environ.get("ODDS_API_EVENT_WINDOW_HOURS", "168"))
MAX_EVENTS = int(os.environ.get("ODDS_API_MAX_EVENTS", "40"))
REGIONS = os.environ.get("ODDS_API_REGIONS", "us")
BOOKMAKERS = os.environ.get("ODDS_API_BOOKMAKERS", "").strip()

# prop markets we care about
CORE_MARKETS = [
    "player_pass_yds", "player_pass_tds",
    "player_rush_yds", "player_rush_attempts",
    "player_receptions", "player_reception_yds",
    "player_rush_reception_yds",   # rush+rec
    "player_anytime_td",
]

ANCHORS = {"over", "yes", "under", "no"}

def _key() -> str:
    k = os.environ.get("ODDS_API_KEY", "")
    if not k: print("[odds_api] ODDS_API_KEY missing")
    else:     print(f"[odds_api] using key suffix ...{k[-4:]}")
    return k

def _get(url: str, params: Dict[str, Any]) -> Any:
    r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    rem = r.headers.get("x-requests-remaining"); used = r.headers.get("x-requests-used"); lim = r.headers.get("x-requests-limit")
    if rem or used or lim:
        print(f"[odds_api] credits remaining={rem} used={used} limit={lim}")
    r.raise_for_status()
    return r.json()

def _list_events(key: str) -> List[Dict[str, Any]]:
    url = f"{BASE}/sports/{SPORT}/odds"
    params = dict(apiKey=key, markets="h2h", regions=REGIONS, oddsFormat="american", dateFormat="iso")
    data = _get(url, params)
    return data if isinstance(data, list) else []

def _parse_iso(ts: str):
    try:
        if ts.endswith("Z"): ts = ts.replace("Z","+00:00")
        return datetime.fromisoformat(ts)
    except Exception:
        return None

def _filter_window(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    now = datetime.now(timezone.utc); cutoff = now + timedelta(hours=EVENT_WINDOW_HOURS)
    kept = []
    for ev in events:
        dt = _parse_iso(ev.get("commence_time") or "")
        if dt and now <= dt <= cutoff:
            kept.append(ev)
    print(f"[odds_api] events listed={len(events)} in_window({EVENT_WINDOW_HOURS}h)={len(kept)} cap={MAX_EVENTS}")
    return kept[:MAX_EVENTS]

def _event_props(key: str, event_id: str, markets: Iterable[str]) -> Dict[str, Any]:
    url = f"{BASE}/sports/{SPORT}/events/{event_id}/odds"
    params = dict(
        apiKey=key,
        regions=REGIONS,
        markets=",".join(markets),
        oddsFormat="american",
        dateFormat="iso",
    )
    if BOOKMAKERS:
        params["bookmakers"] = BOOKMAKERS
    return _get(url, params)

def _classify(player_field: str, desc_field: str) -> Tuple[str,str]:
    """Return (player_name, side) where side in {'over','under','yes','no',''}."""
    name = (player_field or "").strip()
    desc = (desc_field or "").strip()
    nlow, dlow = name.lower(), desc.lower()
    if nlow in ANCHORS:
        return desc, nlow
    if dlow in ANCHORS:
        return name, dlow
    # neither looks like a side → treat `name` as player, empty side
    return name or desc, ""

def _rows_from_event_json(j: Dict[str, Any], allowed: set) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for bm in j.get("bookmakers", []):
        book = bm.get("key")
        for mkt in bm.get("markets", []):
            mkey = mkt.get("key")
            if mkey not in allowed: 
                continue
            for oc in mkt.get("outcomes", []):
                player, side = _classify(oc.get("name"), oc.get("description"))
                if not player or side not in {"over","under","yes","no"}:
                    continue
                price = oc.get("price", None)
                line  = oc.get("point", 0.0)  # anytime TD often has no point
                try: price = float(price); line = float(line)
                except Exception: continue
                rows.append({
                    "event_id": j.get("id"),
                    "home_team": j.get("home_team"),
                    "away_team": j.get("away_team"),
                    "commence_time": j.get("commence_time"),
                    "book": book,
                    "market": mkey,
                    "player": player,
                    "side": side,                # keep both sides; devig later
                    "line": line,
                    "price": price,
                })
    return rows

def fetch_props(date: str = "today", season: str = "2025") -> pd.DataFrame:
    key = _key()
    if not key: return pd.DataFrame()

    DBG = Path("outputs/debug"); DBG.mkdir(parents=True, exist_ok=True)

    evs = _filter_window(_list_events(key))
    if not evs: 
        print("[odds_api] no events within window"); return pd.DataFrame()

    # discover which markets exist on the first event
    first_id = evs[0].get("id")
    j0 = _event_props(key, first_id, CORE_MARKETS)
    (DBG / f"event_{first_id}_discovery.json").write_text(json.dumps(j0, indent=2))
    discovered = {m.get("key") for bm in j0.get("bookmakers", []) for m in bm.get("markets", []) if m.get("key")}
    discovered &= set(CORE_MARKETS)
    print(f"[odds_api] discovered markets on first event: {sorted(discovered) or 'NONE'}")
    if not discovered:
        print("[odds_api] discovery had no usable markets")
        return pd.DataFrame()

    all_rows: List[Dict[str, Any]] = []
    for idx, ev in enumerate(evs, start=1):
        eid = ev.get("id")
        if not eid: continue
        jj = _event_props(key, eid, sorted(discovered))
        if idx <= 3:
            (DBG / f"event_{eid}.json").write_text(json.dumps(jj, indent=2))
        bm_ct = len(jj.get("bookmakers", []))
        m_ct  = sum(len(bm.get("markets", [])) for bm in jj.get("bookmakers", []))
        o_ct  = sum(len(m.get("outcomes", [])) for bm in jj.get("bookmakers", []) for m in bm.get("markets", []))
        print(f"[odds_api] event {idx}/{len(evs)} {eid}: bookmakers={bm_ct} markets={m_ct} outcomes={o_ct}")
        all_rows.extend(_rows_from_event_json(jj, discovered))
        time.sleep(0.12)

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("[odds_api] 0 player-prop rows for this slate")
        return df

    Path("outputs").mkdir(exist_ok=True)
    df.to_csv("outputs/props_raw.csv", index=False)
    print(f"[odds_api] assembled {len(df)} outcome rows from {len(evs)} events; markets={sorted(discovered)}")
    return df

get_props = build_props_frame = load_props = fetch_props
