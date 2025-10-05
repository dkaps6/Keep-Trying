# scripts/odds_api.py
from __future__ import annotations

import os, time, json
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Iterable, List, Set
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"
HEADERS = {"User-Agent": "keep-trying/props-pipeline python-requests"}

# ------- FULL SLATE DEFAULTS (override via env in workflow) -------------------
EVENT_WINDOW_HOURS = int(os.environ.get("ODDS_API_EVENT_WINDOW_HOURS", "168"))  # 7 days
MAX_EVENTS = int(os.environ.get("ODDS_API_MAX_EVENTS", "40"))                   # hard cap
REGIONS = os.environ.get("ODDS_API_REGIONS", "us")                              # default to US only
BOOKMAKERS = os.environ.get("ODDS_API_BOOKMAKERS", "").strip()                  # e.g. "draftkings,fanduel,betmgm,caesars,pointsbetus"
# -----------------------------------------------------------------------------

CORE_MARKETS: List[str] = [
    "player_pass_yds", "player_pass_tds",
    "player_rush_yds", "player_rush_attempts",
    "player_receptions", "player_reception_yds",
    "player_rush_reception_yds", "player_anytime_td",
]
_POS_DESCS = {"over", "yes", ""}

def _key() -> str:
    k = os.environ.get("ODDS_API_KEY", "")
    if not k:
        print("[odds_api] ODDS_API_KEY missing")
    else:
        print(f"[odds_api] using key suffix ...{k[-4:]}")
    return k

def _get(url: str, params: Dict[str, Any]) -> Any:
    r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    rem = r.headers.get("x-requests-remaining"); used = r.headers.get("x-requests-used"); lim = r.headers.get("x-requests-limit")
    if rem or used or lim:
        print(f"[odds_api] credits remaining={rem} used={used} limit={lim}")
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        body = (r.text or "")[:300]
        print(f"[odds_api] {url} -> HTTP {r.status_code}: {body}")
        if r.status_code == 401 and "OUT_OF_USAGE_CREDITS" in body:
            raise RuntimeError("ODDS_API_OUT_OF_CREDITS") from e
        raise
    return r.json()

def _list_events(key: str) -> List[Dict[str, Any]]:
    url = f"{BASE}/sports/{SPORT}/odds"
    params = dict(apiKey=key, markets="h2h", regions=REGIONS, oddsFormat="american", dateFormat="iso")
    return _get(url, params) or []

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
        params["bookmakers"] = BOOKMAKERS  # limit credits + ensure known books
    return _get(url, params)

def _rows_from_event_json(j: Dict[str, Any], allowed: Set[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for bm in j.get("bookmakers", []):
        book = bm.get("key")
        for mkt in bm.get("markets", []):
            mkey = mkt.get("key")
            if mkey not in allowed: continue
            for oc in mkt.get("outcomes", []):
                desc = (oc.get("description") or "").lower()
                if desc not in _POS_DESCS: continue
                player = oc.get("name"); line = oc.get("point"); price = oc.get("price")
                if player is None or line is None or price is None: continue
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
    key = _key()
    if not key: return pd.DataFrame()

    DBG = Path("outputs/debug"); DBG.mkdir(parents=True, exist_ok=True)

    # 1) list + window
    evs_all = _list_events(key)
    evs = _filter_window(evs_all)
    if not evs:
        print("[odds_api] no events within window"); return pd.DataFrame()

    # 2) discovery on first event
    first_id = evs[0].get("id")
    j0 = _event_props(key, first_id, CORE_MARKETS)
    (DBG / f"event_{first_id}_discovery.json").write_text(json.dumps(j0, indent=2))
    discovered: Set[str] = {m.get("key") for bm in j0.get("bookmakers", []) for m in bm.get("markets", []) if m.get("key")}
    discovered &= set(CORE_MARKETS)
    print(f"[odds_api] discovered markets on first event: {sorted(discovered) or 'NONE'}")
    if not discovered:
        print("[odds_api] discovery returned no usable prop markets for this plan/region")
        return pd.DataFrame()

    # 3) fetch loop (also dump first few raw payloads for inspection)
    all_rows: List[Dict[str, Any]] = []
    for idx, ev in enumerate(evs, start=1):
        eid = ev.get("id"); 
        if not eid: continue
        jj = _event_props(key, eid, sorted(discovered))
        # debug: save first 3 events raw
        if idx <= 3:
            (DBG / f"event_{eid}.json").write_text(json.dumps(jj, indent=2))
        bm_ct = len(jj.get("bookmakers", []))
        m_ct = sum(len(bm.get("markets", [])) for bm in jj.get("bookmakers", []))
        o_ct = sum(len(m.get("outcomes", [])) for bm in jj.get("bookmakers", []) for m in bm.get("markets", []))
        print(f"[odds_api] event {idx}/{len(evs)} {eid}: bookmakers={bm_ct} markets={m_ct} outcomes={o_ct}")
        rows = _rows_from_event_json(jj, discovered)
        all_rows.extend(rows)
        time.sleep(0.15)

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("[odds_api] 0 player-prop rows for this slate")
        return df

    keep = ["player","market","line","price","book","event_id","home_team","away_team","commence_time"]
    for k in keep:
        if k not in df.columns: df[k] = None
    df = df[keep].copy()

    Path("outputs").mkdir(exist_ok=True)
    df.to_csv("outputs/props_raw.csv", index=False)
    print(f"[odds_api] assembled {len(df)} rows from {len(evs)} events; markets={sorted(discovered)}")
    return df

get_props = build_props_frame = load_props = fetch_props
