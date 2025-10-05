# scripts/odds_api.py
from __future__ import annotations

import os
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Iterable, List, Set

import pandas as pd
import requests

BASE = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"
HEADERS = {"User-Agent": "keep-trying/props-pipeline python-requests"}

# ---- FULL-SLATE DEFAULTS (you can override with envs in the workflow) -------
# 7 days (full slate window) and a large cap to cover bye weeks / intl games
EVENT_WINDOW_HOURS = int(os.environ.get("ODDS_API_EVENT_WINDOW_HOURS", "168"))  # 7 days
MAX_EVENTS = int(os.environ.get("ODDS_API_MAX_EVENTS", "40"))                   # hard cap
REGIONS = os.environ.get("ODDS_API_REGIONS", "us,us2")                          # regions to request
# -----------------------------------------------------------------------------

# Core player-prop markets that are common across books; keeps URLs compact
CORE_MARKETS: List[str] = [
    # passing
    "player_pass_yds", "player_pass_tds",
    # rushing
    "player_rush_yds", "player_rush_attempts",
    # receiving
    "player_receptions", "player_reception_yds",
    # combos / scorers
    "player_rush_reception_yds", "player_anytime_td",
]

_POS_DESCS = {"over", "yes", ""}  # anchor rows to infer Over/Yes

def _key() -> str:
    k = os.environ.get("ODDS_API_KEY", "")
    if not k:
        print("[odds_api] ODDS_API_KEY missing")
    else:
        print(f"[odds_api] using key suffix ...{k[-4:]}")
    return k

def _get(url: str, params: Dict[str, Any]) -> Any:
    r = requests.get(url, params=params, headers=HEADERS, timeout=30)

    # log credits on every call
    rem = r.headers.get("x-requests-remaining")
    used = r.headers.get("x-requests-used")
    lim  = r.headers.get("x-requests-limit")
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
    """Enumerate events via a cheap featured market (1 request)."""
    url = f"{BASE}/sports/{SPORT}/odds"
    params = dict(
        apiKey=key,
        markets="h2h",
        regions=REGIONS,
        oddsFormat="american",
        dateFormat="iso",
    )
    data = _get(url, params)
    return data if isinstance(data, list) else []

def _parse_iso(ts: str) -> datetime | None:
    try:
        if ts.endswith("Z"):
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return datetime.fromisoformat(ts)
    except Exception:
        return None

def _filter_window(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only events starting within EVENT_WINDOW_HOURS from now (UTC), then cap."""
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=EVENT_WINDOW_HOURS)
    kept = []
    for ev in events:
        dt = _parse_iso(ev.get("commence_time") or "")
        if dt and now <= dt <= cutoff:
            kept.append(ev)
    print(f"[odds_api] events listed={len(events)} in_window({EVENT_WINDOW_HOURS}h)={len(kept)} cap={MAX_EVENTS}")
    return kept[:MAX_EVENTS]

def _event_props(key: str, event_id: str, markets: Iterable[str]) -> Dict[str, Any]:
    """Per-event props (1 request per event)."""
    url = f"{BASE}/sports/{SPORT}/events/{event_id}/odds"
    params = dict(
        apiKey=key,
        regions=REGIONS,
        markets=",".join(markets),
        oddsFormat="american",
        dateFormat="iso",
    )
    return _get(url, params)

def _rows_from_event_json(j: Dict[str, Any], allowed: Set[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for bm in j.get("bookmakers", []):
        book = bm.get("key")
        for mkt in bm.get("markets", []):
            mkey = mkt.get("key")
            if mkey not in allowed:
                continue
            for oc in mkt.get("outcomes", []):
                desc = (oc.get("description") or "").lower()
                if desc not in _POS_DESCS:
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
    Full-slate pull, credit-predictable:
      1 (listing) + 1 (discovery) + N (events in window, capped) requests
    """
    key = _key()
    if not key:
        return pd.DataFrame()

    # 1) list events, window + cap
    try:
        evs_all = _list_events(key)
    except RuntimeError as rte:
        if "ODDS_API_OUT_OF_CREDITS" in str(rte):
            print("[odds_api] out of credits during listing")
            return pd.DataFrame()
        raise

    evs = _filter_window(evs_all)
    if not evs:
        print("[odds_api] no events within window")
        return pd.DataFrame()

    # 2) discovery: ask the FIRST event for CORE_MARKETS (single call)
    first_id = evs[0].get("id")
    if not first_id:
        print("[odds_api] first event missing id")
        return pd.DataFrame()

    try:
        j = _event_props(key, first_id, CORE_MARKETS)
    except RuntimeError as rte:
        if "ODDS_API_OUT_OF_CREDITS" in str(rte):
            print("[odds_api] out of credits at discovery")
            return pd.DataFrame()
        raise

    discovered: Set[str] = set()
    for bm in j.get("bookmakers", []):
        for m in bm.get("markets", []):
            k = m.get("key")
            if k:
                discovered.add(k)

    discovered &= set(CORE_MARKETS)
    print(f"[odds_api] discovered markets on first event: {sorted(discovered) or 'NONE'}")
    if not discovered:
        print("[odds_api] discovery returned no usable prop markets for this plan/region")
        return pd.DataFrame()

    # 3) fetch each event once with the discovered list
    all_rows: List[Dict[str, Any]] = []
    for ev in evs:
        eid = ev.get("id")
        if not eid:
            continue
        try:
            jj = _event_props(key, eid, sorted(discovered))
            all_rows.extend(_rows_from_event_json(jj, discovered))
        except RuntimeError as rte:
            if "ODDS_API_OUT_OF_CREDITS" in str(rte):
                print("[odds_api] out of credits mid-loop; stopping.")
                break
            raise
        except Exception as e:
            print(f"[odds_api] event {eid} failed: {type(e).__name__}: {e}")
        time.sleep(0.12)  # polite pacing

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("[odds_api] 0 player-prop rows for this slate")
        return df

    keep = ["player","market","line","price","book","event_id","home_team","away_team","commence_time"]
    for k in keep:
        if k not in df.columns:
            df[k] = None
    df = df[keep].copy()

    # debug snapshot
    try:
        os.makedirs("outputs", exist_ok=True)
        df.to_csv("outputs/props_raw.csv", index=False)
    except Exception:
        pass

    print(f"[odds_api] assembled {len(df)} rows from {len(evs)} events; markets={sorted(discovered)}")
    return df

# compatibility aliases
get_props = build_props_frame = load_props = fetch_props
