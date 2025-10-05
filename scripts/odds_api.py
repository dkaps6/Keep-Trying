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

# ---- TUNABLE SAFETY LIMITS (via env) ----------------------------------------
EVENT_WINDOW_HOURS = int(os.environ.get("ODDS_API_EVENT_WINDOW_HOURS", "48"))  # only call games in next N hours
MAX_EVENTS = int(os.environ.get("ODDS_API_MAX_EVENTS", "8"))                    # hard cap on events to price
REGIONS = os.environ.get("ODDS_API_REGIONS", "us,us2")                          # regions to request
# -----------------------------------------------------------------------------

# Broad candidate set; we discover what's actually available for your key/region
MARKET_CANDIDATES: List[str] = [
    "player_pass_yds", "player_pass_tds",
    "player_pass_attempts", "player_pass_completions",
    "player_pass_interceptions", "player_pass_longest_completion",
    "player_rush_yds", "player_rush_attempts", "player_rush_longest",
    "player_receptions", "player_reception_yds", "player_reception_longest",
    "player_anytime_td", "player_1st_td", "player_last_td",
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
    url = f"{BASE}/sports/{SPORT}/odds"
    params = dict(
        apiKey=key,
        markets="h2h",             # cheap feature market to enumerate events
        regions=REGIONS,
        oddsFormat="american",
        dateFormat="iso",
    )
    data = _get(url, params)
    return data if isinstance(data, list) else []


def _parse_time(ts: str) -> datetime | None:
    try:
        # Samples look like "2025-10-05T17:00:00Z"
        if ts.endswith("Z"):
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _filter_events_window(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Only keep events starting within EVENT_WINDOW_HOURS from now (UTC)."""
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=EVENT_WINDOW_HOURS)
    kept = []
    for ev in events:
        ct = ev.get("commence_time")
        dt = _parse_time(ct) if ct else None
        if dt and now <= dt <= cutoff:
            kept.append(ev)
    print(f"[odds_api] events visible={len(events)}  in_window({EVENT_WINDOW_HOURS}h)={len(kept)}  cap={MAX_EVENTS}")
    return kept[:MAX_EVENTS]


def _event_once(key: str, event_id: str, markets: Iterable[str]) -> Dict[str, Any]:
    url = f"{BASE}/sports/{SPORT}/events/{event_id}/odds"
    params = dict(
        apiKey=key,
        regions=REGIONS,
        markets=",".join(markets),
        oddsFormat="american",
        dateFormat="iso",
    )
    return _get(url, params)


def _discover_markets(key: str, event_id: str) -> Set[str]:
    """Probe the first event in small chunks; keep only markets we actually see."""
    discovered: Set[str] = set()
    CHUNK = 10
    candidates = list(dict.fromkeys(MARKET_CANDIDATES))
    for i in range(0, len(candidates), CHUNK):
        chunk = candidates[i:i+CHUNK]
        try:
            j = _event_once(key, event_id, chunk)
        except RuntimeError:
            raise
        except Exception as e:
            print(f"[odds_api] discovery chunk failed ({chunk[:3]}...): {e}")
            continue

        for bm in j.get("bookmakers", []):
            for mkt in bm.get("markets", []):
                k = mkt.get("key")
                if k:
                    discovered.add(k)
        time.sleep(0.12)

    discovered &= set(MARKET_CANDIDATES)
    print(f"[odds_api] discovered markets for event {event_id}: {sorted(discovered) or 'NONE'}")
    return discovered


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
                if desc not in {"over", "yes", ""}:
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
    Returns normalized rows:
      [player, market, line, price, book, event_id, home_team, away_team, commence_time]
    Writes outputs/props_raw.csv for inspection.
    """
    key = _key()
    if not key:
        return pd.DataFrame()

    # 1) Enumerate and filter events to a small, near-term set
    try:
        evs = _list_events(key)
    except RuntimeError as rte:
        if "ODDS_API_OUT_OF_CREDITS" in str(rte):
            print("[odds_api] out of usage credits; aborting Odds API pulls this run.")
            return pd.DataFrame()
        raise

    if not evs:
        print("[odds_api] 0 events listed")
        return pd.DataFrame()

    evs = _filter_events_window(evs)
    if not evs:
        print("[odds_api] no events within window; nothing to fetch")
        return pd.DataFrame()

    # 2) Discover prop markets on the first event
    first_id = evs[0].get("id")
    if not first_id:
        print("[odds_api] first event has no id")
        return pd.DataFrame()

    try:
        markets = _discover_markets(key, first_id)
    except RuntimeError as rte:
        if "ODDS_API_OUT_OF_CREDITS" in str(rte):
            print("[odds_api] out of credits during discovery; aborting.")
            return pd.DataFrame()
        raise

    if not markets:
        print("[odds_api] discovery returned no player-prop markets for this plan/region")
        return pd.DataFrame()

    # 3) Fetch those markets for each capped event
    all_rows: List[Dict[str, Any]] = []
    for ev in evs:
        eid = ev.get("id")
        if not eid:
            continue
        try:
            j = _event_once(key, eid, sorted(markets))
            all_rows.extend(_rows_from_event_json(j, markets))
        except RuntimeError as rte:
            if "ODDS_API_OUT_OF_CREDITS" in str(rte):
                print("[odds_api] out of credits mid-loop; stopping.")
                break
            raise
        except Exception as e:
            print(f"[odds_api] event {eid} props failed: {e}")
        time.sleep(0.12)

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("[odds_api] 0 player-prop rows")
        return df

    keep = ["player","market","line","price","book","event_id","home_team","away_team","commence_time"]
    for k in keep:
        if k not in df.columns:
            df[k] = None
    df = df[keep].copy()

    # snapshot for debugging
    try:
        os.makedirs("outputs", exist_ok=True)
        df.to_csv("outputs/props_raw.csv", index=False)
    except Exception:
        pass

    print(f"[odds_api] assembled {len(df)} rows from {len(evs)} events, markets={sorted(markets)}")
    return df


# Back-compat aliases
get_props = build_props_frame = load_props = fetch_props
