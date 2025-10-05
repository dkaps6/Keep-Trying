# scripts/odds_api.py
from __future__ import annotations

import os
import time
from typing import Any, Dict, Iterable, List, Set

import pandas as pd
import requests

BASE = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"
HEADERS = {"User-Agent": "keep-trying/props-pipeline python-requests"}

# A broad superset of NFL player-prop market keys (per docs)
# We'll probe with these once, discover what's actually available,
# then use only the discovered set for the full slate.
MARKET_CANDIDATES: List[str] = [
    # passing
    "player_pass_yds", "player_pass_tds",
    "player_pass_attempts", "player_pass_completions",
    "player_pass_interceptions", "player_pass_longest_completion",
    "player_pass_rush_yds", "player_pass_rush_reception_yds",
    # rushing
    "player_rush_yds", "player_rush_attempts",
    "player_rush_longest", "player_rush_tds",
    "player_rush_reception_yds", "player_rush_reception_tds",
    # receiving
    "player_receptions", "player_reception_yds",
    "player_reception_longest", "player_reception_tds",
    # kickers / defense (sometimes available)
    "player_field_goals", "player_kicking_points",
    "player_sacks", "player_solo_tackles", "player_tackles_assists",
    # scorers
    "player_anytime_td", "player_1st_td", "player_last_td",
]

# Accept any of these "positive" outcomes as our anchor row
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

    # credits log
    rem = r.headers.get("x-requests-remaining")
    used = r.headers.get("x-requests-used")
    lim = r.headers.get("x-requests-limit")
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
    """Enumerate events via featured market (h2h). Costs 1 request."""
    url = f"{BASE}/sports/{SPORT}/odds"
    params = dict(
        apiKey=key,
        markets="h2h",
        regions="us,us2",
        oddsFormat="american",
        dateFormat="iso",
    )
    try:
        data = _get(url, params)
        return data if isinstance(data, list) else []
    except RuntimeError:  # out of credits
        raise
    except Exception as e:
        print(f"[odds_api] events list failed: {e}")
        return []


def _event_props_once(key: str, event_id: str, markets: Iterable[str]) -> Dict[str, Any]:
    """Call event endpoint once for a specific set of markets."""
    url = f"{BASE}/sports/{SPORT}/events/{event_id}/odds"
    params = dict(
        apiKey=key,
        regions="us,us2",
        markets=",".join(markets),
        oddsFormat="american",
        dateFormat="iso",
    )
    return _get(url, params)


def _discover_markets(key: str, event_id: str) -> Set[str]:
    """
    Probe the first event with MARKET_CANDIDATES (chunked) and return the
    subset of market keys that actually appear in the response.
    """
    discovered: Set[str] = set()
    CHUNK = 12  # keep query strings reasonable
    candidates = list(dict.fromkeys(MARKET_CANDIDATES))  # de-dupe but preserve order
    for i in range(0, len(candidates), CHUNK):
        chunk = candidates[i : i + CHUNK]
        try:
            j = _event_props_once(key, event_id, chunk)
        except RuntimeError:
            raise
        except Exception as e:
            print(f"[odds_api] discovery chunk failed ({chunk[:3]}...): {e}")
            continue

        for bm in j.get("bookmakers", []):
            for mkt in bm.get("markets", []):
                discovered.add(mkt.get("key"))

        time.sleep(0.12)

    # Keep only the ones we know how to price downstream
    discovered = {m for m in discovered if m in set(MARKET_CANDIDATES)}
    print(f"[odds_api] discovered markets for event {event_id}: {sorted(discovered)}")
    return discovered


def _rows_from_event_json(j: Dict[str, Any], allowed_markets: Set[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for bm in j.get("bookmakers", []):
        book = bm.get("key")
        for mkt in bm.get("markets", []):
            mkey = mkt.get("key")
            if mkey not in allowed_markets:
                continue
            for oc in mkt.get("outcomes", []):
                desc = (oc.get("description") or "").lower()
                if desc not in _POS_DESCS:
                    continue
                player = oc.get("name")
                line = oc.get("point")
                price = oc.get("price")
                if player is None or line is None or price is None:
                    continue
                rows.append(
                    {
                        "event_id": j.get("id"),
                        "home_team": j.get("home_team"),
                        "away_team": j.get("away_team"),
                        "commence_time": j.get("commence_time"),
                        "book": book,
                        "market": mkey,
                        "player": player,
                        "line": float(line),
                        "price": float(price),
                    }
                )
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

    try:
        events = _list_events(key)
    except RuntimeError as rte:
        if "ODDS_API_OUT_OF_CREDITS" in str(rte):
            print("[odds_api] out of usage credits; aborting Odds API pulls this run.")
            return pd.DataFrame()
        raise

    if not events:
        print("[odds_api] 0 events listed")
        return pd.DataFrame()

    # 1) Discover which prop markets are actually available for this slate
    first_event_id = next((e.get("id") for e in events if e.get("id")), None)
    if not first_event_id:
        print("[odds_api] no valid event id found in listing")
        return pd.DataFrame()

    try:
        discovered = _discover_markets(key, first_event_id)
    except RuntimeError as rte:
        if "ODDS_API_OUT_OF_CREDITS" in str(rte):
            print("[odds_api] out of usage credits during discovery; aborting.")
            return pd.DataFrame()
        raise

    if not discovered:
        print("[odds_api] discovery found no player prop markets (plan/region/book coverage?)")
        return pd.DataFrame()

    # 2) Fetch props for each event using only discovered keys
    all_rows: List[Dict[str, Any]] = []
    for ev in events:
        eid = ev.get("id")
        if not eid:
            continue
        try:
            j = _event_props_once(key, eid, sorted(discovered))
            all_rows.extend(_rows_from_event_json(j, discovered))
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

    keep = [
        "player",
        "market",
        "line",
        "price",
        "book",
        "event_id",
        "home_team",
        "away_team",
        "commence_time",
    ]
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

    print(f"[odds_api] assembled {len(df)} rows")
    return df


# Backwards-compat aliases
get_props = build_props_frame = load_props = fetch_props
