import os
import requests
import pandas as pd
from datetime import datetime

API_KEY = os.getenv("THE_ODDS_API_KEY")
BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
assert API_KEY, "Missing THE_ODDS_API_KEY (GitHub secret)."

MARKETS = (os.getenv("ODDS_MARKETS", "") or "").strip()
ALL = os.getenv("ODDS_ALL_MARKETS", "false").lower() == "true"
PROPS_ONLY = os.getenv("ODDS_PROPS_ONLY", "false").lower() == "true"
SIDES_ONLY = os.getenv("ODDS_SIDES_ONLY", "false").lower() == "true"

os.makedirs("outputs", exist_ok=True)

if ALL or MARKETS == "":
    fetch_game_lines = not PROPS_ONLY
    fetch_props = not SIDES_ONLY
    markets_param = None
else:
    fetch_game_lines = not PROPS_ONLY
    fetch_props = not SIDES_ONLY
    markets_param = MARKETS

def _req(path, **params):
    params.update(dict(apiKey=API_KEY))
    r = requests.get(f"{BASE}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def _flatten_game_lines(data):
    # record_path browsing: bookmakers -> markets -> outcomes
    rows = []
    for ev in data:
        event_id = ev.get("id")
        ct = ev.get("commence_time")
        home = ev.get("home_team")
        away = ev.get("away_team")
        for bk in ev.get("bookmakers", []):
            bname = bk.get("title")
            for mk in bk.get("markets", []):
                market = mk.get("key")
                last = mk.get("last_update")
                for oc in mk.get("outcomes", []):
                    rows.append(dict(
                        event_id=event_id,
                        commence_time=ct,
                        home_team=home,
                        away_team=away,
                        bookmaker=bname,
                        market=market,
                        outcome=oc.get("name"),
                        price=oc.get("price"),
                        point=oc.get("point"),
                        last_update=last,
                    ))
    return pd.DataFrame(rows)

def _flatten_props(data):
    # record_path: bookmakers -> markets -> outcomes; props include labels (player names)
    rows = []
    for ev in data:
        event_id = ev.get("id")
        ct = ev.get("commence_time")
        home = ev.get("home_team")
        away = ev.get("away_team")
        for bk in ev.get("bookmakers", []):
            bname = bk.get("title")
            for mk in bk.get("markets", []):
                market = mk.get("key")
                label = mk.get("description") or mk.get("name")  # may contain player name depending on feed
                last = mk.get("last_update")
                for oc in mk.get("outcomes", []):
                    # try to split team/player (not always present in v4, but description often holds it)
                    outcome_name = oc.get("name")
                    player = oc.get("description") or label
                    rows.append(dict(
                        event_id=event_id,
                        commence_time=ct,
                        team=None,
                        player=player,
                        bookmaker=bname,
                        market=market,
                        label=label,
                        outcome=outcome_name,
                        price=oc.get("price"),
                        point=oc.get("point"),
                        last_update=last,
                    ))
    return pd.DataFrame(rows)

def _build_params(markets=None):
    p = dict(regions="us", bookmakers=BOOKS, oddsFormat="decimal")
    if markets:
        p["markets"] = markets
    if DATE:
        p["date"] = DATE
    return p

if fetch_game_lines:
    try:
        params = _build_params(markets_param)
        data = _req("/odds", **params)
        df = _flatten_game_lines(data)
        df.to_csv("outputs/game_lines.csv", index=False)
        print(f"[odds] game_lines.csv rows={len(df)}")
    except Exception as e:
        print(f"[warn] game lines fetch failed: {e}")
        pd.DataFrame().to_csv("outputs/game_lines.csv", index=False)

if fetch_props:
    try:
        params = _build_params(markets_param)
        data = _req("/player_props", **params)
        df = _flatten_props(data)
        df.to_csv("outputs/props_raw.csv", index=False)
        print(f"[odds] props_raw.csv rows={len(df)}")
    except Exception as e:
        print(f"[warn] props fetch failed: {e}")
        pd.DataFrame().to_csv("outputs/props_raw.csv", index=False)

print("[odds] done.")
