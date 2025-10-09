# external/nflverse_bundle/addons/fetch_odds.py
import os
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

API_KEY = os.getenv("THE_ODDS_API_KEY")
BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
assert API_KEY, "Missing THE_ODDS_API_KEY (GitHub secret)."

# ---- Inputs from workflow (see full-slate.yml) ----
# Markets flags
MARKETS      = (os.getenv("ODDS_MARKETS", "") or "").strip()
ALL_MARKETS  = os.getenv("ODDS_ALL_MARKETS", "false").lower() == "true"
PROPS_ONLY   = os.getenv("ODDS_PROPS_ONLY", "false").lower() == "true"
SIDES_ONLY   = os.getenv("ODDS_SIDES_ONLY", "false").lower() == "true"

# Books & time filters
BOOKS        = (os.getenv("ODDS_BOOKS", "") or "").strip()  # e.g. "draftkings,fanduel,betmgm,caesars"
ODDS_DATE    = (os.getenv("ODDS_DATE", "") or "").strip()   # YYYY-MM-DD (optional)
ODDS_FROM    = (os.getenv("ODDS_FROM", "") or "").strip()   # ISO-8601 datetime (optional)
ODDS_TO      = (os.getenv("ODDS_TO", "") or "").strip()     # ISO-8601 datetime (optional)

# If user picked “all markets” or left blank, treat as ALL for game lines.
all_for_game_lines = ALL_MARKETS or MARKETS == ""

# For props, the API typically requires explicit markets. If user didn’t provide any,
# use a broad default set that includes common player props + anytime TD.
DEFAULT_PROPS_MARKETS = ",".join([
    "player_pass_tds",
    "player_pass_yds",
    "player_rush_yds",
    "player_rec_yds",
    "player_receptions",
    "player_anytime_td",
])

# Decide what to fetch
fetch_game_lines = not PROPS_ONLY
fetch_props      = not SIDES_ONLY

os.makedirs("outputs", exist_ok=True)

def _req(path, **params):
    params.update(dict(apiKey=API_KEY))
    r = requests.get(f"{BASE}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def _flatten_game_lines(data):
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
                label = mk.get("description") or mk.get("name")
                last = mk.get("last_update")
                for oc in mk.get("outcomes", []):
                    player = oc.get("description") or label  # best-effort
                    rows.append(dict(
                        event_id=event_id,
                        commence_time=ct,
                        team=None,
                        player=player,
                        bookmaker=bname,
                        market=market,
                        label=label,
                        outcome=oc.get("name"),
                        price=oc.get("price"),
                        point=oc.get("point"),
                        last_update=last,
                    ))
    return pd.DataFrame(rows)

def _iso(s: str) -> str:
    """Return an ISO-8601 with 'Z' if the input is date-only or missing tz."""
    # Accept YYYY-MM-DD or full ISO; normalize to Z to match the API expectations
    if len(s) == 10:  # YYYY-MM-DD
        dt = datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    # Otherwise just return as-is; caller must pass a valid ISO
    return s

def _window_from_date(date_str: str):
    """Turn YYYY-MM-DD into [from, to) ISO-8601 range at UTC midnight."""
    d0 = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
    d1 = d0 + timedelta(days=1)
    return (d0.isoformat().replace("+00:00", "Z"),
            d1.isoformat().replace("+00:00", "Z"))

def _build_params(markets=None):
    p = dict(regions="us", oddsFormat="decimal")
    if BOOKS:
        p["bookmakers"] = BOOKS
    # Time filters: prefer explicit from/to; otherwise convert date into a 24h window.
    if ODDS_FROM:
        p["commenceTimeFrom"] = _iso(ODDS_FROM)
    if ODDS_TO:
        p["commenceTimeTo"] = _iso(ODDS_TO)
    if (not ODDS_FROM and not ODDS_TO) and ODDS_DATE:
        f, t = _window_from_date(ODDS_DATE)
        p["commenceTimeFrom"] = f
        p["commenceTimeTo"]   = t
    if markets:
        p["markets"] = markets
    return p

# ------------------ Fetch game lines ------------------
if fetch_game_lines:
    try:
        # If "all markets" (or blank), do NOT filter by markets for game lines.
        markets_param = None if all_for_game_lines else MARKETS
        params = _build_params(markets_param)
        data = _req("/odds", **params)
        df = _flatten_game_lines(data)
        df.to_csv("outputs/game_lines.csv", index=False)
        print(f"[odds] game_lines.csv rows={len(df)}")
    except Exception as e:
        print(f"[warn] game lines fetch failed: {e}")
        pd.DataFrame().to_csv("outputs/game_lines.csv", index=False)

# ------------------ Fetch player props ----------------
if fetch_props:
    try:
        # If user didn’t specify markets for props, use the broad default list.
        props_markets = MARKETS if MARKETS else DEFAULT_PROPS_MARKETS
        params = _build_params(props_markets)
        data = _req("/player-props", **params)
        df = _flatten_props(data)
        df.to_csv("outputs/props_raw.csv", index=False)
        print(f"[odds] props_raw.csv rows={len(df)}")
    except Exception as e:
        print(f"[warn] props fetch failed: {e}")
        pd.DataFrame().to_csv("outputs/props_raw.csv", index=False)

print("[odds] done.")

