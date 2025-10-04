# scripts/odds_api.py
import os
import time
import requests
import pandas as pd

BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"

# Game markets
MARKETS_GAME = ["h2h", "spreads", "totals"]

# --- v4 Player prop market keys (per official list) ---
# https://the-odds-api.com/sports-odds-data/betting-markets.html
MARKETS_PROPS = [
    "player_pass_yds",
    "player_pass_tds",
    "player_rush_yds",
    "player_rush_attempts",
    "player_receptions",
    "player_reception_yds",        # <- receiving yards
    "player_rush_reception_yds",   # <- rush + reception yards
    "player_anytime_td",
]

def _get(path: str, **params):
    key = os.getenv("ODDS_API_KEY")
    if not key:
        raise RuntimeError("ODDS_API_KEY not set (add it in GitHub: Settings → Secrets → Actions)")
    params.update({"apiKey": key})
    url = f"{BASE}{path}"
    r = requests.get(url, params=params, timeout=25)
    if r.status_code >= 400:
        # redact key in logs
        safe_url = r.request.url.replace(key, "***")
        print("ERROR calling:", safe_url)
        print("Response:", r.text[:800])
    r.raise_for_status()
    return r.json()

def fetch_game_lines(regions: str = "us") -> pd.DataFrame:
    js = _get(
        "/odds",
        regions=regions,
        markets=",".join(MARKETS_GAME),
        oddsFormat="american",
        dateFormat="iso",
    )
    rows = []
    for ev in js:
        eid = ev["id"]
        home = ev.get("home_team"); away = ev.get("away_team"); start = ev.get("commence_time")
        for bk in ev.get("bookmakers", []):
            book = bk.get("title") or bk.get("key")
            for m in bk.get("markets", []):
                mkey = m.get("key")
                for o in m.get("outcomes", []):
                    rows.append({
                        "event_id": eid, "start": start, "home": home, "away": away,
                        "book": book, "market": mkey,           # h2h / spreads / totals
                        "name": o.get("name"),                   # team/selection
                        "price": o.get("price"),                 # American odds
                        "point": o.get("point"),                 # spread/total number if present
                    })
    return pd.DataFrame(rows)

def fetch_props_all_events(regions: str = "us",
                           markets: list[str] | None = None,
                           sleep: float = 0.25) -> pd.DataFrame:
    if markets is None:
        markets = MARKETS_PROPS

    # 1) enumerate events cheaply
    events = _get(
        "/odds",
        regions=regions,
        markets="h2h",
        oddsFormat="american",
        dateFormat="iso",
    )

    all_rows = []
    for ev in events:
        eid = ev["id"]
        home = ev.get("home_team"); away = ev.get("away_team"); start = ev.get("commence_time")

        # 2) pull props for this event
        js = _get(
            f"/events/{eid}/odds",
            regions=regions,
            markets=",".join(markets),
            oddsFormat="american",
        )

        for bk in js.get("bookmakers", []):
            book = bk.get("title") or bk.get("key")
            for m in bk.get("markets", []):
                mkey = m.get("key")              # e.g., player_reception_yds
                for o in m.get("outcomes", []):
                    all_rows.append({
                        "event_id": eid, "start": start, "home": home, "away": away,
                        "book": book, "market": mkey,
                        "player_name_raw": o.get("description"),  # player name
                        "outcome": o.get("name"),                 # Over/Under or Yes/No
                        "price": o.get("price"),
                        "point": o.get("point"),                  # alt lines appear as multiple rows
                    })
        time.sleep(sleep)

    return pd.DataFrame(all_rows)

def list_sports() -> pd.DataFrame:
    return pd.DataFrame(_get("/sports"))
