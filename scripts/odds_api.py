# scripts/odds_api.py
import os
import time
import requests
import pandas as pd

BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"

MARKETS_GAME = ["h2h", "spreads", "totals"]

# ✅ Correct v4 prop market keys (use *_yds, not *_yards)
MARKETS_PROPS = [
    "player_pass_yds",
    "player_pass_tds",
    "player_rush_yds",
    "player_rush_attempts",
    "player_rec_yds",
    "player_receptions",
    "player_rush_rec_yds",
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
        eid = ev["id"]; home = ev.get("home_team"); away = ev.get("away_team"); start = ev.get("commence_time")
        for bk in ev.get("bookmakers", []):
            book = bk.get("title") or bk.get("key")
            for m in bk.get("markets", []):
                mkey = m.get("key")
                for o in m.get("outcomes", []):
                    rows.append({
                        "event_id": eid, "start": start, "home": home, "away": away,
                        "book": book, "market": mkey,
                        "name": o.get("name"),
                        "price": o.get("price"),
                        "point": o.get("point"),
                    })
    return pd.DataFrame(rows)

def fetch_props_all_events(regions: str = "us", markets: list[str] | None = None, sleep: float = 0.25) -> pd.DataFrame:
    if markets is None:
        markets = MARKETS_PROPS

    # 1) enumerate events
    events = _get(
        "/odds",
        regions=regions,
        markets="h2h",
        oddsFormat="american",
        dateFormat="iso",
    )

    all_rows = []
    for ev in events:
        eid = ev["id"]; home = ev.get("home_team"); away = ev.get("away_team"); start = ev.get("commence_time")

        # 2) per-event props
        js = _get(
            f"/events/{eid}/odds",
            regions=regions,
            markets=",".join(markets),
            oddsFormat="american",
        )

        for bk in js.get("bookmakers", []):
            book = bk.get("title") or bk.get("key")
            for m in bk.get("markets", []):
                mkey = m.get("key")
                for o in m.get("outcomes", []):
                    all_rows.append({
                        "event_id": eid, "start": start, "home": home, "away": away,
                        "book": book, "market": mkey,
                        "player_name_raw": o.get("description"),
                        "outcome": o.get("name"),      # Over/Under or Yes/No
                        "price": o.get("price"),
                        "point": o.get("point"),       # alternates are multiple rows with different points
                    })
        time.sleep(sleep)

    return pd.DataFrame(all_rows)

def list_sports() -> pd.DataFrame:
    return pd.DataFrame(_get("/sports"))
