# scripts/odds_api.py
import os
import time
import requests
import pandas as pd

# ---- Base paths ----
BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"

# Game-line markets are correct as-is
MARKETS_GAME = ["h2h", "spreads", "totals"]

# ✅ Use the official Odds API market keys for player props
MARKETS_PROPS = [
    "player_pass_yards",
    "player_pass_tds",
    "player_rush_yards",
    "player_rush_attempts",
    "player_receiving_yards",
    "player_receptions",
    "player_rush_and_rec_yards",
    "player_anytime_td",
]

def _get(path: str, **params):
    """
    Internal GET helper that:
      - injects the ODDS_API_KEY
      - prints a helpful error (with key redacted) before raising
    """
    key = os.getenv("ODDS_API_KEY")
    if not key:
        raise RuntimeError("ODDS_API_KEY not set (add it in GitHub: Settings → Secrets → Actions)")

    params.update({"apiKey": key})
    url = f"{BASE}{path}"

    r = requests.get(url, params=params, timeout=25)
    if r.status_code >= 400:
        # redact key from the logged URL
        safe_url = r.request.url.replace(key, "***")
        print("ERROR calling:", safe_url)
        print("Response:", r.text[:800])
    r.raise_for_status()
    return r.json()

def fetch_game_lines(regions: str = "us") -> pd.DataFrame:
    """
    Pulls current game lines (moneyline, spreads, totals) for NFL.
    Returns a DataFrame with: event_id, start, home, away, book, market, name, price, point
    """
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
        home = ev.get("home_team")
        away = ev.get("away_team")
        start = ev.get("commence_time")
        for bk in ev.get("bookmakers", []):
            book = bk.get("title") or bk.get("key")
            for m in bk.get("markets", []):
                mkey = m.get("key")
                for o in m.get("outcomes", []):
                    rows.append(
                        {
                            "event_id": eid,
                            "start": start,
                            "home": home,
                            "away": away,
                            "book": book,
                            "market": mkey,          # h2h / spreads / totals
                            "name": o.get("name"),   # team or selection name
                            "price": o.get("price"), # american odds
                            "point": o.get("point"), # spread/total number if present
                        }
                    )

    return pd.DataFrame(rows)

def fetch_props_all_events(
    regions: str = "us",
    markets: list[str] | None = None,
    sleep: float = 0.25,
) -> pd.DataFrame:
    """
    Enumerates NFL events (via /odds h2h) then calls the per-event endpoint to pull player props.
    Returns a DataFrame with: event_id, start, home, away, book, market, player_name_raw, outcome, price, point
    - 'outcome' is Over/Under (or Yes/No for TDs)
    - 'point' carries the line (and alternates appear as multiple rows with different points)
    """
    if markets is None:
        markets = MARKETS_PROPS

    # 1) list events so we have event IDs
    events = _get(
        "/odds",
        regions=regions,
        markets="h2h",           # minimal market just to enumerate events
        oddsFormat="american",
        dateFormat="iso",
    )

    all_rows = []
    for ev in events:
        eid = ev["id"]
        home = ev.get("home_team")
        away = ev.get("away_team")
        start = ev.get("commence_time")

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
                mkey = m.get("key")  # e.g., player_receiving_yards
                for o in m.get("outcomes", []):
                    # For props, Odds API puts the player name in 'description'
                    all_rows.append(
                        {
                            "event_id": eid,
                            "start": start,
                            "home": home,
                            "away": away,
                            "book": book,
                            "market": mkey,
                            "player_name_raw": o.get("description"),
                            "outcome": o.get("name"),  # "Over"/"Under" or "Yes"/"No"
                            "price": o.get("price"),
                            "point": o.get("point"),   # prop line; alternates show up as multiple points
                        }
                    )

        # Be a good citizen with a tiny delay between event calls
        time.sleep(sleep)

    return pd.DataFrame(all_rows)

# (Optional) quick utility if you ever want to list sports programmatically
def list_sports() -> pd.DataFrame:
    """
    Return the Odds API /sports listing as a DataFrame (handy for debugging).
    """
    js = _get("/sports")
    return pd.DataFrame(js)
