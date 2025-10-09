#!/usr/bin/env python3
import os, sys, time, json, csv
from pathlib import Path
import requests

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
SPORT = os.getenv("ODDS_SPORT", "americanfootball_nfl")
REGIONS = os.getenv("ODDS_REGIONS", "us")
BOOKMAKERS = os.getenv("ODDS_BOOKMAKERS", "")  # optional comma-list
MARKETS = os.getenv("ODDS_MARKETS", "player_pass_yds,player_rush_yds,player_rec_yds,player_receptions,player_rush_att,player_pass_tds,player_anytime_td").split(",")

OUT = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT / "props_raw.csv"

BASE = "https://api.the-odds-api.com/v4/sports"

def fetch_market(market: str):
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": REGIONS,
        "markets": market,
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    if BOOKMAKERS:
        params["bookmakers"] = BOOKMAKERS
    url = f"{BASE}/{SPORT}/odds"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def flatten_row(event, bookmaker_key, market_key, out):
    # event: per-game
    # bookmaker: contains markets -> outcomes per player
    for bm in event.get("bookmakers", []):
        if bookmaker_key and bm.get("key") != bookmaker_key:
            continue
        bk = bm.get("key")
        up = bm.get("last_update")
        for m in bm.get("markets", []):
            if m.get("key") != market_key:
                continue
            for o in m.get("outcomes", []):
                row = {
                    "event_id": event.get("id", ""),
                    "commence_time": event.get("commence_time", ""),
                    "home_team": event.get("home_team", ""),
                    "away_team": event.get("away_team", ""),
                    "sport_key": event.get("sport_key", ""),
                    "book": bk,
                    "market": m.get("key", ""),
                    "player": o.get("description", ""),
                    "team": o.get("name", ""),   # some books put team here; sometimes blank
                    "price": o.get("price"),     # American odds
                    "point": o.get("point"),     # line (e.g., 64.5 yards)
                    "last_update": up,
                }
                out.append(row)

def main():
    if not ODDS_API_KEY:
        print("ERROR: ODDS_API_KEY not set", file=sys.stderr)
        sys.exit(2)

    all_rows = []
    # Pull each market separately (Odds API requires specific market key per call)
    for i, mk in enumerate(MARKETS, 1):
        mk = mk.strip()
        if not mk:
            continue
        try:
            print(f"[props] fetching market={mk}")
            data = fetch_market(mk)
            for ev in data:
                # If you want to filter by specific bookmakers, set BOOKMAKERS in env.
                flatten_row(ev, None, mk, all_rows)
            # gentle delay to avoid rate-limit
            time.sleep(0.8)
        except requests.HTTPError as e:
            print(f"WARN: market={mk} HTTP {e}", file=sys.stderr)
        except Exception as e:
            print(f"WARN: market={mk} {e}", file=sys.stderr)

    # write CSV
    cols = ["event_id","commence_time","home_team","away_team","sport_key","book","market","player","team","point","price","last_update"]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in all_rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"[write] {OUT_CSV} rows={len(all_rows)}")

if __name__ == "__main__":
    sys.exit(main())
