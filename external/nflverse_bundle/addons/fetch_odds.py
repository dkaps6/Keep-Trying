import os
import requests
import pandas as pd
from datetime import datetime

# -------------------------------
# CONFIG & ENVIRONMENT VARIABLES
# -------------------------------

API_KEY = os.getenv("THE_ODDS_API_KEY")
BASE_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"

# Inputs from GitHub Actions workflow
DATE = os.getenv("ODDS_DATE", "")               # YYYY-MM-DD optional filter
BOOKS = os.getenv("ODDS_BOOKMAKERS", "draftkings,fanDuel,betmgm,caesars")
MARKETS = (os.getenv("ODDS_MARKETS", "") or "").strip()
ALL = os.getenv("ODDS_ALL_MARKETS", "false").lower() == "true"
PROPS_ONLY = os.getenv("ODDS_PROPS_ONLY", "false").lower() == "true"
SIDES_ONLY = os.getenv("ODDS_SIDES_ONLY", "false").lower() == "true"

# Output directories
os.makedirs("outputs", exist_ok=True)

# -------------------------------
# MARKET LOGIC
# -------------------------------

if ALL or MARKETS == "":
    # Fetch everything
    fetch_game_lines = not PROPS_ONLY
    fetch_props = not SIDES_ONLY
    markets_param = None
else:
    markets_param = MARKETS
    fetch_game_lines = not PROPS_ONLY
    fetch_props = not SIDES_ONLY

print(f"[odds] Fetching data with configuration:")
print(f"  ALL_MARKETS: {ALL}")
print(f"  MARKETS: {MARKETS}")
print(f"  PROPS_ONLY: {PROPS_ONLY}")
print(f"  SIDES_ONLY: {SIDES_ONLY}")
print(f"  BOOKS: {BOOKS}")
print(f"  DATE FILTER: {DATE}")
print()

# -------------------------------
# HELPERS
# -------------------------------

def fetch_odds(endpoint, params, out_file):
    """Generic fetcher with logging and graceful fallback."""
    url = f"{BASE_URL}/{endpoint}/odds/"
    params["apiKey"] = API_KEY

    print(f"[fetch_odds] GET {url}")
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if not data:
            print(f"[warn] No data returned for {endpoint}")
            pd.DataFrame().to_csv(out_file, index=False)
            return
        df = pd.json_normalize(data)
        df.to_csv(out_file, index=False)
        print(f"[ok] Saved {len(df)} rows to {out_file}")
    except Exception as e:
        print(f"[error] Failed to fetch {endpoint}: {e}")
        pd.DataFrame().to_csv(out_file, index=False)

# -------------------------------
# MAIN EXECUTION
# -------------------------------

params = {
    "regions": "us",
    "markets": markets_param,
    "bookmakers": BOOKS,
    "oddsFormat": "decimal"
}

if DATE:
    params["date"] = DATE

if fetch_game_lines:
    fetch_odds("odds", params, "outputs/game_lines.csv")

if fetch_props:
    fetch_odds("player_props", params, "outputs/props_raw.csv")

print("[done] Odds fetching complete.")
