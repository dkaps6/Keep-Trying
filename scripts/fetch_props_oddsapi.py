import os, sys, argparse, pathlib, pandas as pd, requests

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", default="draftkings,fanduel,betmgm,caesars")
    ap.add_argument("--markets", default="")
    ap.add_argument("--date", default="")
    ap.add_argument("--out", default="outputs/props_raw.csv")
    args = ap.parse_args()

    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["sport_key","commence_time","book","market","player","line","over_odds","under_odds"]
    pd.DataFrame(columns=cols).to_csv(out, index=False)

    key = os.getenv("ODDS_API_KEY", "").strip()
    if not key:
        print("[oddsapi] ODDS_API_KEY not set; wrote header-only CSV and exiting cleanly.")
        return 0

    base = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
    params = {
        "regions": "us",
        "oddsFormat": "american",
        "markets": "h2h,spreads,totals,player_pass_yds,player_rec_yds,player_rush_yds,player_receptions",
        "apiKey": key
    }
    if args.date:
        params["dateFormat"] = "iso"
        params["commenceTimeFrom"] = args.date

    try:
        r = requests.get(base, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[oddsapi] request failed: {e}")
        return 0

    rows = []
    for game in data:
        commence = game.get("commence_time")
        for bk in game.get("bookmakers", []):
            book_key = bk.get("key")
            for mk in bk.get("markets", []):
                mkt = mk.get("key")
                for ou in mk.get("outcomes", []):
                    name = ou.get("description") or ou.get("name")
                    line = ou.get("point")
                    price = ou.get("price")
                    if name and line is not None and price is not None:
                        rows.append([game.get("sport_key","nfl"), commence, book_key, mkt, name, line, price, None])
    pd.DataFrame(rows, columns=cols).to_csv(out, index=False)
    print(f"[oddsapi] wrote {out} rows={len(rows)}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
