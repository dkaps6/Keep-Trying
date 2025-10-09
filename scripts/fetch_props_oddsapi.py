#!/usr/bin/env python3
import argparse, os, sys, time, requests
from pathlib import Path

ODDS_API_KEY = os.getenv("ODDS_API_KEY") or ""
BASE = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"
OUT = Path("outputs"); OUT.mkdir(parents=True, exist_ok=True)

def ensure_headers():
    with (OUT/"props_raw.csv").open("w", newline="") as f:
        f.write("event_id,commence_time,home_team,away_team,book,market,player,team,side,line,odds,last_update\n")
    with (OUT/"game_lines.csv").open("w", newline="") as f:
        f.write("event_id,home_team,away_team,home_wp,away_wp\n")

def _get(url, params, retries=2, backoff=0.5):
    for i in range(retries+1):
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (402,403,404):
            try: j = r.json()
            except Exception: j = {"error": r.text}
            print(f"Odds API error {r.status_code}: {j}", file=sys.stderr)
            return []
        time.sleep(backoff * (2**i))
    return []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport", default=SPORT)
    ap.add_argument("--season", default="")
    ap.add_argument("--date", default="")
    ap.add_argument("--books", default="draftkings,fanduel,betmgm,caesars")
    ap.add_argument("--markets", default="player_rec_yds,player_rush_yds,player_pass_yds,player_receptions,player_rush_att,player_anytime_td")
    ap.add_argument("--out", default="outputs/props_raw.csv")
    args = ap.parse_args()

    ensure_headers()
    if not ODDS_API_KEY:
        print("WARN: ODDS_API_KEY not set; writing header-only files.")
        return 0

    # events
    params = {"apiKey": ODDS_API_KEY, "regions": "us", "markets": "h2h"}
    if args.date:
        params["dateFormat"]="iso"; params["commenceTimeFrom"]=f"{args.date}T00:00:00Z"; params["commenceTimeTo"]=f"{args.date}T23:59:59Z"
    events = _get(f"{BASE}/sports/{SPORT}/odds", params) or []
    with (OUT/"game_lines.csv").open("w", newline="") as f:
        f.write("event_id,home_team,away_team,home_wp,away_wp\n")
        for ev in events:
            eid=ev.get("id"); home=ev.get("home_team"); away=ev.get("away_team")
            p_home=p_away=""
            try:
                outcomes = ev.get("bookmakers", [])[0].get("markets", [])[0].get("outcomes", [])
                for o in outcomes:
                    name=o.get("name"); price=o.get("price")
                    if price is None: continue
                    p = 100.0/(price+100.0) if price>0 else (-price)/((-price)+100.0)
                    if name==home: p_home=p
                    if name==away: p_away=p
                if p_home!="" and p_away!="":
                    s=p_home+p_away; p_home/=s; p_away/=s
            except Exception:
                pass
            f.write(f"{eid},{home},{away},{p_home},{p_away}\n")

    # props
    params = {"apiKey": ODDS_API_KEY, "regions":"us", "markets":args.markets}
    if args.books:
        params["bookmakers"]=args.books
    if args.date:
        params["dateFormat"]="iso"; params["commenceTimeFrom"]=f"{args.date}T00:00:00Z"; params["commenceTimeTo"]=f"{args.date}T23:59:59Z"
    rows=[]
    for ev in _get(f"{BASE}/sports/{SPORT}/odds", params) or []:
        eid=ev.get("id"); meta={"event_id":eid, "commence_time":ev.get("commence_time"), "home_team":ev.get("home_team"), "away_team":ev.get("away_team")}
        for bk in ev.get("bookmakers", []):
            book=bk.get("title") or bk.get("key")
            for mk in bk.get("markets", []):
                key=mk.get("key")
                for o in mk.get("outcomes", []):
                    rows.append({**meta,"book":book,"market":key,"player":o.get("description") or o.get("participant") or "", "team":o.get("team") or "","side":o.get("name"),"line":o.get("point"),"odds":o.get("price"),"last_update":mk.get("last_update")})
    with (OUT/"props_raw.csv").open("w", newline="") as f:
        f.write("event_id,commence_time,home_team,away_team,book,market,player,team,side,line,odds,last_update\n")
        for r in rows:
            f.write(",".join([str(r.get(k,"")) for k in ["event_id","commence_time","home_team","away_team","book","market","player","team","side","line","odds","last_update"]])+"\n")
    print("wrote outputs/props_raw.csv rows=", len(rows))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
