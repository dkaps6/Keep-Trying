#!/usr/bin/env python3
"""
Fallback: scrape ESPN injuries page to CSV for a given week (or all).
Note: ESPN does not publish an official free JSON injuries API endpoint;
this scraper pulls the public injuries listings page.
"""
import argparse, os, re
import pandas as pd
import requests
from bs4 import BeautifulSoup

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def parse_injuries_page(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    rows = []
    # ESPN groups injuries by team sections
    for teamsec in soup.select("section.Team"):
        team_name_tag = teamsec.find(["h2","h1"])
        team = team_name_tag.get_text(strip=True) if team_name_tag else "Unknown"
        for tr in teamsec.select("tr"):
            tds = [td.get_text(" ", strip=True) for td in tr.select("td")]
            if len(tds) >= 4:
                player, pos, status, comment = tds[0], tds[1], tds[2], tds[3]
                rows.append({"team": team, "player": player, "pos": pos, "status": status, "comment": comment})
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", default="-1", help="Week number or -1 for ALL (ESPN aggregate view)")
    ap.add_argument("--out", default="outputs/injuries/espn")
    args = ap.parse_args()
    ensure_dir(args.out)
    url = f"https://www.espn.com/nfl/injuries/_/week/{args.week}"
    df = parse_injuries_page(url)
    out = os.path.join(args.out, f"injuries_espn_week_{args.week}.csv")
    df.to_csv(out, index=False)
    print("✅ Wrote", out)

if __name__ == "__main__":
    main()
