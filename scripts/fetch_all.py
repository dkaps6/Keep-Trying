# scripts/fetch_all.py  (tiny orchestrator)
from __future__ import annotations
import argparse, json
from pathlib import Path
from .fetch_nfl_data import build_team_form
from .fetch_weather import fetch_weather

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    args = ap.parse_args()

    Path("metrics").mkdir(parents=True, exist_ok=True)
    team = build_team_form(args.season)
    team.to_csv("metrics/team_form.csv", index=False)
    print("[fetch_all] team_form.csv ✓", len(team))

    wx = fetch_weather()
    wx.to_csv("data/weather.csv", index=False)
    print("[fetch_all] weather.csv ✓", len(wx))

    Path("metrics/fetch_status.json").write_text(json.dumps({
        "season": args.season,
        "team_form_rows": int(len(team)),
        "weather_rows": int(len(wx)),
    }, indent=2))

if __name__ == "__main__":
    main()
