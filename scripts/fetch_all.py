# scripts/fetch_all.py
from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd

from .fetch_nfl_data import build_team_form
from .fetch_weather import fetch_weather

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True, help="Season to build team_form from nfl_data_py")
    ap.add_argument("--write-status", default="metrics/fetch_status.json")
    args = ap.parse_args()

    Path("metrics").mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)

    # 1) Team form (EPA splits, pressure/sack, pace, AY/Att, neutral pass rate, PROE)
    team_form = build_team_form(args.season)
    team_form.to_csv("metrics/team_form.csv", index=False)
    print(f"[fetch_all] team_form.csv ✓ rows={len(team_form)}")

    # 2) Weather (needs data/stadiums.csv + outputs/game_lines.csv)
    weather = fetch_weather()
    weather.to_csv("data/weather.csv", index=False)
    print(f"[fetch_all] weather.csv ✓ rows={len(weather)}")

    status = {
        "season": args.season,
        "paths": {
            "metrics/team_form.csv": len(team_form),
            "data/weather.csv": len(weather),
        }
    }
    Path(args.write_status).write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))

if __name__ == "__main__":
    main()
