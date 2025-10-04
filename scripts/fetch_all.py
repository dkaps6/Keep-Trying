# scripts/fetch_all.py
from __future__ import annotations
import argparse
import os
from scripts.fetch_nfl_data import build_team_form, build_player_form, build_id_map
from scripts.fetch_weather import build_weather_csv

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--weeks", type=str, default="1-18")  # "1-5" or "all"
    return p.parse_args()

def ensure_dirs():
    os.makedirs("metrics", exist_ok=True)
    os.makedirs("inputs", exist_ok=True)
    os.makedirs("data", exist_ok=True)

def main():
    args = parse_args()
    ensure_dirs()

    print("▶ Building team_form.csv ...")
    tf = build_team_form(season=args.season, weeks=args.weeks)
    tf.to_csv("metrics/team_form.csv", index=False)
    print("Wrote metrics/team_form.csv", len(tf), "rows")

    print("▶ Building player_form.csv ...")
    pf = build_player_form(season=args.season, weeks=args.weeks)
    pf.to_csv("metrics/player_form.csv", index=False)
    print("Wrote metrics/player_form.csv", len(pf), "rows")

    print("▶ Building id_map.csv ...")
    ids = build_id_map(season=args.season)
    ids.to_csv("metrics/id_map.csv", index=False)
    print("Wrote metrics/id_map.csv", len(ids), "rows")

    print("▶ Building weather.csv (optional) ...")
    try:
        w = build_weather_csv(season=args.season)
        w.to_csv("inputs/weather.csv", index=False)
        print("Wrote inputs/weather.csv", len(w), "rows")
    except Exception as e:
        print("⚠️ Weather fetch failed (skipping):", e)

if __name__ == "__main__":
    main()
