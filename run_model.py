from __future__ import annotations
import argparse
from engine import run_pipeline

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True, help="e.g., 2025")
    ap.add_argument("--date", default="", help="ISO date YYYY-MM-DD")
    args = ap.parse_args()
    run_pipeline(season=args.season, date=args.date)
