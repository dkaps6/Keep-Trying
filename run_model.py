# run_model.py
from __future__ import annotations
import argparse
from engine import run_pipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True, help="e.g., 2025")
    ap.add_argument("--date", default="", help="ISO date YYYY-MM-DD (optional)")
    ap.add_argument("--books", default="draftkings,fanduel,betmgm,caesars")
    ap.add_argument("--markets", default="", help="optional override, comma-separated")
    args = ap.parse_args()

    run_pipeline(
        season=str(args.season),
        date=str(args.date),
        books=[b.strip() for b in args.books.split(",") if b.strip()],
        markets=[m.strip() for m in args.markets.split(",") if m.strip()] or None,
    )

if __name__ == "__main__":
    main()
