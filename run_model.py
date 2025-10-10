# run_model.py
from __future__ import annotations
import argparse
from engine import run_pipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True, help="Season year, e.g., 2025")
    ap.add_argument("--date", default="", help="ISO date (YYYY-MM-DD) for slate and odds window")
    ap.add_argument("--books", default="draftkings,fanduel,betmgm,caesars",
                    help="Comma-separated sportsbook keys")
    ap.add_argument("--markets", default="",
                    help="Comma-separated markets override (else pipeline default)")
    args = ap.parse_args()

    # Hand through to the orchestrator
    run_pipeline(
        season=args.season,
        date=args.date,
        books=args.books.split(",") if args.books else None,
        markets=args.markets.split(",") if args.markets else None,
    )

if __name__ == "__main__":
    main()
