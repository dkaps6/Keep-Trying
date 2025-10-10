from __future__ import annotations
import argparse
from engine import run_pipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True, help="e.g., 2025")
    ap.add_argument("--date", default="", help="YYYY-MM-DD")
    ap.add_argument("--books", default="draftkings,fanduel,betmgm,caesars")
    ap.add_argument("--markets", default="")
    a = ap.parse_args()
    run_pipeline(
        season=str(a.season),
        date=str(a.date),
        books=[b.strip() for b in a.books.split(",") if b.strip()],
        markets=[m.strip() for m in a.markets.split(",") if m.strip()] or None,
    )

if __name__ == "__main__":
    main()
