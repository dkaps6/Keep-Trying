# scripts/props_to_csv.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

# your existing hybrid getter
from .props_hybrid import get_props

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYY-MM-DD or 'today'")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--hours", type=int, default=36, help="window hours (deprecated name)")
    ap.add_argument("--window_hours", type=int, default=None, help="preferred arg name")
    ap.add_argument("--books", default="draftkings,fanduel,betmgm,caesars")
    ap.add_argument("--markets", default="")
    ap.add_argument("--order", default="odds")
    ap.add_argument("--team-filter", default="")
    ap.add_argument("--out", default="outputs/props_raw.csv")
    args = ap.parse_args()

    window = args.window_hours if args.window_hours is not None else args.hours

    Path("outputs").mkdir(exist_ok=True, parents=True)

    df = get_props(
        date=args.date,
        season=args.season,
        window_hours=window,
        books=args.books,
        markets=args.markets or None,
        order=args.order,
        team_filter=args.team_filter or None,
    )
    df.to_csv(args.out, index=False)
    print(f"[props_to_csv] ✅ wrote {len(df)} rows → {args.out}")

if __name__ == "__main__":
    main()
