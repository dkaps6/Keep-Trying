# scripts/props_to_csv.py
# Helper that uses YOUR existing props_hybrid.get_props(...) to dump tidy CSVs
# without modifying the props_hybrid module or the engine orchestration.

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

# Use your existing module & function (keep signatures intact)
from .props_hybrid import get_props
from .market_keys import NFL_DEFAULT_MARKETS  # your canonical keys set

def _normalize_list_arg(val: str | None) -> list[str] | None:
    if not val:
        return None
    return [x.strip() for x in val.split(",") if x.strip()]

def build_anchor(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep a single 'anchor' row per (event, market_internal, player, line)
    by taking the first book deterministically. Pricing/devig stays downstream.
    """
    if df.empty:
        return df

    keep_cols = [
        "id", "commence_time", "home_team", "away_team",
        "bookmaker_key", "bookmaker_title",
        "market_api", "market_internal",
        "player", "side", "line", "price",
    ]

    dfa = (
        df[keep_cols]
        .sort_values(["id", "market_internal", "player", "line", "bookmaker_key"])
        .drop_duplicates(subset=["id", "market_internal", "player", "line"], keep="first")
    )
    return dfa

def main():
    ap = argparse.ArgumentParser(description="Dump player props to CSV via existing props_hybrid.get_props()")
    ap.add_argument("--date", default="today", help="ISO date (YYYY-MM-DD) or 'today'")
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--hours", type=int, default=96)
    ap.add_argument("--books", default="draftkings,fanduel,betmgm,caesars,pointsbetus")
    ap.add_argument("--markets", default=None, help="Comma-separated list; defaults to NFL_DEFAULT_MARKETS")
    ap.add_argument("--team-filter", default=None, help="Comma-separated list like 'DAL,SF'")
    ap.add_argument("--event-ids", default=None, help="Comma-separated list of OddsAPI event IDs")
    ap.add_argument("--order", default="odds", help="Pass-through to get_props (keep your default)")
    ap.add_argument("--out-raw", default="outputs/props_raw.csv")
    ap.add_argument("--out-anchor", default="outputs/props_market_anchor.csv")
    args = ap.parse_args()

    markets = _normalize_list_arg(args.markets) or list(NFL_DEFAULT_MARKETS)
    team_filter = _normalize_list_arg(args.team_filter)
    event_ids = _normalize_list_arg(args.event_ids)

    # Call YOUR function (no changes to its behavior)
    df = get_props(
        date=args.date,
        season=args.season,
        hours=args.hours,
        books=args.books,
        markets=markets,
        team_filter=team_filter,
        event_ids=event_ids,
        order=args.order,
    )

    Path("outputs").mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_raw, index=False)

    anchor = build_anchor(df)
    anchor.to_csv(args.out_anchor, index=False)

    print(f"[props_to_csv] wrote {len(df):,} rows → {args.out_raw}")
    print(f"[props_to_csv] wrote {len(anchor):,} rows → {args.out_anchor}")

if __name__ == "__main__":
    main()
