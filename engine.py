# engine.py
# Orchestrates fetching sportsbook props, normalizing, pricing, and writing outputs.
# Default behavior: NO TEAM FILTER (fetch all teams for the date window).
# You can still pass a comma-separated team list via teams="Chiefs,Jaguars" to filter if needed.

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Repo-local imports
from scripts.odds_api import get_props  # your existing fetcher
from scripts.normalize_props import normalize_props  # your existing normalizer
from scripts.pricing import price_props, write_outputs  # your pricing & writer


def _to_list_if_csv(s: Optional[str]) -> Optional[List[str]]:
    """Turn 'Chiefs,Jaguars' into ['Chiefs','Jaguars'], keep None/'' as None."""
    if s is None:
        return None
    if isinstance(s, list):
        return s
    s = str(s).strip()
    if not s or s.lower() == "all":
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def _coerce_args(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce run args to sane defaults (without applying model defaults)."""
    out = dict(kwargs)
    out.setdefault("season", None)            # e.g., 2025
    out.setdefault("date", "today")           # 'today' or 'YYYY-MM-DD'
    out.setdefault("window", "168h")          # lookahead window; e.g. '168h' = 7 days
    out.setdefault("cap", 0)                  # optional event cap (0 = no cap)
    out.setdefault("markets", None)           # comma list or None to use your odds_api defaults
    out.setdefault("books", "dk")             # e.g., 'dk' or 'dk,mgm'
    out.setdefault("order", "odds")           # odds_api ordering
    out.setdefault("teams", None)             # None/''/'all' => NO FILTER (fetch all teams)
    out.setdefault("selection", None)         # optional selection name filter (exact/regex in your odds_api)
    out.setdefault("write_dir", "outputs")    # output directory
    out.setdefault("basename", None)          # output file basename (auto if None)
    return out


def _log(msg: str) -> None:
    print(f"[engine] {msg}")


def run_pipeline(**kwargs) -> int:
    """
    Main entry point. Returns 0 on success, non-zero on failure.
    - Removes hardcoded team filters by default (fetch all teams).
    - You can still pass teams="Chiefs,Jaguars" to filter if desired.
    """
    args = _coerce_args(kwargs)

    season: Optional[int] = args["season"]
    date: str = args["date"]
    window: str = args["window"]
    cap: int = int(args["cap"])
    books: str = str(args["books"])
    order: str = str(args["order"])
    selection: Optional[str] = args["selection"]
    write_dir: str = str(args["write_dir"])
    basename: Optional[str] = args["basename"]

    # Parse markets (optional)
    markets = args["markets"]
    if isinstance(markets, str) and markets.strip():
        markets = [m.strip() for m in markets.split(",") if m.strip()]
    elif not markets:
        markets = None  # let odds_api apply its own default set

    # Parse team filter — DEFAULT: None (fetch all teams)
    team_filter = _to_list_if_csv(args.get("teams"))
    if team_filter is None:
        _log("team filter: None (fetching ALL teams for the window)")
    else:
        _log(f"team filter: {team_filter}")

    # Basename for outputs
    if not basename:
        # e.g., props_priced_2025-10-12 (or 'today')
        basename = f"props_priced_{date}"

    try:
        _log(f"fetching props… date={date} season={season}")
        _log(f"window={window} cap={cap}")
        _log(f"markets={','.join(markets) if markets else 'default'} order={order} books={books}")

        # ===========================
        # 1) Fetch sportsbook props
        # ===========================
        # NOTE: This call uses NO team filter by default (team_filter=None).
        # If you pass teams="Chiefs,Jaguars", we'll forward that list.
        df_props = get_props(
            date=date,
            season=season,
            window=window,
            cap=cap,
            markets=markets,
            order=order,
            books=books,
            team_filter=team_filter,
            selection_filter=selection,
        )

        if df_props is None or df_props.empty:
            _log("No props available to price (props fetch returned 0 rows).")
            return 1

        _log(f"fetched {len(df_props)} raw props from odds_api")

        # ===========================
        # 2) Normalize to model schema
        # ===========================
        df_norm = normalize_props(df_props)
        if df_norm is None or df_norm.empty:
            _log("Normalization produced 0 rows. Check provider mapping / markets.")
            return 1

        _log(f"normalized {len(df_norm)} rows")

        # ===========================
        # 3) Price
        # ===========================
        df_priced = price_props(df_norm)
        if df_priced is None or df_priced.empty:
            _log("Pricing produced 0 rows. Check required inputs & strict validators.")
            return 1

        _log(f"priced {len(df_priced)} rows")

        # ===========================
        # 4) Write outputs
        # ===========================
        Path(write_dir).mkdir(parents=True, exist_ok=True)
        write_outputs(df_priced, write_dir, basename)
        _log("pipeline complete.")
        return 0

    except Exception as e:
        _log(f"EXCEPTION: {e}")
        traceback.print_exc()
        return 1


# Optional manual test:
if __name__ == "__main__":
    # Allow quick local testing without run_model.py
    # Example:
    #   python engine.py
    #   python engine.py 2025-10-12 2025
    import argparse

    parser = argparse.ArgumentParser(description="Run pricing pipeline (no team filter by default).")
    parser.add_argument("date", nargs="?", default="today", help="YYYY-MM-DD or 'today'")
    parser.add_argument("season", nargs="?", type=int, default=None, help="Season year (e.g., 2025)")
    parser.add_argument("--window", default="168h", help="Lookahead window like '168h'")
    parser.add_argument("--cap", type=int, default=0, help="Max events to keep (0 = unlimited)")
    parser.add_argument("--markets", default=None, help="Comma-separated markets, or omit for default")
    parser.add_argument("--books", default="dk", help="Comma-separated books (e.g., 'dk,mgm')")
    parser.add_argument("--order", default="odds", help="Sorting for odds_api")
    parser.add_argument("--teams", default=None, help='Comma-separated teams or "all" (default=all)')
    parser.add_argument("--selection", default=None, help="Optional selection filter (exact/regex)")
    parser.add_argument("--write_dir", default="outputs", help="Output directory")
    parser.add_argument("--basename", default=None, help="Output file basename")
    args = parser.parse_args()

    sys.exit(
        run_pipeline(
            date=args.date,
            season=args.season,
            window=args.window,
            cap=args.cap,
            markets=args.markets,
            books=args.books,
            order=args.order,
            teams=args.teams,
            selection=args.selection,
            write_dir=args.write_dir,
            basename=args.basename,
        )
    )
