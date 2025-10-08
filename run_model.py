# run_model.py
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parent
DATA = REPO / "data"

def _read_soft(p: Path) -> pd.DataFrame:
    try:
        if p.exists() and p.stat().st_size > 0:
            return pd.read_csv(p)
    except Exception:
        pass
    return pd.DataFrame()
def _log(msg: str) -> None:
    print(f"[run_model] {msg}", flush=True)

def _to_list(s: Optional[str]) -> Optional[List[str]]:
    """
    Convert a comma-separated string to a list[str].
    Return None for empty/blank/None so engine can apply defaults.
    """
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def _positive_int_or_zero(x: str) -> int:
    v = int(x)
    if v < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return v


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the NFL props pipeline")

    p.add_argument("--date", required=True, help="Logical date (YYYY-MM-DD or 'today')")
    p.add_argument("--season", required=True, help="Season tag (e.g., 2025)")

    p.add_argument("--window", type=_positive_int_or_zero, default=36,
                   help="Only price events starting within N hours (0 = no filter)")
    p.add_argument("--cap", type=_positive_int_or_zero, default=0,
                   help="Hard cap on number of events to fetch (0 = no cap)")

    p.add_argument("--books", default="draftkings,fanduel,betmgm,caesars",
                   help="Comma-separated list of bookmaker keys")
    p.add_argument("--markets", default="",
                   help="Comma-separated markets override (blank = engine defaults)")
    p.add_argument("--order", default="odds",
                   help="Provider sorting (usually 'odds')")

    p.add_argument("--teams", default="",
                   help="Only include games where team name contains any of these (comma separated)")
    p.add_argument("--selection", default="",
                   help="Optional player-name filter (substring/regex)")

    # Optional paths if your engine supports them; safe to leave unused.
    p.add_argument("--write_dir", default="outputs",
                   help="Directory to write outputs (engine may handle)")
    p.add_argument("--basename", default="",
                   help="Optional basename for output files")

    args = p.parse_args(argv)
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # Build kwargs for the engine
    kwargs: Dict[str, Any] = {
        "date": args.date,
        "season": args.season,
        "hours": int(args.window),
        "cap": int(args.cap),
        "books": _to_list(args.books),
        "order": args.order or None,
        "markets": _to_list(args.markets) if args.markets else None,
        "teams": _to_list(args.teams) if args.teams else None,
        "selection": (args.selection or None),
        # Optional—engine can ignore if unsupported:
        "write_dir": args.write_dir,
        "basename": (args.basename or None),
    }

    # NEW: pass odds consensus tables into the engine (optional if your engine accepts them)
    odds_game = _read_soft(DATA / "odds_game_consensus.csv")
    odds_props = _read_soft(DATA / "odds_props_consensus.csv")

    print(f"[model] odds_game rows={len(odds_game)}; odds_props rows={len(odds_props)}")

    kwargs["odds_game_df"]  = odds_game
    kwargs["odds_props_df"] = odds_props
    
    # Import your engine dynamically
    try:
        engine = importlib.import_module("engine")
        _log(f"Loaded engine module from: {Path(getattr(engine, '__file__', 'engine.py')).resolve()}")
    except Exception as e:
        _log(f"failed to import engine: {e}")
        return 1

    # Pick the entrypoint the engine provides
    entry = None
    if hasattr(engine, "run_pipeline") and callable(getattr(engine, "run_pipeline")):
        entry = getattr(engine, "run_pipeline")
    elif hasattr(engine, "main") and callable(getattr(engine, "main")):
        entry = getattr(engine, "main")

    if entry is None:
        _log("engine module must define run_pipeline(**kwargs) or main(**kwargs)")
        return 1

    # Pretty log of what we're about to do
    _log("starting pipeline…")
    _log(f"team filter: {kwargs.get('teams') if kwargs.get('teams') else 'None (ALL teams in the date window)'}")
    _log(f"fetching props… date={args.date} season={args.season}")
    _log(f"window={args.window}h cap={args.cap}")
    _log(f"markets={'default' if not kwargs.get('markets') else ','.join(kwargs['markets'])} "
         f"order={args.order} books={','.join(kwargs['books'] or [])}")

    try:
        result = entry(**kwargs)
    except SystemExit as se:
        # If engine uses sys.exit
        code = int(getattr(se, "code", 1) or 1)
        return code
    except Exception as e:
        _log(f"pipeline crashed: {e}")
        return 1

    # If the engine returns a code, propagate it; otherwise success.
    if isinstance(result, int):
        return result
    return 0


if __name__ == "__main__":
    sys.exit(main())
