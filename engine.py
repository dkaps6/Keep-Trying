# engine.py
# Orchestrates fetching sportsbook props, normalizing, pricing, and writing outputs.
# - No team filter by default (fetch ALL teams in the window)
# - Robust imports: odds fetcher and normalizer function names are auto-detected

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import importlib
import inspect

# ===========================
# Robust odds_api importer
# ===========================
def _import_odds_fetcher():
    """
    Try multiple names so we don't crash if scripts/odds_api.py changed.
    Returns a callable fetcher.
    """
    err = None
    try:
        from scripts.odds_api import get_props as fn  # type: ignore
        return fn
    except Exception as e:
        err = e
    try:
        from scripts.odds_api import fetch_props as fn  # type: ignore
        return fn
    except Exception as e:
        err = e
    try:
        from scripts.odds_api import get_props_df as fn  # type: ignore
        return fn
    except Exception as e:
        err = e
    # Last resort: import module and look for a callable that sounds right
    try:
        mod = importlib.import_module("scripts.odds_api")
        for name in ("get_props", "fetch_props", "get_props_df"):
            fn = getattr(mod, name, None)
            if callable(fn):
                return fn
    except Exception as e:
        err = e
    raise ImportError(
        "Could not import a props fetcher from scripts.odds_api "
        "(tried get_props, fetch_props, get_props_df)"
    ) from err


def _call_odds_fetcher(fn, **kwargs):
    """
    Call the fetcher with only the parameters it accepts.
    Transparently maps team_filter<->teams and selection_filter<->selection if needed.
    """
    sig = inspect.signature(fn)
    params = {}

    canonical = {
        "date": kwargs.get("date"),
        "season": kwargs.get("season"),
        "window": kwargs.get("window"),
        "cap": kwargs.get("cap"),
        "markets": kwargs.get("markets"),
        "order": kwargs.get("order"),
        "books": kwargs.get("books"),
        "team_filter": kwargs.get("team_filter"),
        "selection_filter": kwargs.get("selection"),
    }

    for k, v in canonical.items():
        if k in sig.parameters:
            params[k] = v

    # Friendly mapping: team_filter <-> teams
    if "team_filter" not in sig.parameters and "teams" in sig.parameters:
        params["teams"] = canonical["team_filter"]
    if "teams" not in sig.parameters and "team_filter" in sig.parameters:
        params["team_filter"] = canonical["team_filter"]

    # Friendly mapping: selection_filter <-> selection
    if "selection_filter" not in sig.parameters and "selection" in sig.parameters:
        params["selection"] = canonical["selection_filter"]
    if "selection" not in sig.parameters and "selection_filter" in sig.parameters:
        params["selection_filter"] = canonical["selection_filter"]

    return fn(**params)


# ===========================
# Robust normalizer importer
# ===========================
def _import_normalizer():
    """
    Try common export names in scripts/normalize_props.py and return a callable.
    """
    err = None
    try:
        from scripts.normalize_props import normalize_props as fn  # type: ignore
        return fn
    except Exception as e:
        err = e
    try:
        from scripts.normalize_props import normalize as fn  # type: ignore
        return fn
    except Exception as e:
        err = e
    try:
        from scripts.normalize_props import normalize_df as fn  # type: ignore
        return fn
    except Exception as e:
        err = e
    try:
        from scripts.normalize_props import to_model_schema as fn  # type: ignore
        return fn
    except Exception as e:
        err = e

    # Last resort: look for any callable that sounds like a normalizer
    try:
        mod = importlib.import_module("scripts.normalize_props")
        for name in ("normalize_props", "normalize", "normalize_df", "to_model_schema"):
            fn = getattr(mod, name, None)
            if callable(fn):
                return fn
    except Exception as e:
        err = e

    raise ImportError(
        "Could not import a normalizer from scripts.normalize_props "
        "(tried normalize_props, normalize, normalize_df, to_model_schema)"
    ) from err


# ===========================
# Pricing & writer (fixed names)
# ===========================
from scripts.pricing import price_props, write_outputs  # these exist in your repo

# ===========================
# Helpers
# ===========================
def _to_list_if_csv(s: Optional[str]) -> Optional[List[str]]:
    """'Chiefs,Jaguars' -> ['Chiefs','Jaguars']; None/''/'all' -> None (no filter)."""
    if s is None:
        return None
    if isinstance(s, list):
        return s
    s = str(s).strip()
    if not s or s.lower() == "all":
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def _coerce_args(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(kwargs)
    out.setdefault("season", None)
    out.setdefault("date", "today")
    out.setdefault("window", "168h")
    out.setdefault("cap", 0)
    out.setdefault("markets", None)
    out.setdefault("books", "dk")
    out.setdefault("order", "odds")
    out.setdefault("teams", None)         # default: no team filter
    out.setdefault("selection", None)
    out.setdefault("write_dir", "outputs")
    out.setdefault("basename", None)
    return out


def _log(msg: str) -> None:
    print(f"[engine] {msg}")


# ===========================
# Main
# ===========================
def run_pipeline(**kwargs) -> int:
    """
    Returns 0 on success, non-zero on failure.
    Default: NO TEAM FILTER (fetch all teams in the given date+window).
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

    # Markets
    markets = args["markets"]
    if isinstance(markets, str) and markets.strip():
        markets = [m.strip() for m in markets.split(",") if m.strip()]
    elif not markets:
        markets = None  # let odds_api decide defaults

    # Teams (NO FILTER by default)
    team_filter = _to_list_if_csv(args.get("teams"))
    if team_filter is None:
        _log("team filter: None (ALL teams in the date window)")
    else:
        _log(f"team filter: {team_filter}")

    if not basename:
        basename = f"props_priced_{date}"

    try:
        _log(f"fetching props… date={date} season={season}")
        _log(f"window={window} cap={cap}")
        _log(f"markets={','.join(markets) if markets else 'default'} order={order} books={books}")

        fetch_fn = _import_odds_fetcher()
        norm_fn  = _import_normalizer()

        # 1) Fetch sportsbook props
        df_props = _call_odds_fetcher(
            fetch_fn,
            date=date,
            season=season,
            window=window,
            cap=cap,
            markets=markets,
            order=order,
            books=books,
            team_filter=team_filter,
            selection=selection,
        )

        if df_props is None or (isinstance(df_props, (list, tuple)) and len(df_props) == 0):
            _log("No props available to price (props fetch returned 0 rows).")
            return 1

        if not isinstance(df_props, pd.DataFrame):
            df_props = pd.DataFrame(df_props)
        if df_props.empty:
            _log("No props available to price (props fetch returned empty DataFrame).")
            return 1

        _log(f"fetched {len(df_props)} raw props from odds_api")

        # 2) Normalize to model schema
        df_norm = norm_fn(df_props)
        if df_norm is None or (hasattr(df_norm, "empty") and df_norm.empty):
            _log("Normalization produced 0 rows. Check provider mapping / markets.")
            return 1

        _log(f"normalized {len(df_norm)} rows")

        # 3) Price
        df_priced = price_props(df_norm)
        if df_priced is None or df_priced.empty:
            _log("Pricing produced 0 rows. Check required inputs & strict validators.")
            return 1
        _log(f"priced {len(df_priced)} rows")

        # 4) Write outputs
        Path(write_dir).mkdir(parents=True, exist_ok=True)
        write_outputs(df_priced, write_dir, basename)
        _log("pipeline complete.")
        return 0

    except Exception as e:
        _log(f"EXCEPTION: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run pricing pipeline (no team filter by default).")
    parser.add_argument("date", nargs="?", default="today", help="YYYY-MM-DD or 'today'")
    parser.add_argument("season", nargs="?", type=int, default=None, help="Season year (e.g., 2025)")
    parser.add_argument("--window", default="168h", help="Lookahead window like '24h' or '168h'")
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
