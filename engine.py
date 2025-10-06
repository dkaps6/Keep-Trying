# engine.py
# Orchestrates fetching sportsbook props, normalizing, pricing, and writing outputs.
# - No team filter by default (fetch all teams in the date window)
# - Robust import: works whether odds_api exposes get_props(), fetch_props(), or get_props_df()
# - You can still pass teams="Chiefs,Jaguars" from run_model.py to filter if desired

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# ===========================
# Robust odds_api importer
# ===========================
def _import_odds_fetcher():
    """
    Try multiple common names so we don't crash if your odds_api changed.
    Returns a callable fetcher from scripts.odds_api.
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
    raise ImportError(
        "Could not import a props fetcher from scripts.odds_api "
        "(tried get_props, fetch_props, get_props_df)"
    ) from err


def _call_odds_fetcher(fn, **kwargs):
    """
    Call the fetcher with only the parameters it accepts.
    Transparently maps team_filter<->teams and selection_filter<->selection if needed.
    """
    import inspect

    sig = inspect.signature(fn)
    params = {}

    # Primary args we support at the engine level:
    canonical = {
        "date": kwargs.get("date"),
        "season": kwargs.get("season"),
        "window": kwargs.get("window"),
        "cap": kwargs.get("cap"),
        "markets": kwargs.get("markets"),
        "order": kwargs.get("order"),
        "books": kwargs.get("books"),
        "team_filter": kwargs.get("team_filter"),      # None or list[str]
        "selection_filter": kwargs.get("selection"),   # string or None
    }

    # Direct pass-through where accepted
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
# Repo-local model imports
# ===========================
from scripts.normalize_props import normalize_props  # your existing normalizer
from scripts.pricing import price_props, write_outputs  # pricing & writer


def _to_list_if_csv(s: Optional[str]) -> Optional[List[str]]:
    """Turn 'Chiefs,Jaguars' into ['Chiefs','Jaguars']; None/''/'all' -> None (no filter)."""
    if s is None:
        return None
    if isinstance(s, list):
        return s
    s = str(s).strip()
    if not s or s.lower() == "all":
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def _coerce_args(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce run args to sane pass-through defaults."""
    out = dict(kwargs)
    out.setdefault("season", None)            # e.g. 2025
    out.setdefault("date", "today")           # 'today' or 'YYYY-MM-DD'
    out.setdefault("window", "168h")          # lookahead window (e.g. '24h', '168h')
    out.setdefault("cap", 0)                  # 0 = no cap
    out.setdefault("markets", None)           # None -> odds_api default set
    out.setdefault("books", "dk")             # e.g. 'dk,mgm'
    out.setdefault("order", "odds")           # odds_api ordering
    out.setdefault("teams", None)             # None/''/'all' => NO team filter
    out.setdefault("selection", None)         # optional selection filter
    out.setdefault("write_dir", "outputs")
    out.setdefault("basename", None)
    return out


def _log(msg: str) -> None:
    print(f"[engine] {msg}")


def run_pipeline(**kwargs) -> int:
    """
    Main entry point. Returns 0 on success, non-zero on failure.
    Default: NO TEAM FILTER (fetch all teams within the window).
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
        markets = None  # let odds_api decide default set

    # Teams (NO FILTER by default)
    team_filter = _to_list_if_csv(args.get("teams"))
    if team_filter is None:
        _log("team filter: None (ALL teams in the date window)")
    else:
        _log(f"team filter: {team_filter}")

    # Basename for outputs
    if not basename:
        basename = f"props_priced_{date}"

    try:
        _log(f"fetching props… date={date} season={season}")
        _log(f"window={window} cap={cap}")
        _log(f"markets={','.join(markets) if markets else 'default'} order={order} books={books}")

        fetch_fn = _import_odds_fetcher()

        # 1) Fetch sportsbook props (no team filter by default)
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

        if df_props is None or len(df_props) == 0:
            _log("No props available to price (props fetch returned 0 rows).")
            return 1

        # Normalize input to DataFrame just in case fetcher returned a list
        if not isinstance(df_props, pd.DataFrame):
            df_props = pd.DataFrame(df_props)
        _log(f"fetched {len(df_props)} raw props from odds_api")

        # 2) Normalize to model schema
        df_norm = normalize_props(df_props)
        if df_norm is None or df_norm.empty:
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
    # Quick local test without run_model.py
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
