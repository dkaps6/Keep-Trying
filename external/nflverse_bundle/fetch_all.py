#!/usr/bin/env python3
"""
Fetch all (nflverse bundle) — version-tolerant

- Works with recent nflreadpy/nfl_data_py where `file_type` is no longer accepted.
- Gracefully skips datasets if a loader isn't available in your installed version.
- Writes CSVs under external/nflverse_bundle/outputs/<dataset_group>/...

Usage:
  python external/nflverse_bundle/fetch_all.py --season 2025
  python external/nflverse_bundle/fetch_all.py --start 2019 --end 2025
  python external/nflverse_bundle/fetch_all.py --seasons 2019,2020,2021
  python external/nflverse_bundle/fetch_all.py --skip-pbp
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path
from typing import Iterable, List, Dict, Any, Callable, Optional

import pandas as pd

# ------------------------------------------------------------------------------
# Imports: prefer nflreadpy, fall back to nfl_data_py if needed
# ------------------------------------------------------------------------------
nfl = None  # type: ignore
_load_source = "unknown"

try:
    import nflreadpy as nfl  # modern wrapper
    _load_source = "nflreadpy"
except Exception:
    try:
        import nfl_data_py as nfl  # older lib
        _load_source = "nfl_data_py"
    except Exception:
        nfl = None  # type: ignore


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------

def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_csv(df: Optional[pd.DataFrame], out_path: Path) -> None:
    """Write df to CSV (or create a 0-row file with headers if df is None/empty)."""
    safe_mkdir(out_path.parent)
    if df is None:
        # create an empty file (no headers) so downstream validators can flag it
        out_path.write_text("")
        print(f"[write] empty → {out_path}")
        return
    try:
        if df.empty:
            # write header if present, so pandas can read columns later
            df.head(0).to_csv(out_path, index=False)
            print(f"[write] 0 rows (headers only) → {out_path}")
        else:
            df.to_csv(out_path, index=False)
            print(f"[write] {len(df):,} rows → {out_path}")
    except Exception as e:
        print(f"[write] failed {out_path.name}: {e}")
        out_path.write_text("")


def safe_load(func: Callable[..., pd.DataFrame], **kwargs) -> Optional[pd.DataFrame]:
    """
    Try with file_type=csv for old versions, fallback to no file_type for new versions.
    Returns a DataFrame or None (on failure).
    """
    try:
        # some very old versions expect file_type
        return func(**kwargs, file_type="csv")
    except TypeError:
        # modern versions don't accept file_type
        try:
            kwargs.pop("file_type", None)
            return func(**kwargs)
        except Exception as e:
            print(f"[load] {getattr(func, '__name__', 'callable')} failed: {e}")
            return None
    except Exception as e:
        print(f"[load] {getattr(func, '__name__', 'callable')} failed: {e}")
        return None


def try_call(module: Any, name: str) -> Optional[Callable[..., pd.DataFrame]]:
    """Return a callable if it exists on the module else None."""
    if module is None:
        return None
    fn = getattr(module, name, None)
    if callable(fn):
        return fn
    return None


def seasons_from_args(args) -> List[int]:
    if args.seasons:
        return [int(s.strip()) for s in args.seasons.split(",") if s.strip()]
    if args.start and args.end:
        return list(range(int(args.start), int(args.end) + 1))
    return [int(args.season)]


# ------------------------------------------------------------------------------
# Fetch groups
# ------------------------------------------------------------------------------

def fetch_team_stats(root: Path, seasons: List[int]) -> None:
    outdir = root / "outputs" / "team_stats"
    safe_mkdir(outdir)

    for scope, fname in [
        ("week", "team_stats_week_{season}.csv"),
        ("reg",  "team_stats_reg_{season}.csv"),
        ("post", "team_stats_post_{season}.csv"),
    ]:
        fn = try_call(nfl, "load_team_stats")
        if fn is None:
            print(f"[team_stats/{scope}] skipped: loader not available in {_load_source}")
            continue
        for season in seasons:
            try:
                df = safe_load(fn, seasons=[season], scope=scope)
                write_csv(df, outdir / fname.format(season=season))
            except TypeError as te:
                # older API may not support scope kw; try without it
                print(f"[team_stats/{scope}] retry without scope: {te}")
                df = safe_load(fn, seasons=[season])
                write_csv(df, outdir / fname.format(season=season))


def fetch_player_stats(root: Path, seasons: List[int]) -> None:
    outdir = root / "outputs" / "player_stats"
    safe_mkdir(outdir)

    for scope, fname in [
        ("week", "player_stats_week_{season}.csv"),
        ("reg",  "player_stats_reg_{season}.csv"),
        ("post", "player_stats_post_{season}.csv"),
    ]:
        fn = try_call(nfl, "load_player_stats")
        if fn is None:
            print(f"[player_stats/{scope}] skipped: loader not available in {_load_source}")
            continue
        for season in seasons:
            try:
                df = safe_load(fn, seasons=[season], scope=scope)
                write_csv(df, outdir / fname.format(season=season))
            except TypeError as te:
                print(f"[player_stats/{scope}] retry without scope: {te}")
                df = safe_load(fn, seasons=[season])
                write_csv(df, outdir / fname.format(season=season))


def fetch_nextgen(root: Path, seasons: List[int]) -> None:
    outdir = root / "outputs" / "nextgen"
    safe_mkdir(outdir)
    fn = try_call(nfl, "load_nextgen_stats")
    if fn is None:
        print(f"[nextgen/*] skipped: loader not available in {_load_source}")
        return

    # Some newer libs want stat_type, others have separate functions. We try stat_type first.
    for stat_type, fname in [
        ("passing",   "nextgen_passing_{season}.csv"),
        ("receiving", "nextgen_receiving_{season}.csv"),
        ("rushing",   "nextgen_rushing_{season}.csv"),
    ]:
        for season in seasons:
            try:
                df = safe_load(fn, seasons=[season], stat_type=stat_type)
                write_csv(df, outdir / fname.format(season=season))
            except TypeError as te:
                print(f"[nextgen/{stat_type}] retry without stat_type: {te}")
                df = safe_load(fn, seasons=[season])
                write_csv(df, outdir / fname.format(season=season))


def fetch_depth_charts(root: Path, seasons: List[int]) -> None:
    outdir = root / "outputs" / "depth_charts"
    safe_mkdir(outdir)
    fn = try_call(nfl, "load_depth_charts")
    if fn is None:
        print(f"[depth_charts] skipped: loader not available in {_load_source}")
        return
    for season in seasons:
        df = safe_load(fn, seasons=[season])
        write_csv(df, outdir / f"depth_charts_{season}.csv")


def fetch_snap_counts(root: Path, seasons: List[int]) -> None:
    outdir = root / "outputs" / "snap_counts"
    safe_mkdir(outdir)
    fn = try_call(nfl, "load_snap_counts")
    if fn is None:
        print(f"[snap_counts] skipped: loader not available in {_load_source}")
        return
    for season in seasons:
        df = safe_load(fn, seasons=[season])
        write_csv(df, outdir / f"snap_counts_{season}.csv")


def fetch_rosters(root: Path, seasons: List[int]) -> None:
    outdir = root / "outputs" / "rosters"
    safe_mkdir(outdir)

    fn = try_call(nfl, "load_rosters")
    if fn is None:
        print(f"[rosters] skipped: loader not available in {_load_source}")
    else:
        for season in seasons:
            df = safe_load(fn, seasons=[season])
            write_csv(df, outdir / f"rosters_{season}.csv")

    fn_weekly = try_call(nfl, "load_rosters_weekly")
    if fn_weekly is None:
        print(f"[rosters_weekly] skipped: loader not available in {_load_source}")
    else:
        outdir_w = root / "outputs" / "rosters_weekly"
        safe_mkdir(outdir_w)
        for season in seasons:
            df = safe_load(fn_weekly, seasons=[season])
            write_csv(df, outdir_w / f"rosters_weekly_{season}.csv")


def fetch_schedules(root: Path, seasons: List[int]) -> None:
    outdir = root / "outputs" / "schedules"
    safe_mkdir(outdir)
    # different libs name it differently
    fn = (try_call(nfl, "load_schedules")
          or try_call(nfl, "import_schedules"))
    if fn is None:
        print(f"[schedules] skipped: loader not available in {_load_source}")
        return
    for season in seasons:
        df = safe_load(fn, seasons=[season])
        write_csv(df, outdir / f"schedules_{season}.csv")


def fetch_pfr_advstats(root: Path, seasons: List[int]) -> None:
    outdir = root / "outputs" / "pfr_advstats"
    safe_mkdir(outdir)
    fn = try_call(nfl, "load_pfr_advstats")
    if fn is None:
        print(f"[pfr_advstats] skipped: loader not available in {_load_source}")
        return
    for season in seasons:
        df = safe_load(fn, seasons=[season])
        write_csv(df, outdir / f"pfr_advstats_{season}.csv")


def fetch_espn_qbr(root: Path, seasons: List[int]) -> None:
    outdir = root / "outputs" / "espn_qbr"
    safe_mkdir(outdir)
    fn = try_call(nfl, "load_espn_qbr")
    if fn is None:
        print(f"[espn_qbr] skipped: loader not available in {_load_source}")
        return
    for season in seasons:
        df = safe_load(fn, seasons=[season])
        write_csv(df, outdir / f"espn_qbr_{season}.csv")


def fetch_participation(root: Path, seasons: List[int]) -> None:
    outdir = root / "outputs" / "participation"
    safe_mkdir(outdir)
    fn = try_call(nfl, "load_participation")
    if fn is None:
        print(f"[participation] skipped: loader not available in {_load_source}")
        return
    for season in seasons:
        df = safe_load(fn, seasons=[season])
        write_csv(df, outdir / f"participation_{season}.csv")


def fetch_officials(root: Path, seasons: List[int]) -> None:
    outdir = root / "outputs" / "officials"
    safe_mkdir(outdir)
    fn = try_call(nfl, "load_officials")
    if fn is None:
        print(f"[officials] skipped: loader not available in {_load_source}")
        return
    for season in seasons:
        df = safe_load(fn, seasons=[season])
        write_csv(df, outdir / f"officials_{season}.csv")


def fetch_trades(root: Path, seasons: List[int]) -> None:
    outdir = root / "outputs" / "trades"
    safe_mkdir(outdir)
    # Some versions want year, some want seasons
    fn = try_call(nfl, "load_trades") or try_call(nfl, "import_transactions")
    if fn is None:
        print(f"[trades] skipped: loader not available in {_load_source}")
        return
    for season in seasons:
        try:
            df = safe_load(fn, seasons=[season])
        except TypeError:
            df = safe_load(fn, year=season)
        write_csv(df, outdir / f"trades_{season}.csv")


def fetch_pbp(root: Path, seasons: List[int], enabled: bool) -> None:
    if not enabled:
        print("[pbp] skipped by flag --skip-pbp")
        return
    outdir = root / "outputs" / "pbp"
    safe_mkdir(outdir)
    fn = try_call(nfl, "load_pbp") or try_call(nfl, "import_pbp")
    if fn is None:
        print(f"[pbp] skipped: loader not available in {_load_source}")
        return
    # Many versions accept a list for seasons
    df = safe_load(fn, seasons=seasons)
    write_csv(df, outdir / f"pbp_{min(seasons)}_{max(seasons)}.csv")


def fetch_ftn_charting(root: Path, seasons: List[int]) -> None:
    outdir = root / "outputs" / "ftn_charting"
    safe_mkdir(outdir)
    fn = try_call(nfl, "load_ftn_charting")
    if fn is None:
        print(f"[ftn_charting] skipped: loader not available in {_load_source}")
        return
    for season in seasons:
        df = safe_load(fn, seasons=[season])
        write_csv(df, outdir / f"ftn_charting_{season}.csv")


def fetch_injuries(root: Path, seasons: List[int]) -> None:
    outdir = root / "outputs" / "injuries"
    safe_mkdir(outdir)
    fn = try_call(nfl, "load_injuries")
    if fn is None:
        print(f"[injuries] skipped: loader not available in {_load_source}")
        return
    for season in seasons:
        df = safe_load(fn, seasons=[season])
        write_csv(df, outdir / f"injuries_{season}.csv")


# ------------------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025, help="Single season (if start/end/seasons not provided)")
    ap.add_argument("--start", type=int, default=None, help="Start season (inclusive)")
    ap.add_argument("--end", type=int, default=None, help="End season (inclusive)")
    ap.add_argument("--seasons", type=str, default=None, help="Comma-separated seasons (e.g. 2019,2020,2021)")
    ap.add_argument("--skip-pbp", action="store_true", help="Skip play-by-play fetch (big file)")
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parent
    os.chdir(ROOT)

    if nfl is None:
        print("::error ::Neither nflreadpy nor nfl_data_py could be imported. Install one in requirements.txt")
        return 2

    seasons = seasons_from_args(args)
    print(f"[env] loader={_load_source}, seasons={seasons}")

    try:
        fetch_team_stats(ROOT, seasons)
        fetch_player_stats(ROOT, seasons)
        fetch_nextgen(ROOT, seasons)
        fetch_depth_charts(ROOT, seasons)
        fetch_snap_counts(ROOT, seasons)
        fetch_rosters(ROOT, seasons)
        fetch_schedules(ROOT, seasons)
        fetch_pfr_advstats(ROOT, seasons)
        fetch_espn_qbr(ROOT, seasons)
        fetch_participation(ROOT, seasons)
        fetch_officials(ROOT, seasons)
        fetch_trades(ROOT, seasons)
        fetch_pbp(ROOT, seasons, enabled=(not args.skip_pbp))
        fetch_ftn_charting(ROOT, seasons)
        fetch_injuries(ROOT, seasons)
        print("✅ Done. CSVs saved under: outputs")
        return 0
    except Exception as e:
        print(f"::error ::fetch_all failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
