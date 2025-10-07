#!/usr/bin/env python3
"""
Fetch all (nflverse bundle) — version/engine tolerant

- Handles old/new nflreadpy/nfl_data_py signatures (no 'file_type', some lack 'scope')
- Works with pandas or polars DataFrames
- Writes CSVs under external/nflverse_bundle/outputs/** (creates empty headers if empty/None)

Usage:
  python external/nflverse_bundle/fetch_all.py --season 2025
  python external/nflverse_bundle/fetch_all.py --start 2019 --end 2025
  python external/nflverse_bundle/fetch_all.py --seasons 2019,2020,2021
  python external/nflverse_bundle/fetch_all.py --skip-pbp
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional

# try to import modern first
_loader_name = "unknown"
nfl = None  # type: ignore
try:
    import nflreadpy as nfl  # modern
    _loader_name = "nflreadpy"
except Exception:
    try:
        import nfl_data_py as nfl  # older
        _loader_name = "nfl_data_py"
    except Exception:
        nfl = None  # type: ignore

# pandas is needed even when source is polars (we convert)
import pandas as pd


# ----------------------- utils -----------------------

def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def to_pandas(df: Any) -> Optional[pd.DataFrame]:
    """Return a pandas DataFrame or None."""
    if df is None:
        return None
    try:
        if isinstance(df, pd.DataFrame):
            return df
        # polars or other: try common adapters
        if hasattr(df, "to_pandas"):
            return df.to_pandas()
        if hasattr(df, "to_dicts"):
            import pandas as _pd
            return _pd.DataFrame(df.to_dicts())  # polars alt
        # last resort: duck typing via iterator of mappings
        return pd.DataFrame(df)
    except Exception:
        return None


def nrows(df: Any) -> int:
    try:
        return len(df)
    except Exception:
        return 0


def write_csv(df: Any, out_path: Path) -> None:
    safe_mkdir(out_path.parent)
    pdf = to_pandas(df)
    if pdf is None:
        out_path.write_text("")  # no headers: validator will flag
        print(f"[write] empty → {out_path}")
        return
    try:
        if pdf.empty:
            pdf.head(0).to_csv(out_path, index=False)
            print(f"[write] 0 rows (headers only) → {out_path}")
        else:
            pdf.to_csv(out_path, index=False)
            print(f"[write] {len(pdf):,} rows → {out_path}")
    except Exception as e:
        print(f"[write] failed {out_path.name}: {e}")
        out_path.write_text("")


def try_call(module: Any, name: str) -> Optional[Callable[..., Any]]:
    if module is None:
        return None
    fn = getattr(module, name, None)
    return fn if callable(fn) else None


def safe_load(func: Callable[..., Any], **kwargs) -> Optional[Any]:
    """
    Call loader in a way that works across versions:
    1) Try with file_type="csv" (ancient)
    2) Fallback to no file_type
    Return object or None (and print a concise reason).
    """
    try:
        return func(**kwargs, file_type="csv")
    except TypeError:
        try:
            kwargs.pop("file_type", None)
            return func(**kwargs)
        except Exception as e:
            print(f"[load] {getattr(func, '__name__','fn')} failed: {e}")
            return None
    except Exception as e:
        print(f"[load] {getattr(func, '__name__','fn')} failed: {e}")
        return None


def seasons_from_args(args) -> List[int]:
    if args.seasons:
        return [int(s.strip()) for s in args.seasons.split(",") if s.strip()]
    if args.start and args.end:
        return list(range(int(args.start), int(args.end) + 1))
    return [int(args.season)]


# ----------------------- fetchers -----------------------

def fetch_team_stats(root: Path, seasons: List[int]) -> None:
    outdir = root / "outputs" / "team_stats"; safe_mkdir(outdir)
    fn = try_call(nfl, "load_team_stats")
    if fn is None:
        print(f"[team_stats] skipped: loader not available in {_loader_name}")
        return
    for scope, fname in [("week","team_stats_week_{s}.csv"),
                         ("reg","team_stats_reg_{s}.csv"),
                         ("post","team_stats_post_{s}.csv")]:
        for s in seasons:
            df = safe_load(fn, seasons=[s], scope=scope)
            if df is None:  # older signature without 'scope'
                df = safe_load(fn, seasons=[s])
            write_csv(df, outdir / fname.format(s=s))


def fetch_player_stats(root: Path, seasons: List[int]) -> None:
    outdir = root / "outputs" / "player_stats"; safe_mkdir(outdir)
    fn = try_call(nfl, "load_player_stats")
    if fn is None:
        print(f"[player_stats] skipped: loader not available in {_loader_name}")
        return
    for scope, fname in [("week","player_stats_week_{s}.csv"),
                         ("reg","player_stats_reg_{s}.csv"),
                         ("post","player_stats_post_{s}.csv")]:
        for s in seasons:
            df = safe_load(fn, seasons=[s], scope=scope)
            if df is None:
                df = safe_load(fn, seasons=[s])
            write_csv(df, outdir / fname.format(s=s))


def fetch_nextgen(root: Path, seasons: List[int]) -> None:
    outdir = root / "outputs" / "nextgen"; safe_mkdir(outdir)
    fn = try_call(nfl, "load_nextgen_stats")
    if fn is None:
        print(f"[nextgen] skipped: loader not available in {_loader_name}")
        return
    for stat_type, fname in [("passing","nextgen_passing_{s}.csv"),
                             ("receiving","nextgen_receiving_{s}.csv"),
                             ("rushing","nextgen_rushing_{s}.csv")]:
        for s in seasons:
            df = safe_load(fn, seasons=[s], stat_type=stat_type)
            if df is None:
                df = safe_load(fn, seasons=[s])
            write_csv(df, outdir / fname.format(s=s))


def fetch_depth_charts(root: Path, seasons: List[int]) -> None:
    outdir = root / "outputs" / "depth_charts"; safe_mkdir(outdir)
    fn = try_call(nfl, "load_depth_charts")
    if fn is None:
        print(f"[depth_charts] skipped: loader not available in {_loader_name}")
        return
    for s in seasons:
        write_csv(safe_load(fn, seasons=[s]), outdir / f"depth_charts_{s}.csv")


def fetch_snap_counts(root: Path, seasons: List[int]) -> None:
    outdir = root / "outputs" / "snap_counts"; safe_mkdir(outdir)
    fn = try_call(nfl, "load_snap_counts")
    if fn is None:
        print(f"[snap_counts] skipped: loader not available in {_loader_name}")
        return
    for s in seasons:
        write_csv(safe_load(fn, seasons=[s]), outdir / f"snap_counts_{s}.csv")


def fetch_rosters(root: Path, seasons: List[int]) -> None:
    fn_r = try_call(nfl, "load_rosters")
    fn_rw = try_call(nfl, "load_rosters_weekly")
    if fn_r is None and fn_rw is None:
        print(f"[rosters] skipped: loader not available in {_loader_name}")
        return
    out_r = root / "outputs" / "rosters"; safe_mkdir(out_r)
    out_rw = root / "outputs" / "rosters_weekly"; safe_mkdir(out_rw)
    for s in seasons:
        if fn_r:
            write_csv(safe_load(fn_r, seasons=[s]), out_r / f"rosters_{s}.csv")
        if fn_rw:
            write_csv(safe_load(fn_rw, seasons=[s]), out_rw / f"rosters_weekly_{s}.csv")


def fetch_schedules(root: Path, seasons: List[int]) -> None:
    outdir = root / "outputs" / "schedules"; safe_mkdir(outdir)
    fn = try_call(nfl, "load_schedules") or try_call(nfl, "import_schedules")
    if fn is None:
        print(f"[schedules] skipped: loader not available in {_loader_name}")
        return
    for s in seasons:
        write_csv(safe_load(fn, seasons=[s]), outdir / f"schedules_{s}.csv")


def fetch_pfr_advstats(root: Path, seasons: List[int]) -> None:
    outdir = root / "outputs" / "pfr_advstats"; safe_mkdir(outdir)
    fn = try_call(nfl, "load_pfr_advstats")
    if fn is None:
        print(f"[pfr_advstats] skipped: loader not available in {_loader_name}")
        return
    for s in seasons:
        write_csv(safe_load(fn, seasons=[s]), outdir / f"pfr_advstats_{s}.csv")


def fetch_espn_qbr(root: Path, seasons: List[int]) -> None:
    outdir = root / "outputs" / "espn_qbr"; safe_mkdir(outdir)
    fn = try_call(nfl, "load_espn_qbr")
    if fn is None:
        print(f"[espn_qbr] skipped: loader not available in {_loader_name}")
        return
    for s in seasons:
        write_csv(safe_load(fn, seasons=[s]), outdir / f"espn_qbr_{s}.csv")


def fetch_participation(root: Path, seasons: List[int]) -> None:
    outdir = root / "outputs" / "participation"; safe_mkdir(outdir)
    fn = try_call(nfl, "load_participation")
    if fn is None:
        print(f"[participation] skipped: loader not available in {_loader_name}")
        return
    for s in seasons:
        # some libs only support <= 2024; gracefully skip newer
        if s > 2024:
            print(f"[participation] no data for {s} in current loader; skipping")
            write_csv(None, outdir / f"participation_{s}.csv")
            continue
        write_csv(safe_load(fn, seasons=[s]), outdir / f"participation_{s}.csv")


def fetch_officials(root: Path, seasons: List[int]) -> None:
    outdir = root / "outputs" / "officials"; safe_mkdir(outdir)
    fn = try_call(nfl, "load_officials")
    if fn is None:
        print(f"[officials] skipped: loader not available in {_loader_name}")
        return
    for s in seasons:
        write_csv(safe_load(fn, seasons=[s]), outdir / f"officials_{s}.csv")


def fetch_trades(root: Path, seasons: List[int]) -> None:
    outdir = root / "outputs" / "trades"; safe_mkdir(outdir)
    fn = try_call(nfl, "load_trades") or try_call(nfl, "import_transactions")
    if fn is None:
        print(f"[trades] skipped: loader not available in {_loader_name}")
        return
    for s in seasons:
        df = safe_load(fn, seasons=[s])
        if df is None:
            df = safe_load(fn, year=s)  # alt signature
        write_csv(df, outdir / f"trades_{s}.csv")


def fetch_pbp(root: Path, seasons: List[int], enabled: bool) -> None:
    if not enabled:
        print("[pbp] skipped by flag --skip-pbp")
        return
    outdir = root / "outputs" / "pbp"; safe_mkdir(outdir)
    fn = try_call(nfl, "load_pbp") or try_call(nfl, "import_pbp")
    if fn is None:
        print(f"[pbp] skipped: loader not available in {_loader_name}")
        return
    write_csv(safe_load(fn, seasons=seasons), outdir / f"pbp_{min(seasons)}_{max(seasons)}.csv")


def fetch_ftn_charting(root: Path, seasons: List[int]) -> None:
    outdir = root / "outputs" / "ftn_charting"; safe_mkdir(outdir)
    fn = try_call(nfl, "load_ftn_charting")
    if fn is None:
        print(f"[ftn_charting] skipped: loader not available in {_loader_name}")
        return
    for s in seasons:
        write_csv(safe_load(fn, seasons=[s]), outdir / f"ftn_charting_{s}.csv")


def fetch_injuries(root: Path, seasons: List[int]) -> None:
    outdir = root / "outputs" / "injuries"; safe_mkdir(outdir)
    fn = try_call(nfl, "load_injuries")
    if fn is None:
        print(f"[injuries] skipped: loader not available in {_loader_name}")
        return
    for s in seasons:
        df = safe_load(fn, seasons=[s])
        # github parquet for current season might not exist yet => None
        write_csv(df, outdir / f"injuries_{s}.csv")


# ----------------------- orchestration -----------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--seasons", type=str, default=None)
    ap.add_argument("--skip-pbp", action="store_true")
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parent
    os.chdir(ROOT)

    if nfl is None:
        print("::error ::Neither nflreadpy nor nfl_data_py could be imported. Add to requirements.txt")
        return 2

    seasons = seasons_from_args(args)
    print(f"[env] loader={_loader_name}, seasons={seasons}")

    fetch_team_stats(ROOT, seasons)
    fetch_player_stats(ROOT, seasons)
    fetch_nextgen(ROOT, seasons)
    fetch_depth_charts(ROOT, seasons)
    fetch_snap_counts(ROOT, seasons)
    fetch_rosters(ROOT, seasons)
    fetch_schedules(ROOT, seasons)
    fetch_pfr_advstats(ROOT, seasons)
    fetch_espn_qbr(ROOT, seasons)
    fetch_participation(ROOT, seasons)   # guarded for >2024
    fetch_officials(ROOT, seasons)
    fetch_trades(ROOT, seasons)
    fetch_pbp(ROOT, seasons, enabled=(not args.skip_pbp))
    fetch_ftn_charting(ROOT, seasons)
    fetch_injuries(ROOT, seasons)

    print("✅ Done. CSVs saved under: outputs")
    return 0


if __name__ == "__main__":
    sys.exit(main())

