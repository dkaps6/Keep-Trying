#!/usr/bin/env python3
"""
Fetch nflverse data via nflreadpy and write CSVs.
"""
from __future__ import annotations
import argparse
import os
from typing import Iterable, List

# nflreadpy uses Polars under the hood
import polars as pl
import nflreadpy as nfl

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def write_csv(df: "pl.DataFrame", path: str) -> None:
    if df is None or (hasattr(df, "height") and df.height == 0):
        return
    # ensure parent exists
    ensure_dir(os.path.dirname(path))
    df.write_csv(path)

def seasons_from_args(vals: List[str]) -> List[int]:
    out: List[int] = []
    for v in vals:
        out.append(int(v))
    return out

def save_team_stats(seasons: List[int], outdir: str) -> None:
    base = os.path.join(outdir, "team_stats")
    ensure_dir(base)
    for level, suffix in [("week","week"), ("reg","reg"), ("post","post")]:
        try:
            df = nfl.load_team_stats(seasons=seasons, summary_level=level, file_type="csv")
            write_csv(df, os.path.join(base, f"team_stats_{suffix}_{'-'.join(map(str,seasons))}.csv"))
        except Exception as e:
            print(f"[team_stats/{level}] skipped: {e}")

def save_player_stats(seasons: List[int], outdir: str) -> None:
    base = os.path.join(outdir, "player_stats")
    ensure_dir(base)
    for level, suffix in [("week","week"), ("reg","reg"), ("post","post")]:
        try:
            df = nfl.load_player_stats(seasons=seasons, summary_level=level, file_type="csv")
            write_csv(df, os.path.join(base, f"player_stats_{suffix}_{'-'.join(map(str,seasons))}.csv"))
        except Exception as e:
            print(f"[player_stats/{level}] skipped: {e}")

def save_nextgen(seasons: List[int], outdir: str) -> None:
    base = os.path.join(outdir, "nextgen")
    ensure_dir(base)
    for stat_type in ["passing","receiving","rushing"]:
        try:
            df = nfl.load_nextgen_stats(seasons=seasons, stat_type=stat_type, file_type="csv")
            write_csv(df, os.path.join(base, f"nextgen_{stat_type}_{'-'.join(map(str,seasons))}.csv"))
        except Exception as e:
            print(f"[nextgen/{stat_type}] skipped: {e}")

def save_ftn_charting(seasons: List[int], outdir: str) -> None:
    base = os.path.join(outdir, "ftn_charting")
    ensure_dir(base)
    try:
        df = nfl.load_ftn_charting(seasons=seasons, file_type="csv")
        write_csv(df, os.path.join(base, f"ftn_charting_{'-'.join(map(str,seasons))}.csv"))
    except Exception as e:
        print(f"[ftn_charting] skipped: {e}")

def save_injuries(seasons: List[int], outdir: str) -> None:
    base = os.path.join(outdir, "injuries")
    ensure_dir(base)
    try:
        df = nfl.load_injuries(seasons=seasons, file_type="csv")
        write_csv(df, os.path.join(base, f"injuries_{'-'.join(map(str,seasons))}.csv"))
    except Exception as e:
        print(f"[injuries] skipped: {e}")

def save_depth_charts(seasons: List[int], outdir: str) -> None:
    base = os.path.join(outdir, "depth_charts")
    ensure_dir(base)
    try:
        df = nfl.load_depth_charts(seasons=seasons, file_type="csv")
        write_csv(df, os.path.join(base, f"depth_charts_weekly_{'-'.join(map(str,seasons))}.csv"))
    except Exception as e:
        print(f"[depth_charts] skipped: {e}")

def save_snap_counts(seasons: List[int], outdir: str) -> None:
    base = os.path.join(outdir, "snap_counts")
    ensure_dir(base)
    try:
        df = nfl.load_snap_counts(seasons=seasons, file_type="csv")
        write_csv(df, os.path.join(base, f"snap_counts_{'-'.join(map(str,seasons))}.csv"))
    except Exception as e:
        print(f"[snap_counts] skipped: {e}")

def save_rosters(seasons: List[int], outdir: str) -> None:
    base = os.path.join(outdir, "rosters")
    ensure_dir(base)
    try:
        df = nfl.load_rosters(seasons=seasons, file_type="csv")
        write_csv(df, os.path.join(base, f"rosters_{'-'.join(map(str,seasons))}.csv"))
    except Exception as e:
        print(f"[rosters] skipped: {e}")
    try:
        dfw = nfl.load_rosters_weekly(seasons=seasons, file_type="csv")
        write_csv(dfw, os.path.join(base, f"rosters_weekly_{'-'.join(map(str,seasons))}.csv"))
    except Exception as e:
        print(f"[rosters_weekly] skipped: {e}")

def save_schedules(seasons: List[int], outdir: str) -> None:
    base = os.path.join(outdir, "schedules")
    ensure_dir(base)
    try:
        df = nfl.load_schedules(seasons=seasons, file_type="csv")
        write_csv(df, os.path.join(base, f"schedules_{'-'.join(map(str,seasons))}.csv"))
    except Exception as e:
        print(f"[schedules] skipped: {e}")

def save_pfr_advstats(seasons: List[int], outdir: str) -> None:
    base = os.path.join(outdir, "pfr_advstats")
    ensure_dir(base)
    try:
        df = nfl.load_pfr_advstats(seasons=seasons, file_type="csv")
        write_csv(df, os.path.join(base, f"pfr_advstats_{'-'.join(map(str,seasons))}.csv"))
    except Exception as e:
        print(f"[pfr_advstats] skipped: {e}")

def save_espn_qbr(seasons: List[int], outdir: str) -> None:
    base = os.path.join(outdir, "espn_qbr")
    ensure_dir(base)
    try:
        df = nfl.load_espn_qbr(seasons=seasons, file_type="csv")
        write_csv(df, os.path.join(base, f"espn_qbr_{'-'.join(map(str,seasons))}.csv"))
    except Exception as e:
        print(f"[espn_qbr] skipped: {e}")

def save_participation(seasons: List[int], outdir: str) -> None:
    base = os.path.join(outdir, "participation")
    ensure_dir(base)
    try:
        df = nfl.load_participation(seasons=seasons, file_type="csv")
        write_csv(df, os.path.join(base, f"participation_{'-'.join(map(str,seasons))}.csv"))
    except Exception as e:
        print(f"[participation] skipped: {e}")

def save_officials(seasons: List[int], outdir: str) -> None:
    base = os.path.join(outdir, "officials")
    ensure_dir(base)
    try:
        df = nfl.load_officials(seasons=seasons, file_type="csv")
        write_csv(df, os.path.join(base, f"officials_{'-'.join(map(str,seasons))}.csv"))
    except Exception as e:
        print(f"[officials] skipped: {e}")

def save_trades(seasons: List[int], outdir: str) -> None:
    base = os.path.join(outdir, "trades")
    ensure_dir(base)
    try:
        df = nfl.load_trades(seasons=seasons, file_type="csv")
        write_csv(df, os.path.join(base, f"trades_{'-'.join(map(str,seasons))}.csv"))
    except Exception as e:
        print(f"[trades] skipped: {e}")

def save_pbp(seasons: List[int], outdir: str) -> None:
    base = os.path.join(outdir, "pbp")
    ensure_dir(base)
    for season in seasons:
        try:
            df = nfl.load_pbp(seasons=[season], file_type="csv")
            write_csv(df, os.path.join(base, f"pbp_{season}.csv"))
        except Exception as e:
            print(f"[pbp/{season}] skipped: {e}")

def main():
    ap = argparse.ArgumentParser(description="Fetch nflverse datasets to CSV via nflreadpy")
    ap.add_argument("--season", nargs="+", default=["2025"], help="Season(s) to fetch (e.g., 2025 or 2024 2025)")
    ap.add_argument("--out", default="outputs", help="Output directory")
    ap.add_argument("--file-type", default="csv", choices=["csv","parquet","rds","qs"], help="Preferred file type to request upstream (we still write CSV)")
    ap.add_argument("--skip-pbp", action="store_true", help="Skip play-by-play (large files)")
    args = ap.parse_args()

    seasons = seasons_from_args(args.season)
    outdir = args.out
    ensure_dir(outdir)

    # Team / player aggregates
    save_team_stats(seasons, outdir)
    save_player_stats(seasons, outdir)

    # NGS & FTN
    save_nextgen(seasons, outdir)
    save_ftn_charting(seasons, outdir)

    # Roster / injuries / depth / snaps / schedules
    save_injuries(seasons, outdir)
    save_depth_charts(seasons, outdir)
    save_snap_counts(seasons, outdir)
    save_rosters(seasons, outdir)
    save_schedules(seasons, outdir)

    # Advanced summaries
    save_pfr_advstats(seasons, outdir)
    save_espn_qbr(seasons, outdir)
    save_participation(seasons, outdir)
    save_officials(seasons, outdir)
    save_trades(seasons, outdir)

    if not args.skip_pbp:
        save_pbp(seasons, outdir)

    print(f"✅ Done. CSVs saved under: {outdir}")

if __name__ == "__main__":
    main()
