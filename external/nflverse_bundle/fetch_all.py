#!/usr/bin/env python3
import argparse, sys
from pathlib import Path
import pandas as pd

try:
    import nfl_data_py as nfl
except Exception as e:
    print("ERROR: nfl_data_py not installed:", e, file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

def _w_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(df, pd.DataFrame):
        df.to_csv(path, index=False)
        print("[write]", path, "rows=", len(df))
    else:
        pd.DataFrame().to_csv(path, index=False)
        print("[write-empty]", path, "rows=0")

def _w_parquet(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(df, pd.DataFrame) and not df.empty:
        df.to_parquet(path, index=False)
        print("[write]", path, "rows=", len(df))
    else:
        csvp = path.with_suffix(".csv")
        pd.DataFrame().to_csv(csvp, index=False)
        print("[write-empty]", csvp, "rows=0")

def must_nonempty(df, name):
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise SystemExit(f"FATAL: {name} returned 0 rows")

def fetch_all(season: int):
    years = [int(season)]
    pbp = nfl.import_pbp_data(years)
    _w_parquet(pbp, OUT / "pbp" / f"pbp_{season}.parquet")
    must_nonempty(pbp, "import_pbp_data")

    sched = nfl.import_schedules(years)
    _w_csv(sched, OUT / "schedules" / f"schedules_{season}.csv")
    must_nonempty(sched, "import_schedules")

    rosters = nfl.import_rosters(years)
    _w_csv(rosters, OUT / "rosters" / f"rosters_{season}.csv")
    must_nonempty(rosters, "import_rosters")

    weekly = nfl.import_weekly_data(years)
    _w_csv(weekly, OUT / "player_stats" / f"player_stats_week_{season}.csv")
    must_nonempty(weekly, "import_weekly_data")

    snaps = nfl.import_snap_counts(years)
    _w_csv(snaps, OUT / "snap_counts" / f"snap_counts_{season}.csv")
    must_nonempty(snaps, "import_snap_counts")

    try:
        team_week = weekly.groupby(["season","team","week"], as_index=False).agg(cnt=("player_id","count"))
    except Exception:
        team_week = pd.DataFrame()
    _w_csv(team_week, OUT / "team_stats" / f"team_stats_week_{season}.csv")
    print("✅ fetch_all complete")
    return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True, type=int)
    args = ap.parse_args()
    return fetch_all(args.season)

if __name__ == "__main__":
    raise SystemExit(main())
