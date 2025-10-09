#!/usr/bin/env python3
from __future__ import annotations

import sys, os, argparse
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

def _import_or_none(name: str):
    try:
        return __import__(name, fromlist=["*"])
    except Exception:
        return None

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

nflreadpy = _import_or_none("nflreadpy")
nfl_data_py = _import_or_none("nfl_data_py")

espn = _import_or_none("scripts.providers.espn_pbp")
msf = _import_or_none("scripts.providers.msf")
apis = _import_or_none("scripts.providers.apisports")
gsis = _import_or_none("scripts.providers.nflgsis")

def _ok_df(df) -> bool:
    return isinstance(df, pd.DataFrame) and not df.empty

def _write_csv(df: Optional[pd.DataFrame], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if _ok_df(df):
        df.to_csv(path, index=False)
        print(f"[write] {path} rows={len(df)}")
    else:
        pd.DataFrame().to_csv(path, index=False)
        print(f"[write-empty] {path} rows=0")

def get_pbp(season: int) -> pd.DataFrame:
    try:
        if nflreadpy and hasattr(nflreadpy, "load_pbp"):
            df = nflreadpy.load_pbp(seasons=[season])
            if _ok_df(df):
                print("[pbp] nflreadpy ok"); return df
    except Exception as e:
        print(f"[pbp] nflreadpy failed: {e}")
    try:
        if nfl_data_py:
            from nfl_data_py import import_pbp_data
            df = import_pbp_data([season])
            if _ok_df(df):
                print("[pbp] nfl_data_py ok"); return df
    except Exception as e:
        print(f"[pbp] nfl_data_py failed: {e}")
    try:
        if espn and hasattr(espn, "pbp"):
            df = espn.pbp(season)
            if _ok_df(df):
                print("[pbp] espn ok"); return df
    except Exception as e:
        print(f"[pbp] espn failed: {e}")
    try:
        if msf and hasattr(msf, "pbp"):
            df = msf.pbp(season)
            if _ok_df(df):
                print("[pbp] msf ok"); return df
    except Exception as e:
        print(f"[pbp] msf failed: {e}")
    try:
        if apis and hasattr(apis, "pbp"):
            df = apis.pbp(season)
            if _ok_df(df):
                print("[pbp] apisports ok"); return df
    except Exception as e:
        print(f"[pbp] apisports failed: {e}")
    try:
        if gsis and hasattr(gsis, "pbp"):
            df = gsis.pbp(season)
            if _ok_df(df):
                print("[pbp] gsis ok"); return df
    except Exception as e:
        print(f"[pbp] gsis failed: {e}")
    print("[pbp] all providers failed -> empty")
    return pd.DataFrame()

def _fallback_chain(name: str, season: int) -> pd.DataFrame:
    order = []
    if name == "injuries":
        order = [
            ("nflreadpy", getattr(nflreadpy, "load_injuries", None)),
            ("nfl_data_py", getattr(nfl_data_py, "import_injuries", None) if nfl_data_py else None),
            ("espn", getattr(espn, "injuries", None)),
            ("msf", getattr(msf, "injuries", None)),
            ("apis", getattr(apis, "injuries", None)),
            ("gsis", getattr(gsis, "injuries", None)),
        ]
    elif name == "schedules":
        order = [
            ("nflreadpy", getattr(nflreadpy, "load_schedule", None)),
            ("nfl_data_py", getattr(nfl_data_py, "import_schedules", None) if nfl_data_py else None),
            ("espn", getattr(espn, "schedules", None)),
            ("msf", getattr(msf, "schedules", None)),
            ("apis", getattr(apis, "schedules", None)),
            ("gsis", getattr(gsis, "schedules", None)),
        ]
    elif name == "rosters":
        order = [
            ("nflreadpy", getattr(nflreadpy, "load_players", None)),
            ("nfl_data_py", getattr(nfl_data_py, "import_rosters", None) if nfl_data_py else None),
            ("espn", getattr(espn, "rosters", None)),
            ("msf", getattr(msf, "rosters", None)),
            ("apis", getattr(apis, "rosters", None)),
            ("gsis", getattr(gsis, "rosters", None)),
        ]
    elif name == "depth_charts":
        order = [
            ("espn", getattr(espn, "depth_charts", None)),
            ("msf", getattr(msf, "depth_charts", None)),
            ("apis", getattr(apis, "depth_charts", None)),
            ("gsis", getattr(gsis, "depth_charts", None)),
        ]
    elif name == "snap_counts":
        order = [
            ("nflreadpy", getattr(nflreadpy, "load_weekly_snap_counts", None)),
            ("nfl_data_py", getattr(nfl_data_py, "import_weekly_snap_counts", None) if nfl_data_py else None),
            ("msf", getattr(msf, "snap_counts", None)),
            ("espn", getattr(espn, "snap_counts", None)),
            ("apis", getattr(apis, "snap_counts", None)),
        ]
    elif name == "team_stats_week":
        order = [
            ("nflreadpy", getattr(nflreadpy, "load_team_stats", None)),
            ("nfl_data_py", getattr(nfl_data_py, "import_ngs_team_data", None) if nfl_data_py else None),
            ("espn", getattr(espn, "team_stats_week", None)),
            ("msf", getattr(msf, "team_stats_week", None)),
            ("apis", getattr(apis, "team_stats_week", None)),
            ("gsis", getattr(gsis, "team_stats_week", None)),
        ]
    elif name == "player_stats_week":
        order = [
            ("nflreadpy", getattr(nflreadpy, "load_player_stats", None)),
            ("nfl_data_py", getattr(nfl_data_py, "import_ngs_qb_data", None) if nfl_data_py else None),
            ("espn", getattr(espn, "player_stats_week", None)),
            ("msf", getattr(msf, "player_stats_week", None)),
            ("apis", getattr(apis, "player_stats_week", None)),
            ("gsis", getattr(gsis, "player_stats_week", None)),
        ]
    else:
        order = []

    for tag, fn in order:
        try:
            if fn:
                df = fn(season) if fn.__code__.co_argcount >= 1 else fn()
                if _ok_df(df):
                    print(f"[{name}] {tag} ok"); return df
        except Exception as e:
            print(f"[{name}] {tag} failed: {e}")
    print(f"[{name}] all providers failed -> empty")
    return pd.DataFrame()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    args = ap.parse_args()
    season = int(args.season)

    pbp = get_pbp(season)
    _write_csv(pbp, OUT / "pbp" / f"pbp_{season}_{season}.csv")

    _write_csv(_fallback_chain("injuries", season), OUT / "injuries" / f"injuries_{season}.csv")
    _write_csv(_fallback_chain("schedules", season), OUT / "schedules" / f"schedules_{season}.csv")
    _write_csv(_fallback_chain("rosters", season), OUT / "rosters" / f"rosters_{season}.csv")
    _write_csv(_fallback_chain("depth_charts", season), OUT / "depth_charts" / f"depth_charts_{season}.csv")
    _write_csv(_fallback_chain("snap_counts", season), OUT / "snap_counts" / f"snap_counts_{season}.csv")

    _write_csv(_fallback_chain("team_stats_week", season), OUT / "team_stats" / f"team_stats_week_{season}.csv")
    _write_csv(_fallback_chain("player_stats_week", season), OUT / "player_stats" / f"player_stats_week_{season}.csv")

    print("✅ fetch_all completed")

if __name__ == "__main__":
    sys.exit(main())
