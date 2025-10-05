# scripts/providers/nflverse_free.py
from __future__ import annotations
import pandas as pd

def _try_import(module: str):
    try:
        return __import__(module, fromlist=["*"])
    except Exception:
        return None

_nfl = _try_import("nfl_data_py")

def import_pbp_2025() -> pd.DataFrame:
    if not _nfl:
        return pd.DataFrame()
    try:
        # nflverse PBP for current season (fastR data)
        # For most installs: import_pbp_data accepts a list of seasons
        df = _nfl.import_pbp_data([2025])
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    except Exception as e:
        print(f"[nflverse_free] import_pbp_data failed: {e}")
        return pd.DataFrame()

def team_week_form_from_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Derive neutral-situation team-week features:
    - pace (secs/play)
    - pass_rate (neutral)
    - pass EPA / rush EPA (neutral)
    - redzone pass rate (proxied)
    Returns columns: team, week, pace_sec_play, pr_neutral, epa_pass_neutral, epa_rush_neutral, rz_pass_rate
    """
    if pbp.empty:
        return pd.DataFrame()

    df = pbp.copy()

    # Basic neutral filter (adjust as you wish)
    neutral = (
        (df["qtr"].between(1, 3, inclusive="both")) &
        (df["score_differential"].abs() <= 7) &
        (df["play_type"].isin(["pass", "run"])) &
        (df["no_play"] == 0)
    )
    df = df.loc[neutral, ["posteam","defteam","week","play_type","epa","game_seconds_remaining","yardline_100"]].copy()
    df.rename(columns={"posteam":"team","defteam":"opp"}, inplace=True)

    # Pace: approx seconds per play using deltas in game time within team-week
    df["sec"] = 3600 - df["game_seconds_remaining"]
    df.sort_values(["team","week","sec"], inplace=True)
    df["dsec"] = df.groupby(["team","week"])["sec"].diff().fillna(0)
    pace = df.groupby(["team","week"], as_index=False).agg(
        plays=("play_type","size"),
        sec_sum=("dsec","sum"),
        pr_neutral=("play_type", lambda s: (s=="pass").mean()),
        epa_pass_neutral=("epa", lambda s: s[df.loc[s.index,"play_type"]=="pass"].mean()),
        epa_rush_neutral=("epa", lambda s: s[df.loc[s.index,"play_type"]=="run"].mean()),
        rz_pass_rate=("yardline_100", lambda s: ((s<=20) & (df.loc[s.index,"play_type"]=="pass")).mean()),
    )
    pace["pace_sec_play"] = (pace["sec_sum"] / pace["plays"]).clip(lower=10, upper=60)
    keep = pace[["team","week","pace_sec_play","pr_neutral","epa_pass_neutral","epa_rush_neutral","rz_pass_rate"]]
    return keep
