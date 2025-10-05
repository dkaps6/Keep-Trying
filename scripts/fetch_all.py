#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fetch_all.py
Pulls external metrics needed by the engine without scraping PFR (which blocks bots).
Order of operations:
  1) Try nflverse via nfl_data_py for PBP (primary).
  2) Compute team-season and team-week features:
      - neutral pass rate (off/def)
      - explosive pass/rush rates (15+ pass, 10+ rush)
      - red zone proxies (plays inside 20, RZ play share, RZ TD rate)
      - drive counts & scoring rate (from PBP)
  3) Write:
      metrics/team_form.csv
      metrics/team_week_form.csv
      metrics/player_form.csv      (schema stub; 0 rows is okay)
      inputs/id_map.csv            (schema stub; 0 rows is okay)
      inputs/weather.csv           (left as-is for now)

Run:
  python -m scripts.fetch_all --season 2025
"""

import argparse
import sys
import os
from typing import Tuple
import pandas as pd
import numpy as np

# Soft import: nfl_data_py may not be present on all runners
try:
    import nfl_data_py as nfl
    HAS_NFL = True
except Exception:
    HAS_NFL = False

OUT_METRICS = "metrics"
OUT_INPUTS = "inputs"

pd.options.mode.copy_on_write = True


def _ensure_dirs():
    os.makedirs(OUT_METRICS, exist_ok=True)
    os.makedirs(OUT_INPUTS, exist_ok=True)


def _readable_team(x: str) -> str:
    # Normalize team codes to 2–3 letter standard (nfl_data_py already uses 3-letter)
    if pd.isna(x):
        return x
    return str(x).strip().upper()


def _safe_mean(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce")
    return float(s.mean()) if s.notna().any() else np.nan


def _compute_explosive_flags(pbp: pd.DataFrame) -> pd.DataFrame:
    """Add flags for explosive plays: 15+ pass, 10+ rush."""
    pbp = pbp.copy()
    # yards_gained exists in nflfastR schema
    yg = pd.to_numeric(pbp.get("yards_gained"), errors="coerce")
    is_pass = pbp.get("pass") == 1
    is_rush = pbp.get("rush") == 1
    pbp["is_explosive_pass"] = (is_pass) & (yg >= 15)
    pbp["is_explosive_rush"] = (is_rush) & (yg >= 10)
    return pbp


def _compute_neutral_flag(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Neutral situation heuristic:
      - score differential abs <= 7
      - between 1:00 and 9:00 left in quarter (avoid pure 2-minute / dead time)
      - exclude 4th & very short/long? (we keep all downs to avoid bias)
    """
    pbp = pbp.copy()
    # score differential pre-snap: home_score - away_score (nflfastR has 'score_differential' offense-side)
    sd = pd.to_numeric(pbp.get("score_differential"), errors="coerce")
    pbp["neutral_score"] = sd.abs() <= 7

    # game_seconds_remaining exists in nflfastR; map to quarter seconds
    gsr = pd.to_numeric(pbp.get("game_seconds_remaining"), errors="coerce")
    qtr = pd.to_numeric(pbp.get("qtr"), errors="coerce")
    # seconds elapsed in quarter = 900 - (gsr - 900*(qtr-1))
    # We’ll instead compute “seconds left in quarter”
    sec_left_q = 900 - ((4 - qtr) * 900 - (gsr - (4 - qtr) * 900))
    # Filter between 60 and 540 secs left (1:00 to 9:00)
    pbp["neutral_clock"] = (sec_left_q >= 60) & (sec_left_q <= 540)
    pbp["neutral"] = pbp["neutral_score"] & pbp["neutral_clock"]
    return pbp


def _compute_red_zone(pbp: pd.DataFrame) -> pd.DataFrame:
    """Add red-zone indicator (yardline_100 <= 20 for offense)."""
    pbp = pbp.copy()
    yl = pd.to_numeric(pbp.get("yardline_100"), errors="coerce")
    pbp["is_rz_snap"] = yl <= 20
    # RZ TD proxy: touchdown == 1 within RZ
    pbp["is_rz_td"] = (pbp.get("touchdown") == 1) & pbp["is_rz_snap"]
    return pbp


def _team_keys(pbp: pd.DataFrame) -> Tuple[str, str]:
    # Offense team column is usually 'posteam'
    off = "posteam" if "posteam" in pbp.columns else "pos_team"
    defn = "defteam" if "defteam" in pbp.columns else "def_team"
    return off, defn


def _drives_from_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    """Approximate drive summaries from PBP (when dedicated drives aren’t available)."""
    # nflfastR already includes drive info (drive, drive_result, etc.) on each row
    if "drive" not in pbp.columns:
        return pd.DataFrame(columns=["team", "week", "drives", "scoring_drives", "drive_scoring_rate"])
    off_col, _ = _team_keys(pbp)
    wk = pd.to_numeric(pbp.get("week"), errors="coerce")
    tmp = pbp.loc[:, ["game_id", off_col, "drive", "drive_result"]].copy()
    tmp["week"] = wk
    tmp = tmp.dropna(subset=["drive"])
    # count unique drives per team/week
    g = tmp.groupby([off_col, "week"], dropna=True)
    dd = g.agg(
        drives=("drive", "nunique"),
        scoring_drives=("drive_result", lambda s: (s.astype(str).str.contains("Touchdown|Field Goal", case=False, na=False)).sum())
    ).reset_index()
    dd["drive_scoring_rate"] = dd["scoring_drives"] / dd["drives"].replace(0, np.nan)
    dd = dd.rename(columns={off_col: "team"})
    return dd


def _season_agg_from_pbp(pbp: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build team_week_form and team_form from play-by-play."""
    if pbp.empty:
        cols_w = ["team", "week", "neutral_pass_rate_off", "neutral_pass_rate_def",
                  "explosive_pass_rate_off", "explosive_rush_rate_off",
                  "explosive_pass_rate_def", "explosive_rush_rate_def",
                  "rz_play_share_off", "rz_td_rate_off", "drives", "drive_scoring_rate"]
        cols_s = [c for c in cols_w if c not in ("week",)]
        return pd.DataFrame(columns=cols_w), pd.DataFrame(columns=cols_s)

    off_col, def_col = _team_keys(pbp)
    wk = pd.to_numeric(pbp.get("week"), errors="coerce")

    # Flags
    pbp = _compute_explosive_flags(pbp)
    pbp = _compute_neutral_flag(pbp)
    pbp = _compute_red_zone(pbp)
    pbp["week"] = wk

    # Neutral pass rate (offense): among neutral snaps, % that are passes
    def _neutral_pass_rate(df, team_col):
        d = df[df["neutral"] & (df[team_col].notna())]
        g = d.groupby([team_col, "week"], dropna=True)["pass"].mean().reset_index()
        g = g.rename(columns={team_col: "team", "pass": "neutral_pass_rate_off"})
        return g

    # Defensive neutral pass **against**: use defense team perspective
    def _neutral_pass_rate_def(df, team_col):
        d = df[df["neutral"] & (df[team_col].notna())]
        g = d.groupby([team_col, "week"], dropna=True)["pass"].mean().reset_index()
        g = g.rename(columns={team_col: "team", "pass": "neutral_pass_rate_def"})
        return g

    off_npr = _neutral_pass_rate(pbp, off_col)
    def_npr = _neutral_pass_rate_def(pbp.rename(columns={def_col: "team_def"}), "team_def")
    def_npr = def_npr.rename(columns={"team_def": "team"})

    # Explosive rates (offense)
    ex_off = pbp.groupby([off_col, "week"], dropna=True).agg(
        explosive_pass_rate_off=("is_explosive_pass", "mean"),
        explosive_rush_rate_off=("is_explosive_rush", "mean")
    ).reset_index().rename(columns={off_col: "team"})

    # Explosive rates (defense allowed)
    ex_def = pbp.groupby([def_col, "week"], dropna=True).agg(
        explosive_pass_rate_def=("is_explosive_pass", "mean"),
        explosive_rush_rate_def=("is_explosive_rush", "mean")
    ).reset_index().rename(columns={def_col: "team"})

    # Red zone proxies (offense)
    rz_off = pbp.groupby([off_col, "week"], dropna=True).agg(
        rz_play_share_off=("is_rz_snap", "mean"),
        rz_td_rate_off=("is_rz_td", "mean")
    ).reset_index().rename(columns={off_col: "team"})

    # Drive metrics (offense)
    drives = _drives_from_pbp(pbp)

    # Merge weekly form
    tw = (
        off_npr
        .merge(def_npr, on=["team", "week"], how="outer")
        .merge(ex_off, on=["team", "week"], how="outer")
        .merge(ex_def, on=["team", "week"], how="outer")
        .merge(rz_off, on=["team", "week"], how="outer")
        .merge(drives, on=["team", "week"], how="outer")
        .sort_values(["team", "week"])
        .reset_index(drop=True)
    )

    # Season aggregates
    tf = (
        tw.groupby("team", dropna=True)
          .agg({c: "mean" for c in tw.columns if c not in ("team", "week")})
          .reset_index()
          .rename(columns=lambda c: c if c == "team" else f"{c}")
    )
    return tw, tf


def _import_pbp(season: int) -> pd.DataFrame:
    """
    Try to bring play-by-play from nflverse via nfl_data_py.
    If not available (e.g., future season), return empty DF but explain.
    """
    if not HAS_NFL:
        print("ℹ nfl_data_py not installed/available; cannot pull PBP.")
        return pd.DataFrame()

    try:
        print(f"► Downloading play-by-play for season={season} …")
        # nfl_data_py.import_pbp_data accepts seasons=[season]
        pbp = nfl.import_pbp_data([season])
        if isinstance(pbp, pd.DataFrame):
            # Standardize a couple of key columns that sometimes differ by version
            for col in ("pass", "rush", "touchdown"):
                if col in pbp.columns:
                    pbp[col] = pd.to_numeric(pbp[col], errors="coerce")
            print(f"✓ PBP rows: {len(pbp):,}")
            return pbp
        print("⚠ nfl_data_py returned a non-DataFrame object; treating as empty.")
        return pd.DataFrame()
    except Exception as e:
        print(f"⚠ nfl_data_py failed for {season}: {e}")
        return pd.DataFrame()


def _write_csv(df: pd.DataFrame, path: str):
    # Keep consistent floats
    if not df.empty:
        for c in df.select_dtypes(include=[np.number]).columns:
            df[c] = df[c].astype(float)
    df.to_csv(path, index=False)
    print(f"Wrote {path} ({len(df)} rows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    args = parser.parse_args()

    _ensure_dirs()

    # 1) Pull PBP (primary) from nflverse
    pbp = _import_pbp(args.season)

    # 2) Build team-week and team-season form
    team_week_form, team_form = _season_agg_from_pbp(pbp)

    # 3) Minimal player_form schema (kept empty by default, but with stable columns)
    player_form = pd.DataFrame(columns=[
        "player_id", "player_name", "team", "week",
        "route_share", "tgt_share", "rush_share", "red_zone_touches",
        "air_yards_share"
    ])

    # 4) id_map kept minimal (if you have a cache process, wire it here)
    id_map = pd.DataFrame(columns=["player_name", "gsis_id", "recent_team", "position"])

    # 5) Weather (left optional)
    weather = pd.DataFrame(columns=["game_id", "stadium", "temp_f", "wind_mph", "precip", "roof", "surface"])

    # 6) Write all outputs
    _write_csv(team_form, os.path.join(OUT_METRICS, "team_form.csv"))
    _write_csv(team_week_form, os.path.join(OUT_METRICS, "team_week_form.csv"))
    _write_csv(player_form, os.path.join(OUT_METRICS, "player_form.csv"))
    _write_csv(id_map, os.path.join(OUT_INPUTS, "id_map.csv"))
    _write_csv(weather, os.path.join(OUT_INPUTS, "weather.csv"))

    # 7) Verification summary
    print("\n=== Data verification summary ===")
    print(f"team_form.csv:       {len(team_form)} rows")
    print(f"team_week_form.csv:  {len(team_week_form)} rows")
    print(f"player_form.csv:     {len(player_form)} rows")
    print(f"id_map.csv:          {len(id_map)} rows")
    print(f"weather.csv:         {len(weather)} rows")

    if pbp.empty:
        print("\n⚠ NOTE: PBP was empty for this season. "
              "If you’re aiming at an in-progress future season, nflverse may not have published data yet. "
              "The pipeline will still run, but outputs will rely more on market priors.")
    else:
        print("✓ PBP successfully loaded; metrics computed from play-by-play.")


if __name__ == "__main__":
    sys.exit(main())
