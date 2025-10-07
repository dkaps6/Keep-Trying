# scripts/fetch_nfl_data.py
# Builds data/team_form.csv using nfl_data_py (preferred) with robust fallbacks to nflverse mirrors.
# - Historical seasons (e.g., 2019-2024) + current 2025 in-progress
# - Derives DEF/OPP EPA splits, pace, neutral pass rate, PROE, plays, red-zone rate
# - Outputs z-scores the rest of your pipeline expects
#
# Usage:
#   python -m scripts.fetch_nfl_data --season 2025 --write data/team_form.csv
#   python -m scripts.fetch_nfl_data --history 2019-2024 --season 2025 --write data/team_form.csv

from __future__ import annotations

import argparse
import datetime as dt
import io
import math
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

# Optional dependency, but we guard it
try:
    import nfl_data_py as nfl
    HAS_NFL_DATA_PY = True
except Exception:
    HAS_NFL_DATA_PY = False

# ---------- Helpers to fetch PBP ----------

NFLVERSE_URLS = [
    # season parquet (common location)
    "https://github.com/nflverse/nflfastR-data/raw/master/data/play_by_play_{season}.parquet",
    # sometimes released under "releases"
    "https://github.com/nflverse/nflfastR-data/releases/latest/download/play_by_play_{season}.parquet",
]

def _try_read_parquet(url: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(url)
    except Exception:
        return pd.DataFrame()

def load_pbp_for_season(season: int) -> pd.DataFrame:
    """
    Try nfl_data_py -> nflverse mirrors. Returns DataFrame (may be empty).
    """
    # 1) nfl_data_py
    if HAS_NFL_DATA_PY:
        try:
            print(f"[fetch_nfl_data] nfl_data_py.import_pbp_data({season}) ...")
            df = nfl.import_pbp_data([season])
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception as e:
            print(f"[fetch_nfl_data] nfl_data_py failed ({type(e).__name__}): {e}")

    # 2) nflverse mirrors
    for base in NFLVERSE_URLS:
        url = base.format(season=season)
        print(f"[fetch_nfl_data] trying nflverse: {url}")
        df = _try_read_parquet(url)
        if not df.empty:
            return df

    print(f"[fetch_nfl_data] ⚠️ no PBP available for season {season}")
    return pd.DataFrame()

def load_pbp(history: Iterable[int], current_season: int) -> pd.DataFrame:
    """
    Concatenate historical + current season pbp. Drops obvious junk rows.
    """
    frames: List[pd.DataFrame] = []
    for yr in list(history) + [current_season]:
        df = load_pbp_for_season(yr)
        if df is None or df.empty:
            continue
        # minimal cleaning
        need_cols = {"season","week","posteam","defteam","play_type","pass","rush","epa",
                     "qtr","half_seconds_remaining","wp","ydstogo","yardline_100",
                     "game_id","home_team","away_team","game_seconds_remaining"}
        missing = need_cols - set(df.columns)
        if missing:
            print(f"[fetch_nfl_data] season {yr} missing columns: {missing} (continuing)")
        df = df.assign(season=yr)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # drop non-plays / penalties if needed
    if "play_type" in out.columns:
        out = out[out["play_type"].notna()]
    return out

# ---------- Feature engineering ----------

def _neutral_filter(df: pd.DataFrame) -> pd.Series:
    """
    Neutral situation: quarters 1-3, and 4th before 5:00, win prob 0.20-0.80, distance <= 10.
    """
    q = (df.get("qtr", 0).between(1, 3)) | ((df.get("qtr", 0) == 4) & (df.get("half_seconds_remaining", 0) > 300))
    wp = df.get("wp", 0.5).between(0.20, 0.80)
    dist = df.get("ydstogo", 10) <= 10
    return q & wp & dist

def _is_pass(df: pd.DataFrame) -> pd.Series:
    if "pass" in df.columns:
        return df["pass"] == 1
    # fallback: classify by play_type
    return df.get("play_type", "").astype(str).str.contains("pass", case=False, na=False)

def _is_rush(df: pd.DataFrame) -> pd.Series:
    if "rush" in df.columns:
        return df["rush"] == 1
    return df.get("play_type", "").astype(str).str.contains("rush", case=False, na=False)

def _pace_seconds_per_play(df: pd.DataFrame) -> float:
    # crude: average time between plays via game_seconds_remaining deltas
    d = df.sort_values(["game_id", "game_seconds_remaining"]).copy()
    d["dt"] = d.groupby("game_id")["game_seconds_remaining"].diff(-1).abs()
    val = d["dt"].median()
    try:
        return float(val)
    except Exception:
        return float("nan")

def _agg_team(df: pd.DataFrame, side_col: str) -> pd.DataFrame:
    """
    side_col: 'posteam' (offense) to compute offense EPA; then we will pivot to DEF by swapping perspective.
    """
    if df.empty:
        return pd.DataFrame(columns=["team","plays","pass_plays","rush_plays","pass_rate","epa_mean",
                                     "epa_pass","epa_rush","pace_sec_play","rz_rate","neutral_pass_rate"])
    team = (df
            .groupby(side_col)
            .apply(lambda g: pd.Series({
                "plays": len(g),
                "pass_plays": _is_pass(g).sum(),
                "rush_plays": _is_rush(g).sum(),
                "epa_mean": g["epa"].mean(skipna=True),
                "epa_pass": g.loc[_is_pass(g), "epa"].mean(skipna=True),
                "epa_rush": g.loc[_is_rush(g), "epa"].mean(skipna=True),
                "pace_sec_play": _pace_seconds_per_play(g),
                "rz_rate": (g.get("yardline_100", 100) <= 20).mean(),  # fraction of plays in RZ
                "neutral_pass_rate": (_is_pass(g) & _neutral_filter(g)).mean()
            }))
            .reset_index()
            .rename(columns={side_col: "team"}))
    team["pass_rate"] = team["pass_plays"] / team["plays"].clip(lower=1)
    return team

def compute_team_form(pbp: pd.DataFrame, current_season: int) -> pd.DataFrame:
    """
    Compute offense & defense aggregates, then produce DEF z-scores per your model.
    """
    if pbp.empty:
        return pd.DataFrame(columns=[
            "team","opp_team","event_id",
            "def_pressure_rate_z","def_pass_epa_z","def_rush_epa_z","def_sack_rate_z",
            "pace_z","light_box_rate_z","heavy_box_rate_z","ay_per_att_z",
            "plays_est","proe","rz_rate"
        ])

    # Offense perspective
    off = _agg_team(pbp[pbp["season"] == current_season], "posteam")
    # Defense perspective: compute offense stats vs each defense, then attribute to defense
    opp = (pbp[pbp["season"] == current_season]
           .assign(is_pass=_is_pass(pbp), is_rush=_is_rush(pbp))
           .groupby("defteam")
           .apply(lambda g: pd.Series({
               "def_pass_epa": g.loc[g["is_pass"], "epa"].mean(skipna=True),
               "def_rush_epa": g.loc[g["is_rush"], "epa"].mean(skipna=True),
               "def_plays": len(g),
               # proxies for boxes/pressure if you later wire true sources:
               "light_box_rate": 0.0,
               "heavy_box_rate": 0.0,
               "def_sack_rate": 0.0,
           }))
           .reset_index()
           .rename(columns={"defteam": "team"}))

    # Join offense neutral pass rate to compute PROE:
    # PROE ≈ team neutral pass rate - league neutral pass rate (current season)
    league_neutral = float(off["neutral_pass_rate"].mean(skipna=True)) if not off.empty else float("nan")
    off["proe"] = off["neutral_pass_rate"] - league_neutral

    # Plays estimate per game (current season only)
    # Rough: plays per game (all plays by offense / games played)
    games = (pbp[pbp["season"] == current_season][["game_id", "posteam"]]
             .drop_duplicates()
             .groupby("posteam").size()
             .rename("games")).reset_index().rename(columns={"posteam":"team"})
    off = off.merge(games, on="team", how="left")
    off["plays_est"] = (off["plays"] / off["games"].clip(lower=1)).fillna(off["plays"])
    off["rz_rate"] = off["rz_rate"].fillna(0.0)

    team = off.merge(opp, on="team", how="outer")

    # z-scores for the columns your pricing consumes as *_z (defense-oriented):
    def _z(col):
        s = team[col]
        return (s - s.mean(skipna=True)) / (s.std(ddof=0, skipna=True) or 1.0)

    team["def_pass_epa_z"] = _z("def_pass_epa")
    team["def_rush_epa_z"] = _z("def_rush_epa")
    team["def_sack_rate_z"] = _z("def_sack_rate")       # currently zeros; placeholder until you wire real pressure
    team["def_pressure_rate_z"] = 0.0                   # placeholder until wired
    team["pace_z"] = _z("pace_sec_play")
    team["light_box_rate_z"] = _z("light_box_rate")
    team["heavy_box_rate_z"] = _z("heavy_box_rate")
    team["ay_per_att_z"] = 0.0                          # placeholder (air yards/team)

    # keep schema your pricing expects
    keep = [
        "team",
        "def_pressure_rate_z","def_pass_epa_z","def_rush_epa_z","def_sack_rate_z",
        "pace_z","light_box_rate_z","heavy_box_rate_z","ay_per_att_z",
        "plays_est","proe","rz_rate"
    ]
    out = team[keep].copy()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True, help="Current season (e.g., 2025)")
    ap.add_argument("--history", default="2019-2024",
                    help="Historical seasons range for priors (e.g., '2019-2024' or '2021-2023')")
    ap.add_argument("--write", default="data/team_form.csv")
    args = ap.parse_args()

    Path("data").mkdir(parents=True, exist_ok=True)

    # Parse history range
    hist = []
    if args.history:
        if "-" in args.history:
            a,b = args.history.split("-", 1)
            hist = list(range(int(a), int(b)+1))
        else:
            hist = [int(x) for x in args.history.split(",") if x.strip().isdigit()]

    pbp = load_pbp(hist, args.season)
    df = compute_team_form(pbp, args.season)

    df.to_csv(args.write, index=False)
    print(f"[fetch_nfl_data] ✅ wrote {len(df)} rows → {args.write}")

if __name__ == "__main__":
    main()
