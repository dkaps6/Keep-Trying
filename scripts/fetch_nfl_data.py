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

# --- add at top of file if not present ---
import pandas as pd
try:
    import nfl_data_py as nfl
    HAS_NFL_DATA_PY = True
except Exception:
    HAS_NFL_DATA_PY = False

# ---------- NEW: build team form from weekly (fallback when PBP missing) ----------
def _team_form_from_weekly(season: int) -> pd.DataFrame:
    """
    Fallback using nfl_data_py.import_weekly_data(season).
    Produces: team, plays_est, proe (proxy), rz_rate (proxy), and neutral 0 z-scores for DEF fields.
    """
    if not HAS_NFL_DATA_PY:
        return pd.DataFrame()

    try:
        print(f"[fetch_nfl_data] weekly fallback: nfl_data_py.import_weekly_data({season}) ...")
        wk = nfl.import_weekly_data([season])
    except Exception as e:
        print(f"[fetch_nfl_data] weekly fallback failed: {type(e).__name__}: {e}")
        return pd.DataFrame()

    if wk is None or wk.empty:
        return pd.DataFrame()

    # Normalize expected columns
    # Common columns: team, opponent_team, attempts, completions, targets, receptions, carries, rush_yards, rec_yards, pass_yards, week
    # We'll derive team totals per game then per team per week.
    # 'team' naming in nfl_data_py weekly is usually team abbreviation.
    # plays ≈ team pass attempts + team rush attempts
    # PROE proxy ≈ (team_pass_rate - league_pass_rate)
    # RZ rate proxy unavailable here → approximate from TD opportunities:
    cols_needed = ["team","opponent_team","week","attempts","carries","targets","receptions",
                   "pass_yards","rush_yards","rec_yards","games"]
    # Compute team totals per week
    # Many rows per player; sum by team/week
    grp = (wk.groupby(["team","week"], dropna=True)
             .agg(team_pass_att=("attempts","sum"),
                  team_rush_att=("carries","sum"),
                  team_targets=("targets","sum"),
                  team_receptions=("receptions","sum"))
             .reset_index())
    if grp.empty:
        return pd.DataFrame()

    grp["team_plays"] = grp["team_pass_att"].fillna(0) + grp["team_rush_att"].fillna(0)
    # Per-team games played
    games = grp.groupby("team")["week"].nunique().rename("games").reset_index()
    team = grp.groupby("team").agg(
        plays_est=("team_plays", "mean"),
        pass_att=("team_pass_att","sum"),
        rush_att=("team_rush_att","sum"),
    ).reset_index().merge(games, on="team", how="left")

    team["pass_rate"] = team["pass_att"] / (team["pass_att"] + team["rush_att"]).replace({0: pd.NA})
    league_pass = float(team["pass_rate"].mean(skipna=True))
    team["proe"] = (team["pass_rate"] - league_pass).fillna(0.0)

    # rz_rate proxy (we don’t have RZ plays here): use receptions near end zone proxy via touchdowns would be better;
    # keep conservative constant so pipeline runs; pricing still uses weather/script/market anchors.
    team["rz_rate"] = 0.20

    # Fill defensive z-scores with neutral zeros (until PBP/pressure feeds arrive)
    out = team[["team","plays_est","proe","rz_rate"]].copy()
    out["def_pressure_rate_z"] = 0.0
    out["def_pass_epa_z"] = 0.0
    out["def_rush_epa_z"] = 0.0
    out["def_sack_rate_z"] = 0.0
    out["pace_z"] = 0.0
    out["light_box_rate_z"] = 0.0
    out["heavy_box_rate_z"] = 0.0
    out["ay_per_att_z"] = 0.0

    # Order to your schema
    out = out[[
        "team",
        "def_pressure_rate_z","def_pass_epa_z","def_rush_epa_z","def_sack_rate_z",
        "pace_z","light_box_rate_z","heavy_box_rate_z","ay_per_att_z",
        "plays_est","proe","rz_rate"
    ]]
    return out

# ---------- REPLACE your compute_team_form(...) with this ----------
def compute_team_form(pbp: pd.DataFrame, current_season: int) -> pd.DataFrame:
    """
    Preferred: build from PBP. If current-season PBP is unavailable, fall back to weekly stats.
    """
    cols = [
        "team","opp_team","event_id",
        "def_pressure_rate_z","def_pass_epa_z","def_rush_epa_z","def_sack_rate_z",
        "pace_z","light_box_rate_z","heavy_box_rate_z","ay_per_att_z",
        "plays_est","proe","rz_rate"
    ]
    if pbp.empty:
        # No PBP at all → try weekly
        wk = _team_form_from_weekly(current_season)
        if not wk.empty:
            wk["opp_team"] = pd.NA
            wk["event_id"] = pd.NA
            return wk[cols]
        return pd.DataFrame(columns=cols)

    cur = pbp[pbp["season"] == current_season].copy()
    if cur.empty:
        # Current season missing → weekly fallback just for 2025 while keeping historical priors
        wk = _team_form_from_weekly(current_season)
        if not wk.empty:
            wk["opp_team"] = pd.NA
            wk["event_id"] = pd.NA
            return wk[cols]
        # As a last resort, make neutral rows for all teams seen in pbp
        teams = sorted(set(pbp["posteam"].dropna()))
        out = pd.DataFrame({"team": teams})
        for c in ["def_pressure_rate_z","def_pass_epa_z","def_rush_epa_z","def_sack_rate_z",
                  "pace_z","light_box_rate_z","heavy_box_rate_z","ay_per_att_z","plays_est","proe","rz_rate"]:
            out[c] = 0.0
        out["opp_team"] = pd.NA
        out["event_id"] = pd.NA
        return out[cols]

    # ----- your existing PBP-based logic (fixed for defteam collision) -----
    cur["is_pass"] = (cur.get("pass", 0) == 1) if "pass" in cur.columns else cur.get("play_type","").astype(str).str.contains("pass", case=False, na=False)
    cur["is_rush"] = (cur.get("rush", 0) == 1) if "rush" in cur.columns else cur.get("play_type","").astype(str).str.contains("rush", case=False, na=False)
    cur["is_rz"]   = (cur.get("yardline_100", 100) <= 20)

    off = (cur
           .groupby("posteam", as_index=False)
           .agg(
               plays=("posteam", "size"),
               pass_plays=("is_pass", "sum"),
               rush_plays=("is_rush", "sum"),
               rz_rate=("is_rz","mean"),
           )
           .rename(columns={"posteam":"team"}))

    # neutral PROE
    mask_neutral = ((cur.get("qtr",0).between(1,3)) | ((cur.get("qtr",0)==4) & (cur.get("half_seconds_remaining",0)>300))) & (cur.get("ydstogo",10)<=10)
    league_neutral = float(cur.loc[mask_neutral, "is_pass"].mean()) if mask_neutral.any() else 0.56
    team_neutral = (cur.loc[mask_neutral].groupby("posteam")["is_pass"].mean()
                    .rename("neutral_pass_rate")).reset_index().rename(columns={"posteam":"team"})
    off = off.merge(team_neutral, on="team", how="left")
    off["neutral_pass_rate"] = off["neutral_pass_rate"].fillna(league_neutral)
    off["proe"] = off["neutral_pass_rate"] - league_neutral

    # pace proxy
    d = cur.sort_values(["game_id","game_seconds_remaining"]).copy()
    d["dt"] = d.groupby("game_id")["game_seconds_remaining"].diff(-1).abs()
    pace = d.groupby("posteam")["dt"].median().rename("pace_sec_play").reset_index().rename(columns={"posteam":"team"})
    off = off.merge(pace, on="team", how="left")

    games = (cur[["game_id","posteam"]].drop_duplicates().groupby("posteam").size()
             .rename("games")).reset_index().rename(columns={"posteam":"team"})
    off = off.merge(games, on="team", how="left")
    off["plays_est"] = (off["plays"] / off["games"].clip(lower=1)).fillna(off["plays"])

    # defense without collision
    g_pass = (cur[cur["is_pass"]].groupby("defteam", as_index=False)["epa"].mean().rename(columns={"epa":"def_pass_epa"}))
    g_rush = (cur[cur["is_rush"]].groupby("defteam", as_index=False)["epa"].mean().rename(columns={"epa":"def_rush_epa"}))
    g_cnt  = (cur.groupby("defteam", as_index=False).size().rename(columns={"size":"def_plays"}))
    opp = g_cnt.merge(g_pass, on="defteam", how="left").merge(g_rush, on="defteam", how="left").rename(columns={"defteam":"team"})
    opp["light_box_rate"] = 0.0; opp["heavy_box_rate"] = 0.0; opp["def_sack_rate"] = 0.0

    team = off.merge(opp, on="team", how="outer")

    def _z(name):
        s = team[name].astype(float)
        mu = s.mean(skipna=True); sd = s.std(ddof=0, skipna=True)
        if sd == 0 or pd.isna(sd): return s*0.0
        return (s - mu) / sd

    team["def_pass_epa_z"] = _z("def_pass_epa")
    team["def_rush_epa_z"] = _z("def_rush_epa")
    team["def_sack_rate_z"] = _z("def_sack_rate")
    team["def_pressure_rate_z"] = 0.0
    team["pace_z"] = _z("pace_sec_play")
    team["light_box_rate_z"] = _z("light_box_rate")
    team["heavy_box_rate_z"] = _z("heavy_box_rate")
    team["ay_per_att_z"] = 0.0

    out = team[[
        "team",
        "def_pressure_rate_z","def_pass_epa_z","def_rush_epa_z","def_sack_rate_z",
        "pace_z","light_box_rate_z","heavy_box_rate_z","ay_per_att_z",
        "plays_est","proe","rz_rate"
    ]].copy()
    for c in ["plays_est","proe","rz_rate"]:
        out[c] = out[c].fillna(0.0)
    return out

    # flags
    cur["is_pass"] = _is_pass(cur)
    cur["is_rush"] = _is_rush(cur)
    cur["is_rz"]   = (cur.get("yardline_100", 100) <= 20)

    # ---------- OFFENSE (by posteam) ----------
    off = (cur
           .groupby("posteam", as_index=False)
           .agg(
               plays=("posteam", "size"),
               pass_plays=("is_pass", "sum"),
               rush_plays=("is_rush", "sum"),
               epa_mean=("epa", "mean"),
               epa_pass=("epa", lambda s: cur.loc[s.index & cur["is_pass"], "epa"].mean()),
               epa_rush=("epa", lambda s: cur.loc[s.index & cur["is_rush"], "epa"].mean()),
               rz_rate=("is_rz", "mean"),
           )
           .rename(columns={"posteam": "team"}))

    # neutral filter for PROE
    neutral_mask = _neutral_filter(cur)
    league_neutral = float(
        (cur.loc[neutral_mask, "is_pass"].mean())
        if neutral_mask.any() else 0.56
    )
    team_neutral = (cur.loc[neutral_mask]
                      .groupby("posteam")["is_pass"].mean()
                      .rename("neutral_pass_rate")).reset_index().rename(columns={"posteam":"team"})
    off = off.merge(team_neutral, on="team", how="left")
    off["neutral_pass_rate"] = off["neutral_pass_rate"].fillna(league_neutral)
    off["proe"] = off["neutral_pass_rate"] - league_neutral

    # Pace & plays per game estimate
    # (median time between plays as a crude pace proxy)
    d = cur.sort_values(["game_id", "game_seconds_remaining"]).copy()
    d["dt"] = d.groupby("game_id")["game_seconds_remaining"].diff(-1).abs()
    pace_sec = d.groupby("posteam")["dt"].median().rename("pace_sec_play").reset_index().rename(columns={"posteam":"team"})
    off = off.merge(pace_sec, on="team", how="left")

    games = (cur[["game_id","posteam"]].drop_duplicates()
             .groupby("posteam").size().rename("games")
             ).reset_index().rename(columns={"posteam":"team"})
    off = off.merge(games, on="team", how="left")
    off["plays_est"] = (off["plays"] / off["games"].clip(lower=1)).fillna(off["plays"])

    # ---------- DEFENSE (by defteam) ----------
    # Avoid the reset_index/duplicate-column bug: build with as_index=False + separate pass/rush means
    g_pass = (cur[cur["is_pass"]]
              .groupby("defteam", as_index=False)["epa"]
              .mean().rename(columns={"epa":"def_pass_epa"}))
    g_rush = (cur[cur["is_rush"]]
              .groupby("defteam", as_index=False)["epa"]
              .mean().rename(columns={"epa":"def_rush_epa"}))
    g_count = (cur.groupby("defteam", as_index=False)
               .size().rename(columns={"size":"def_plays"}))

    opp = g_count.merge(g_pass, on="defteam", how="left").merge(g_rush, on="defteam", how="left")
    # placeholders; wire real sources later
    opp["light_box_rate"] = 0.0
    opp["heavy_box_rate"] = 0.0
    opp["def_sack_rate"]  = 0.0
    opp = opp.rename(columns={"defteam": "team"})

    # ---------- Combine + Z-scores ----------
    team = off.merge(opp, on="team", how="outer")

    def _z(name):
        s = team[name].astype(float)
        mu = s.mean(skipna=True)
        sd = s.std(ddof=0, skipna=True)
        if sd == 0 or pd.isna(sd):
            return s*0.0
        return (s - mu) / sd

    team["def_pass_epa_z"]    = _z("def_pass_epa")
    team["def_rush_epa_z"]    = _z("def_rush_epa")
    team["def_sack_rate_z"]   = _z("def_sack_rate")
    team["def_pressure_rate_z"]= 0.0
    team["pace_z"]            = _z("pace_sec_play")
    team["light_box_rate_z"]  = _z("light_box_rate")
    team["heavy_box_rate_z"]  = _z("heavy_box_rate")
    team["ay_per_att_z"]      = 0.0

    # keep schema your pricing expects
    out = team[[
        "team",
        "def_pressure_rate_z","def_pass_epa_z","def_rush_epa_z","def_sack_rate_z",
        "pace_z","light_box_rate_z","heavy_box_rate_z","ay_per_att_z",
        "plays_est","proe","rz_rate"
    ]].copy()

    # fill any missing numerics with 0 so downstream joins don’t break
    for c in ["plays_est","proe","rz_rate"]:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = out[c].fillna(0.0)

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
