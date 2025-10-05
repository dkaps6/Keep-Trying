# scripts/fetch_all.py
# Robust “free sources” fetcher:
# - Team form and team-week form from nfl_data_py play-by-play
# - Player usage (player_form) derived from play-by-play (targets/rush share)
# - ID map from nfl_data_py rosters (fallback: empty but well-formed)
# - Optional PFR pages (graceful if 403/blocked)
# - Writes metrics/fetch_status.json with row counts & providers

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Dict, Any, Tuple

import pandas as pd

# nfl_data_py is already in your requirements
try:
    import nfl_data_py as nfl
except Exception as e:
    nfl = None

# --- IO helpers --------------------------------------------------------------

METRICS_DIR = "metrics"
INPUTS_DIR = "inputs"
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(INPUTS_DIR, exist_ok=True)

def _write_csv(df: pd.DataFrame, path: str) -> int:
    df.to_csv(path, index=False)
    return len(df)

def _status_init() -> Dict[str, Any]:
    return {
        "season": None,
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "providers": {},
        "rows": {},
        "notes": [],
    }

def _status_upsert(status: Dict[str, Any], key: str, rows: int, provider: str):
    status["rows"][key] = rows
    status["providers"][key] = provider

def _empty_df(cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})

# --- PFR best-effort helpers (optional) --------------------------------------

def _pfr_read_html(url: str) -> Tuple[pd.DataFrame | None, str]:
    """
    Best-effort read_html wrapper. Many runners are blocked by PFR.
    Returns (df, provider_str). If blocked or empty, returns (None, "pfr:blocked").
    """
    try:
        tables = pd.read_html(url)  # will raise if blocked/changed
        if tables and len(tables[0]) > 0:
            return tables[0], "pfr"
        return None, "pfr:empty"
    except Exception:
        return None, "pfr:blocked"

# --- Feature Builders from PBP ------------------------------------------------

def _load_pbp(season: int) -> Tuple[pd.DataFrame, str]:
    if nfl is None:
        return _empty_df(["play_id"]), "nflverse:not_installed"

    try:
        # One season; nfl_data_py handles paging
        pbp = nfl.import_pbp_data([season])
        # Light downcast to quiet warnings and speed grouping
        for c in ("yards_gained", "yardline_100", "score_differential"):
            if c in pbp.columns:
                pbp[c] = pd.to_numeric(pbp[c], errors="coerce")
        return pbp, "nflverse"
    except Exception:
        return _empty_df(["play_id"]), "nflverse:error"

def _build_team_week_form(pbp: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Basic team-week context:
      - pass Plays, rush plays
      - neutral pass rate (score diff between -7..7, quarters 1-3)
      - explosive rates: pass >= 20y, rush >= 10y
      - red-zone attempts proxy (plays with yardline_100 <= 20)
    """
    if pbp.empty:
        return _empty_df(["team", "week"])

    df = pbp.copy()

    df["is_pass"] = df.get("pass", False).astype("int")
    df["is_rush"] = df.get("rush", False).astype("int")

    # Neutral situations: score diff in [-7, 7] and before Q4
    df["neutral"] = (
        (df.get("score_differential", 0).abs() <= 7)
        & (df.get("qtr", 0) <= 3)
    ).astype("int")

    # Explosive proxies
    df["exp_pass"] = ((df["is_pass"] == 1) & (df.get("yards_gained", 0) >= 20)).astype("int")
    df["exp_rush"] = ((df["is_rush"] == 1) & (df.get("yards_gained", 0) >= 10)).astype("int")

    # Red-zone proxy
    df["rz_play"] = (df.get("yardline_100", 100) <= 20).astype("int")

    # offense_team: prefer 'posteam' then 'pos_team'
    team_col = "posteam" if "posteam" in df.columns else ("pos_team" if "pos_team" in df.columns else None)
    if team_col is None:
        return _empty_df(["team", "week"])

    week_col = "week" if "week" in df.columns else "game_week"
    if week_col not in df.columns:
        df["week"] = df.get("game_week", 1)

    g = df.groupby([team_col, "week"], dropna=False).agg(
        pass_plays=("is_pass", "sum"),
        rush_plays=("is_rush", "sum"),
        neutral_pass=("is_pass", lambda s: (s[df.loc[s.index, "neutral"] == 1].sum() or 0)),
        neutral_total=("neutral", "sum"),
        exp_pass=("exp_pass", "sum"),
        exp_rush=("exp_rush", "sum"),
        rz_plays=("rz_play", "sum"),
        total_plays=("is_pass", "count"),
    ).reset_index()

    g.rename(columns={team_col: "team"}, inplace=True)
    # Rates
    g["neutral_pass_rate"] = g["neutral_pass"].div(g["neutral_total"]).fillna(0.0)
    g["exp_pass_rate"] = g["exp_pass"].div(g["pass_plays"].clip(lower=1)).fillna(0.0)
    g["exp_rush_rate"] = g["exp_rush"].div(g["rush_plays"].clip(lower=1)).fillna(0.0)
    g["rz_rate"] = g["rz_plays"].div(g["total_plays"].clip(lower=1)).fillna(0.0)
    g["season"] = season
    return g[[
        "season","team","week","pass_plays","rush_plays","neutral_pass_rate",
        "exp_pass_rate","exp_rush_rate","rz_rate","total_plays"
    ]]

def _build_team_form(team_week: pd.DataFrame) -> pd.DataFrame:
    if team_week.empty:
        return _empty_df(["team"])

    g = team_week.groupby(["season", "team"], dropna=False).agg(
        weeks=("week", "count"),
        pass_plays=("pass_plays", "sum"),
        rush_plays=("rush_plays", "sum"),
        neutral_pass_rate=("neutral_pass_rate", "mean"),
        exp_pass_rate=("exp_pass_rate", "mean"),
        exp_rush_rate=("exp_rush_rate", "mean"),
        rz_rate=("rz_rate", "mean"),
        total_plays=("total_plays","sum"),
    ).reset_index()
    return g

def _build_player_form(pbp: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Very robust, no-scrape player usage from PBP:
      - targets (receiver), rush att (rusher)
      - team shares per week
    """
    if pbp.empty:
        return _empty_df(["player","team","week"])

    df = pbp.copy()

    # Identify receivers & rushers by player name fields
    recv = df[["posteam","week","receiver_player_name"]].copy()
    recv = recv[recv["receiver_player_name"].notna()]
    recv["targets"] = 1

    rush = df[["posteam","week","rusher_player_name"]].copy()
    rush = rush[rush["rusher_player_name"].notna()]
    rush["rush_att"] = 1

    # Team totals for shares
    team_pass = recv.groupby(["posteam","week"], dropna=False)["targets"].sum().rename("team_targets")
    team_rush = rush.groupby(["posteam","week"], dropna=False)["rush_att"].sum().rename("team_rushatt")

    # Player weekly
    p_tgt = recv.groupby(["posteam","week","receiver_player_name"], dropna=False)["targets"].sum().reset_index()
    p_tgt.rename(columns={"receiver_player_name":"player"}, inplace=True)
    p_rush = rush.groupby(["posteam","week","rusher_player_name"], dropna=False)["rush_att"].sum().reset_index()
    p_rush.rename(columns={"rusher_player_name":"player"}, inplace=True)

    # Combine
    players = pd.merge(p_tgt, p_rush, on=["posteam","week","player"], how="outer").fillna(0)
    players = players.merge(team_pass, on=["posteam","week"], how="left")
    players = players.merge(team_rush, on=["posteam","week"], how="left")
    players["tgt_share"] = players["targets"].div(players["team_targets"].clip(lower=1))
    players["rush_share"] = players["rush_att"].div(players["team_rushatt"].clip(lower=1))
    players["season"] = season
    players.rename(columns={"posteam":"team"}, inplace=True)

    # Stable columns
    cols = ["season","team","week","player","targets","rush_att","tgt_share","rush_share"]
    return players[cols].sort_values(["team","player","week"])

def _build_id_map_from_rosters(season: int) -> Tuple[pd.DataFrame, str]:
    if nfl is None:
        return _empty_df(["player","team","position"]), "nflverse:not_installed"
    try:
        rost = nfl.import_rosters([season])
        if rost.empty:
            return _empty_df(["player","team","position"]), "nflverse:empty"
        df = rost.rename(columns={
            "recent_team": "team",
            "player_name": "player",
            "position": "position"
        })[["player","team","position"]].dropna(subset=["player","team"])
        return df.drop_duplicates(), "nflverse"
    except Exception:
        return _empty_df(["player","team","position"]), "nflverse:error"

# --- Main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    args = ap.parse_args()

    status = _status_init()
    status["season"] = args.season

    # 1) PBP
    pbp, provider = _load_pbp(args.season)
    status["providers"]["pbp"] = provider
    status["rows"]["pbp"] = len(pbp)

    # 2) Team-week & Team form
    team_week = _build_team_week_form(pbp, args.season)
    rows = _write_csv(team_week, os.path.join(METRICS_DIR, "team_week_form.csv"))
    _status_upsert(status, "team_week_form.csv", rows, provider)

    team_form = _build_team_form(team_week)
    rows = _write_csv(team_form, os.path.join(METRICS_DIR, "team_form.csv"))
    _status_upsert(status, "team_form.csv", rows, provider)

    # 3) Player_form from PBP
    player_form = _build_player_form(pbp, args.season)
    rows = _write_csv(player_form, os.path.join(METRICS_DIR, "player_form.csv"))
    _status_upsert(status, "player_form.csv", rows, "pbp-derived")

    # 4) ID map from rosters
    id_map, id_provider = _build_id_map_from_rosters(args.season)
    rows = _write_csv(id_map, os.path.join(INPUTS_DIR, "id_map.csv"))
    _status_upsert(status, "id_map.csv", rows, id_provider)

    # 5) Optional PFR calls (best-effort). Keep disabled for now; enable later if needed:
    # pfr_df, pfr_src = _pfr_read_html(f"https://www.pro-football-reference.com/years/{args.season}/opp.htm")
    # if pfr_df is None:
    #     status["notes"].append(f"PFR opponent splits {pfr_src}")
    # else:
    #     # you can persist selected columns to metrics/ if desired
    #     pass

    # 6) Weather (optional placeholder; we’ll fill later)
    weather = _empty_df(["team","game_id","is_dome","temp_f","wind_mph"])
    rows = _write_csv(weather, os.path.join(INPUTS_DIR, "weather.csv"))
    _status_upsert(status, "weather.csv", rows, "placeholder")

    # 7) Status json
    with open(os.path.join(METRICS_DIR, "fetch_status.json"), "w") as f:
        json.dump(status, f, indent=2)

    print("Fetch complete:", json.dumps(status, indent=2))

if __name__ == "__main__":
    main()
