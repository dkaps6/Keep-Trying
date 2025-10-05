# scripts/fetch_all.py
# Free-data builder with robust fallbacks + stadium weather inference:
# - Team + team-week context from nfl_data_py play-by-play
# - Player usage (targets / rush share) derived from PBP
# - ID map from nfl_data_py rosters; if empty → ESPN roster API fallback (with espn_player_id)
# - Stadium-based weather inference (uses inputs/stadiums.csv)
# - Writes metrics/fetch_status.json with rowcounts + providers

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict, Any, Tuple, List

import pandas as pd
import requests

try:
    import nfl_data_py as nfl
except Exception:
    nfl = None

METRICS_DIR = "metrics"
INPUTS_DIR = "inputs"
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(INPUTS_DIR, exist_ok=True)

# ---------- small IO helpers --------------------------------------------------

def _write_csv(df: pd.DataFrame, path: str) -> int:
    df.to_csv(path, index=False)
    return len(df)

def _status_init(season: int) -> Dict[str, Any]:
    return {
        "season": season,
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "providers": {},
        "rows": {},
        "notes": [],
    }

def _up(status: Dict[str, Any], name: str, rows: int, provider: str):
    status["rows"][name] = rows
    status["providers"][name] = provider

def _empty_df(cols: List[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})

def _read_csv_safe(path: str, cols: List[str] | None = None) -> pd.DataFrame:
    if not os.path.exists(path):
        return _empty_df(cols or [])
    try:
        df = pd.read_csv(path)
        if cols:
            for c in cols:
                if c not in df.columns:
                    df[c] = pd.NA
        return df
    except Exception:
        return _empty_df(cols or [])

# ---------- nflverse: PBP + builders -----------------------------------------

def _load_pbp(season: int) -> Tuple[pd.DataFrame, str]:
    if nfl is None:
        return _empty_df(["play_id"]), "nflverse:not_installed"
    try:
        df = nfl.import_pbp_data([season])
        for c in ("yards_gained", "yardline_100", "score_differential"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        # normalize week column
        if "week" not in df.columns and "game_week" in df.columns:
            df["week"] = df["game_week"]
        return df, "nflverse"
    except Exception:
        return _empty_df(["play_id"]), "nflverse:error"

def _build_team_week_form(pbp: pd.DataFrame, season: int) -> pd.DataFrame:
    if pbp.empty:
        return _empty_df(["team", "week"])

    df = pbp.copy()
    df["is_pass"] = (df.get("pass", False)).astype("int")
    df["is_rush"] = (df.get("rush", False)).astype("int")
    df["neutral"] = ((df.get("score_differential", 0).abs() <= 7) & (df.get("qtr", 0) <= 3)).astype("int")
    df["exp_pass"] = ((df["is_pass"] == 1) & (df.get("yards_gained", 0) >= 20)).astype("int")
    df["exp_rush"] = ((df["is_rush"] == 1) & (df.get("yards_gained", 0) >= 10)).astype("int")
    df["rz_play"] = (df.get("yardline_100", 100) <= 20).astype("int")

    team_col = "posteam" if "posteam" in df.columns else ("pos_team" if "pos_team" in df.columns else None)
    if team_col is None:
        return _empty_df(["team", "week"])
    if "week" not in df.columns:
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
    g = team_week.groupby(["season","team"], dropna=False).agg(
        weeks=("week","count"),
        pass_plays=("pass_plays","sum"),
        rush_plays=("rush_plays","sum"),
        neutral_pass_rate=("neutral_pass_rate","mean"),
        exp_pass_rate=("exp_pass_rate","mean"),
        exp_rush_rate=("exp_rush_rate","mean"),
        rz_rate=("rz_rate","mean"),
        total_plays=("total_plays","sum"),
    ).reset_index()
    return g

def _build_player_form(pbp: pd.DataFrame, season: int) -> pd.DataFrame:
    if pbp.empty:
        return _empty_df(["player","team","week"])

    df = pbp.copy()
    # targets
    recv = df[["posteam","week","receiver_player_name"]].copy()
    recv = recv[recv["receiver_player_name"].notna()]
    recv["targets"] = 1
    # rush attempts
    rush = df[["posteam","week","rusher_player_name"]].copy()
    rush = rush[rush["rusher_player_name"].notna()]
    rush["rush_att"] = 1

    team_pass = recv.groupby(["posteam","week"], dropna=False)["targets"].sum().rename("team_targets")
    team_rush = rush.groupby(["posteam","week"], dropna=False)["rush_att"].sum().rename("team_rushatt")

    p_tgt = recv.groupby(["posteam","week","receiver_player_name"], dropna=False)["targets"].sum().reset_index()
    p_tgt.rename(columns={"receiver_player_name":"player"}, inplace=True)
    p_rush = rush.groupby(["posteam","week","rusher_player_name"], dropna=False)["rush_att"].sum().reset_index()
    p_rush.rename(columns={"rusher_player_name":"player"}, inplace=True)

    players = pd.merge(p_tgt, p_rush, on=["posteam","week","player"], how="outer").fillna(0)
    players = players.merge(team_pass, on=["posteam","week"], how="left")
    players = players.merge(team_rush, on=["posteam","week"], how="left")
    players["tgt_share"] = players["targets"].div(players["team_targets"].clip(lower=1))
    players["rush_share"] = players["rush_att"].div(players["team_rushatt"].clip(lower=1))
    players["season"] = season
    players.rename(columns={"posteam":"team"}, inplace=True)

    cols = ["season","team","week","player","targets","rush_att","tgt_share","rush_share"]
    return players[cols].sort_values(["team","player","week"])

# ---------- nflverse rosters → id_map ----------------------------------------

def _id_map_from_nflverse(season: int) -> Tuple[pd.DataFrame, str]:
    if nfl is None:
        return _empty_df(["player","team","position","espn_player_id"]), "nflverse:not_installed"
    try:
        rost = nfl.import_rosters([season])
        if rost.empty:
            return _empty_df(["player","team","position","espn_player_id"]), "nflverse:empty"
        df = rost.rename(columns={
            "recent_team": "team",
            "player_name": "player",
            "position": "position",
            "espn_id": "espn_player_id",
        })[["player","team","position","espn_player_id"]].dropna(subset=["player","team"])
        return df.drop_duplicates(), "nflverse"
    except Exception:
        return _empty_df(["player","team","position","espn_player_id"]), "nflverse:error"

# ---------- ESPN roster fallback (with espn_player_id) ------------------------

_ESPN_TEAMS = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
_ESPN_TEAM_ROSTER = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{tid}?enable=roster"

def _req_json(url: str, tries: int = 3, timeout: int = 20) -> Dict[str, Any] | None:
    for i in range(tries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 503):
                time.sleep(1.5 * (i + 1))
            else:
                time.sleep(0.5)
        except requests.RequestException:
            time.sleep(0.5)
    return None

def _id_map_from_espn() -> Tuple[pd.DataFrame, str]:
    base = _req_json(_ESPN_TEAMS)
    if not base:
        return _empty_df(["player","team","position","espn_player_id"]), "espn:error_teams"

    teams: list[tuple[str,str]] = []
    try:
        for t in base["sports"][0]["leagues"][0]["teams"]:
            team = t["team"]
            abbr = team.get("abbreviation") or team.get("shortDisplayName")
            teams.append((team["id"], abbr))
    except Exception:
        return _empty_df(["player","team","position","espn_player_id"]), "espn:parse_teams"

    rows = []
    for tid, abbr in teams:
        j = _req_json(_ESPN_TEAM_ROSTER.format(tid=tid))
        if not j:
            continue
        try:
            roster = j["team"]["athletes"]
        except Exception:
            continue
        for group in roster:
            pos = group.get("position", {}).get("abbreviation") or group.get("position", {}).get("name")
            for ath in group.get("items", []):
                name = ath.get("fullName") or ath.get("displayName")
                espn_pid = ath.get("id")  # ESPN athlete id (string)
                if name and abbr:
                    rows.append((name, abbr, pos, espn_pid))
        time.sleep(0.05)  # small courtesy pause

    if not rows:
        return _empty_df(["player","team","position","espn_player_id"]), "espn:empty"

    df = pd.DataFrame(rows, columns=["player","team","position","espn_player_id"]).drop_duplicates()
    return df, "espn"

# ---------- Stadium-based weather inference ----------------------------------

STADIUM_COLS = ["team","abbr","stadium","city","state","roof","is_dome","surface","altitude_ft"]

def _infer_weather_from_stadiums(stadiums: pd.DataFrame) -> pd.DataFrame:
    """
    Per-team baseline weather inference:
      - dome/retractable: wind=0, temp=70F, precip=0
      - outdoor: NaNs (to be filled by a live weather step later)
    Carries surface + altitude to downstream features.
    """
    if stadiums.empty:
        return _empty_df(["team","abbr","is_dome","surface","altitude_ft","temp_f","wind_mph","precip_in"])

    df = stadiums.copy()
    # normalize booleans
    dome = df.get("is_dome")
    if dome is None:
        dome = df.get("roof", "").astype(str).str.contains("Fixed|Retractable", case=False, regex=True)
    else:
        dome = dome.astype(str).str.upper().isin(["TRUE","1","YES","Y"])

    out = pd.DataFrame({
        "team": df["team"].astype(str),
        "abbr": df.get("abbr", df["team"]).astype(str),
        "is_dome": dome,
        "surface": df.get("surface", "Grass"),
        "altitude_ft": pd.to_numeric(df.get("altitude_ft", 0), errors="coerce").fillna(0).astype(int),
    })

    # defaults
    out["temp_f"] = None
    out["wind_mph"] = None
    out["precip_in"] = None

    # dome/retractable inference
    dome_mask = out["is_dome"] == True
    out.loc[dome_mask, "temp_f"] = 70
    out.loc[dome_mask, "wind_mph"] = 0
    out.loc[dome_mask, "precip_in"] = 0.0

    return out

# ---------- main --------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    args = ap.parse_args()

    status = _status_init(args.season)

    # 1) PBP & core form
    pbp, pbp_provider = _load_pbp(args.season)
    status["providers"]["pbp"] = pbp_provider
    status["rows"]["pbp"] = len(pbp)

    team_week = _build_team_week_form(pbp, args.season)
    _up(status, "team_week_form.csv", _write_csv(team_week, os.path.join(METRICS_DIR,"team_week_form.csv")), pbp_provider)

    team_form = _build_team_form(team_week)
    _up(status, "team_form.csv", _write_csv(team_form, os.path.join(METRICS_DIR,"team_form.csv")), pbp_provider)

    player_form = _build_player_form(pbp, args.season)
    _up(status, "player_form.csv", _write_csv(player_form, os.path.join(METRICS_DIR,"player_form.csv")), "pbp-derived")

    # 2) id_map: nflverse → fallback espn (with espn_player_id)
    id_map, id_src = _id_map_from_nflverse(args.season)
    rows = _write_csv(id_map, os.path.join(INPUTS_DIR,"id_map.csv"))
    _up(status, "id_map.csv", rows, id_src)

    if rows == 0:
        espn_map, espn_src = _id_map_from_espn()
        rows2 = _write_csv(espn_map, os.path.join(INPUTS_DIR,"id_map.csv"))
        _up(status, "id_map.csv", rows2, espn_src)
        if rows2 == 0:
            status["notes"].append("Both nflverse and ESPN rosters empty; id_map.csv will be empty")

    # 3) Stadium-based weather inference
    stadiums = _read_csv_safe(os.path.join(INPUTS_DIR, "stadiums.csv"), STADIUM_COLS)
    if stadiums.empty:
        status["notes"].append("inputs/stadiums.csv not found or empty; weather inference skipped")
        weather = _empty_df(["team","abbr","is_dome","surface","altitude_ft","temp_f","wind_mph","precip_in"])
        provider = "placeholder"
    else:
        weather = _infer_weather_from_stadiums(stadiums)
        provider = "stadium_inference"

    _up(status, "weather.csv", _write_csv(weather, os.path.join(INPUTS_DIR,"weather.csv")), provider)

    with open(os.path.join(METRICS_DIR,"fetch_status.json"), "w") as f:
        json.dump(status, f, indent=2)

    print("Fetch complete:\n", json.dumps(status, indent=2))

if __name__ == "__main__":
    main()
