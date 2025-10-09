#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = REPO / "outputs" / "metrics"
OUT.mkdir(parents=True, exist_ok=True)

def _read_csv(p):
    try: return pd.read_csv(p)
    except Exception: return pd.DataFrame()

def _read_parquet(p):
    try: return pd.read_parquet(p)
    except Exception: return pd.DataFrame()

def zscore(s):
    a = pd.to_numeric(s, errors="coerce").astype(float)
    mu, sd = np.nanmean(a), np.nanstd(a)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(a)), index=s.index)
    return (a - mu) / sd

def build_team_form(season):
    pbp = _read_parquet(DATA / f"pbp_{season}.parquet")
    if pbp.empty:
        return pd.DataFrame(columns=["season","defense_team","def_pressure_rate_z","def_sack_rate_z","def_pass_epa_z","def_rush_epa_z","pace_z","ay_per_att_z"])

    pbp["is_pass"] = (pbp.get("pass", 0) == 1) | (pbp.get("pass", False) == True)
    pbp["is_rush"] = (pbp.get("rush", 0) == 1) | (pbp.get("rush", False) == True)
    pbp["is_db"]   = pbp["is_pass"]
    pbp["prs_event"] = (pbp.get("sack", 0).fillna(0).astype(int) > 0) | (pbp.get("qb_hit", 0).fillna(0).astype(int) > 0)

    grp = pbp.groupby(["season","defteam"], as_index=False)

    def pass_epa(g):
        m = g.loc[g["is_pass"], "epa"]
        return np.nan if m.empty else np.nanmean(m)

    def rush_epa(g):
        m = g.loc[g["is_rush"], "epa"]
        return np.nan if m.empty else np.nanmean(m)

    epa_pass = grp.apply(pass_epa).reset_index(name="def_pass_epa")
    epa_rush = grp.apply(rush_epa).reset_index(name="def_rush_epa")
    base = grp.agg(dropbacks=("is_db","sum"), prs_events=("prs_event","sum"), sacks=("sack","sum"))
    base = base.merge(epa_pass, on=["season","defteam"], how="left").merge(epa_rush, on=["season","defteam"], how="left")

    base["def_pressure_rate"] = np.where(base["dropbacks"]>0, base["prs_events"]/base["dropbacks"], np.nan)
    base["def_sack_rate"]     = np.where(base["dropbacks"]>0, base["sacks"]/base["dropbacks"], np.nan)

    try:
        pbp["secs_elapsed"] = (60*15) - pbp["game_seconds_remaining"].fillna(0)
        pace = pbp.groupby(["season","defteam"], as_index=False).agg(plays=("play_id","count"), secs=("secs_elapsed","max"))
        pace["pace_sec_per_play"] = np.where(pace["plays"]>0, pace["secs"]/pace["plays"], np.nan)
        base = base.merge(pace[["season","defteam","pace_sec_per_play"]], on=["season","defteam"], how="left")
    except Exception:
        base["pace_sec_per_play"] = np.nan

    try:
        sub = pbp.loc[pbp["is_pass"] & pbp["air_yards"].notna(), ["season","posteam","defteam","air_yards"]]
        ay = sub.groupby(["season","posteam"], as_index=False)["air_yards"].mean().rename(columns={"air_yards":"ay_per_att"})
        ay_def = sub.merge(ay, on=["season","posteam"], how="left").groupby(["season","defteam"], as_index=False)["ay_per_att"].mean()
        base = base.merge(ay_def, on=["season","defteam"], how="left")
    except Exception:
        base["ay_per_att"] = np.nan

    base = base.rename(columns={"defteam":"defense_team"})
    for c in ["def_pressure_rate","def_sack_rate","def_pass_epa","def_rush_epa","pace_sec_per_play","ay_per_att"]:
        base[c+"_z"] = zscore(base[c])
    keep = ["season","defense_team","def_pressure_rate_z","def_sack_rate_z","def_pass_epa_z","def_rush_epa_z","pace_z","ay_per_att_z"]
    return base[keep]

def build_player_form(season):
    players = _read_csv(DATA / "player_stats_week.csv")
    rosters = _read_csv(DATA / "rosters.csv")
    if players.empty:
        return pd.DataFrame(columns=["season","player","team","position","targets","target_share","rush_att","rush_share","routes","route_share"])
    if "targets" not in players.columns and "target" in players.columns:
        players = players.rename(columns={"target":"targets"})
    if "rushing_attempts" not in players.columns and "rush_attempts" in players.columns:
        players = players.rename(columns={"rush_attempts":"rushing_attempts"})
    week_team = players.groupby(["season","team","week"], as_index=False).agg(tm_targets=("targets","sum"), tm_rush_att=("rushing_attempts","sum"))
    df = players.merge(week_team, on=["season","team","week"], how="left")
    df["target_share_week"] = (df["targets"]/df["tm_targets"]).where(df["tm_targets"]>0)
    df["rush_share_week"]   = (df["rushing_attempts"]/df["tm_rush_att"]).where(df["tm_rush_att"]>0)
    agg = df.groupby(["season","player_display_name","team"], as_index=False).agg(
        targets=("targets","sum"),
        routes=("routes_run","sum") if "routes_run" in df.columns else ("receptions","sum"),
        rush_att=("rushing_attempts","sum"),
        target_share=("target_share_week","mean"),
        rush_share=("rush_share_week","mean"),
    ).rename(columns={"player_display_name":"player"})
    if not rosters.empty:
        pos = rosters[["player_name","position"]].drop_duplicates().rename(columns={"player_name":"player"})
        agg = agg.merge(pos, on="player", how="left")
    else:
        agg["position"] = ""
    team_routes = agg.groupby(["season","team"])["routes"].transform(lambda s: s.max() if s.max()>0 else 1)
    agg["route_share"] = (agg["routes"]/team_routes).where(team_routes>0)
    return agg[["season","player","team","position","targets","target_share","rush_att","rush_share","routes","route_share"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True)
    args = ap.parse_args()
    team_form = build_team_form(args.season)
    player_form = build_player_form(args.season)
    OUT.mkdir(parents=True, exist_ok=True)
    team_form.to_csv(OUT / "team_form.csv", index=False)
    player_form.to_csv(OUT / "player_form.csv", index=False)
    (REPO / "data").mkdir(parents=True, exist_ok=True)
    team_form.to_csv(REPO / "data" / "team_form.csv", index=False)
    player_form.to_csv(REPO / "data" / "player_form.csv", index=False)
    if team_form.empty or player_form.empty:
        raise SystemExit("FATAL: metrics empty — upstream fetch likely failed")
    print("✅ metrics built")
if __name__ == "__main__":
    main()
