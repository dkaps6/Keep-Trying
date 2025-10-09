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
    if not np.isfinite(sd) or sd == 0: return pd.Series(np.zeros(len(a)), index=s.index)
    return (a - mu) / sd

def build_team_form(season):
    pbp = _read_parquet(DATA / f"pbp_{season}.parquet")
    if pbp.empty:
        return pd.DataFrame(columns=["season","defense_team","def_pressure_rate_z","def_sack_rate_z","def_pass_epa_z","def_rush_epa_z","pace_z","ay_per_att_z"])
    pbp["is_pass"] = (pbp.get("pass", 0) == 1) | (pbp.get("pass", False) == True)
    pbp["is_rush"] = (pbp.get("rush", 0) == 1) | (pbp.get("rush", False) == True)
    pbp["is_db"] = pbp["is_pass"]
    pbp["prs_event"] = (pbp.get("sack", 0).fillna(0).astype(int) > 0) | (pbp.get("qb_hit", 0).fillna(0).astype(int) > 0)
    grp = pbp.groupby(["season","defteam"], as_index=False)
    def avg(g, maskcol): 
        m = g.loc[g[maskcol], "epa"]
        return float(m.mean()) if len(m)>0 else np.nan
    pass_epa = grp.apply(lambda g: avg(g, "is_pass")).reset_index(name="def_pass_epa")
    rush_epa = grp.apply(lambda g: avg(g, "is_rush")).reset_index(name="def_rush_epa")
    base = grp.agg(dropbacks=("is_db","sum"), prs_events=("prs_event","sum"), sacks=("sack","sum")).merge(pass_epa, on=["season","defteam"]).merge(rush_epa, on=["season","defteam"])
    base["def_pressure_rate"] = (base["prs_events"]/base["dropbacks"]).where(base["dropbacks"]>0)
    base["def_sack_rate"] = (base["sacks"]/base["dropbacks"]).where(base["dropbacks"]>0)
    base = base.rename(columns={"defteam":"defense_team"})
    for c in ["def_pressure_rate","def_sack_rate","def_pass_epa","def_rush_epa"]:
        base[c+"_z"] = zscore(base[c])
    base["pace_z"] = 0.0; base["ay_per_att_z"]=0.0  # placeholders if not available
    keep = ["season","defense_team","def_pressure_rate_z","def_sack_rate_z","def_sack_rate_z","def_pass_epa_z","def_rush_epa_z","pace_z","ay_per_att_z"]
    # fix duplicate keep entry
    keep = ["season","defense_team","def_pressure_rate_z","def_sack_rate_z","def_pass_epa_z","def_rush_epa_z","pace_z","ay_per_att_z"]
    return base[keep]

def build_player_form(season):
    players = _read_csv(DATA / "player_stats_week.csv")
    rosters = _read_csv(DATA / "rosters.csv")
    if players.empty:
        return pd.DataFrame(columns=["season","player","team","position","targets","target_share","rush_att","rush_share","routes","route_share"])
    if "targets" not in players.columns and "target" in players.columns: players = players.rename(columns={"target":"targets"})
    if "rushing_attempts" not in players.columns and "rush_attempts" in players.columns: players = players.rename(columns={"rush_attempts":"rushing_attempts"})
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
    team = build_team_form(args.season)
    player = build_player_form(args.season)
    OUT.mkdir(parents=True, exist_ok=True)
    team.to_csv(OUT/"team_form.csv", index=False)
    player.to_csv(OUT/"player_form.csv", index=False)
    (REPO/"data").mkdir(parents=True, exist_ok=True)
    team.to_csv(REPO/"data"/"team_form.csv", index=False)
    player.to_csv(REPO/"data"/"player_form.csv", index=False)
    if team.empty or player.empty:
        raise SystemExit("FATAL: metrics empty — upstream fetch likely failed")
    print("✅ metrics built")
if __name__ == "__main__":
    main()
