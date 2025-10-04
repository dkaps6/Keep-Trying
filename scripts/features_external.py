# scripts/features_external.py
import pandas as pd
import numpy as np
import nfl_data_py as nfl

ROLL = 4

def _pick(colnames, *candidates):
    """Return the first candidate that exists in colnames, else None."""
    for c in candidates:
        if c in colnames:
            return c
    return None

def _safe_ids() -> pd.DataFrame:
    """Load ids and standardize to columns: player_name, gsis_id, recent_team, position."""
    ids_raw = nfl.import_ids()

    cols = set(ids_raw.columns)

    name_c = _pick(cols, "player_name", "display_name", "full_name", "name")
    gsis_c = _pick(cols, "gsis_id", "gsis")
    team_c = _pick(cols, "recent_team", "team", "recentTeam")
    pos_c  = _pick(cols, "position", "pos")

    # Build a minimal, standardized frame; fill missing with NA
    out = pd.DataFrame()
    out["player_name"] = ids_raw[name_c] if name_c else pd.Series(dtype=str)
    out["gsis_id"]     = ids_raw[gsis_c] if gsis_c else pd.Series(dtype=str)
    out["recent_team"] = ids_raw[team_c] if team_c else pd.Series(dtype=str)
    out["position"]    = ids_raw[pos_c]  if pos_c  else pd.Series(dtype=str)

    return out.dropna(subset=["gsis_id"]).drop_duplicates()

def build_external(season: int) -> dict:
    """
    Returns a dict with:
      - ids (player_name, gsis_id, recent_team, position)
      - sched (schedule w/ home, away, game_date)
      - team_form (rolling L4 team EPA/SR + volume splits)
      - player_form (rolling L4 player stats)
      - inj (latest injuries snapshot)
      - depth (latest depth chart snapshot)
    All joins are optional—we keep this robust so the pipeline always runs.
    """
    # ----- IDs (robust to schema changes) -----
    ids = _safe_ids()

    # ----- Schedules -----
    try:
        sched = nfl.import_schedules([season]).rename(columns={
            "home_team": "home", "away_team": "away", "gameday": "game_date"
        })
    except Exception:
        sched = pd.DataFrame(columns=["home","away","game_date"])
    if "game_date" in sched.columns:
        sched["game_date"] = pd.to_datetime(sched["game_date"], errors="coerce")

    # ----- Weekly player data (for L4 form) -----
    try:
        wk = nfl.import_weekly_data([season], downcast=True)
    except Exception:
        wk = pd.DataFrame()

    if not wk.empty and "player_name" in wk.columns:
        wk = wk.merge(ids, on="player_name", how="left")
    else:
        # ensure keys exist so downstream code doesn't crash
        for c in ["gsis_id","team","week","targets","receptions","receiving_yards",
                  "rushing_attempts","rushing_yards","attempts","passing_yards"]:
            if c not in wk.columns:
                wk[c] = np.nan

    # team play-by-play across (season-1, season) for rolling form
    try:
        pbp = nfl.import_pbp_data([season-1, season], downcast=True)
    except Exception:
        pbp = pd.DataFrame()

    if not pbp.empty:
        # Some versions store season type in a different column; filter only if present
        if "season_type" in pbp.columns:
            pbp = pbp[pbp["season_type"].eq("REG")].copy()

        def _agg_team(p, group_cols, mask=None):
            q = p if mask is None else p[mask]
            return q.groupby(group_cols, as_index=False).agg(
                epa=("epa","mean"),
                sr=("success","mean"),
                plays=("play_id","count")
            )

        off_all = _agg_team(pbp, ["posteam","week"]).rename(columns={"posteam":"team","epa":"off_epa","sr":"off_sr","plays":"off_plays"})
        off_pass = _agg_team(pbp, ["posteam","week"], mask=(pbp.get("pass") == 1)).rename(columns={"posteam":"team","epa":"off_pass_epa","sr":"off_pass_sr","plays":"off_dropbacks"})
        off_rush = _agg_team(pbp, ["posteam","week"], mask=(pbp.get("rush") == 1)).rename(columns={"posteam":"team","epa":"off_rush_epa","sr":"off_rush_sr","plays":"off_rushes"})

        de_all = _agg_team(pbp, ["defteam","week"]).rename(columns={"defteam":"team","epa":"def_epa","sr":"def_sr","plays":"def_plays"})
        de_pass = _agg_team(pbp, ["defteam","week"], mask=(pbp.get("pass") == 1)).rename(columns={"defteam":"team","epa":"def_pass_epa","sr":"def_pass_sr","plays":"def_dropbacks"})
        de_rush = _agg_team(pbp, ["defteam","week"], mask=(pbp.get("rush") == 1)).rename(columns={"defteam":"team","epa":"def_rush_epa","sr":"def_rush_sr","plays":"def_rushes"})

        team_week = off_all.merge(off_pass, on=["team","week"], how="left") \
                           .merge(off_rush, on=["team","week"], how="left") \
                           .merge(de_all, on=["team","week"], how="left") \
                           .merge(de_pass, on=["team","week"], how="left") \
                           .merge(de_rush, on=["team","week"], how="left") \
                           .sort_values(["team","week"])

        def _roll(df, cols):
            return (df.groupby("team")[cols]
                      .rolling(ROLL, min_periods=1).mean()
                      .reset_index().drop(columns=["level_1"]))

        base_cols = ["off_epa","off_sr","off_pass_epa","off_rush_epa",
                     "def_epa","def_sr","def_pass_epa","def_rush_epa",
                     "off_plays","def_plays","off_dropbacks","off_rushes"]

        # ensure all present
        for c in base_cols:
            if c not in team_week.columns:
                team_week[c] = np.nan

        team_form = team_week[["team","week"] + base_cols]
        rolled = _roll(team_form, base_cols)
        # merge back to include *_l4 columns (suffix handled by caller)
        team_form = team_form.merge(
            rolled.add_suffix("_l4").rename(columns={"team_l4":"team","week_l4":"week"}),
            left_index=True, right_index=True, how="left"
        )
    else:
        team_form = pd.DataFrame(columns=["team","week"])

    # ----- Player L4 form -----
    if not wk.empty:
        pweek = (wk.groupby(["gsis_id","team","week"], as_index=False)
                   .agg(tgt=("targets","sum"),
                        rec=("receptions","sum"),
                        rec_yds=("receiving_yards","sum"),
                        ra=("rushing_attempts","sum"),
                        ry=("rushing_yards","sum"),
                        pass_att=("attempts","sum"),
                        pass_yds=("passing_yards","sum")))
        pform = (pweek.sort_values(["gsis_id","week"])
                 .groupby("gsis_id")[["tgt","rec","rec_yds","ra","ry","pass_att","pass_yds"]]
                 .rolling(ROLL, min_periods=1).mean()
                 .reset_index().drop(columns=["level_1"])
                 .rename(columns={
                     "tgt":"tgt_l4","rec":"rec_l4","rec_yds":"rec_yds_l4",
                     "ra":"ra_l4","ry":"ry_l4","pass_att":"pass_att_l4","pass_yds":"pass_yds_l4"}))
        pform = pform.merge(pweek[["gsis_id","team","week"]], on=["gsis_id","week"], how="left")
    else:
        pform = pd.DataFrame(columns=["gsis_id","team","week","tgt_l4","rec_l4","rec_yds_l4","ra_l4","ry_l4","pass_att_l4","pass_yds_l4"])

    # ----- Injuries / Depth (best-effort) -----
    try:
        inj = nfl.import_injuries([season])
        inj_latest = (inj.sort_values("report_date")
                        .drop_duplicates(subset=["gsis_id"], keep="last")[["gsis_id","status","practice_status"]])
    except Exception:
        inj_latest = pd.DataFrame(columns=["gsis_id","status","practice_status"])

    try:
        depth = nfl.import_depth_charts([season])
        depth_latest = (depth.sort_values("updated")
                          .drop_duplicates(subset=["gsis_id"], keep="last")[["gsis_id","depth_team","depth_position","position_depth"]])
    except Exception:
        depth_latest = pd.DataFrame(columns=["gsis_id","depth_team","depth_position","position_depth"])

    return {
        "ids": ids,
        "sched": sched,
        "team_form": team_form,
        "player_form": pform,
        "inj": inj_latest,
        "depth": depth_latest,
    }
