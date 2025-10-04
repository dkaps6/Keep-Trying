import pandas as pd
import numpy as np
import nfl_data_py as nfl

ROLL = 4

def build_external(season: int) -> dict:
    # schedules (optional for advanced joins)
    sched = nfl.import_schedules([season]).rename(columns={
        "home_team": "home", "away_team": "away", "gameday": "game_date"
    })
    sched["game_date"] = pd.to_datetime(sched["game_date"], errors="coerce")

    # ids for mapping
    ids = nfl.import_ids()[["player_name","gsis_id","recent_team","position"]].drop_duplicates()

    # weekly player stats
    wk = nfl.import_weekly_data([season], downcast=True).merge(ids, on="player_name", how="left")

    # team pbp → offense/defense EPA & SR, plus PASS/RUSH splits; rolling L4
    pbp = nfl.import_pbp_data([season-1, season], downcast=True)
    pbp = pbp[pbp["season_type"] == "REG"].copy()

    def _agg_team(p, group_cols, mask=None):
        q = p if mask is None else p[mask]
        return q.groupby(group_cols, as_index=False).agg(
            epa=("epa","mean"),
            sr=("success","mean"),
            plays=("play_id","count")
        )

    off_all = _agg_team(pbp, ["posteam","week"]).rename(columns={"posteam":"team","epa":"off_epa","sr":"off_sr","plays":"off_plays"})
    off_pass = _agg_team(pbp, ["posteam","week"], mask=pbp["pass"]==1).rename(columns={"posteam":"team","epa":"off_pass_epa","sr":"off_pass_sr","plays":"off_dropbacks"})
    off_rush = _agg_team(pbp, ["posteam","week"], mask=pbp["rush"]==1).rename(columns={"posteam":"team","epa":"off_rush_epa","sr":"off_rush_sr","plays":"off_rushes"})

    de_all = _agg_team(pbp, ["defteam","week"]).rename(columns={"defteam":"team","epa":"def_epa","sr":"def_sr","plays":"def_plays"})
    de_pass = _agg_team(pbp, ["defteam","week"], mask=pbp["pass"]==1).rename(columns={"defteam":"team","epa":"def_pass_epa","sr":"def_pass_sr","plays":"def_dropbacks"})
    de_rush = _agg_team(pbp, ["defteam","week"], mask=pbp["rush"]==1).rename(columns={"defteam":"team","epa":"def_rush_epa","sr":"def_rush_sr","plays":"def_rushes"})

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

    team_form = team_week[["team","week","off_epa","off_sr","off_pass_epa","off_rush_epa","def_epa","def_sr","def_pass_epa","def_rush_epa","off_plays","def_plays","off_dropbacks","off_rushes"]]\
        .pipe(lambda d: d.merge(_roll(d, ["off_epa","off_sr","off_pass_epa","off_rush_epa","def_epa","def_sr","def_pass_epa","def_rush_epa","off_plays","def_plays","off_dropbacks","off_rushes"]),
                                left_index=True, right_index=True, suffixes=("","_l4")))

    # player L4 form
    pweek = (wk.groupby(["gsis_id","position","team","week"], as_index=False)
               .agg(tgt=("targets","sum"), rec=("receptions","sum"), rec_yds=("receiving_yards","sum"),
                    ra=("rushing_attempts","sum"), ry=("rushing_yards","sum"),
                    pass_att=("attempts","sum"), pass_yds=("passing_yards","sum")))
    pform = (pweek.sort_values(["gsis_id","week"])
             .groupby("gsis_id")[["tgt","rec","rec_yds","ra","ry","pass_att","pass_yds"]]
             .rolling(ROLL, min_periods=1).mean()
             .reset_index().drop(columns=["level_1"])
             .rename(columns={
                 "tgt":"tgt_l4","rec":"rec_l4","rec_yds":"rec_yds_l4",
                 "ra":"ra_l4","ry":"ry_l4","pass_att":"pass_att_l4","pass_yds":"pass_yds_l4"}))
    pform = pform.merge(pweek[["gsis_id","position","team","week"]], on=["gsis_id","week"], how="left")

    # injuries (latest snapshot)
    inj = nfl.import_injuries([season])
    inj_latest = (inj.sort_values("report_date")
                    .drop_duplicates(subset=["gsis_id"], keep="last")[["gsis_id","status","practice_status"]]
                  if not inj.empty else pd.DataFrame(columns=["gsis_id","status","practice_status"]))

    # depth charts (latest)
    depth = nfl.import_depth_charts([season])
    depth_latest = (depth.sort_values("updated")
                      .drop_duplicates(subset=["gsis_id"], keep="last")[["gsis_id","depth_team","depth_position","position_depth"]]
                    if not depth.empty else pd.DataFrame(columns=["gsis_id","depth_team","depth_position","position_depth"]))

    return {"ids": ids, "sched": sched, "team_form": team_form, "player_form": pform,
            "inj": inj_latest, "depth": depth_latest}

