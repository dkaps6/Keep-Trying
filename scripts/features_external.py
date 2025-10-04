import pandas as pd
import numpy as np
import nfl_data_py as nfl

ROLL = 4

def build_external(season: int) -> dict:
    # schedules for joins
    sched = nfl.import_schedules([season]).rename(columns={
        "home_team": "home", "away_team": "away", "gameday": "game_date"
    })
    sched["game_date"] = pd.to_datetime(sched["game_date"], errors="coerce")

    # ids for mapping
    ids = nfl.import_ids()[["player_name","gsis_id","recent_team"]].drop_duplicates()

    # weekly player stats
    wk = nfl.import_weekly_data([season], downcast=True).merge(ids, on="player_name", how="left")

    # team pbp → offense/defense EPA/SR; rolling L4
    pbp = nfl.import_pbp_data([season-1, season], downcast=True)
    off = pbp.groupby(["posteam","week"], as_index=False).agg(off_epa=("epa","mean"), off_sr=("success","mean"))
    off = off.rename(columns={"posteam":"team"})
    de  = pbp.groupby(["defteam","week"], as_index=False).agg(def_epa=("epa","mean"), def_sr=("success","mean"))
    de  = de.rename(columns={"defteam":"team"})
    team_week = off.merge(de, on=["team","week"], how="left").sort_values(["team","week"])
    team_form = (team_week.groupby("team")[["off_epa","off_sr","def_epa","def_sr"]]
                 .rolling(ROLL, min_periods=1).mean().reset_index().drop(columns=["level_1"]))

    # player L4 form
    pweek = (wk.groupby(["gsis_id","team","week"], as_index=False)
               .agg(tgt=("targets","sum"), rec=("receptions","sum"), rec_yds=("receiving_yards","sum"),
                    ra=("rushing_attempts","sum"), ry=("rushing_yards","sum")))
    pform = (pweek.sort_values(["gsis_id","week"])
             .groupby("gsis_id")[["tgt","rec","rec_yds","ra","ry"]]
             .rolling(ROLL, min_periods=1).mean().reset_index().drop(columns=["level_1"])
             .rename(columns={"tgt":"tgt_l4","rec":"rec_l4","rec_yds":"rec_yds_l4","ra":"ra_l4","ry":"ry_l4"}))
    pform = pform.merge(pweek[["gsis_id","team","week"]], on=["gsis_id","week"], how="left")

    # injuries (latest snapshot)
    inj = nfl.import_injuries([season])
    inj_latest = (inj.sort_values("report_date")
                    .drop_duplicates(subset=["gsis_id"], keep="last")[["gsis_id","status","practice_status"]]
                    if not inj.empty else pd.DataFrame(columns=["gsis_id","status","practice_status"]))

    # depth charts (latest)
    depth = nfl.import_depth_charts([season])
    depth_latest = (depth.sort_values("updated")
                      .drop_duplicates(subset=["gsis_id"], keep="last")[
                          ["gsis_id","depth_team","depth_position","position_depth"]
                      ] if not depth.empty else pd.DataFrame(columns=["gsis_id","depth_team","depth_position","position_depth"]))

    return {"ids": ids, "sched": sched, "team_form": team_form, "player_form": pform,
            "inj": inj_latest, "depth": depth_latest}
