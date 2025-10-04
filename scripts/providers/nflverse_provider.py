from __future__ import annotations
import pandas as pd, numpy as np
import nfl_data_py as nfl
from typing import List
from .base import WeeklyProvider, coerce_weekly_schema, empty_weekly_df

class NFLVerseProvider(WeeklyProvider):
    name = "nflverse"

    def fetch_weekly(self, season: int) -> pd.DataFrame:
        try:
            weekly = nfl.import_weekly_data([season])
        except Exception as e:
            print(f"⚠️ nflverse weekly failed for {season}: {e}")
            return empty_weekly_df()
        if weekly is None or weekly.empty:
            return empty_weekly_df()
        weekly = weekly.sort_values(["player_id","week"]).copy()
        grp = weekly.groupby("player_id", as_index=False)

        def l4(s): return s.rolling(4, min_periods=1).sum()

        def get(col):
            return col if col in weekly.columns else None

        weekly["rec_l4"]      = grp[get("rec")]            ["rec"].transform(l4)            if get("rec") else 0
        weekly["rec_yds_l4"]  = grp[get("rec_yds")]        ["rec_yds"].transform(l4)        if get("rec_yds") else 0
        weekly["ra_l4"]       = grp[get("rush_att")]       ["rush_att"].transform(l4)       if get("rush_att") else 0
        weekly["ry_l4"]       = grp[get("rush_yds")]       ["rush_yds"].transform(l4)       if get("rush_yds") else 0
        weekly["pass_att_l4"] = grp[get("attempts")]       ["attempts"].transform(l4)       if get("attempts") else 0
        weekly["pass_yds_l4"] = grp[get("passing_yards")]  ["passing_yards"].transform(l4)  if get("passing_yards") else 0

        if "targets" in weekly.columns and "redzone_targets" in weekly.columns:
            tgt_l4 = grp["targets"].transform(l4)
            rz_l4  = grp["redzone_targets"].transform(l4)
            weekly["rz_tgt_share_l4"] = (rz_l4 / tgt_l4.replace(0, np.nan)).fillna(0.0)
        else:
            weekly["rz_tgt_share_l4"] = 0.15

        out = weekly.rename(columns={
            "player_id":"gsis_id",
            "player_name":"player_name",
            "team":"recent_team",
            "position":"position"
        })
        cols = ["gsis_id","week","player_name","recent_team","position",
                "rec_l4","rec_yds_l4","ra_l4","ry_l4","pass_att_l4","pass_yds_l4","rz_tgt_share_l4"]
        out = out[[c for c in cols if c in out.columns]].dropna(subset=["gsis_id"])
        return coerce_weekly_schema(out)
