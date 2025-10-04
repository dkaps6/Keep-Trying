# scripts/fetch_nfl_data.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, List
import nfl_data_py as nfl

def _parse_weeks(weeks: str) -> List[int]:
    if weeks.lower() == "all":
        return list(range(1, 19))
    if "-" in weeks:
        a, b = weeks.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(weeks)]

# ---------- TEAM FORM ----------
def build_team_form(season: int, weeks: str = "1-18") -> pd.DataFrame:
    wks = _parse_weeks(weeks)
    print(f"Downloading pbp for season={season}, weeks={wks} ...")
    pbp = nfl.import_pbp_data([season], downcast=True)
    pbp = pbp[pbp["week"].isin(wks)]

    # Offense/Defense team labels
    pbp = pbp.assign(
        offense = pbp["posteam"].fillna(""),
        defense = pbp["defteam"].fillna(""),
        pass_play = pbp["pass"].fillna(0).astype(int),
        rush_play = pbp["rush"].fillna(0).astype(int),
        db = (pbp["pass"] == 1).astype(int)
    )

    # Early-down neutral (proxy for PROE): Q1-3, score diff in [-7,7], not garbage time
    neutral = pbp[
        (pbp["qtr"].between(1,3)) &
        (pbp["score_differential"].fillna(0).between(-7,7)) &
        (pbp["down"].isin([1,2]))
    ].copy()

    # DEF pressure proxy: sacks / dropbacks (dropbacks approx = pass attempts + sacks)
    def_agg = pbp.groupby("defense").agg(
        def_pass_plays=("pass_play","sum"),
        def_dropbacks=("qb_dropback","sum"),
        def_sacks=("sack","sum"),
        def_pass_epa=("epa","mean")  # EPA allowed per play (mixed); tighten to pass snaps only below
    ).reset_index().rename(columns={"defense":"team"})

    # EPA against PASS only
    def_pass = pbp[pbp["pass_play"]==1].groupby("defense").agg(
        def_pass_epa=("epa","mean")
    ).reset_index().rename(columns={"defense":"team"})
    def_agg = def_agg.drop(columns=["def_pass_epa"], errors="ignore").merge(def_pass, on="team", how="left")

    def_agg["def_dropbacks"] = def_agg["def_dropbacks"].replace(0, np.nan)
    def_agg["pressure_rate"] = def_agg["def_sacks"] / def_agg["def_dropbacks"]

    # Neutral pass rate by offense (proxy for PROE)
    off_neutral = neutral.groupby("offense").agg(
        neutral_plays=("play_id","count"),
        neutral_pass=("pass_play","sum")
    ).reset_index().rename(columns={"offense":"team"})
    off_neutral["neutral_pass_rate"] = off_neutral["neutral_pass"] / off_neutral["neutral_plays"].replace(0,np.nan)

    # League means
    pr_mu = def_agg["pressure_rate"].mean(skipna=True)
    pr_sd = def_agg["pressure_rate"].std(ddof=0, skipna=True)
    epa_mu = def_agg["def_pass_epa"].mean(skipna=True)
    epa_sd = def_agg["def_pass_epa"].std(ddof=0, skipna=True)
    npr_mu = off_neutral["neutral_pass_rate"].mean(skipna=True)

    # Z-scores & PROE proxy
    def_agg["pressure_z"] = (def_agg["pressure_rate"] - pr_mu) / (pr_sd if pr_sd else 1.0)
    def_agg["pass_epa_z"]  = (def_agg["def_pass_epa"] - epa_mu) / (epa_sd if epa_sd else 1.0)

    tf = def_agg.merge(off_neutral[["team","neutral_pass_rate"]], on="team", how="left")
    tf["proe"] = tf["neutral_pass_rate"] - npr_mu

    # Placeholders (None) for coverage/box counts (safe in rules)
    tf["light_box_share"] = np.nan
    tf["heavy_box_share"] = np.nan
    tf["man_rate_z"] = np.nan
    tf["zone_rate_z"] = np.nan

    # Clean
    tf = tf[[
        "team","pressure_rate","pressure_z","def_pass_epa","pass_epa_z",
        "neutral_pass_rate","proe","light_box_share","heavy_box_share","man_rate_z","zone_rate_z"
    ]].sort_values("team")
    return tf

# ---------- PLAYER FORM ----------
def build_player_form(season: int, weeks: str = "1-18") -> pd.DataFrame:
    wks = _parse_weeks(weeks)
    weekly = nfl.import_weekly_data([season])
    weekly = weekly[weekly["week"].isin(wks)].copy()

    # Keep core fields (names vary by version; be defensive)
    cols = weekly.columns
    def get(c): return c if c in cols else None

    # Rolling last-4 by player
    weekly = weekly.sort_values(["player_id","week"])
    grp = weekly.groupby("player_id", as_index=False)

    def last4_sum(s): return s.rolling(4, min_periods=1).sum()

    weekly["rec_l4"]       = grp[get("rec")]        ["rec"].transform(last4_sum) if get("rec") else 0
    weekly["rec_yds_l4"]   = grp[get("rec_yds")]    ["rec_yds"].transform(last4_sum) if get("rec_yds") else 0
    weekly["ra_l4"]        = grp[get("rush_att")]   ["rush_att"].transform(last4_sum) if get("rush_att") else 0
    weekly["ry_l4"]        = grp[get("rush_yds")]   ["rush_yds"].transform(last4_sum) if get("rush_yds") else 0
    weekly["pass_att_l4"]  = grp[get("attempts")]   ["attempts"].transform(last4_sum) if get("attempts") else 0
    weekly["pass_yds_l4"]  = grp[get("passing_yards")]["passing_yards"].transform(last4_sum) if get("passing_yards") else 0

    # Red zone target share (approx): use rzs targets if present, else estimate from inside-20 targets
    if "targets" in cols:
        weekly["tgt_l4"] = grp["targets"].transform(last4_sum)
    else:
        weekly["tgt_l4"] = 0

    # crude RZ share: if there is a 'redzone_targets' column use it; else fallback to 15% baseline
    if "redzone_targets" in cols:
        weekly["rz_tgts_l4"] = grp["redzone_targets"].transform(last4_sum)
        weekly["rz_tgt_share_l4"] = weekly["rz_tgts_l4"] / weekly["tgt_l4"].replace(0,np.nan)
    else:
        weekly["rz_tgt_share_l4"] = 0.15

    out = weekly.rename(columns={
        "player_id":"gsis_id",
        "recent_team":"recent_team",
        "position":"position",
        "player_name":"player_name",
    })
    keep = ["gsis_id","week","player_name","recent_team","position",
            "rec_l4","rec_yds_l4","ra_l4","ry_l4","pass_att_l4","pass_yds_l4","rz_tgt_share_l4"]
    out = out[[c for c in keep if c in out.columns]].dropna(subset=["gsis_id"])
    return out

# ---------- ID MAP ----------
def build_id_map(season: int) -> pd.DataFrame:
    rosters = nfl.import_rosters([season])
    out = rosters.rename(columns={
        "gsis_id":"gsis_id",
        "player_name":"player_name",
        "team":"recent_team",
        "position":"position"
    })
    return out[["player_name","gsis_id","recent_team","position"]].dropna(subset=["gsis_id"])
