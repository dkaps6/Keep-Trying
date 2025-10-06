# scripts/fetch_nfl_data.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def _import_pbp(season: int) -> pd.DataFrame:
    import nfl_data_py as nfl
    # nfl.import_pbp_data takes a list of seasons
    pbp = nfl.import_pbp_data([season])
    # Ensure lower/consistent dtypes
    for b in ["pass", "rush", "qb_dropback", "qb_hit", "sack", "pass_attempt", "qb_scramble"]:
        if b in pbp.columns:
            pbp[b] = pbp[b].astype("float").fillna(0.0)
    return pbp

def _is_neutral(p: pd.Series) -> pd.Series:
    """
    Rough neutrality filter:
      - periods 1-3
      - win prob between 0.20 and 0.80
      - not two-minute (game_seconds_remaining % 900 > 120)
      - normal downs
    """
    q_ok  = (p.get("qtr", pd.Series(index=p.index, dtype=float)) <= 3)
    wp    = p.get("wp", pd.Series(index=p.index, dtype=float)).astype(float)
    wp_ok = (wp >= 0.20) & (wp <= 0.80)
    gsr   = p.get("game_seconds_remaining", pd.Series(index=p.index, dtype=float)).astype(float)
    not_two_min = (gsr % 900.0) > 120.0
    down  = p.get("down", pd.Series(index=p.index, dtype=float)).astype(float)
    down_ok = down.isin([1,2,3])
    return q_ok & wp_ok & not_two_min & down_ok

def _sec_per_play_neutral(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Offense-neutral pace: median sec/play for each offensive team in neutral script.
    """
    need = ["game_id","play_id","posteam","game_seconds_remaining"]
    for c in need:
        if c not in pbp.columns:
            return pd.DataFrame(columns=["team","pace"])
    df = pbp.loc[_is_neutral(pbp), need].dropna(subset=["posteam"]).copy()
    df = df.sort_values(["game_id","play_id"])
    # time between consecutive plays by same offense within a game (approx)
    df["delta"] = -df.groupby(["game_id","posteam"])["game_seconds_remaining"].diff(-1)
    # bound outliers
    df["delta"] = df["delta"].clip(lower=8, upper=60)
    pace = df.groupby("posteam")["delta"].median().reset_index()
    pace.rename(columns={"posteam":"team","delta":"pace"}, inplace=True)
    return pace

def _def_epa_split(pbp: pd.DataFrame) -> pd.DataFrame:
    need = ["defteam","epa","pass","rush"]
    for c in need:
        if c not in pbp.columns:
            return pd.DataFrame(columns=["team","def_pass_epa","def_rush_epa"])
    df = pbp.loc[:, ["defteam","epa","pass","rush"]].copy()
    df["pass"] = df["pass"].fillna(0).astype(int)
    df["rush"] = df["rush"].fillna(0).astype(int)
    pass_df = df.loc[df["pass"]==1].groupby("defteam")["epa"].mean().reset_index().rename(columns={"defteam":"team","epa":"def_pass_epa"})
    rush_df = df.loc[df["rush"]==1].groupby("defteam")["epa"].mean().reset_index().rename(columns={"defteam":"team","epa":"def_rush_epa"})
    out = pass_df.merge(rush_df, on="team", how="outer")
    return out

def _pressure_and_sack_rates(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    pressure_rate ≈ (sacks + qb_hits) / dropbacks
    sack_rate     = sacks / dropbacks
    dropbacks: prefer qb_dropback; else pass_attempt + qb_scramble + sack
    """
    cols = ["defteam","qb_dropback","pass_attempt","qb_scramble","sack","qb_hit"]
    for c in cols:
        if c not in pbp.columns:
            pbp[c] = 0.0
    pbp = pbp.copy()
    drop = pbp["qb_dropback"]
    # Fallback dropback if qb_dropback missing/zeroed
    if drop.sum() == 0:
        drop = pbp["pass_attempt"].fillna(0) + pbp["qb_scramble"].fillna(0) + pbp["sack"].fillna(0)
    sacks = pbp["sack"].fillna(0)
    hits  = pbp["qb_hit"].fillna(0)
    g = pd.DataFrame({
        "team": pbp["defteam"],
        "dropbacks": drop,
        "sacks": sacks,
        "hits": hits
    }).groupby("team", as_index=False).sum(numeric_only=True)
    g["def_sack_rate"] = np.where(g["dropbacks"]>0, g["sacks"]/g["dropbacks"], np.nan)
    g["def_pressure_rate"] = np.where(g["dropbacks"]>0, (g["sacks"]+g["hits"])/g["dropbacks"], np.nan)
    return g[["team","def_pressure_rate","def_sack_rate"]]

def _air_yards_per_att_offense(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Offense AY/Att used to sanity-cap WR1 YPR vs scheme.
    """
    for c in ["posteam","air_yards","pass_attempt"]:
        if c not in pbp.columns:
            return pd.DataFrame(columns=["team","ay_per_att"])
    df = pbp.loc[pbp["pass_attempt"]==1, ["posteam","air_yards","pass_attempt"]].copy()
    # air_yards can be NaN on some plays; treat missing as 0 contrib to numerator
    df["air_yards"] = df["air_yards"].fillna(0.0)
    agg = df.groupby("posteam", as_index=False).agg(air_yards_sum=("air_yards","sum"),
                                                    pass_att=("pass_attempt","sum"))
    agg["ay_per_att"] = np.where(agg["pass_att"]>0, agg["air_yards_sum"]/agg["pass_att"], np.nan)
    agg.rename(columns={"posteam":"team"}, inplace=True)
    return agg[["team","ay_per_att"]]

def _neutral_pass_rate(pbp: pd.DataFrame) -> pd.DataFrame:
    mask = _is_neutral(pbp)
    cols = ["posteam","pass_attempt","rush"]
    for c in cols:
        if c not in pbp.columns:
            return pd.DataFrame(columns=["team","neutral_pass_rate"])
    df = pbp.loc[mask, ["posteam","pass_attempt","rush"]].copy()
    df["pass_attempt"] = df["pass_attempt"].fillna(0).astype(float)
    df["rush"] = df["rush"].fillna(0).astype(float)
    agg = df.groupby("posteam", as_index=False).sum(numeric_only=True)
    agg["neutral_pass_rate"] = np.where((agg["pass_attempt"]+agg["rush"])>0,
                                        agg["pass_attempt"]/(agg["pass_attempt"]+agg["rush"]),
                                        np.nan)
    agg.rename(columns={"posteam":"team"}, inplace=True)
    return agg[["team","neutral_pass_rate"]]

def build_team_form(season: int) -> pd.DataFrame:
    pbp = _import_pbp(season)

    pace  = _sec_per_play_neutral(pbp)
    epa   = _def_epa_split(pbp)
    rushp = _pressure_and_sack_rates(pbp)
    ay    = _air_yards_per_att_offense(pbp)
    npr   = _neutral_pass_rate(pbp)

    # merge
    out = (
        epa
        .merge(rushp, on="team", how="outer")
        .merge(pace,  on="team", how="outer")
        .merge(ay,    on="team", how="outer")
        .merge(npr,   on="team", how="outer")
    )

    # clean team codes if present as strings like 'NA'
    out["team"] = out["team"].astype(str)

    # Make it obvious what's missing (coverage/boxes can be filled by other scripts later)
    out["light_box_rate"] = np.nan
    out["heavy_box_rate"] = np.nan

    # Clip/guard some metrics
    if "pace" in out.columns:
        out["pace"] = out["pace"].clip(lower=18, upper=40)

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--write", default="metrics/team_form.csv")
    args = ap.parse_args()

    df = build_team_form(args.season)
    Path(args.write).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.write, index=False)
    print(f"[team_form] wrote {len(df)} rows to {args.write}")

if __name__ == "__main__":
    main()

