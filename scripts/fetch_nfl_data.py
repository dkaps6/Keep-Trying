# scripts/fetch_nfl_data.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def _import_pbp(season: int) -> pd.DataFrame:
    import nfl_data_py as nfl
    pbp = nfl.import_pbp_data([season])
    # normalize types we use
    for b in ["pass", "rush", "qb_dropback", "qb_hit", "sack", "pass_attempt", "qb_scramble"]:
        if b in pbp.columns:
            pbp[b] = pbp[b].astype(float).fillna(0.0)
    return pbp

def _neutral_mask(df: pd.DataFrame) -> pd.Series:
    q_ok = (df.get("qtr", pd.Series([], dtype=float)) <= 3)
    wp = df.get("wp", pd.Series([], dtype=float)).astype(float)
    wp_ok = (wp >= 0.20) & (wp <= 0.80)
    gsr = df.get("game_seconds_remaining", pd.Series([], dtype=float)).astype(float)
    not_two_min = (gsr % 900.0) > 120.0
    down = df.get("down", pd.Series([], dtype=float)).astype(float)
    down_ok = down.isin([1, 2, 3])
    return q_ok & wp_ok & not_two_min & down_ok

def _sec_per_play_neutral(pbp: pd.DataFrame) -> pd.DataFrame:
    need = ["game_id", "play_id", "posteam", "game_seconds_remaining"]
    if not set(need).issubset(pbp.columns):
        return pd.DataFrame(columns=["team", "pace"])
    df = pbp.loc[_neutral_mask(pbp), need].dropna(subset=["posteam"]).copy()
    df = df.sort_values(["game_id", "play_id"])
    df["delta"] = -df.groupby(["game_id", "posteam"])["game_seconds_remaining"].diff(-1)
    df["delta"] = df["delta"].clip(lower=8, upper=60)
    pace = df.groupby("posteam")["delta"].median().reset_index()
    pace.rename(columns={"posteam": "team", "delta": "pace"}, inplace=True)
    return pace

def _def_epa_split(pbp: pd.DataFrame) -> pd.DataFrame:
    if not {"defteam", "epa", "pass", "rush"}.issubset(pbp.columns):
        return pd.DataFrame(columns=["team", "def_pass_epa", "def_rush_epa"])
    df = pbp[["defteam", "epa", "pass", "rush"]].copy()
    df["pass"] = df["pass"].fillna(0).astype(int)
    df["rush"] = df["rush"].fillna(0).astype(int)
    p = df.loc[df["pass"] == 1].groupby("defteam")["epa"].mean().rename("def_pass_epa")
    r = df.loc[df["rush"] == 1].groupby("defteam")["epa"].mean().rename("def_rush_epa")
    out = pd.concat([p, r], axis=1).reset_index().rename(columns={"defteam": "team"})
    return out

def _pressure_and_sack_rates(pbp: pd.DataFrame) -> pd.DataFrame:
    cols = ["defteam", "qb_dropback", "pass_attempt", "qb_scramble", "sack", "qb_hit"]
    for c in cols:
        if c not in pbp.columns:
            pbp[c] = 0.0
    drop = pbp["qb_dropback"]
    if drop.sum() == 0:
        drop = pbp["pass_attempt"].fillna(0) + pbp["qb_scramble"].fillna(0) + pbp["sack"].fillna(0)
    sacks = pbp["sack"].fillna(0)
    hits = pbp["qb_hit"].fillna(0)
    g = pd.DataFrame({
        "team": pbp["defteam"],
        "dropbacks": drop,
        "sacks": sacks,
        "hits": hits
    }).groupby("team", as_index=False).sum(numeric_only=True)
    g["def_sack_rate"] = np.where(g["dropbacks"] > 0, g["sacks"] / g["dropbacks"], np.nan)
    g["def_pressure_rate"] = np.where(g["dropbacks"] > 0, (g["sacks"] + g["hits"]) / g["dropbacks"], np.nan)
    return g[["team", "def_pressure_rate", "def_sack_rate"]]

def _air_yards_per_att_offense(pbp: pd.DataFrame) -> pd.DataFrame:
    cols = ["posteam", "air_yards", "pass_attempt"]
    if not set(cols).issubset(pbp.columns):
        return pd.DataFrame(columns=["team", "ay_per_att"])
    df = pbp.loc[pbp["pass_attempt"] == 1, cols].copy()
    df["air_yards"] = df["air_yards"].fillna(0.0)
    agg = df.groupby("posteam", as_index=False).agg(air_yards_sum=("air_yards", "sum"),
                                                    pass_att=("pass_attempt", "sum"))
    agg["ay_per_att"] = np.where(agg["pass_att"] > 0, agg["air_yards_sum"] / agg["pass_att"], np.nan)
    agg.rename(columns={"posteam": "team"}, inplace=True)
    return agg[["team", "ay_per_att"]]

def _neutral_pass_rate_and_proe(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    PROE ≈ (team neutral pass rate) - (expected neutral pass rate).
    Expected rate is computed by applying league baseline pass rates per
    (down, ytg bucket, yardline bucket) to each team's neutral plays.
    """
    mask = _neutral_mask(pbp)
    cols = ["posteam", "down", "ydstogo", "yardline_100", "pass_attempt", "rush"]
    if not set(cols).issubset(pbp.columns):
        return pd.DataFrame(columns=["team", "neutral_pass_rate", "proe"])

    df = pbp.loc[mask, cols].dropna(subset=["posteam", "down", "ydstogo", "yardline_100"]).copy()
    df["pass_attempt"] = df["pass_attempt"].fillna(0).astype(int)
    df["rush"] = df["rush"].fillna(0).astype(int)

    # bucketize context
    df["ytg_bin"] = pd.cut(df["ydstogo"], [0, 2, 5, 10, 99], labels=["0-2", "3-5", "6-10", "11+"])
    df["yl_bin"]  = pd.cut(df["yardline_100"], [0, 20, 50, 80, 100], labels=["RZ", "Own-50", "Opp-50", "GL"])

    # league baseline pass probability by bucket
    base = (df.groupby(["down", "ytg_bin", "yl_bin"], dropna=False)["pass_attempt"]
              .mean().rename("p_pass_base").reset_index())

    # merge baseline into each neutral play
    df = df.merge(base, on=["down", "ytg_bin", "yl_bin"], how="left")

    # per-team neutral pass rate
    team_agg = (df.groupby("posteam", as_index=False)
                  .agg(neutral_pass_rate=("pass_attempt", "mean"),
                       exp_pass_rate=("p_pass_base", "mean")))

    team_agg["proe"] = team_agg["neutral_pass_rate"] - team_agg["exp_pass_rate"]
    team_agg.rename(columns={"posteam": "team"}, inplace=True)
    return team_agg[["team", "neutral_pass_rate", "proe"]]

def build_team_form(season: int) -> pd.DataFrame:
    pbp = _import_pbp(season)

    pace  = _sec_per_play_neutral(pbp)
    epa   = _def_epa_split(pbp)
    rushp = _pressure_and_sack_rates(pbp)
    ay    = _air_yards_per_att_offense(pbp)
    nprp  = _neutral_pass_rate_and_proe(pbp)

    out = (epa
           .merge(rushp, on="team", how="outer")
           .merge(pace,  on="team", how="outer")
           .merge(ay,    on="team", how="outer")
           .merge(nprp,  on="team", how="outer"))

    out["team"] = out["team"].astype(str)
    # placeholders (optional fields referenced elsewhere)
    out["light_box_rate"] = np.nan
    out["heavy_box_rate"] = np.nan

    # guardrails
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
