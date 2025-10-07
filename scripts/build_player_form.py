# scripts/build_player_form.py
# Builds data/player_form.csv from historical + current PBP:
# - target_share (player targets / team attempts)
# - rush_share   (player rushes / team rushes)
# - redzone_tgt_share / redzone_carry_share (inside 20)
# - yprr proxy   (receiving yards / team dropbacks)  [true routes unavailable; proxy still informative]
# - ypc          (rushing)
# - ypt          (receiving)
# - qb_ypa       (passing)
#
# Usage:
#   python -m scripts.build_player_form --season 2025 --history 2019-2024 --write data/player_form.csv

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

try:
    import nfl_data_py as nfl
    HAS_NFL_DATA_PY = True
except Exception:
    HAS_NFL_DATA_PY = False

from .fetch_nfl_data import load_pbp  # reuse the loader and mirrors

def _is_pass(df: pd.DataFrame) -> pd.Series:
    if "pass" in df.columns:
        return df["pass"] == 1
    return df.get("play_type", "").astype(str).str.contains("pass", case=False, na=False)

def _is_rush(df: pd.DataFrame) -> pd.Series:
    if "rush" in df.columns:
        return df["rush"] == 1
    return df.get("play_type", "").astype(str).str.contains("rush", case=False, na=False)

def build_player_form(pbp: pd.DataFrame, season: int) -> pd.DataFrame:
    if pbp.empty:
        return pd.DataFrame(columns=[
            "player","team","position",
            "target_share","rush_share","rz_tgt_share","rz_carry_share",
            "yprr_proxy","ypc","ypt","qb_ypa"
        ])

    cur = pbp[pbp["season"] == season].copy()

    # --- receiving (targets, yards) ---
    # nflfastR schema: 'passer_player_name', 'receiver_player_name', 'rusher_player_name'
    cur["is_pass"] = _is_pass(cur)
    cur["is_rush"] = _is_rush(cur)
    cur["is_rz"] = (cur.get("yardline_100", 100) <= 20)

    rec = (cur[cur["is_pass"]]
           .groupby(["posteam","receiver_player_name"], dropna=True)
           .agg(
               targets=("receiver_player_name","count"),
               rec_yards=("yards_gained","sum"),
               rz_targets=("is_rz","sum")
           )
           .reset_index()
           .rename(columns={"posteam":"team","receiver_player_name":"player"}))

    team_pass = (cur[cur["is_pass"]]
                 .groupby("posteam").size()
                 .rename("team_attempts")
                 ).reset_index().rename(columns={"posteam":"team"})

    rec = rec.merge(team_pass, on="team", how="left")
    rec["target_share"] = rec["targets"] / rec["team_attempts"].clip(lower=1)
    rec["ypt"] = rec["rec_yards"] / rec["targets"].clip(lower=1)
    rec["rz_tgt_share"] = rec["rz_targets"] / rec["targets"].clip(lower=1)

    # yprr proxy: receiving yards / team dropbacks (since true routes unavailable)
    dropbacks = (cur[cur["is_pass"]]
                 .groupby("posteam").size()
                 .rename("team_dropbacks")
                 ).reset_index().rename(columns={"posteam":"team"})
    rec = rec.merge(dropbacks, on="team", how="left")
    rec["yprr_proxy"] = rec["rec_yards"] / rec["team_dropbacks"].clip(lower=1)

    # --- rushing (carries, yards) ---
    rush = (cur[cur["is_rush"]]
            .groupby(["posteam","rusher_player_name"], dropna=True)
            .agg(
                carries=("rusher_player_name","count"),
                rush_yards=("yards_gained","sum"),
                rz_carries=("is_rz","sum")
            )
            .reset_index()
            .rename(columns={"posteam":"team","rusher_player_name":"player"}))

    team_rush = (cur[cur["is_rush"]]
                 .groupby("posteam").size()
                 .rename("team_rushes")
                 ).reset_index().rename(columns={"posteam":"team"})
    rush = rush.merge(team_rush, on="team", how="left")
    rush["rush_share"] = rush["carries"] / rush["team_rushes"].clip(lower=1)
    rush["ypc"] = rush["rush_yards"] / rush["carries"].clip(lower=1)
    rush["rz_carry_share"] = rush["rz_carries"] / rush["carries"].clip(lower=1)

    # --- QB YPA ---
    qb = (cur[cur["is_pass"]]
          .groupby(["posteam","passer_player_name"], dropna=True)
          .agg(pass_yards=("yards_gained","sum"),
               attempts=("pass","sum"))
          .reset_index()
          .rename(columns={"posteam":"team","passer_player_name":"player"}))
    qb["qb_ypa"] = qb["pass_yards"] / qb["attempts"].clip(lower=1)

    # Combine (outer-join; players may appear in only one role)
    out = pd.merge(rec[["player","team","target_share","rz_tgt_share","yprr_proxy","ypt"]],
                   rush[["player","team","rush_share","rz_carry_share","ypc"]],
                   on=["player","team"], how="outer")
    out = pd.merge(out, qb[["player","team","qb_ypa"]], on=["player","team"], how="outer")

    # Position inference (light)
    out["position"] = out["position"] = out.apply(
        lambda r: ("QB" if pd.notna(r.get("qb_ypa")) else
                   ("RB" if pd.notna(r.get("rush_share")) else
                    ("WR/TE" if pd.notna(r.get("target_share")) else ""))), axis=1)

    # NaNs -> 0 for shares/efficiency
    for col in ["target_share","rush_share","rz_tgt_share","rz_carry_share","yprr_proxy","ypc","ypt","qb_ypa"]:
        out[col] = out[col].fillna(0.0)

    out = out.sort_values(["team","player"]).reset_index(drop=True)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--history", default="2019-2024")
    ap.add_argument("--write", default="data/player_form.csv")
    args = ap.parse_args()

    Path("data").mkdir(parents=True, exist_ok=True)

    # Parse history range (not used directly here, but we reuse loader’s signature)
    hist = []
    if args.history:
        if "-" in args.history:
            a,b = args.history.split("-", 1)
            hist = list(range(int(a), int(b)+1))
        else:
            hist = [int(x) for x in args.history.split(",") if x.strip().isdigit()]

    pbp = load_pbp(hist, args.season)  # same loader as team_form
    df = build_player_form(pbp, args.season)
    df.to_csv(args.write, index=False)
    print(f"[build_player_form] ✅ wrote {len(df)} rows → {args.write}")

if __name__ == "__main__":
    main()
