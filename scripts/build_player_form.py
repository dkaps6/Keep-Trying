# scripts/build_player_form.py
from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd

try:
    import nfl_data_py as nfl
    HAS_NFL_DATA_PY = True
except Exception:
    HAS_NFL_DATA_PY = False

from .fetch_nfl_data import load_pbp  # reuse your loader (for history), might be empty for 2025

def _is_pass(df: pd.DataFrame) -> pd.Series:
    if "pass" in df.columns:
        return df["pass"] == 1
    return df.get("play_type","").astype(str).str.contains("pass", case=False, na=False)

def _is_rush(df: pd.DataFrame) -> pd.Series:
    if "rush" in df.columns:
        return df["rush"] == 1
    return df.get("play_type","").astype(str).str.contains("rush", case=False, na=False)

def _from_pbp(pbp: pd.DataFrame, season: int) -> pd.DataFrame:
    cur = pbp[pbp["season"] == season].copy()
    if cur.empty:
        return pd.DataFrame()
    cur["is_pass"] = _is_pass(cur)
    cur["is_rush"] = _is_rush(cur)
    cur["is_rz"]   = (cur.get("yardline_100", 100) <= 20)

    rec = (cur[cur["is_pass"]]
           .groupby(["posteam","receiver_player_name"], dropna=True)
           .agg(targets=("receiver_player_name","count"),
                rec_yards=("yards_gained","sum"),
                rz_targets=("is_rz","sum"))
           .reset_index().rename(columns={"posteam":"team","receiver_player_name":"player"}))
    team_attempts = (cur[cur["is_pass"]].groupby("posteam").size()
                     .rename("team_attempts")).reset_index().rename(columns={"posteam":"team"})
    rec = rec.merge(team_attempts, on="team", how="left")
    rec["target_share"] = rec["targets"] / rec["team_attempts"].clip(lower=1)
    rec["ypt"] = rec["rec_yards"] / rec["targets"].clip(lower=1)
    dropbacks = team_attempts.rename(columns={"team_attempts":"team_dropbacks"})
    rec = rec.merge(dropbacks, on="team", how="left")
    rec["yprr_proxy"] = rec["rec_yards"] / rec["team_dropbacks"].clip(lower=1)
    rec["rz_tgt_share"] = rec["rz_targets"] / rec["targets"].clip(lower=1)

    rush = (cur[cur["is_rush"]]
            .groupby(["posteam","rusher_player_name"], dropna=True)
            .agg(carries=("rusher_player_name","count"),
                 rush_yards=("yards_gained","sum"),
                 rz_carries=("is_rz","sum"))
            .reset_index().rename(columns={"posteam":"team","rusher_player_name":"player"}))
    team_rushes = (cur[cur["is_rush"]].groupby("posteam").size()
                   .rename("team_rushes")).reset_index().rename(columns={"posteam":"team"})
    rush = rush.merge(team_rushes, on="team", how="left")
    rush["rush_share"] = rush["carries"] / rush["team_rushes"].clip(lower=1)
    rush["ypc"] = rush["rush_yards"] / rush["carries"].clip(lower=1)
    rush["rz_carry_share"] = rush["rz_carries"] / rush["carries"].clip(lower=1)

    qb = (cur[cur["is_pass"]]
          .groupby(["posteam","passer_player_name"], dropna=True)
          .agg(pass_yards=("yards_gained","sum"),
               attempts=("is_pass","sum"))
          .reset_index().rename(columns={"posteam":"team","passer_player_name":"player"}))
    qb["qb_ypa"] = qb["pass_yards"] / qb["attempts"].clip(lower=1)

    out = pd.merge(rec[["player","team","target_share","rz_tgt_share","yprr_proxy","ypt"]],
                   rush[["player","team","rush_share","rz_carry_share","ypc"]],
                   on=["player","team"], how="outer")
    out = pd.merge(out, qb[["player","team","qb_ypa"]], on=["player","team"], how="outer")
    out["position"] = out.apply(lambda r:
        "QB" if pd.notna(r.get("qb_ypa"))
        else ("RB" if pd.notna(r.get("rush_share"))
              else ("WR/TE" if pd.notna(r.get("target_share")) else "")), axis=1)
    for c in ["target_share","rush_share","rz_tgt_share","rz_carry_share","yprr_proxy","ypc","ypt","qb_ypa"]:
        out[c] = out[c].fillna(0.0)
    return out.sort_values(["team","player"]).reset_index(drop=True)

def _from_weekly(season: int) -> pd.DataFrame:
    if not HAS_NFL_DATA_PY:
        return pd.DataFrame()
    try:
        print(f"[build_player_form] weekly fallback: nfl_data_py.import_weekly_data({season}) ...")
        wk = nfl.import_weekly_data([season])
    except Exception as e:
        print(f"[build_player_form] weekly fallback failed: {type(e).__name__}: {e}")
        return pd.DataFrame()
    if wk is None or wk.empty:
        return pd.DataFrame()

    # Per-player aggregation across weeks
    # For receivers: targets/rec_yards → shares need team totals; compute team totals first
    team_tot = (wk.groupby(["team"])
                  .agg(team_targets=("targets","sum"),
                       team_rushes=("carries","sum"),
                       team_attempts=("attempts","sum"))
                  .reset_index())
    rec = (wk.groupby(["team","player"])
             .agg(targets=("targets","sum"),
                  receptions=("receptions","sum"),
                  rec_yards=("receiving_yards","sum") if "receiving_yards" in wk.columns else ("rec_yards","sum"),
                  carries=("carries","sum"),
                  rush_yards=("rushing_yards","sum") if "rushing_yards" in wk.columns else ("rush_yards","sum"),
                  pass_yards=("passing_yards","sum") if "passing_yards" in wk.columns else ("pass_yards","sum"),
                  attempts=("attempts","sum"))
             .reset_index())
    out = rec.merge(team_tot, on="team", how="left")

    out["target_share"] = out["targets"] / out["team_targets"].clip(lower=1)
    out["rush_share"]   = out["carries"] / out["team_rushes"].clip(lower=1)
    out["ypt"]          = out["rec_yards"] / out["targets"].clip(lower=1)
    out["ypc"]          = out["rush_yards"] / out["carries"].clip(lower=1)
    out["qb_ypa"]       = out["pass_yards"] / out["attempts"].clip(lower=1)
    # yprr proxy: use rec_yards per team attempts (proxy for dropbacks)
    out["yprr_proxy"]   = out["rec_yards"] / out["team_attempts"].clip(lower=1)

    # We don’t have true RZ shares from weekly; set conservative 0 (your pricing handles absence)
    out["rz_tgt_share"] = 0.0
    out["rz_carry_share"] = 0.0

    out["position"] = out.apply(lambda r:
        "QB" if (r.get("attempts",0) or 0) > 0 else
        ("RB" if (r.get("carries",0) or 0) > 0 else
         ("WR/TE" if (r.get("targets",0) or 0) > 0 else "")), axis=1)

    keep = ["player","team","position","target_share","rush_share","rz_tgt_share","rz_carry_share","yprr_proxy","ypc","ypt","qb_ypa"]
    for c in keep:
        if c not in out.columns: out[c] = 0.0
    return out[keep].fillna(0.0).sort_values(["team","player"]).reset_index(drop=True)

def build_player_form(season: int, history: str) -> pd.DataFrame:
    # Try PBP for current season first (best quality)
    pbp = load_pbp([], season)
    pf = _from_pbp(pbp, season)
    if pf.empty:
        # Weekly fallback
        pf = _from_weekly(season)
    return pf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--history", default="2019-2024")  # kept for interface parity
    ap.add_argument("--write", default="data/player_form.csv")
    args = ap.parse_args()

    Path("data").mkdir(parents=True, exist_ok=True)
    df = build_player_form(args.season, args.history)
    df.to_csv(args.write, index=False)
    print(f"[build_player_form] ✅ wrote {len(df)} rows → {args.write}")

if __name__ == "__main__":
    main()
