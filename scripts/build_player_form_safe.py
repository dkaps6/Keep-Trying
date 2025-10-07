#!/usr/bin/env python3
"""
Safe fallback builder for player_form.csv that does NOT replace your existing builder.
It composes player_form from the outputs produced by make_all.py (nflverse + add-ons).

Output schema (exact order):
  team, player, position, target_share, rush_share, rz_tgt_share, rz_carry_share,
  yprr_proxy, ypc, ypt, qb_ypa
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

EXPECT = [
    "team","player","position",
    "target_share","rush_share",
    "rz_tgt_share","rz_carry_share",
    "yprr_proxy","ypc","ypt","qb_ypa"
]

def exists_nonempty(p: Path) -> bool:
    return p.exists() and p.is_file() and p.stat().st_size > 0

def latest_csv(folder: Path, stem: str) -> Path | None:
    if not folder.exists():
        return None
    cands = sorted([p for p in folder.glob("*.csv") if stem in p.name],
                   key=lambda q: q.stat().st_mtime, reverse=True)
    return cands[0] if cands else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--write", default="data/player_form.csv")
    args = ap.parse_args()

    out_dir = Path(args.write).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    root = Path("outputs")  # where make_all.py stages raw pulls
    pstats_dir = root / "player_stats"
    roles_dir  = root / "roles"
    rbm_dir    = root / "rb_metrics"

    # base from player weekly stats
    base = pd.DataFrame(columns=[
        "season","week","team","player_display_name","position",
        "targets","receptions","receiving_yards","rush_attempts",
        "rushing_yards","air_yards"
    ])
    pw = latest_csv(pstats_dir, f"player_stats_week_{args.season}")
    if pw and exists_nonempty(pw):
        dfp = pd.read_csv(pw)
        keep = [c for c in base.columns if c in dfp.columns]
        base = dfp[keep].copy()

    # target share
    if not base.empty and {"targets","team"}.issubset(base.columns):
        team_tot = base.groupby(["season","week","team"], as_index=False)["targets"].sum()\
                       .rename(columns={"targets":"team_targets"})
        base = base.merge(team_tot, on=["season","week","team"], how="left")
        base["target_share"] = (base["targets"] / base["team_targets"]).fillna(0.0)
    else:
        base["target_share"] = 0.0

    # roles
    roles_csv = latest_csv(roles_dir, f"roles_weekly_{args.season}")
    if roles_csv and exists_nonempty(roles_csv):
        rr = pd.read_csv(roles_csv).rename(columns={"role_label":"role"})
        base = base.merge(rr[["season","week","team","player_display_name","role"]],
                          on=["season","week","team","player_display_name"], how="left")

    # RB metrics
    rbw = latest_csv(rbm_dir, f"rb_metrics_week_{args.season}")
    if rbw and exists_nonempty(rbw):
        r = pd.read_csv(rbw).rename(columns={
            "posteam":"team",
            "rusher_player_name":"player_display_name",
            "yards_per_carry":"ypc"
        })
        base = base.merge(r[["season","week","team","player_display_name","ypc"]],
                          on=["season","week","team","player_display_name"], how="left")
    else:
        base["ypc"] = 0.0

    # final, exact schema for your engine
    out_df = pd.DataFrame()
    out_df["team"]           = base.get("team", "")
    out_df["player"]         = base.get("player_display_name", "")
    out_df["position"]       = base.get("position", "")
    out_df["target_share"]   = base.get("target_share", 0.0).fillna(0.0)
    out_df["rush_share"]     = 0.0
    out_df["rz_tgt_share"]   = 0.0
    out_df["rz_carry_share"] = 0.0
    out_df["yprr_proxy"]     = 0.0
    out_df["ypc"]            = base.get("ypc", 0.0).fillna(0.0)
    out_df["ypt"]            = 0.0
    out_df["qb_ypa"]         = 0.0

    out_df = out_df[EXPECT].fillna(0.0)
    out_df.to_csv(args.write, index=False)
    print(f"[build_player_form_safe] ✅ wrote {len(out_df)} rows → {args.write}")

if __name__ == "__main__":
    main()
