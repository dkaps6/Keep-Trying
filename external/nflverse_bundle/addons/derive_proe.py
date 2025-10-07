#!/usr/bin/env python3
"""
Derive team-level PROE (Pass Rate Over Expectation) from nflverse play-by-play.
- Uses nflreadpy.load_pbp() which contains xpass / pass indicators (since nflfastR models).
- Outputs team-week and season aggregates.
"""
import argparse, os
import polars as pl
import nflreadpy as nfl

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def compute_proe(pbp: pl.DataFrame) -> pl.DataFrame:
    # Filter to offensive team pass/run plays (drop penalties, no-plays)
    df = pbp.filter(
        (pl.col("pass").is_in([0,1])) &
        (pl.col("rush").is_in([0,1])) &
        (pl.col("play_type").is_in(["run","pass"])) &
        (pl.col("posteam").is_not_null())
    )
    # Expected pass probability (xpass) and actual pass
    # nflfastR adds 'xpass' (expected pass probability) and 'pass' (1 if pass)
    if "xpass" not in df.columns:
        raise RuntimeError("No xpass in PBP; cannot compute PROE. Ensure seasons >= 2006 and nflfastR-style pbp.")
    df = df.with_columns([
        pl.col("xpass").cast(pl.Float64),
        pl.col("pass").cast(pl.Float64)
    ])
    # Aggregate to team-week
    team_week = df.group_by(["season","week","posteam"]).agg([
        pl.mean("xpass").alias("exp_pass_rate"),
        pl.mean("pass").alias("act_pass_rate"),
        pl.count().alias("plays")
    ]).with_columns((pl.col("act_pass_rate") - pl.col("exp_pass_rate")).alias("proe"))
    return team_week

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", nargs="+", default=["2025"])
    ap.add_argument("--out", default="outputs/proe")
    args = ap.parse_args()
    seasons = [int(s) for s in args.season]
    ensure_dir(args.out)
    # Load PBP for seasons (may be large)
    pbp = nfl.load_pbp(seasons=seasons, file_type="csv")
    tw = compute_proe(pbp)
    tw.write_csv(os.path.join(args.out, f"team_proe_week_{'-'.join(map(str,seasons))}.csv"))
    # Season aggregate
    ts = tw.group_by(["season","posteam"]).agg([
        pl.col("exp_pass_rate").mean().alias("season_exp_pass_rate"),
        pl.col("act_pass_rate").mean().alias("season_act_pass_rate"),
        pl.col("proe").mean().alias("season_proe"),
        pl.col("plays").sum().alias("season_plays")
    ])
    ts.write_csv(os.path.join(args.out, f"team_proe_season_{'-'.join(map(str,seasons))}.csv"))
    print("✅ Wrote PROE CSVs to", args.out)

if __name__ == "__main__":
    main()
