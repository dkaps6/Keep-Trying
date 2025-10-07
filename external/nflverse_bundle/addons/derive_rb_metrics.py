#!/usr/bin/env python3
"""
Derive RB metrics from nflverse play-by-play:
- success rate (EPA > 0 or by down-distance success)
- explosive run rate (>= 10 yards)
- yards after contact proxy not available in pbp; we output YPC and EPA/rush.
Outputs per-player weekly and season aggregates.
"""
import argparse, os
import polars as pl
import nflreadpy as nfl

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", nargs="+", default=["2025"])
    ap.add_argument("--out", default="outputs/rb_metrics")
    args = ap.parse_args()
    seasons = [int(s) for s in args.season]
    ensure_dir(args.out)

    pbp = nfl.load_pbp(seasons=seasons, file_type="csv")

    # Keep designed runs only
    runs = pbp.filter((pl.col("play_type")=="run") & (pl.col("rusher_player_name").is_not_null()))
    # Success definition: EPA > 0 (simple) – adjust if you prefer down-distance success
    runs = runs.with_columns([
        (pl.col("epa") > 0).cast(pl.Int8).alias("success"),
        (pl.col("yards_gained") >= 10).cast(pl.Int8).alias("explosive_run")
    ])

    # Player-week aggregation
    pw = runs.group_by(["season","week","posteam","rusher_player_name"]).agg([
        pl.count().alias("rushes"),
        pl.mean("epa").alias("epa_per_rush"),
        pl.sum("epa").alias("total_epa"),
        pl.sum("yards_gained").alias("rush_yards"),
        (pl.col("rush_yards")/pl.col("rushes")).alias("yards_per_carry"),
        pl.mean("success").alias("success_rate"),
        pl.mean("explosive_run").alias("explosive_rate")
    ])

    pw.write_csv(os.path.join(args.out, f"rb_metrics_week_{'-'.join(map(str,seasons))}.csv"))

    # Season aggregate
    ps = pw.group_by(["season","posteam","rusher_player_name"]).agg([
        pl.sum("rushes").alias("rushes"),
        pl.mean("epa_per_rush").alias("epa_per_rush"),
        pl.sum("total_epa").alias("total_epa"),
        pl.sum("rush_yards").alias("rush_yards"),
        (pl.col("rush_yards")/pl.col("rushes")).alias("yards_per_carry"),
        pl.mean("success_rate").alias("success_rate"),
        pl.mean("explosive_rate").alias("explosive_rate")
    ])
    ps.write_csv(os.path.join(args.out, f"rb_metrics_season_{'-'.join(map(str,seasons))}.csv"))
    print("✅ Wrote RB metrics to", args.out)

if __name__ == "__main__":
    main()
