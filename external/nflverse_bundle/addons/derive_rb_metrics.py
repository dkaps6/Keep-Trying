#!/usr/bin/env python3
import argparse, sys
import polars as pl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True, type=int)
    args = ap.parse_args()
    season=int(args.season)
    try:
        df = pl.read_parquet(f"external/nflverse_bundle/outputs/pbp/pbp_{season}.parquet")
    except Exception:
        try:
            df = pl.read_csv(f"external/nflverse_bundle/outputs/pbp/pbp_{season}.csv")
        except Exception as e:
            print(f"[derive_rb_metrics] no pbp available: {e}", file=sys.stderr)
            df = pl.DataFrame(schema={"season":pl.Int64, "week":pl.Int64, "posteam":pl.Utf8, "rusher_player_name":pl.Utf8, "rushing_yards":pl.Float64})

    if df.height == 0 or "rusher_player_name" not in df.columns:
        out = pl.DataFrame({"season":[], "week":[], "posteam":[], "rusher_player_name":[], "rushes":[], "yards":[]})
        out.write_csv("outputs/proe/rb_metrics.csv")
        print("wrote outputs/proe/rb_metrics.csv rows=0")
        return

    runs = df.filter((pl.col("rusher_player_name").is_not_null()))
    pw = runs.group_by(["season","week","posteam","rusher_player_name"]).agg([
        pl.len().alias("rushes"),
        pl.col("rushing_yards").sum().alias("yards")
    ])
    pw.write_csv("outputs/proe/rb_metrics.csv")
    print("wrote outputs/proe/rb_metrics.csv rows=", pw.height)

if __name__ == "__main__":
    main()
