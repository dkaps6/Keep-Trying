#!/usr/bin/env python3
"""
Aggregate box count rates from nflverse Participation data.
- Uses defenders_in_box from load_participation()
- Emits team-level offensive light/heavy box rates and defense-facing rates.
"""
import argparse, os
import polars as pl
import nflreadpy as nfl

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def tag_box(x: int) -> str:
    if x is None: return "unknown"
    try:
        xi = int(x)
    except Exception:
        return "unknown"
    if xi <= 6: return "light"
    if xi >= 8: return "heavy"
    return "neutral"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", nargs="+", default=["2025"])
    ap.add_argument("--out", default="outputs/box_counts")
    args = ap.parse_args()
    seasons = [int(s) for s in args.season]
    ensure_dir(args.out)

    part = nfl.load_participation(seasons=seasons, file_type="csv")
    # Keep run/pass plays only
    part = part.filter(pl.col("play_type").is_in(["run","pass"]))
    # Tag box
    part = part.with_columns(pl.col("defenders_in_box").apply(tag_box).alias("box_tag"))

    # Offensive team rates (what boxes they *face* when on offense)
    off = part.group_by(["season","week","posteam","box_tag"]).agg(pl.count().alias("plays")).pivot(
        values="plays", index=["season","week","posteam"], columns="box_tag"
    ).fill_null(0)
    for col in ["light","neutral","heavy"]:
        if col not in off.columns: off = off.with_columns(pl.lit(0).alias(col))
    off = off.with_columns((pl.col("light")+pl.col("neutral")+pl.col("heavy")).alias("total_plays"))
    off = off.with_columns([
        (pl.col("light")/pl.col("total_plays")).alias("light_box_rate"),
        (pl.col("heavy")/pl.col("total_plays")).alias("heavy_box_rate")
    ])
    off.write_csv(os.path.join(args.out, f"offense_box_rates_week_{'-'.join(map(str,seasons))}.csv"))

    # Defensive team rates (what boxes they *show* on defense)
    # Map 'defteam' as defensive team field in participation
    defn = part.group_by(["season","week","defteam","box_tag"]).agg(pl.count().alias("plays")).pivot(
        values="plays", index=["season","week","defteam"], columns="box_tag"
    ).fill_null(0)
    for col in ["light","neutral","heavy"]:
        if col not in defn.columns: defn = defn.with_columns(pl.lit(0).alias(col))
    defn = defn.with_columns((pl.col("light")+pl.col("neutral")+pl.col("heavy")).alias("total_plays"))
    defn = defn.with_columns([
        (pl.col("light")/pl.col("total_plays")).alias("def_light_box_rate"),
        (pl.col("heavy")/pl.col("total_plays")).alias("def_heavy_box_rate")
    ])
    defn = defn.rename({"defteam":"team"})
    defn.write_csv(os.path.join(args.out, f"defense_box_rates_week_{'-'.join(map(str,seasons))}.csv"))

    print("✅ Wrote box count rate CSVs to", args.out)

if __name__ == "__main__":
    main()
