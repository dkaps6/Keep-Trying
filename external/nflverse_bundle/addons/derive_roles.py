#!/usr/bin/env python3
"""
Derive roles (WR1/WR2/SLOT/RB1/TE1) per team from weekly depth charts + usage.
Heuristic:
- Start with depth chart positions (nfl.load_depth_charts)
- Join weekly player stats (targets, receptions, routes where available)
- Rank team WRs by targets per week to label WR1/WR2; highest TE as TE1; RB rush attempts to label RB1.
- Assign SLOT to WR with highest slot snap share if available; else leave blank.
Outputs weekly roles CSV.
"""
import argparse, os
import polars as pl
import nflreadpy as nfl

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", nargs="+", default=["2025"])
    ap.add_argument("--out", default="outputs/roles")
    args = ap.parse_args()
    seasons = [int(s) for s in args.season]
    ensure_dir(args.out)

    dc = nfl.load_depth_charts(seasons=seasons, file_type="csv")
    ps = nfl.load_player_stats(seasons=seasons, summary_level="week", file_type="csv")

    # Keep relevant fields
    ps_keep = ps.select([
        "season","week","team","player_display_name","position","targets","receptions","rush_attempts"
    ])
    dc_keep = dc.select([
        "season","week","team","player_display_name","position","depth_chart_position","depth_chart_order"
    ])

    df = dc_keep.join(ps_keep, on=["season","week","team","player_display_name","position"], how="outer")

    # WR roles by targets
    wr = df.filter(pl.col("position")=="WR")
    wr = wr.with_columns(pl.col("targets").fill_null(0))
    wr_rank = wr.group_by(["season","week","team"]).apply(lambda g: (
        g.sort(by="targets", descending=True)
         .with_columns(pl.arange(1, g.height+1).alias("wr_rank"))
    ))
    wr_rank = wr_rank.with_columns(pl.when(pl.col("wr_rank")==1).then("WR1")
                        .when(pl.col("wr_rank")==2).then("WR2")
                        .otherwise(pl.lit(None)).alias("role_label"))

    # TE1 by targets
    te = df.filter(pl.col("position")=="TE").with_columns(pl.col("targets").fill_null(0))
    te_rank = te.group_by(["season","week","team"]).apply(lambda g: (
        g.sort(by="targets", descending=True)
         .with_columns(pl.arange(1, g.height+1).alias("te_rank"))
    ))
    te_rank = te_rank.with_columns(pl.when(pl.col("te_rank")==1).then("TE1").otherwise(pl.lit(None)).alias("role_label"))

    # RB1 by rush attempts
    rb = df.filter(pl.col("position")=="RB").with_columns(pl.col("rush_attempts").fill_null(0))
    rb_rank = rb.group_by(["season","week","team"]).apply(lambda g: (
        g.sort(by="rush_attempts", descending=True)
         .with_columns(pl.arange(1, g.height+1).alias("rb_rank"))
    ))
    rb_rank = rb_rank.with_columns(pl.when(pl.col("rb_rank")==1).then("RB1").otherwise(pl.lit(None)).alias("role_label"))

    roles = pl.concat([
        wr_rank.select(["season","week","team","player_display_name","position","role_label"]),
        te_rank.select(["season","week","team","player_display_name","position","role_label"]),
        rb_rank.select(["season","week","team","player_display_name","position","role_label"])
    ], how="diagonal")

    roles = roles.filter(pl.col("role_label").is_not_null())
    roles.write_csv(os.path.join(args.out, f"roles_weekly_{'-'.join(map(str,seasons))}.csv"))
    print("✅ Wrote roles CSV to", args.out)

if __name__ == "__main__":
    main()
