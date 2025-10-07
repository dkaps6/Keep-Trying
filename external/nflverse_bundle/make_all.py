#!/usr/bin/env python3
"""
Run the full pipeline:
1) Fetch core nflverse datasets to CSV (team/player stats, NGS, FTN, rosters, schedules, etc.).
2) Build add-ons:
   - PROE (from PBP xpass)
   - Box-count rates (from Participation)
   - Roles (WR1/WR2/RB1/TE1 from depth charts + usage)
   - RB metrics (EPA/Success/Explosive/YPC from PBP)
   - ESPN injuries fallback (optional, user flag)
3) Compose neat model-ready CSVs in ./outputs/metrics:
   - team_form.csv  (EPA splits, PROE, light/heavy box rates, sack rate if present)
   - player_form.csv (basic usage + roles + RB metrics; best-effort for shares)
Notes:
- This script is resilient to missing upstream columns; fields may be blank if not available.
"""
from __future__ import annotations
import argparse, os, subprocess, sys, glob
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))

def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=False)

def latest(path_glob: str) -> str | None:
    files = glob.glob(path_glob)
    if not files: return None
    return max(files, key=os.path.getmtime)

def safe_read(path: str) -> pl.DataFrame:
    if not path or not os.path.exists(path):
        return pl.DataFrame()
    try:
        return pl.read_csv(path, ignore_errors=True)
    except Exception as e:
        print(f"[read_csv] {path} skipped: {e}")
        return pl.DataFrame()

def compose_team_form(out_root: str, seasons: list[int]) -> None:
    metrics_dir = os.path.join(out_root, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # Load sources
    team_week = safe_read(latest(os.path.join(out_root, "team_stats", f"team_stats_week_*{seasons[-1]}*.csv")))
    team_reg  = safe_read(latest(os.path.join(out_root, "team_stats", f"team_stats_reg_*{seasons[-1]}*.csv")))
    proe_season = safe_read(latest(os.path.join(out_root, "proe", f"team_proe_season_*{seasons[-1]}*.csv")))
    def_box = safe_read(latest(os.path.join(out_root, "box_counts", f"defense_box_rates_week_*{seasons[-1]}*.csv")))

    # Normalize team key
    for df in [team_week, team_reg]:
        if "team" not in df.columns and "club_code" in df.columns:
            df = df.rename({"club_code":"team"})
    # Build season-level EPA splits if present
    keep_cols = []
    for col in ["team","season","passing_epa","rushing_epa","def_pass_epa","def_rush_epa","epa_per_play","def_epa_per_play","sack_rate"]:
        if col in team_reg.columns: keep_cols.append(col)
    tf = team_reg.select(keep_cols) if keep_cols else pl.DataFrame()

    # Join PROE (season)
    if not proe_season.is_empty():
        # proe dataframe uses posteam
        proe = proe_season.rename({"posteam":"team"})
        tf = tf.join(proe.select(["season","team","season_proe"]), on=["season","team"], how="left")

    # Join defensive box rates season-avg
    if not def_box.is_empty():
        # average by season, team
        def_season = (def_box
                      .group_by(["season","team"])
                      .agg([pl.mean("def_light_box_rate").alias("def_light_box_rate"),
                            pl.mean("def_heavy_box_rate").alias("def_heavy_box_rate")]))
        tf = tf.join(def_season, on=["season","team"], how="left")

    # Rename toward the user's expected schema (best-effort)
    ren = {}
    if "passing_epa" in tf.columns: ren["passing_epa"] = "off_pass_epa"
    if "rushing_epa" in tf.columns: ren["rushing_epa"] = "off_rush_epa"
    if "epa_per_play" in tf.columns: ren["epa_per_play"] = "off_epa_play"
    if "def_pass_epa" in tf.columns: ren["def_pass_epa"] = "def_pass_epa"
    if "def_rush_epa" in tf.columns: ren["def_rush_epa"] = "def_rush_epa"
    if "def_epa_per_play" in tf.columns: ren["def_epa_per_play"] = "def_epa_play"
    if "season_proe" in tf.columns: ren["season_proe"] = "proe"
    if "sack_rate" in tf.columns: ren["sack_rate"] = "def_sack_rate"
    tf = tf.rename(ren)

    # Output
    out_path = os.path.join(metrics_dir, "team_form.csv")
    tf.write_csv(out_path)
    print("✅ Wrote", out_path)

def compose_player_form(out_root: str, seasons: list[int]) -> None:
    metrics_dir = os.path.join(out_root, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # Load sources
    ps_week = safe_read(latest(os.path.join(out_root, "player_stats", f"player_stats_week_*{seasons[-1]}*.csv")))
    roles = safe_read(latest(os.path.join(out_root, "roles", f"roles_weekly_*{seasons[-1]}*.csv")))
    rbw = safe_read(latest(os.path.join(out_root, "rb_metrics", f"rb_metrics_week_*{seasons[-1]}*.csv")))

    # Build player-week base
    # Keep a minimal set, robust to column name drift
    keep = [c for c in ["season","week","team","player_display_name","position","targets","receptions","receiving_yards","rush_attempts","rushing_yards","air_yards"] if c in ps_week.columns]
    base = ps_week.select(keep) if keep else pl.DataFrame()

    # Compute simple shares per team-week (if targets present)
    if not base.is_empty() and "targets" in base.columns:
        team_tot = base.group_by(["season","week","team"]).agg(pl.sum("targets").alias("team_targets"))
        base = base.join(team_tot, on=["season","week","team"], how="left")
        base = base.with_columns((pl.col("targets")/pl.col("team_targets")).fill_null(0).alias("target_share"))

    # aDOT if air_yards present
    if "air_yards" in base.columns and "targets" in base.columns:
        base = base.with_columns((pl.when(pl.col("targets")>0).then(pl.col("air_yards")/pl.col("targets")).otherwise(None)).alias("aDOT"))

    # Merge roles
    if not roles.is_empty():
        roles2 = roles.rename({"player_display_name":"player_display_name"})
        base = base.join(roles2.select(["season","week","team","player_display_name","role_label"]), on=["season","week","team","player_display_name"], how="left").rename({"role_label":"role"})

    # Merge RB metrics (per player-week)
    if not rbw.is_empty():
        rbw2 = rbw.rename({"posteam":"team","rusher_player_name":"player_display_name"})
        base = base.join(rbw2.select(["season","week","team","player_display_name","epa_per_rush","yards_per_carry","success_rate","explosive_rate"]), on=["season","week","team","player_display_name"], how="left")

    # Output
    out_path = os.path.join(metrics_dir, "player_form.csv")
    base.write_csv(out_path)
    print("✅ Wrote", out_path)

def main():
    ap = argparse.ArgumentParser(description="Run full nflverse fetch + addons + metrics composer")
    ap.add_argument("--season", nargs="+", default=["2025"])
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--skip-pbp", action="store_true")
    ap.add_argument("--skip-espn-inj", action="store_true", help="Skip ESPN injuries scrape")
    args = ap.parse_args()

    seasons = [int(s) for s in args.season]
    outdir = os.path.join(HERE, args.out)
    os.makedirs(outdir, exist_ok=True)

    # 1) core fetch
    run([sys.executable, os.path.join(HERE, "fetch_all.py"), "--season", *[str(s) for s in seasons], "--out", outdir] + (["--skip-pbp"] if args.skip_pbp else []))

    # 2) add-ons
    run([sys.executable, os.path.join(HERE, "addons", "derive_proe.py"), "--season", *[str(s) for s in seasons], "--out", os.path.join(outdir, "proe")])
    run([sys.executable, os.path.join(HERE, "addons", "aggregate_box_counts.py"), "--season", *[str(s) for s in seasons], "--out", os.path.join(outdir, "box_counts")])
    run([sys.executable, os.path.join(HERE, "addons", "derive_roles.py"), "--season", *[str(s) for s in seasons], "--out", os.path.join(outdir, "roles")])
    run([sys.executable, os.path.join(HERE, "addons", "derive_rb_metrics.py"), "--season", *[str(s) for s in seasons], "--out", os.path.join(outdir, "rb_metrics")])

    if not args.skip_espn_inj:
        run([sys.executable, os.path.join(HERE, "addons", "fetch_injuries_espn.py"), "--week", "-1", "--out", os.path.join(outdir, "injuries", "espn")])

    # 3) compose model-ready CSVs
    compose_team_form(outdir, seasons)
    compose_player_form(outdir, seasons)

    print("✅ All done. See:", outdir)

if __name__ == "__main__":
    main()
