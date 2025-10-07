#!/usr/bin/env python3
"""
Unified driver for the nflverse bundle that lives under external/nflverse_bundle/.

What this does:
  1) cd into this folder so relative imports/paths work.
  2) Run the local free fetcher: fetch_all.py (NOT nflverse_csv_fetcher/...).
  3) Run addon derivations (PROE, box counts, roles, RB metrics, injuries).
  4) Compose season-level team_form and player_form.
  5) Write outputs to BOTH:
        - repo_root/outputs/metrics/{team_form.csv,player_form.csv}
        - repo_root/data/{team_form.csv,player_form.csv}   (for your existing model steps)
"""

from __future__ import annotations
import argparse
import os
import sys
import subprocess
from pathlib import Path
import pandas as pd

# ---------- helpers ----------

def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(rc)

def exists_nonempty(p: Path) -> bool:
    return p.exists() and p.is_file() and p.stat().st_size > 0

def latest_csv(folder: Path, stem_contains: str) -> Path | None:
    if not folder.exists():
        return None
    c = sorted([p for p in folder.glob("*.csv") if stem_contains in p.name],
               key=lambda q: q.stat().st_mtime, reverse=True)
    return c[0] if c else None

def safe_left_join(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if left is None or left.empty:
        return right.copy()
    if right is None or right.empty:
        return left.copy()
    k = [c for c in keys if c in left.columns and c in right.columns]
    return left if not k else left.merge(right, on=k, how="left")

# ---------- composition ----------

def compose_team_form(root: Path, repo_root: Path, season: int) -> pd.DataFrame:
    """
    Inputs this script expects to find under:
      root/outputs/team_stats/
      root/outputs/proe/
      root/outputs/box_counts/
    """
    out_root = root / "outputs"
    team_stats_dir = out_root / "team_stats"
    proe_dir       = out_root / "proe"
    box_dir        = out_root / "box_counts"

    tf = pd.DataFrame()

    # team regular-season EPA and sacks, etc.
    reg = latest_csv(team_stats_dir, f"team_stats_reg_{season}")
    if reg and exists_nonempty(reg):
        t = pd.read_csv(reg)
        keep = [c for c in t.columns if c in (
            "team","season","passing_epa","rushing_epa","def_pass_epa",
            "def_rush_epa","epa_per_play","def_epa_per_play","sack_rate"
        )]
        if keep:
            tf = t[keep].copy()

    # PROE (season-level)
    proe = latest_csv(proe_dir, f"team_proe_season_{season}")
    if proe and exists_nonempty(proe):
        p = pd.read_csv(proe).rename(columns={"posteam": "team"})
        tf = safe_left_join(tf, p[["season", "team", "season_proe"]], ["season", "team"])

    # defensive box counts aggregated to season avg
    dbox = latest_csv(box_dir, f"defense_box_rates_week_{season}")
    if dbox and exists_nonempty(dbox):
        d = pd.read_csv(dbox)
        g = d.groupby(["season", "team"], as_index=False).agg({
            "def_light_box_rate": "mean",
            "def_heavy_box_rate": "mean",
        })
        tf = safe_left_join(tf, g, ["season", "team"])

    tf = tf.rename(columns={
        "passing_epa":      "off_pass_epa",
        "rushing_epa":      "off_rush_epa",
        "epa_per_play":     "off_epa_play",
        "def_epa_per_play": "def_epa_play",
        "season_proe":      "proe",
        "sack_rate":        "def_sack_rate",
    })

    # write to repo root
    metrics_dir = repo_root / "outputs" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_csv = metrics_dir / "team_form.csv"
    tf.to_csv(out_csv, index=False)
    print(f"[compose] wrote {out_csv} ({len(tf)} rows)")

    # mirror to data/ for your legacy steps
    data_dir = repo_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    tf.to_csv(data_dir / "team_form.csv", index=False)

    return tf

def compose_player_form(root: Path, repo_root: Path, season: int) -> pd.DataFrame:
    """
    Inputs expected under root/outputs:
      - player_stats/
      - roles/
      - rb_metrics/
    Produces exact columns your engine expects later.
    """
    out_root = root / "outputs"
    pstats_dir = out_root / "player_stats"
    roles_dir  = out_root / "roles"
    rbm_dir    = out_root / "rb_metrics"

    EXPECT = [
        "team","player","position",
        "target_share","rush_share",
        "rz_tgt_share","rz_carry_share",
        "yprr_proxy","ypc","ypt","qb_ypa"
    ]

    base = pd.DataFrame(columns=[
        "season","week","team","player_display_name","position",
        "targets","receptions","receiving_yards","rush_attempts","rushing_yards","air_yards"
    ])

    pw = latest_csv(pstats_dir, f"player_stats_week_{season}")
    if pw and exists_nonempty(pw):
        dfp = pd.read_csv(pw)
        have = [c for c in base.columns if c in dfp.columns]
        base = dfp[have].copy()

    # target share
    if not base.empty and {"targets","team"}.issubset(base.columns):
        team_tot = base.groupby(["season","week","team"], as_index=False)["targets"].sum()\
                       .rename(columns={"targets": "team_targets"})
        base = base.merge(team_tot, on=["season","week","team"], how="left")
        base["target_share"] = (base["targets"] / base["team_targets"]).fillna(0.0)
    else:
        base["target_share"] = 0.0

    # roles (optional)
    roles_csv = latest_csv(roles_dir, f"roles_weekly_{season}")
    if roles_csv and exists_nonempty(roles_csv):
        rr = pd.read_csv(roles_csv).rename(columns={"role_label": "role"})
        base = base.merge(rr[["season","week","team","player_display_name","role"]],
                          on=["season","week","team","player_display_name"], how="left")

    # RB metrics (ypc proxy)
    rbw = latest_csv(rbm_dir, f"rb_metrics_week_{season}")
    if rbw and exists_nonempty(rbw):
        r = pd.read_csv(rbw).rename(columns={
            "posteam": "team",
            "rusher_player_name": "player_display_name",
            "yards_per_carry": "ypc",
        })
        base = base.merge(r[["season","week","team","player_display_name","ypc"]],
                          on=["season","week","team","player_display_name"], how="left")
    else:
        base["ypc"] = 0.0

    base = base.rename(columns={"player_display_name": "player"})

    out_df = pd.DataFrame()
    out_df["team"]           = base.get("team", "")
    out_df["player"]         = base.get("player", "")
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

    # write to repo root
    metrics_dir = repo_root / "outputs" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_csv = metrics_dir / "player_form.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"[compose] wrote {out_csv} ({len(out_df)} rows)")

    # mirror to data/ for your legacy steps
    data_dir = repo_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(data_dir / "player_form.csv", index=False)

    return out_df

# ---------- main ----------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--skip-pbp", action="store_true")
    args = ap.parse_args()

    # normalize working dir and import path
    ROOT = Path(__file__).resolve().parent
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))

    # repo root (two levels up from this file)
    REPO_ROOT = ROOT.parents[1]

    # 1) Run local free fetcher (was wrongly pointing to nflverse_csv_fetcher/)
    fetcher = ROOT / "fetch_all.py"
    if fetcher.exists():
        cmd = [sys.executable, str(fetcher), "--season", str(args.season)]
        if args.skip_pbp:
            cmd.append("--skip-pbp")
        run(cmd)
    else:
        print("::warning ::fetch_all.py missing — free fetch skipped.")

    # 2) Run add-ons (they should write into ROOT/outputs/*)
    addons = ROOT / "addons"
    # ensure it's a package
    (addons / "__init__.py").touch(exist_ok=True)

    add_scripts = [
        ("derive_proe.py",        []),
        ("aggregate_box_counts.py", []),
        ("derive_roles.py",       []),
        ("derive_rb_metrics.py",  []),
        ("fetch_injuries_espn.py", []),  # produces injuries baseline
    ]
    for script, extra in add_scripts:
        p = addons / script
        if p.exists():
            run([sys.executable, str(p), "--season", str(args.season), *extra])
        else:
            print(f"::warning ::addon missing: {p.name}")

    # 3) Compose + mirror to repo_root/outputs/metrics and data/
    compose_team_form(ROOT, REPO_ROOT, args.season)
    compose_player_form(ROOT, REPO_ROOT, args.season)

    print("✅ make_all completed.")
    return 0

if __name__ == "__main__":
    sys.exit(main())

