#!/usr/bin/env python3
"""
Unified driver that:
1) Runs nflverse fetches (and addons) via ./nflverse_csv_fetcher/make_all.py
2) Pulls optional paid feeds when secrets are present (MSF, NFLGSIS, Odds, API-Sports) — stubs here; won’t crash if missing.
3) Composes model-ready CSVs:
      outputs/metrics/team_form.csv
      outputs/metrics/player_form.csv
4) Validates non-empty outputs (your workflow will also run scripts/validate_inputs.py).
"""
from __future__ import annotations
import argparse, os, sys, subprocess
from pathlib import Path
import pandas as pd


# ---------- small helpers ----------
def run(cmd: list[str], allow_fail: bool = False) -> int:
    print(">>", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0 and not allow_fail:
        raise SystemExit(rc)
    return rc

def exists_nonempty(p: str | Path) -> bool:
    p = Path(p)
    return p.exists() and p.is_file() and p.stat().st_size > 0

def latest_csv(folder: Path, stem_contains: str) -> Path | None:
    if not folder.exists():
        return None
    cands = sorted(
        [p for p in folder.glob("*.csv") if stem_contains in p.name],
        key=lambda q: q.stat().st_mtime,
        reverse=True,
    )
    return cands[0] if cands else None

def safe_left_join(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if left is None or left.empty:
        return right.copy()
    if right is None or right.empty:
        return left.copy()
    k = [c for c in keys if c in left.columns and c in right.columns]
    if not k:
        return left
    return left.merge(right, on=k, how="left")


# ---------- composition ----------
def compose_team_form(out_root: Path, season: int) -> None:
    """Team-level EPA splits + PROE + defensive box rates (season-averaged)."""
    out_root.mkdir(parents=True, exist_ok=True)
    team_stats_dir = out_root / "team_stats"
    proe_dir      = out_root / "proe"
    box_dir       = out_root / "box_counts"

    # start with regular-season team stats if present
    tf = pd.DataFrame()
    reg = latest_csv(team_stats_dir, f"team_stats_reg_{season}")
    if reg and exists_nonempty(reg):
        t = pd.read_csv(reg)
        keep = [c for c in t.columns if c in (
            "team","season","passing_epa","rushing_epa",
            "def_pass_epa","def_rush_epa","epa_per_play",
            "def_epa_per_play","sack_rate"
        )]
        if keep:
            tf = t[keep].copy()

    # add PROE (season)
    proe = latest_csv(proe_dir, f"team_proe_season_{season}")
    if proe and exists_nonempty(proe):
        p = pd.read_csv(proe).rename(columns={"posteam":"team"})
        tf = safe_left_join(tf, p[["season","team","season_proe"]], ["season","team"])

    # add defensive box rate season averages
    dbox = latest_csv(box_dir, f"defense_box_rates_week_{season}")
    if dbox and exists_nonempty(dbox):
        d = pd.read_csv(dbox)
        g = d.groupby(["season","team"], as_index=False).agg({
            "def_light_box_rate":"mean",
            "def_heavy_box_rate":"mean"
        })
        tf = safe_left_join(tf, g, ["season","team"])

    # rename toward your schema
    ren = {
        "passing_epa":"off_pass_epa",
        "rushing_epa":"off_rush_epa",
        "epa_per_play":"off_epa_play",
        "def_epa_per_play":"def_epa_play",
        "season_proe":"proe",
        "sack_rate":"def_sack_rate",
    }
    tf = tf.rename(columns=ren)

    out = out_root / "metrics" / "team_form.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    tf.to_csv(out, index=False)
    print(f"[compose] wrote {out} ({len(tf)} rows)")


def compose_player_form(out_root: Path, season: int) -> None:
    """
    Player-level usage + roles + RB metrics.
    Produces your expected columns exactly (filled with 0.0/'' if a source is missing):
      team, player, position, target_share, rush_share, rz_tgt_share, rz_carry_share,
      yprr_proxy, ypc, ypt, qb_ypa
    """
    out_root.mkdir(parents=True, exist_ok=True)
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
        keep = [c for c in base.columns if c in dfp.columns]
        base = dfp[keep].copy()

    # shares
    if not base.empty and {"targets","team"}.issubset(base.columns):
        team_tot = base.groupby(["season","week","team"], as_index=False)["targets"].sum()\
                       .rename(columns={"targets":"team_targets"})
        base = base.merge(team_tot, on=["season","week","team"], how="left")
        base["target_share"] = (base["targets"] / base["team_targets"]).fillna(0.0)
    else:
        base["target_share"] = 0.0

    # simple aDOT proxy if air_yards present
    if "air_yards" in base.columns and "targets" in base.columns:
        base["aDOT"] = base.apply(lambda r: (r["air_yards"]/r["targets"]) if r.get("targets",0)>0 else 0.0, axis=1)
    else:
        base["aDOT"] = 0.0

    # roles
    roles_csv = latest_csv(roles_dir, f"roles_weekly_{season}")
    if roles_csv and exists_nonempty(roles_csv):
        rr = pd.read_csv(roles_csv).rename(columns={"role_label":"role"})
        base = base.merge(
            rr[["season","week","team","player_display_name","role"]],
            on=["season","week","team","player_display_name"],
            how="left"
        )

    # RB metrics
    rbw = latest_csv(rbm_dir, f"rb_metrics_week_{season}")
    if rbw and exists_nonempty(rbw):
        r = pd.read_csv(rbw).rename(columns={
            "posteam":"team",
            "rusher_player_name":"player_display_name"
        })
        base = base.merge(
            r[["season","week","team","player_display_name","yards_per_carry"]],
            on=["season","week","team","player_display_name"],
            how="left"
        )
        base = base.rename(columns={"yards_per_carry":"ypc"})
    else:
        base["ypc"] = 0.0

    # final projection columns (zeros if you don't compute them elsewhere)
    base = base.rename(columns={"player_display_name":"player"})
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

    # ensure exact order
    out_df = out_df[EXPECT]
    out =
