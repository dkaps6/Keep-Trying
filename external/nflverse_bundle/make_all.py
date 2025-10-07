#!/usr/bin/env python3
"""
make_all.py — unified data-pack driver + composer (with provider fallbacks)

What this script does:
1) Runs the local fetcher to pull nflverse tables + addons (safe across versions).
2) Uses provider fallbacks (API-Sports, MySportsFeeds, optional NFLGSIS) to fill gaps
   when a primary file is missing or empty (e.g., current-season injuries/schedules).
3) Composes the minimum inputs your model expects:
   - outputs/metrics/team_form.csv   (also mirrored to data/team_form.csv)
   - outputs/metrics/player_form.csv (also mirrored to data/player_form.csv)

Provider fallbacks are ONLY used when primary inputs are missing or have 0 rows.
Your existing model & structure remain intact.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
import numpy as np

# ============================================================
# 🔌 PROVIDER FALLBACKS (INSERTION POINT #1 — imports & helpers)
# ============================================================

# Central 'use_if_empty' helper
def _use_if_empty(primary: Optional[pd.DataFrame],
                  *providers) -> pd.DataFrame:
    def _ok(df: Optional[pd.DataFrame]) -> bool:
        try:
            return isinstance(df, pd.DataFrame) and not df.empty
        except Exception:
            return False
    if _ok(primary):
        return primary
    for fn in providers:
        try:
            df = fn()
        except Exception:
            df = None
        if _ok(df):
            return df
    return primary if primary is not None else pd.DataFrame()

# Import provider modules
try:
    import scripts.providers.apisports as _apis
except Exception:
    _apis = None

try:
    import scripts.providers.msf as _msf
except Exception:
    _msf = None

try:
    import scripts.providers.nflgsis as _gsis
except Exception:
    _gsis = None

# ============================================================
# Utilities
# ============================================================

ROOT = Path(__file__).resolve().parent              # external/nflverse_bundle
REPO = ROOT.parent.parent                           # repo root
OUT_BUNDLE = ROOT / "outputs"                       # raw bundle outputs
OUT_METRICS = REPO / "outputs" / "metrics"          # composed metrics
DATA_MIRROR = REPO / "data"                         # mirror for your model

def _mkdirs():
    for p in [OUT_BUNDLE, OUT_METRICS, DATA_MIRROR]:
        p.mkdir(parents=True, exist_ok=True)

def _run(cmd: List[str]) -> int:
    print(">>", " ".join(cmd))
    return subprocess.call(cmd)

def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        # allow polars-to-csv quirks / empty headers
        try:
            return pd.read_csv(path, engine="python")
        except Exception:
            return pd.DataFrame()

def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df is None or df.empty:
        # Write headers if present, so downstream validators can see schema
        try:
            (df.head(0) if isinstance(df, pd.DataFrame) else pd.DataFrame()).to_csv(path, index=False)
        except Exception:
            path.write_text("")  # totally empty
    else:
        df.to_csv(path, index=False)
    print(f"[write] {path.relative_to(REPO)}  rows={0 if df is None else len(df)}")

def _zscore(col: pd.Series) -> pd.Series:
    try:
        mu = np.nanmean(col.values.astype(float))
        sd = np.nanstd(col.values.astype(float))
        if sd == 0 or np.isnan(sd):
            return pd.Series(np.zeros(len(col)), index=col.index)
        return (col - mu) / sd
    except Exception:
        return pd.Series(np.zeros(len(col)), index=col.index)

def _first_nonempty(*dfs: pd.DataFrame) -> pd.DataFrame:
    for df in dfs:
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    return pd.DataFrame()

# ============================================================
# Fetch & addons
# ============================================================

def fetch_bundle(seasons: List[int]) -> None:
    """Call the local fetcher and addons; keep going even if something fails."""
    fetch = ROOT / "fetch_all.py"
    if fetch.exists():
        for s in seasons:
            rc = _run([sys.executable, str(fetch), "--season", str(s)])
            if rc != 0:
                print(f"[warn] fetch_all.py returned {rc} for {s}, continuing")

    # Run addons (derive_proe, box_counts, roles, rb_metrics, injuries espn mirror)
    addons = [
        ("addons/derive_proe.py", []),
        ("addons/aggregate_box_counts.py", []),
        ("addons/derive_roles.py", []),
        ("addons/derive_rb_metrics.py", []),
        ("addons/fetch_injuries_espn.py", []),  # safe if missing
    ]
    for rel, extra in addons:
        script = ROOT / rel
        if script.exists():
            for s in seasons:
                rc = _run([sys.executable, str(script), "--season", str(s), *extra])
                if rc != 0:
                    print(f"[warn] {rel} returned {rc} for {s}, continuing")

# ============================================================
# Compose: TEAM FORM
# ============================================================

def compose_team_form(season: int) -> pd.DataFrame:
    """
    Produce team_form.csv with the columns your validator expects:
        team,
        def_pressure_rate_z, def_pass_epa_z, def_rush_epa_z, def_sack_rate_z,
        pace_z, light_box_rate_z, heavy_box_rate_z, ay_per_att_z,
        plays_est, proe, rz_rate
    We’re intentionally permissive; if a source is missing we fill with zeros and z-score to 0.
    """

    # primary bundle sources (read whatever we have)
    # You can extend these if you later materialize more outputs in the bundle.
    schedules_p = OUT_BUNDLE / "schedules" / f"schedules_{season}.csv"
    team_stats_week = OUT_BUNDLE / "team_stats" / f"team_stats_week_{season}.csv"
    team_stats_reg  = OUT_BUNDLE / "team_stats" / f"team_stats_reg_{season}.csv"
    box_week        = OUT_BUNDLE / "box_counts" / f"defense_box_rates_week_{season}.csv"
    proe_week       = ROOT / "outputs" / "proe" / f"proe_week_{season}.csv"  # from derive_proe

    sched_primary = _read_csv(schedules_p)

    # ============================================================
    # 🔁 PROVIDER FALLBACKS (INSERTION POINT #2 — schedules)
    # ============================================================
    if _apis:
        sched = _use_if_empty(
            sched_primary,
            lambda: _apis.schedules(season),
        )
    else:
        sched = sched_primary

    tsw = _read_csv(team_stats_week)
    tsr = _read_csv(team_stats_reg)
    box = _read_csv(box_week)
    proe = _read_csv(proe_week)

    # Minimal team frame
    teams = set()
    for df, col in [(sched, "home_team"), (sched, "away_team"),
                    (tsw, "team"), (tsr, "team"),
                    (box, "team"), (proe, "team")]:
        if not df.empty and col in df.columns:
            teams.update(df[col].dropna().unique().tolist())

    tf = pd.DataFrame({"team": sorted(teams)}) if teams else pd.DataFrame({"team": []})

    # --- Defensive EPA / pressure / sack (placeholders if absent) ---
    # (If you wire a proper source for EPA/sack later, just fill raw cols and z-score.)
    tf["def_pressure_rate"] = 0.0
    tf["def_pass_epa"] = 0.0
    tf["def_rush_epa"] = 0.0
    tf["def_sack_rate"] = 0.0
    if not tsr.empty:
        # Try to pull some columns if they exist in your team_stats table
        for raw, out in [
            ("def_pressure_rate", "def_pressure_rate"),
            ("def_pass_epa", "def_pass_epa"),
            ("def_rush_epa", "def_rush_epa"),
            ("def_sack_rate", "def_sack_rate"),
        ]:
            if raw in tsr.columns and "team" in tsr.columns:
                m = tsr[["team", raw]].groupby("team", as_index=False).mean()
                tf = tf.merge(m.rename(columns={raw: out}), on="team", how="left")

    # z-scores
    tf["def_pressure_rate_z"] = _zscore(tf["def_pressure_rate"].fillna(0.0))
    tf["def_pass_epa_z"]      = _zscore(tf["def_pass_epa"].fillna(0.0))
    tf["def_rush_epa_z"]      = _zscore(tf["def_rush_epa"].fillna(0.0))
    tf["def_sack_rate_z"]     = _zscore(tf["def_sack_rate"].fillna(0.0))

    # --- Pace proxy & plays_est ---
    tf["pace"] = 0.0
    if not tsw.empty and "plays" in tsw.columns and "team" in tsw.columns:
        pace_by_team = tsw.groupby("team", as_index=False)["plays"].mean()
        pace_by_team["pace"] = -pace_by_team["plays"]  # inverse proxy (more plays -> faster -> smaller seconds/snap)
        tf = tf.merge(pace_by_team[["team", "pace"]], on="team", how="left")
    tf["pace_z"]   = _zscore(tf["pace"].fillna(0.0))
    tf["plays_est"] = tf["pace"].fillna(0.0).abs() * 60.0  # cheap stable estimate

    # --- Box counts (light/heavy) ---
    tf["def_light_box_rate"] = 0.0
    tf["def_heavy_box_rate"] = 0.0
    if not box.empty and {"team", "def_light_box_rate", "def_heavy_box_rate"} <= set(box.columns):
        box_agg = box.groupby("team", as_index=False)[["def_light_box_rate", "def_heavy_box_rate"]].mean()
        tf = tf.merge(box_agg, on="team", how="left")
    tf["light_box_rate_z"] = _zscore(tf["def_light_box_rate"].fillna(0.0))
    tf["heavy_box_rate_z"] = _zscore(tf["def_heavy_box_rate"].fillna(0.0))

    # --- Air-yards per att proxy (placeholder) ---
    tf["ay_per_att"]   = 0.0
    tf["ay_per_att_z"] = _zscore(tf["ay_per_att"])

    # --- PROE (from addon) + red-zone rate placeholder ---
    tf["proe"] = 0.0
    if not proe.empty and {"team", "proe"} <= set(proe.columns):
        p = proe.groupby("team", as_index=False)["proe"].mean()
        tf = tf.merge(p, on="team", how="left", suffixes=("", "_from_proe"))
        tf["proe"] = tf["proe"].fillna(tf.pop("proe_from_proe"))
    tf["rz_rate"] = 0.0  # fill later when you wire a red-zone source

    # order columns
    cols = [
        "team",
        "def_pressure_rate_z","def_pass_epa_z","def_rush_epa_z","def_sack_rate_z",
        "pace_z","light_box_rate_z","heavy_box_rate_z","ay_per_att_z",
        "plays_est","proe","rz_rate",
    ]
    for c in cols:
        if c not in tf.columns:
            tf[c] = 0.0 if c != "team" else ""

    return tf[cols]

# ============================================================
# Compose: PLAYER FORM
# ============================================================

def compose_player_form(season: int) -> pd.DataFrame:
    """
    Produce player_form.csv with the columns your validator expects:
      player, team, position,
      target_share, rush_share, rz_tgt_share, rz_carry_share,
      yprr_proxy, ypc, ypt, qb_ypa
    We merge injuries (with provider fallback) to provide 'inj_status' (optional).
    """

    # Try to mine roles/usage from addons if present
    roles_p = ROOT / "outputs" / "roles" / f"roles_{season}.csv"
    rbm_p   = ROOT / "outputs" / "rb_metrics" / f"rb_metrics_{season}.csv"
    inj_p   = OUT_BUNDLE / "injuries" / f"injuries_{season}.csv"

    roles = _read_csv(roles_p)       # expected columns: player, team, position, role, target_share, rush_share, ...
    rbm   = _read_csv(rbm_p)         # optional metrics for RBs
    inj_primary = _read_csv(inj_p)   # might be empty for 2025

    # ============================================================
    # 🔁 PROVIDER FALLBACKS (INSERTION POINT #3 — injuries)
    # ============================================================
    def _msf_inj():
        return _msf.injuries(season) if _msf else None
    def _apis_inj():
        return _apis.injuries(season) if _apis else None
    def _gsis_inj():
        return _gsis.injuries(season) if _gsis else None

    injuries = _use_if_empty(inj_primary, _msf_inj, _apis_inj, _gsis_inj)

    # Base player frame
    pf = pd.DataFrame(columns=[
        "player","team","position",
        "target_share","rush_share","rz_tgt_share","rz_carry_share",
        "yprr_proxy","ypc","ypt","qb_ypa"
    ])

    if not roles.empty:
        # keep only the columns we need; provide defaults if missing
        for col, default in [
            ("player",""), ("team",""), ("position",""),
            ("target_share",0.0), ("rush_share",0.0),
            ("rz_tgt_share",0.0), ("rz_carry_share",0.0),
            ("yprr_proxy",0.0), ("ypc",0.0), ("ypt",0.0), ("qb_ypa",0.0),
        ]:
            if col not in roles.columns:
                roles[col] = default
        pf = roles[pf.columns].copy()

    # merge RB metrics if present
    if not rbm.empty and {"player","team"} <= set(rbm.columns):
        for col in ["success_rate","yac","explosive_rate"]:
            if col not in rbm.columns:
                rbm[col] = np.nan
        pf = pf.merge(rbm[["player","team","success_rate","yac","explosive_rate"]],
                      on=["player","team"], how="left")

    # attach injuries if present
    if not injuries.empty:
        inj = injuries.rename(columns={"status":"inj_status"}).copy()
        for c in ["player","team","inj_status"]:
            if c not in inj.columns:
                inj[c] = "" if c != "inj_status" else ""
        pf = pf.merge(inj[["player","team","inj_status"]], on=["player","team"], how="left")

    # defaults if still empty
    for c in pf.columns:
        if pf[c].dtype == object:
            pf[c] = pf[c].fillna("")
        else:
            pf[c] = pf[c].fillna(0.0)

    return pf

# ============================================================
# Main
# ============================================================

def seasons_from_args(args) -> List[int]:
    if args.seasons:
        return [int(x.strip()) for x in args.seasons.split(",") if x.strip()]
    if args.start and args.end:
        return list(range(int(args.start), int(args.end) + 1))
    return [int(args.season)]

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--seasons", type=str, default=None)
    args = ap.parse_args()

    _mkdirs()

    seasons = seasons_from_args(args)
    print(f"[make_all] seasons={seasons}")

    # 1) Fetch bundle + addons
    fetch_bundle(seasons)

    # 2) Compose per season, mirror to outputs/metrics and data/
    for s in seasons:
        print(f"[compose] team_form & player_form for {s}")

        tf = compose_team_form(s)
        pf = compose_player_form(s)

        # write to outputs/metrics
        _write_csv(tf, OUT_METRICS / "team_form.csv")
        _write_csv(pf, OUT_METRICS / "player_form.csv")

        # mirror to data/
        _write_csv(tf, DATA_MIRROR / "team_form.csv")
        _write_csv(pf, DATA_MIRROR / "player_form.csv")

    print("✅ make_all completed.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
