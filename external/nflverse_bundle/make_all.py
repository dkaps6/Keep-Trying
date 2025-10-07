#!/usr/bin/env python3
"""
make_all.py — unified data-pack driver + universal fallbacks

What this does:
1) Runs the local fetcher (nflverse bundle) + addons (PROE, roles, RB metrics, etc.)
2) Provides a *universal* fallback mechanism for ANY dataset:
     - Try primary (CSV written by the bundle)
     - If empty/missing → try providers in order (API-Sports, MSF, GSIS, etc.)
     - If still empty → try a compute/proxy function (optional)
3) Composes the two metrics files your model expects and mirrors them to /data.

You control fallbacks by editing the FALLBACKS registry below.
"""

from __future__ import annotations

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd
import numpy as np

# ============================================================
# Basic paths
# ============================================================

ROOT = Path(__file__).resolve().parent            # external/nflverse_bundle
REPO = ROOT.parent.parent
OUT_BUNDLE = ROOT / "outputs"                     # raw bundle outputs
OUT_METRICS = REPO / "outputs" / "metrics"
DATA_MIRROR = REPO / "data"

def _mkdirs():
    for p in (OUT_BUNDLE, OUT_METRICS, DATA_MIRROR):
        p.mkdir(parents=True, exist_ok=True)

# ============================================================
# Provider imports (these are safe; file will still run if missing)
# ============================================================

def _import_or_none(modname: str):
    try:
        return __import__(modname, fromlist=["*"])
    except Exception:
        return None

apis = _import_or_none("scripts.providers.apisports")
msf  = _import_or_none("scripts.providers.msf")
gsis = _import_or_none("scripts.providers.nflgsis")

# ============================================================
# Helpers
# ============================================================

def _ok(df: Optional[pd.DataFrame]) -> bool:
    try:
        return isinstance(df, pd.DataFrame) and not df.empty
    except Exception:
        return False

def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, engine="python")
        except Exception:
            return pd.DataFrame()

def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not _ok(df):
        try:
            (df.head(0) if isinstance(df, pd.DataFrame) else pd.DataFrame()).to_csv(path, index=False)
        except Exception:
            path.write_text("")
        print(f"[write] {path.relative_to(REPO)}  rows=0")
    else:
        df.to_csv(path, index=False)
        print(f"[write] {path.relative_to(REPO)}  rows={len(df)}")

def _run(cmd: List[str]) -> int:
    print(">>", " ".join(cmd))
    return subprocess.call(cmd)

def _z(col: pd.Series) -> pd.Series:
    try:
        arr = col.astype(float).values
        mu, sd = np.nanmean(arr), np.nanstd(arr)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.zeros(len(col)), index=col.index)
        return (col - mu) / sd
    except Exception:
        return pd.Series(np.zeros(len(col)), index=col.index)

# ============================================================
# Universal fallback layer
# ============================================================

# 1) Keys → where the bundle would have written the primary file(s)
PRIMARY_PATHS: Dict[str, Callable[[int], List[Path]]] = {
    "schedules":      lambda s: [OUT_BUNDLE / "schedules" / f"schedules_{s}.csv"],
    "injuries":       lambda s: [OUT_BUNDLE / "injuries" / f"injuries_{s}.csv"],
    "team_stats_reg": lambda s: [OUT_BUNDLE / "team_stats" / f"team_stats_reg_{s}.csv"],
    "team_stats_week":lambda s: [OUT_BUNDLE / "team_stats" / f"team_stats_week_{s}.csv"],
    "player_stats_reg":lambda s:[OUT_BUNDLE / "player_stats" / f"player_stats_reg_{s}.csv"],
    "player_stats_week":lambda s:[OUT_BUNDLE / "player_stats" / f"player_stats_week_{s}.csv"],
    "depth_charts":   lambda s: [OUT_BUNDLE / "depth_charts" / f"depth_charts_{s}.csv"],
    "snap_counts":    lambda s: [OUT_BUNDLE / "snap_counts" / f"snap_counts_{s}.csv"],
    "rosters":        lambda s: [OUT_BUNDLE / "rosters" / f"rosters_{s}.csv"],
    "rosters_weekly": lambda s: [OUT_BUNDLE / "rosters_weekly" / f"rosters_weekly_{s}.csv"],
    "participation":  lambda s: [OUT_BUNDLE / "participation" / f"participation_{s}.csv"],
    "pbp":            lambda s: [OUT_BUNDLE / "pbp" / f"pbp_{s}_{s}.csv"],   # if you fetch multiple seasons, adjust
    "proe_week":      lambda s: [ROOT / "outputs" / "proe" / f"proe_week_{s}.csv"],
    "box_week":       lambda s: [OUT_BUNDLE / "box_counts" / f"defense_box_rates_week_{s}.csv"],
}

# 2) Provider registry: add functions that return a DataFrame or None.
#    I’ve wired concrete ones for schedules + injuries (we already wrote those).
FALLBACKS: Dict[str, List[Callable[[int], Optional[pd.DataFrame]]]] = {
    "schedules": [
        (lambda season: apis.schedules(season)) if apis else None,   # API-Sports
        # add more providers here if you have them
    ],
    "injuries": [
        (lambda season: msf.injuries(season)) if msf else None,      # MySportsFeeds
        (lambda season: apis.injuries(season)) if apis else None,    # API-Sports
        (lambda season: gsis.injuries(season)) if gsis else None,    # your GSIS client (stub by default)
    ],
    # Examples (stubbed: return None until you implement provider functions)
    "rosters": [],
    "rosters_weekly": [],
    "depth_charts": [],
    "snap_counts": [],
    "team_stats_reg": [],
    "team_stats_week": [],
    "player_stats_reg": [],
    "player_stats_week": [],
    "participation": [],  # many libs only up to 2024
    "pbp": [],
    "proe_week": [],   # normally produced by addon derive_proe.py
    "box_week": [],    # normally produced by addon aggregate_box_counts.py
}

# 3) Optional computed/proxy fallbacks if providers fail
def _compute_from_pbp(key: str, season: int, df_map: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Drop-in place to compute simple proxies if providers are empty.
    Examples:
      - team_stats_week from pbp: plays per team/week as a pace proxy
      - player_stats_week from pbp: targets/receptions if your pbp has them
    Keep these conservative; they’re only used as a last resort.
    """
    if key == "team_stats_week" and _ok(df_map.get("pbp")):
        pbp = df_map["pbp"].copy()
        # Try to infer columns; write a minimal weekly plays table
        team_col = "posteam" if "posteam" in pbp.columns else ("offense_team" if "offense_team" in pbp.columns else None)
        week_col = "week" if "week" in pbp.columns else None
        if team_col and week_col:
            out = (
                pbp.groupby([team_col, week_col], as_index=False)
                   .size()
                   .rename(columns={team_col: "team", week_col: "week", "size": "plays"})
            )
            return out
        return pd.DataFrame(columns=["team","week","plays"])

    # Add more computed fallbacks as needed
    return None

def resolve_table(key: str, season: int, cache: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Universal resolver:
      1) Try primary csv(s)
      2) Try providers in order
      3) Try computed/proxy from other tables in cache
    """
    # 1) primary
    for p in PRIMARY_PATHS.get(key, lambda s: [])(season):
        df = _read_csv(p)
        if _ok(df):
            return df

    # 2) providers
    for fn in FALLBACKS.get(key, []):
        if fn is None:
            continue
        try:
            df = fn(season)
        except Exception:
            df = None
        if _ok(df):
            return df

    # 3) computed/proxy (needs other tables resolved first)
    computed = _compute_from_pbp(key, season, cache)
    if _ok(computed):
        return computed

    # final: empty with no rows
    return pd.DataFrame()

# ============================================================
# Fetch bundle + addons
# ============================================================

def fetch_bundle(seasons: List[int]) -> None:
    fetch = ROOT / "fetch_all.py"
    if fetch.exists():
        for s in seasons:
            rc = _run([sys.executable, str(fetch), "--season", str(s)])
            if rc != 0:
                print(f"[warn] fetch_all.py returned {rc} for {s}; continuing")

    # Run addons (safe if a script is missing; we keep going)
    addons = [
        ("addons/derive_proe.py", []),
        ("addons/aggregate_box_counts.py", []),
        ("addons/derive_roles.py", []),
        ("addons/derive_rb_metrics.py", []),
        ("addons/fetch_injuries_espn.py", []),
    ]
    for rel, extra in addons:
        script = ROOT / rel
        if not script.exists():
            continue
        for s in seasons:
            rc = _run([sys.executable, str(script), "--season", str(s), *extra])
            if rc != 0:
                print(f"[warn] {rel} returned {rc} for {s}; continuing")

# ============================================================
# Compose TEAM FORM
# ============================================================

def compose_team_form(season: int, cache: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Columns required by your validator:
      team,
      def_pressure_rate_z, def_pass_epa_z, def_rush_epa_z, def_sack_rate_z,
      pace_z, light_box_rate_z, heavy_box_rate_z, ay_per_att_z,
      plays_est, proe, rz_rate
    """
    # Resolve each ingredient universally
    sched = cache["schedules"] = cache.get("schedules") or resolve_table("schedules", season, cache)
    tsw   = cache["team_stats_week"] = cache.get("team_stats_week") or resolve_table("team_stats_week", season, cache)
    tsr   = cache["team_stats_reg"]  = cache.get("team_stats_reg")  or resolve_table("team_stats_reg",  season, cache)
    box   = cache["box_week"]        = cache.get("box_week")        or resolve_table("box_week",        season, cache)
    proe  = cache["proe_week"]       = cache.get("proe_week")       or resolve_table("proe_week",       season, cache)

    # Team universe
    teams = set()
    for df, col in [(sched, "home_team"), (sched, "away_team"),
                    (tsw, "team"), (tsr, "team"),
                    (box, "team"), (proe, "team")]:
        if _ok(df) and col in df.columns:
            teams.update(df[col].dropna().unique().tolist())
    tf = pd.DataFrame({"team": sorted(teams)}) if teams else pd.DataFrame({"team": []})

    # Defensive metrics (placeholder unless you wire true sources)
    tf["def_pressure_rate"] = 0.0
    tf["def_pass_epa"] = 0.0
    tf["def_rush_epa"] = 0.0
    tf["def_sack_rate"] = 0.0
    if _ok(tsr) and "team" in tsr.columns:
        for raw, out in [
            ("def_pressure_rate", "def_pressure_rate"),
            ("def_pass_epa", "def_pass_epa"),
            ("def_rush_epa", "def_rush_epa"),
            ("def_sack_rate", "def_sack_rate"),
        ]:
            if raw in tsr.columns:
                m = tsr.groupby("team", as_index=False)[raw].mean()
                tf = tf.merge(m.rename(columns={raw: out}), on="team", how="left")

    tf["def_pressure_rate_z"] = _z(tf["def_pressure_rate"].fillna(0.0))
    tf["def_pass_epa_z"]      = _z(tf["def_pass_epa"].fillna(0.0))
    tf["def_rush_epa_z"]      = _z(tf["def_rush_epa"].fillna(0.0))
    tf["def_sack_rate_z"]     = _z(tf["def_sack_rate"].fillna(0.0))

    # Pace & plays (weekly plays proxy)
    tf["pace"] = 0.0
    if _ok(tsw) and {"team","plays"} <= set(tsw.columns):
        pace = tsw.groupby("team", as_index=False)["plays"].mean()
        pace["pace"] = -pace["plays"]
        tf = tf.merge(pace[["team","pace"]], on="team", how="left")
    tf["pace_z"] = _z(tf["pace"].fillna(0.0))
    tf["plays_est"] = tf["pace"].abs().fillna(0.0) * 60.0

    # Box counts
    tf["def_light_box_rate"] = 0.0
    tf["def_heavy_box_rate"] = 0.0
    if _ok(box) and {"team","def_light_box_rate","def_heavy_box_rate"} <= set(box.columns):
        b = box.groupby("team", as_index=False)[["def_light_box_rate","def_heavy_box_rate"]].mean()
        tf = tf.merge(b, on="team", how="left")
    tf["light_box_rate_z"] = _z(tf["def_light_box_rate"].fillna(0.0))
    tf["heavy_box_rate_z"] = _z(tf["def_heavy_box_rate"].fillna(0.0))

    # Air-yards per attempt (proxy until you wire a real source)
    tf["ay_per_att"] = 0.0
    tf["ay_per_att_z"] = _z(tf["ay_per_att"])

    # PROE + RZ rate placeholder
    tf["proe"] = 0.0
    if _ok(proe) and {"team","proe"} <= set(proe.columns):
        p = proe.groupby("team", as_index=False)["proe"].mean()
        tf = tf.merge(p, on="team", how="left", suffixes=("", "_from_proe"))
        tf["proe"] = tf["proe"].fillna(tf.pop("proe_from_proe"))
    tf["rz_rate"] = 0.0

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
# Compose PLAYER FORM
# ============================================================

def compose_player_form(season: int, cache: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Columns required by your validator:
      player, team, position,
      target_share, rush_share, rz_tgt_share, rz_carry_share,
      yprr_proxy, ypc, ypt, qb_ypa
    """
    roles   = _read_csv(ROOT / "outputs" / "roles" / f"roles_{season}.csv")
    rbm     = _read_csv(ROOT / "outputs" / "rb_metrics" / f"rb_metrics_{season}.csv")

    injuries = cache.get("injuries") or resolve_table("injuries", season, cache)
    cache["injuries"] = injuries

    pf = pd.DataFrame(columns=[
        "player","team","position",
        "target_share","rush_share","rz_tgt_share","rz_carry_share",
        "yprr_proxy","ypc","ypt","qb_ypa"
    ])
    if _ok(roles):
        for col, default in [
            ("player",""), ("team",""), ("position",""),
            ("target_share",0.0), ("rush_share",0.0),
            ("rz_tgt_share",0.0), ("rz_carry_share",0.0),
            ("yprr_proxy",0.0), ("ypc",0.0), ("ypt",0.0), ("qb_ypa",0.0),
        ]:
            if col not in roles.columns:
                roles[col] = default
        pf = roles[pf.columns].copy()

    if _ok(rbm) and {"player","team"} <= set(rbm.columns):
        for c in ["success_rate","yac","explosive_rate"]:
            if c not in rbm.columns: rbm[c] = np.nan
        pf = pf.merge(rbm[["player","team","success_rate","yac","explosive_rate"]],
                      on=["player","team"], how="left")

    if _ok(injuries):
        inj = injuries.rename(columns={"status":"inj_status"})
        for c in ["player","team","inj_status"]:
            if c not in inj.columns:
                inj[c] = "" if c != "inj_status" else ""
        pf = pf.merge(inj[["player","team","inj_status"]], on=["player","team"], how="left")

    for c in pf.columns:
        pf[c] = pf[c].fillna("" if pf[c].dtype == object else 0.0)

    return pf

# ============================================================
# Driver
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

    fetch_bundle(seasons)

    for s in seasons:
        print(f"[compose] season {s}")
        cache: Dict[str, pd.DataFrame] = {}

        # Resolve PBP early so proxy functions can use it
        cache["pbp"] = resolve_table("pbp", s, cache)

        tf = compose_team_form(s, cache)
        pf = compose_player_form(s, cache)

        _write_csv(tf, OUT_METRICS / "team_form.csv")
        _write_csv(pf, OUT_METRICS / "player_form.csv")

        _write_csv(tf, DATA_MIRROR / "team_form.csv")
        _write_csv(pf, DATA_MIRROR / "player_form.csv")

    print("✅ make_all completed.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
