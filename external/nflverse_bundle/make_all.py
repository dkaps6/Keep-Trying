#!/usr/bin/env python3
from __future__ import annotations

import argparse, sys, subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
OUT_BUNDLE = ROOT / "outputs"
OUT_METRICS = REPO / "outputs" / "metrics"
DATA_MIRROR = REPO / "data"

def _mkdirs():
    for p in (OUT_BUNDLE, OUT_METRICS, DATA_MIRROR):
        p.mkdir(parents=True, exist_ok=True)

def _import_or_none(modname: str):
    try:
        return __import__(modname, fromlist=["*"])
    except Exception:
        return None

nflverse = _import_or_none("scripts.providers.nflverse")
msf  = _import_or_none("scripts.providers.msf")
apis = _import_or_none("scripts.providers.apisports")
gsis = _import_or_none("scripts.providers.nflgsis")

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

# ---------- primary bundle paths ----------
PRIMARY_PATHS: Dict[str, Callable[[int], List[Path]]] = {
    "schedules":        lambda s: [OUT_BUNDLE / "schedules" / f"schedules_{s}.csv"],
    "injuries":         lambda s: [OUT_BUNDLE / "injuries" / f"injuries_{s}.csv"],
    "team_stats_week":  lambda s: [OUT_BUNDLE / "team_stats" / f"team_stats_week_{s}.csv"],
    "team_stats_reg":   lambda s: [OUT_BUNDLE / "team_stats" / f"team_stats_reg_{s}.csv"],
    "player_stats_week":lambda s: [OUT_BUNDLE / "player_stats" / f"player_stats_week_{s}.csv"],
    "player_stats_reg": lambda s: [OUT_BUNDLE / "player_stats" / f"player_stats_reg_{s}.csv"],
    "depth_charts":     lambda s: [OUT_BUNDLE / "depth_charts" / f"depth_charts_{s}.csv"],
    "snap_counts":      lambda s: [OUT_BUNDLE / "snap_counts" / f"snap_counts_{s}.csv"],
    "rosters":          lambda s: [OUT_BUNDLE / "rosters" / f"rosters_{s}.csv"],
    "rosters_weekly":   lambda s: [OUT_BUNDLE / "rosters_weekly" / f"rosters_weekly_{s}.csv"],
    "participation":    lambda s: [OUT_BUNDLE / "participation" / f"participation_{s}.csv"],
    "pbp":              lambda s: [OUT_BUNDLE / "pbp" / f"pbp_{s}_{s}.csv"],
    "proe_week":        lambda s: [ROOT / "outputs" / "proe" / f"proe_week_{s}.csv"],
    "box_week":         lambda s: [OUT_BUNDLE / "box_counts" / f"defense_box_rates_week_{s}.csv"],
}

# ---------- fallbacks registry (priority: nflverse -> msf -> apis -> gsis) ----------
FALLBACKS: Dict[str, List] = {
    "schedules": [
        (lambda s: nflverse.schedules(s)) if nflverse else None,
        (lambda s: msf.schedules(s)) if msf and hasattr(msf, "schedules") else None,
        (lambda s: apis.schedules(s)) if apis else None,
        (lambda s: gsis.schedules(s)) if gsis else None,
    ],
    "injuries": [
        (lambda s: nflverse.injuries(s)) if nflverse and hasattr(nflverse, "injuries") else None,
        (lambda s: msf.injuries(s)) if msf else None,
        (lambda s: apis.injuries(s)) if apis else None,
        (lambda s: gsis.injuries(s)) if gsis else None,
    ],
    "pbp": [
        (lambda s: nflverse.pbp(s)) if nflverse else None,
    ],
    "team_stats_week": [
        (lambda s: nflverse.team_stats_week(s)) if nflverse else None,
        (lambda s: msf.team_stats_week(s)) if msf else None,
    ],
    "team_stats_reg": [
        (lambda s: nflverse.team_stats_reg(s)) if nflverse else None,
        (lambda s: msf.team_stats_reg(s)) if msf else None,
    ],
    "player_stats_week": [
        (lambda s: nflverse.player_stats_week(s)) if nflverse else None,
        (lambda s: msf.player_stats_week(s)) if msf else None,
    ],
    "player_stats_reg": [
        (lambda s: nflverse.player_stats_reg(s)) if nflverse else None,
        (lambda s: msf.player_stats_reg(s)) if msf else None,
    ],
    "rosters": [
        (lambda s: nflverse.rosters(s)) if nflverse else None,
        (lambda s: msf.rosters(s)) if msf and hasattr(msf, "rosters") else None,
    ],
    "rosters_weekly": [
        (lambda s: nflverse.rosters_weekly(s)) if nflverse else None,
    ],
    "depth_charts": [
        (lambda s: nflverse.depth_charts(s)) if nflverse else None,
        (lambda s: gsis.depth_charts(s)) if gsis else None,
    ],
    "snap_counts": [
        (lambda s: nflverse.snap_counts(s)) if nflverse else None,
    ],
    "participation": [
        (lambda s: nflverse.participation(s)) if nflverse else None,
    ],
    "proe_week": [],
    "box_week":  [],
}

def _compute_proxy(key: str, season: int, cache: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    if key == "team_stats_week" and _ok(cache.get("pbp")):
        pbp = cache["pbp"]
        tcol = "posteam" if "posteam" in pbp.columns else ("offense_team" if "offense_team" in pbp.columns else None)
        if tcol and "week" in pbp.columns:
            out = (pbp.groupby([tcol, "week"], as_index=False).size()
                      .rename(columns={tcol: "team", "size": "plays"}))
            return out
    if key == "team_stats_reg":
        wk = cache.get("team_stats_week")
        if _ok(wk):
            return wk.groupby(["team"], as_index=False).sum(numeric_only=True)
    if key == "player_stats_reg":
        wk = cache.get("player_stats_week")
        if _ok(wk):
            keys = [c for c in ["team","player","position"] if c in wk.columns]
            return wk.groupby(keys, as_index=False).sum(numeric_only=True)
    return None

def resolve_table(key: str, season: int, cache: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    # 1) primary
    for p in PRIMARY_PATHS.get(key, lambda s: [])(season):
        df = _read_csv(p)
        if _ok(df): return df
    # 2) providers
    for fn in FALLBACKS.get(key, []):
        if fn is None: continue
        try:
            df = fn(season)
        except Exception:
            df = None
        if _ok(df): return df
    # 3) computed
    df = _compute_proxy(key, season, cache)
    if _ok(df): return df
    return pd.DataFrame()

def fetch_bundle(seasons: List[int]) -> None:
    fetch = ROOT / "fetch_all.py"
    if fetch.exists():
        for s in seasons:
            rc = _run([sys.executable, str(fetch), "--season", str(s)])
            if rc != 0:
                print(f"[warn] fetch_all.py returned {rc} for {s}; continuing")
    addons = [
        ("addons/derive_proe.py", []),
        ("addons/aggregate_box_counts.py", []),
        ("addons/derive_roles.py", []),
        ("addons/derive_rb_metrics.py", []),
        ("addons/fetch_injuries_espn.py", []),
    ]
    for rel, extra in addons:
        scr = ROOT / rel
        if not scr.exists(): continue
        for s in seasons:
            rc = _run([sys.executable, str(scr), "--season", str(s), *extra])
            if rc != 0:
                print(f"[warn] {rel} returned {rc} for {s}; continuing")

def compose_team_form(season: int, cache: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    sched = cache.get("schedules") or resolve_table("schedules", season, cache); cache["schedules"] = sched
    tsw   = cache.get("team_stats_week") or resolve_table("team_stats_week", season, cache); cache["team_stats_week"] = tsw
    tsr   = cache.get("team_stats_reg")  or resolve_table("team_stats_reg",  season, cache); cache["team_stats_reg"]  = tsr
    box   = cache.get("box_week")        or resolve_table("box_week",        season, cache); cache["box_week"]        = box
    proe  = cache.get("proe_week")       or resolve_table("proe_week",       season, cache); cache["proe_week"]       = proe

    teams = set()
    for df, col in [(sched,"home_team"), (sched,"away_team"), (tsw,"team"), (tsr,"team"), (box,"team"), (proe,"team")]:
        if _ok(df) and col in df.columns:
            teams.update(df[col].dropna().unique().tolist())
    tf = pd.DataFrame({"team": sorted(teams)}) if teams else pd.DataFrame({"team": []})

    tf["def_pressure_rate"] = 0.0
    tf["def_pass_epa"] = 0.0
    tf["def_rush_epa"] = 0.0
    tf["def_sack_rate"] = 0.0
    if _ok(tsr) and "team" in tsr.columns:
        for raw,out in [("def_pressure_rate","def_pressure_rate"),
                        ("def_pass_epa","def_pass_epa"),
                        ("def_rush_epa","def_rush_epa"),
                        ("def_sack_rate","def_sack_rate")]:
            if raw in tsr.columns:
                m = tsr.groupby("team", as_index=False)[raw].mean()
                tf = tf.merge(m.rename(columns={raw:out}), on="team", how="left")

    tf["def_pressure_rate_z"] = _z(tf["def_pressure_rate"].fillna(0.0))
    tf["def_pass_epa_z"]      = _z(tf["def_pass_epa"].fillna(0.0))
    tf["def_rush_epa_z"]      = _z(tf["def_rush_epa"].fillna(0.0))
    tf["def_sack_rate_z"]     = _z(tf["def_sack_rate"].fillna(0.0))

    tf["pace"] = 0.0
    if _ok(tsw) and {"team","plays"} <= set(tsw.columns):
        pace = tsw.groupby("team", as_index=False)["plays"].mean()
        pace["pace"] = -pace["plays"]
        tf = tf.merge(pace[["team","pace"]], on="team", how="left")
    tf["pace_z"]   = _z(tf["pace"].fillna(0.0))
    tf["plays_est"] = tf["pace"].abs().fillna(0.0) * 60.0

    tf["def_light_box_rate"] = 0.0
    tf["def_heavy_box_rate"] = 0.0
    if _ok(box) and {"team","def_light_box_rate","def_heavy_box_rate"} <= set(box.columns):
        b = box.groupby("team", as_index=False)[["def_light_box_rate","def_heavy_box_rate"]].mean()
        tf = tf.merge(b, on="team", how="left")
    tf["light_box_rate_z"] = _z(tf["def_light_box_rate"].fillna(0.0))
    tf["heavy_box_rate_z"] = _z(tf["def_heavy_box_rate"].fillna(0.0))

    tf["ay_per_att"] = 0.0
    tf["ay_per_att_z"] = _z(tf["ay_per_att"])

    tf["proe"] = 0.0
    if _ok(proe) and {"team","proe"} <= set(proe.columns):
        p = proe.groupby("team", as_index=False)["proe"].mean()
        tf = tf.merge(p, on="team", how="left", suffixes=("", "_from_proe"))
        tf["proe"] = tf["proe"].fillna(tf.pop("proe_from_proe"))
    tf["rz_rate"] = 0.0

    cols = ["team",
            "def_pressure_rate_z","def_pass_epa_z","def_rush_epa_z","def_sack_rate_z",
            "pace_z","light_box_rate_z","heavy_box_rate_z","ay_per_att_z",
            "plays_est","proe","rz_rate"]
    for c in cols:
        if c not in tf.columns:
            tf[c] = 0.0 if c != "team" else ""
    return tf[cols]

def compose_player_form(season: int, cache: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    roles = _read_csv(ROOT / "outputs" / "roles" / f"roles_{season}.csv")
    rbm   = _read_csv(ROOT / "outputs" / "rb_metrics" / f"rb_metrics_{season}.csv")
    inj   = cache.get("injuries") or resolve_table("injuries", season, cache); cache["injuries"] = inj

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
            if col not in roles.columns: roles[col] = default
        pf = roles[pf.columns].copy()

    if _ok(rbm) and {"player","team"} <= set(rbm.columns):
        for c in ["success_rate","yac","explosive_rate"]:
            if c not in rbm.columns: rbm[c] = np.nan
        pf = pf.merge(rbm[["player","team","success_rate","yac","explosive_rate"]],
                      on=["player","team"], how="left")

    if _ok(inj):
        inj2 = inj.rename(columns={"status":"inj_status"})
        for c in ["player","team","inj_status"]:
            if c not in inj2.columns: inj2[c] = "" if c != "inj_status" else ""
        pf = pf.merge(inj2[["player","team","inj_status"]], on=["player","team"], how="left")

    for c in pf.columns:
        pf[c] = pf[c].fillna("" if pf[c].dtype == object else 0.0)
    return pf

def seasons_from_args(args) -> List[int]:
    if args.seasons:
        return [int(x.strip()) for x in args.seasons.split(",") if x.strip()]
    if args.start and args.end:
        return list(range(int(args.start), int(args.end) + 1))
    return [int(args.season)]

def fetch_bundle(seasons: List[int]) -> None:
    fetch = ROOT / "fetch_all.py"
    if fetch.exists():
        for s in seasons:
            rc = _run([sys.executable, str(fetch), "--season", str(s)])
            if rc != 0:
                print(f"[warn] fetch_all.py returned {rc} for {s}; continuing")
    addons = [
        ("addons/derive_proe.py", []),
        ("addons/aggregate_box_counts.py", []),
        ("addons/derive_roles.py", []),
        ("addons/derive_rb_metrics.py", []),
        ("addons/fetch_injuries_espn.py", []),
    ]
    for rel, extra in addons:
        scr = ROOT / rel
        if not scr.exists(): continue
        for s in seasons:
            rc = _run([sys.executable, str(scr), "--season", str(s), *extra])
            if rc != 0:
                print(f"[warn] {rel} returned {rc} for {s}; continuing")

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
        cache["pbp"] = resolve_table("pbp", s, cache)  # early for proxies

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
