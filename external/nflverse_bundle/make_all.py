#!/usr/bin/env python3
from __future__ import annotations

import argparse, sys, subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import math

# ----------------------------
# Paths
# ----------------------------
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
OUT_BUNDLE = ROOT / "outputs"
OUT_METRICS = REPO / "outputs" / "metrics"
DATA_MIRROR = REPO / "data"

# ----------------------------
# Dirs
# ----------------------------
def _mkdirs():
    for p in (OUT_BUNDLE, OUT_METRICS, DATA_MIRROR):
        p.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Strict data quality helpers
# ----------------------------
class DataQualityError(RuntimeError):
    pass

def require_nonempty(df: Optional[pd.DataFrame], name: str, strict: bool, issues: List[str]):
    ok = isinstance(df, pd.DataFrame) and not df.empty
    if ok:
        return
    msg = f"[data] required table '{name}' is empty"
    if strict:
        raise DataQualityError(msg)
    issues.append(msg)

def require_cols(df: pd.DataFrame, needed: List[str], where: str, strict: bool, issues: List[str]):
    miss = [c for c in needed if c and c not in df.columns]
    if not miss:
        return
    msg = f"[data] missing columns in {where}: {miss}"
    if strict:
        raise DataQualityError(msg)
    issues.append(msg)

# ----------------------------
# Dynamic imports for providers
# ----------------------------
def _import_or_none(modname: str):
    try:
        return __import__(modname, fromlist=["*"])
    except Exception:
        return None

nflverse = _import_or_none("scripts.providers.nflverse")
msf  = _import_or_none("scripts.providers.msf")
apis = _import_or_none("scripts.providers.apisports")
gsis = _import_or_none("scripts.providers.nflgsis")
espn = _import_or_none("scripts.providers.espn_pbp")

# ----------------------------
# IO helpers
# ----------------------------
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
        arr = pd.to_numeric(col, errors="coerce").values
        mu, sd = np.nanmean(arr), np.nanstd(arr)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.zeros(len(col)), index=col.index)
        return (col - mu) / sd
    except Exception:
        return pd.Series(np.zeros(len(col)), index=col.index)

# ----------------------------
# Safe resolver
# ----------------------------
def _get_or_resolve(name: str, season: int, cache: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = cache.get(name, None)
    if df is None or (hasattr(df, "empty") and df.empty):
        df = resolve_table(name, season, cache)
    cache[name] = df
    return df

# ----------------------------
# Primary bundle paths
# ----------------------------
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
    # expanded to find your artifacts:
    "proe_week":        lambda s: [
        ROOT / "outputs" / "proe" / f"proe_week_{s}.csv",
        ROOT / "outputs" / "proe" / f"team_proe_week_{s}.csv",
        ROOT / "outputs" / "proe" / f"team_proe_season_{s}.csv",
    ],
    "box_week":         lambda s: [OUT_BUNDLE / "box_counts" / f"defense_box_rates_week_{s}.csv"],
}

# ----------------------------
# Fallbacks (nflverse -> msf -> apisports -> gsis)
# ----------------------------
FALLBACKS: Dict[str, List] = {
    "schedules": [
        (lambda s: nflverse.schedules(s)) if nflverse else None,
        (lambda s: msf.schedules(s)) if msf and hasattr(msf, "schedules") else None,
        (lambda s: apis.schedules(s)) if apis and hasattr(apis, "schedules") else None,
        (lambda s: gsis.schedules(s)) if gsis and hasattr(gsis, "schedules") else None,
    ],
    "injuries": [
        (lambda s: nflverse.injuries(s)) if nflverse and hasattr(nflverse, "injuries") else None,
        (lambda s: msf.injuries(s)) if msf and hasattr(msf, "injuries") else None,
        (lambda s: apis.injuries(s)) if apis and hasattr(apis, "injuries") else None,
        (lambda s: gsis.injuries(s)) if gsis and hasattr(gsis, "injuries") else None,
    ],
    "pbp": [
        (lambda s: nflverse.pbp(s)) if nflverse else None,
        (lambda s: msf.pbp(s)) if msf and hasattr(msf, "pbp") else None,
        (lambda s: apis.pbp(s)) if apis and hasattr(apis, "pbp") else None,
        (lambda s: gsis.pbp(s)) if gsis and hasattr(gsis, "pbp") else None,
        (lambda s: espn.pbp(s)) if espn else None,
    ],
    "team_stats_week": [
        (lambda s: nflverse.team_stats_week(s)) if nflverse and hasattr(nflverse, "team_stats_week") else None,
        (lambda s: msf.team_stats_week(s)) if msf and hasattr(msf, "team_stats_week") else None,
        (lambda s: apis.team_stats_week(s)) if apis and hasattr(apis, "team_stats_week") else None,
        (lambda s: gsis.team_stats_week(s)) if gsis and hasattr(gsis, "team_stats_week") else None,
    ],
    "team_stats_reg": [
        (lambda s: nflverse.team_stats_reg(s)) if nflverse and hasattr(nflverse, "team_stats_reg") else None,
        (lambda s: msf.team_stats_reg(s)) if msf and hasattr(msf, "team_stats_reg") else None,
        (lambda s: apis.team_stats_reg(s)) if apis and hasattr(apis, "team_stats_reg") else None,
        (lambda s: gsis.team_stats_reg(s)) if gsis and hasattr(gsis, "team_stats_reg") else None,
    ],
    "player_stats_week": [
        (lambda s: nflverse.player_stats_week(s)) if nflverse and hasattr(nflverse, "player_stats_week") else None,
        (lambda s: msf.player_stats_week(s)) if msf and hasattr(msf, "player_stats_week") else None,
        (lambda s: apis.player_stats_week(s)) if apis and hasattr(apis, "player_stats_week") else None,
        (lambda s: gsis.player_stats_week(s)) if gsis and hasattr(gsis, "player_stats_week") else None,
    ],
    "player_stats_reg": [
        (lambda s: nflverse.player_stats_reg(s)) if nflverse and hasattr(nflverse, "player_stats_reg") else None,
        (lambda s: msf.player_stats_reg(s)) if msf and hasattr(msf, "player_stats_reg") else None,
        (lambda s: apis.player_stats_reg(s)) if apis and hasattr(apis, "player_stats_reg") else None,
        (lambda s: gsis.player_stats_reg(s)) if gsis and hasattr(gsis, "player_stats_reg") else None,
    ],
    "rosters": [
        (lambda s: nflverse.rosters(s)) if nflverse and hasattr(nflverse, "rosters") else None,
        (lambda s: msf.rosters(s)) if msf and hasattr(msf, "rosters") else None,
        (lambda s: apis.rosters(s)) if apis and hasattr(apis, "rosters") else None,
        (lambda s: gsis.rosters(s)) if gsis and hasattr(gsis, "rosters") else None,
    ],
    "rosters_weekly": [
        (lambda s: nflverse.rosters_weekly(s)) if nflverse and hasattr(nflverse, "rosters_weekly") else None,
        (lambda s: msf.rosters_weekly(s)) if msf and hasattr(msf, "rosters_weekly") else None,
    ],
    "depth_charts": [
        (lambda s: nflverse.depth_charts(s)) if nflverse and hasattr(nflverse, "depth_charts") else None,
        (lambda s: msf.depth_charts(s)) if msf and hasattr(msf, "depth_charts") else None,
        (lambda s: gsis.depth_charts(s)) if gsis and hasattr(gsis, "depth_charts") else None,
    ],
    "snap_counts": [
        (lambda s: nflverse.snap_counts(s)) if nflverse and hasattr(nflverse, "snap_counts") else None,
        (lambda s: msf.snap_counts(s)) if msf and hasattr(msf, "snap_counts") else None,
    ],
    "participation": [
        (lambda s: nflverse.participation(s)) if nflverse and hasattr(nflverse, "participation") else None,
        (lambda s: msf.participation(s)) if msf and hasattr(msf, "participation") else None,
    ],
    "proe_week": [],  # derived addon (kept empty; you already have an addon for this)
    "box_week":  [],  # derived addon (kept empty)
}

# ----------------------------
# Computed / last-resort
# ----------------------------
def _compute_proxy(key: str, season: int, cache: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    if key == "team_stats_week" and _ok(cache.get("pbp")):
        pbp = cache["pbp"]
        tcol = "posteam" if "posteam" in pbp.columns else ("offense_team" if "offense_team" in pbp.columns else None)
        if tcol and "week" in pbp.columns:
            out = (
                pbp.loc[pbp[tcol].notna()]
                   .groupby([tcol, "week"], as_index=False)
                   .size()
                   .rename(columns={tcol: "team", "size": "plays"})
            )
            return out
    if key == "team_stats_reg":
        wk = cache.get("team_stats_week")
        if _ok(wk):
            gcols = [c for c in ["team"] if c in wk.columns]
            return wk.groupby(gcols, as_index=False).sum(numeric_only=True)
    if key == "player_stats_reg":
        wk = cache.get("player_stats_week")
        if _ok(wk):
            keys = [c for c in ["team","player","position"] if c in wk.columns]
            return wk.groupby(keys, as_index=False).sum(numeric_only=True)
    return None

def _normalize_pbp_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Map various provider column names → the canonical names our composers expect."""
    if not _ok(df):
        return df

    col_map = {}
    # teams
    if "posteam" not in df.columns:
        for a in ["offense_team", "team_offense", "poss_team", "possession_team"]:
            if a in df.columns: col_map[a] = "posteam"; break
    if "defteam" not in df.columns:
        for a in ["defense_team", "team_defense"]:
            if a in df.columns: col_map[a] = "defteam"; break

    # ids / week
    if "game_id" not in df.columns:
        for a in ["old_game_id","gameid","gameId","gameId_str","game_identifier"]:
            if a in df.columns: col_map[a] = "game_id"; break
    if "week" not in df.columns:
        for a in ["game_week","week_number","wk"]:
            if a in df.columns: col_map[a] = "week"; break

    # receiver/rusher/passer
    if "receiver_player_name" not in df.columns:
        for a in ["receiver","receiver_name","receiver_full_name","target_player_name"]:
            if a in df.columns: col_map[a] = "receiver_player_name"; break
    if "rusher_player_name" not in df.columns:
        for a in ["rusher","rusher_name","runner_name"]:
            if a in df.columns: col_map[a] = "rusher_player_name"; break
    if "passer_player_name" not in df.columns:
        for a in ["passer","passer_name","qb_player_name","quarterback_name"]:
            if a in df.columns: col_map[a] = "passer_player_name"; break

    # yardline / yards
    if "yardline_100" not in df.columns:
        for a in ["yardline","yard_line_100","yards_to_goal","yardlineNumber"]:
            if a in df.columns: col_map[a] = "yardline_100"; break
    if "yards_gained" not in df.columns and "yards" in df.columns:
        col_map["yards"] = "yards_gained"

    # pass/rush flags
    if "pass" not in df.columns:
        for a in ["is_pass","qb_dropback","pass_flag"]:
            if a in df.columns: col_map[a] = "pass"; break
    if "rush" not in df.columns:
        for a in ["is_rush","rush_flag"]:
            if a in df.columns: col_map[a] = "rush"; break

    if col_map:
        df = df.rename(columns=col_map)

    # types
    for c in ["week","down"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ["yardline_100","yards_gained","ydstogo"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")

    # Best-effort defteam inference if missing: "other team in same game"
    if "defteam" in df.columns and df["defteam"].isna().all() and {"game_id","posteam"}.issubset(df.columns):
        # per game, collect teams appeared as offense; infer defense as the other team seen in that game
        try:
            teams_by_game = (df[["game_id","posteam"]].dropna().groupby("game_id")["posteam"]
                              .agg(lambda x: list(pd.unique(x))).to_dict())
            def infer_def(row):
                plist = teams_by_game.get(row["game_id"], [])
                if len(plist) == 2 and row["posteam"] in plist:
                    return plist[1] if plist[0] == row["posteam"] else plist[0]
                return np.nan
            df["defteam"] = df.apply(infer_def, axis=1)
        except Exception:
            pass

    return df

# --- helpers ---
def _read_csv_soft(path: Path) -> pd.DataFrame:
    try:
        if path.exists() and path.stat().st_size > 0:
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()

def _decimal_to_impl(p):
    # Odds API decimals: implied prob = 1 / price
    try:
        p = float(p)
        return 1.0 / p if p and p > 1e-9 else None
    except Exception:
        return None

def _de_vig(impl_series: pd.Series) -> pd.Series:
    # normalize so each event/market’s outcomes sum to 1.0
    s = impl_series.astype(float)
    tot = s.groupby(level=0).transform("sum")  # assumes MultiIndex with grouping key at level 0
    return s / tot

def _alias(df: pd.DataFrame, mapping: Dict[str, List[str]]) -> pd.DataFrame:
    if not _ok(df): return df
    ren = {}
    for canon, alts in mapping.items():
        if canon in df.columns: continue
        for a in alts:
            if a in df.columns:
                ren[a] = canon
                break
    return df.rename(columns=ren)

def _ensure_types(df: pd.DataFrame, ints: List[str]=None, nums: List[str]=None) -> pd.DataFrame:
    ints = ints or []; nums = nums or []
    for c in ints:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in nums:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _require(df: pd.DataFrame, cols: List[str], key: str) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"[normalize:{key}] Missing required columns: {missing}")
    return df

def _normalize_pbp_cols(df: pd.DataFrame) -> pd.DataFrame:
    if not _ok(df): return df
    df = _alias(df, {
        "posteam": ["offense_team","team_offense","poss_team","possession_team"],
        "defteam": ["defense_team","team_defense"],
        "game_id": ["old_game_id","gameid","gameId","game_identifier"],
        "week":    ["game_week","week_number","wk"],
        "receiver_player_name": ["receiver","receiver_name","receiver_full_name","target_player_name","targeted_receiver","target_name"],
        "rusher_player_name":   ["rusher","rusher_name","runner_name"],
        "passer_player_name":   ["passer","passer_name","qb_player_name","quarterback_name"],
        "yardline_100":         ["yardline","yard_line_100","yards_to_goal","yardlineNumber","yardline_200"],
        "yards_gained":         ["yards","gained_yards"],
        "pass":                 ["is_pass","qb_dropback","pass_flag"],
        "rush":                 ["is_rush","rush_flag"],
        "down":                 ["downs","down_number"],
        "ydstogo":              ["yards_to_go","yds_to_go","distance"],
    })
    df = _ensure_types(df, ints=["week","down"], nums=["yardline_100","yards_gained","ydstogo"])

    # best-effort defteam inference if truly missing
    if "defteam" in df.columns and df["defteam"].isna().all() and {"game_id","posteam"}.issubset(df.columns):
        try:
            pairs = (df[["game_id","posteam"]].dropna()
                     .groupby("game_id")["posteam"].agg(lambda x: list(pd.unique(x))).to_dict())
            df["defteam"] = df.apply(
                lambda r: (pairs.get(r["game_id"], [None,None])[1]
                           if pairs.get(r["game_id"], [None,None])[0] == r["posteam"]
                           else pairs.get(r["game_id"], [None,None])[0]),
                axis=1
            )
        except Exception:
            pass

    # Do **not** hard-require yardline/receiver here; composers handle strict vs. non-strict.
    return df

def _normalize_schedules(df: pd.DataFrame) -> pd.DataFrame:
    df = _alias(df, {
        "game_id": ["old_game_id","gameid","gid","gameId"],
        "week":    ["game_week","wk"],
        "home_team": ["home","homeTeam","team_home"],
        "away_team": ["away","awayTeam","team_away"],
        "start_time": ["start","game_start","start_date_time","game_datetime"],
        "event_id": ["eid","event","match_id"],
    })
    return _require(df, ["game_id","week","home_team","away_team"], "schedules")

def _normalize_injuries(df: pd.DataFrame) -> pd.DataFrame:
    df = _alias(df, {
        "player": ["player_name","athlete","name"],
        "team":   ["team_abbr","team_name","club"],
        "status": ["injury_status","game_status","designation"],
    })
    return _require(df, ["player","team","status"], "injuries")

def _normalize_rosters(df: pd.DataFrame) -> pd.DataFrame:
    df = _alias(df, {
        "player":   ["player_name","name"],
        "team":     ["team_abbr","team_name","club"],
        "position": ["pos","position_group"],
    })
    return _require(df, ["player","team"], "rosters")

def _normalize_rosters_weekly(df: pd.DataFrame) -> pd.DataFrame:
    df = _alias(df, {
        "player": ["player_name","name"],
        "team":   ["team_abbr","team_name","club"],
        "week":   ["wk","game_week"],
    })
    return _require(df, ["player","team","week"], "rosters_weekly")

def _normalize_depth(df: pd.DataFrame) -> pd.DataFrame:
    df = _alias(df, {
        "player": ["player_name","name"],
        "team":   ["team_abbr","team_name","club"],
        "position": ["pos","position_group"],
        "depth": ["depth_order","chart_pos","slot"],
    })
    return _require(df, ["player","team","position"], "depth_charts")

def _normalize_snaps(df: pd.DataFrame) -> pd.DataFrame:
    df = _alias(df, {
        "player": ["player_name","name"],
        "team":   ["team_abbr","team_name","club"],
        "week":   ["wk","game_week"],
        "offense_snaps": ["off_snaps","offensive_snaps","snaps_offense"],
        "routes": ["routes_run","route_runs"],
    })
    return _require(df, ["player","team","week"], "snap_counts")

def _normalize_participation(df: pd.DataFrame) -> pd.DataFrame:
    df = _alias(df, {
        "player": ["player_name","name"],
        "team":   ["team_abbr","team_name","club"],
        "week":   ["wk","game_week"],
        "on_field": ["participation","active","played"],
    })
    return _require(df, ["player","team","week"], "participation")

def _normalize_team_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = _alias(df, {"team":["team_abbr","club","abbr"]})
    return _require(df, ["team"], "team_stats")

def _normalize_player_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = _alias(df, {
        "player":["player_name","name"],
        "team":  ["team_abbr","club","abbr"],
        "position":["pos","position_group"]
    })
    return _require(df, ["player","team"], "player_stats")

def _normalize_box_week(df: pd.DataFrame) -> pd.DataFrame:
    df = _alias(df, {
        "team": ["defense_team","team_abbr","club"],
        "def_light_box_rate": ["light_box_rate","def_light_box_pct"],
        "def_heavy_box_rate": ["heavy_box_rate","def_heavy_box_pct"],
        "week": ["wk","game_week"],
    })
    return _require(df, ["team","week"], "box_week")

def _normalize_proe_week(df: pd.DataFrame) -> pd.DataFrame:
    df = _alias(df, {"team":["team_abbr","club"], "week":["wk","game_week"]})
    return _require(df, ["team","week","proe"], "proe_week")

# registry
NORMALIZERS: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "pbp": _normalize_pbp_cols,
    "schedules": _normalize_schedules,
    "injuries": _normalize_injuries,
    "rosters": _normalize_rosters,
    "rosters_weekly": _normalize_rosters_weekly,
    "depth_charts": _normalize_depth,
    "snap_counts": _normalize_snaps,
    "participation": _normalize_participation,
    "team_stats_week": _normalize_team_stats,
    "team_stats_reg":  _normalize_team_stats,
    "player_stats_week": _normalize_player_stats,
    "player_stats_reg":  _normalize_player_stats,
    "box_week": _normalize_box_week,
    "proe_week": _normalize_proe_week,
}

def resolve_table(key: str, season: int, cache: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    norm = NORMALIZERS.get(key, lambda d: d)

    # 1) primary cache/files
    for p in PRIMARY_PATHS.get(key, lambda s: [])(season):
        df = _read_csv(p)
        if _ok(df):
            return norm(df)

    # 2) providers (fallback chain)
    for fn in FALLBACKS.get(key, []):
        if fn is None: continue
        try:
            df = fn(season)
        except Exception:
            df = None
        if _ok(df):
            return norm(df)

    # 3) computed last-resort
    df = _compute_proxy(key, season, cache)
    if _ok(df):
        return norm(df)

    return pd.DataFrame()

# ============================
# === DERIVED METRICS LAYER ==
# ============================
def _pick(colnames: List[str], df: pd.DataFrame) -> Optional[str]:
    for c in colnames:
        if c in df.columns:
            return c
    return None

def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _bool_col(df: pd.DataFrame, names: List[str]) -> pd.Series:
    c = _pick(names, df)
    if c is None: return pd.Series(False, index=df.index)
    v = df[c]
    if v.dtype == bool: return v.fillna(False)
    return _safe_num(v).fillna(0) > 0

def _name_col(df: pd.DataFrame, names: List[str], default: str="") -> pd.Series:
    c = _pick(names, df)
    return df[c].astype(str).fillna(default) if c else pd.Series(default, index=df.index)

# ----------------------------
# Team derivations
# ----------------------------
def derive_team_from_pbp(pbp: pd.DataFrame, strict: bool, issues: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      team_env  : team-level env metrics (EPA splits, pace, plays_est, ay_per_att, proe, rz_rate)
      proe_week : optional weekly proe table (team, week, proe)
    """
    if not _ok(pbp):
        if strict:
            raise DataQualityError("[derive_team_from_pbp] pbp is empty")
        return pd.DataFrame(), pd.DataFrame()

    # drop duplicate-named columns (e.g., duplicated 'game_id')
    if pbp.columns.duplicated().any():
        pbp = pbp.loc[:, ~pbp.columns.duplicated()].copy()

    # Column normalization (nflfastR style preferred)
    posteam = _pick(["posteam","offense_team"], pbp) or "posteam"
    defteam = _pick(["defteam","defense_team"], pbp) or "defteam"
    weekcol = _pick(["week","game_week"], pbp) or "week"
    game_id = _pick(["game_id","old_game_id","gameid","gameId"], pbp) or "game_id"
    require_cols(pbp, [posteam, defteam, weekcol, game_id], "pbp base", strict, issues)

    epa = _pick(["epa"], pbp)
    pass_flag = _bool_col(pbp, ["pass","is_pass","qb_dropback"])
    rush_flag = _bool_col(pbp, ["rush","is_rush"]) & ~pass_flag

    # EPA splits (defense is negative of offense EPA allowed)
    teams = pd.unique(pd.concat([pbp[posteam], pbp[defteam]], ignore_index=True)).astype(str)
    env = pd.DataFrame({"team": sorted([t for t in teams if t and t != "nan"])})

    env["def_pass_epa"] = 0.0
    env["def_rush_epa"] = 0.0
    if epa:
        off_pass = (
            pbp.loc[pass_flag & pbp[posteam].notna(), [posteam, epa]]
               .groupby(posteam, as_index=False)[epa].mean()
               .rename(columns={posteam: "team", epa: "off_pass_epa"})
        )
        off_rush = (
            pbp.loc[rush_flag & pbp[posteam].notna(), [posteam, epa]]
               .groupby(posteam, as_index=False)[epa].mean()
               .rename(columns={posteam: "team", epa: "off_rush_epa"})
        )
        env = env.merge(off_pass, on="team", how="left").merge(off_rush, on="team", how="left")
        env["def_pass_epa"] = -env["off_pass_epa"].fillna(0.0)
        env["def_rush_epa"] = -env["off_rush_epa"].fillna(0.0)

    # aDOT / air_yards per attempt
    air = _pick(["air_yards"], pbp)
    attempts = (
        pbp.loc[pbp[posteam].notna(), [posteam]]
           .assign(att=pass_flag.astype(int))
           .groupby(posteam, as_index=False)["att"].sum()
           .rename(columns={posteam:"team"})
    )
    if air:
        air_sum = (
            pbp.loc[pass_flag & pbp[posteam].notna(), [posteam, air]]
               .groupby(posteam, as_index=False)[air].sum()
               .rename(columns={posteam:"team", air:"air_sum"})
        )
        env = env.merge(attempts, on="team", how="left").merge(air_sum, on="team", how="left")
        env["ay_per_att"] = env["air_sum"].fillna(0.0) / env["att"].replace(0, np.nan)
    else:
        env["ay_per_att"] = np.nan

    # Neutral pace (sec/play in neutral score)
    sec_rem = _pick(["game_seconds_remaining","game_seconds"], pbp)
    score_diff = _pick(["score_differential","score_diff"], pbp)
    qtr = _pick(["qtr","quarter"], pbp)
    env["pace"] = 0.0
    if sec_rem and score_diff:
        p = pbp.loc[pbp[posteam].notna() & (pbp[score_diff].between(-7, 7, inclusive="both"))].copy()
        order_cols = [c for c in ["game_id","old_game_id","drive","play_id","index"] if c in p.columns]
        order_cols = list(dict.fromkeys(order_cols))
        if order_cols:
            p = p.sort_values(order_cols)
        p["sec"] = _safe_num(p[sec_rem])
        p["delta"] = -p.groupby([posteam])["sec"].diff().fillna(np.nan)
        pace = p.groupby(posteam, as_index=False)["delta"].median()
        pace = pace.rename(columns={posteam:"team","delta":"sec_per_play_neutral"})
        env = env.merge(pace, on="team", how="left")
        env["pace"] = env["sec_per_play_neutral"].fillna(env["sec_per_play_neutral"].median())
        env["pace"] = -env["pace"].fillna(28.0)  # default ~28 sec/play

    # PROE (league expectation by down/dist/field/score/quarter/time)
    down = _pick(["down"], pbp)
    ytg  = _pick(["ydstogo","yards_to_go","yds_to_go"], pbp)
    yline= _pick(["yardline_100","yardline"], pbp)
    time = sec_rem
    proe_week = pd.DataFrame(columns=["team","week","proe"])
    env["proe"] = 0.0

    if strict:
        require_cols(pbp, [c for c in [down, ytg, yline, weekcol] if c], "pbp for PROE", strict, issues)

    if down and ytg and yline and weekcol:
        df = pbp.loc[pbp[posteam].notna() & pbp[down].notna(), [posteam, weekcol, down, ytg, yline]].copy()
        df["is_pass"] = pass_flag.loc[df.index].astype(int)

        # bins
        df["down_b"]   = df[down].clip(1,4)
        df["ytg_b"]    = pd.cut(_safe_num(df[ytg]), bins=[-1,2,6,10,25,99], labels=["short","med","long","xlong","hail"])
        df["yl_b"]     = pd.cut(_safe_num(df[yline]), bins=[-1,20,50,80,110], labels=["rz","mid","deep","backed"])

        # add score bins, quarter bins, time bins
        if score_diff in pbp.columns:
            df2 = pbp.loc[df.index, [score_diff]].copy()
            df["score_b"] = pd.cut(_safe_num(df2[score_diff]), bins=[-99,-14,-7,-3,3,7,14,99],
                                   labels=["<-14","-14..-7","-7..-3","-3..3","3..7","7..14",">14"])
        else:
            df["score_b"] = "0"

        if qtr:
            qmap = pd.Series(pbp[qtr], index=pbp.index).reindex(df.index)
            df["qtr_b"] = qmap.clip(1,4).fillna(1).astype(int)
        else:
            df["qtr_b"] = 1

        if time:
            tmap = pd.Series(_safe_num(pbp[time]), index=pbp.index).reindex(df.index)
            df["time_b"] = pd.cut(tmap, bins=[-1,120,600,1200,1800,3600], labels=["last2m","late","Q3","Q2","Q1"])
        else:
            df["time_b"] = "Q1"

        # league baseline expectation
        exp_keys = ["down_b","ytg_b","yl_b","score_b","qtr_b","time_b"]
        exp_tbl = (
            df.groupby(exp_keys, as_index=False)["is_pass"]
              .mean()
              .rename(columns={"is_pass":"exp_pass"})
        )
        df = df.merge(exp_tbl, on=exp_keys, how="left")
        if df["exp_pass"].isna().any():
            exp_tbl_simple = (
                df.groupby(["down_b","ytg_b","yl_b"], as_index=False)["is_pass"]
                  .mean()
                  .rename(columns={"is_pass":"exp_pass_simple"})
            )
            df = df.merge(exp_tbl_simple, on=["down_b","ytg_b","yl_b"], how="left")
            df["exp_pass"] = df["exp_pass"].fillna(df["exp_pass_simple"]).fillna(df["is_pass"].mean())
            df = df.drop(columns=["exp_pass_simple"], errors="ignore")

        grp = df.groupby([posteam, weekcol], as_index=False)[["is_pass","exp_pass"]].mean()
        grp["proe"] = grp["is_pass"] - grp["exp_pass"]
        grp = grp.rename(columns={posteam: "team", weekcol: "week"})
        proe_week = grp[["team","week","proe"]]
        proe_season = grp.groupby("team", as_index=False)["proe"].mean()
        env = env.merge(proe_season, on="team", how="left")

    # Red-zone trip rate (first entries per game)
    yline_use = yline
    env["rz_rate"] = 0.0

    if strict:
        require_cols(pbp, [c for c in [yline_use, game_id, posteam] if c], "pbp for red-zone trips", strict, issues)

    if yline_use:
        # minimal columns we always need
        base_cols = [game_id, posteam, yline_use]

        df = (
            pbp.loc[pbp[posteam].notna() & pbp[yline_use].notna(), base_cols]
            .copy()
            .rename(columns={posteam: "team"})
        )

        # optional ordering cols, avoid duplicating base columns
        order_cols = [
            c for c in ["game_id", "old_game_id", "drive", "play_id", "index"]
            if c in pbp.columns and c not in base_cols
        ]
        order_cols = list(dict.fromkeys(order_cols))
        if order_cols:
            df = (
                pbp.loc[df.index, base_cols + order_cols]
                   .copy()
                   .rename(columns={posteam: "team"})
            )
            df = df.sort_values(order_cols)

        # mark red-zone on each play
        df["is_rz"] = _safe_num(df[yline_use]) <= 20
        # first entry within each game/team is when previous is_rz is False
        df["prev_is_rz"] = df.groupby([game_id, "team"])["is_rz"].shift(1).fillna(False)
        df["rz_entry"] = df["is_rz"] & (~df["prev_is_rz"])

        # trips per game, then average across games
        trips = (
            df.groupby([game_id, "team"], as_index=False)["rz_entry"]
              .sum()
              .rename(columns={"rz_entry": "rz_trips"})
        )
        rz = (
            trips.groupby("team", as_index=False)["rz_trips"]
                 .mean()
                 .rename(columns={"rz_trips": "rz_trips_per_game"})
        )

        env = env.merge(rz, on="team", how="left")
        env["rz_rate"] = env["rz_trips_per_game"].fillna(0.0)

    # plays_est from pace
    env["plays_est"] = (
        3600.0 / env.get("sec_per_play_neutral", pd.Series(np.nan)).replace(0, np.nan)
    ).fillna(0.0)

    # SAFETY: ensure these columns exist
    for col, default in [
        ("sec_per_play_neutral", np.nan),
        ("pace", np.nan),
        ("ay_per_att", np.nan),
        ("proe", 0.0),
        ("rz_rate", 0.0),
    ]:
        if col not in env.columns:
            if strict:
                issues.append(f"[data] '{col}' not produced; derived team table incomplete")
            env[col] = default

    # final assemble
    out = pd.DataFrame({
        "team": env["team"],
        "def_pressure_rate_z": 0.0,
        "def_pass_epa_z": _z(env["def_pass_epa"].fillna(0.0)) if "def_pass_epa" in env.columns else 0.0,
        "def_rush_epa_z": _z(env["def_rush_epa"].fillna(0.0)) if "def_rush_epa" in env.columns else 0.0,
        "def_sack_rate_z": 0.0,
        "pace_z": _z(env["pace"].fillna(0.0)),
        "light_box_rate_z": 0.0,
        "heavy_box_rate_z": 0.0,
        "ay_per_att_z": _z(env["ay_per_att"].fillna(0.0)),
        "plays_est": env["plays_est"].fillna(0.0),
        "proe": env["proe"].fillna(0.0),
        "rz_rate": env["rz_rate"].fillna(0.0),
    })
    return out, proe_week

# ----------------------------
# Player derivations
# ----------------------------
def derive_player_from_pbp(pbp: pd.DataFrame, strict: bool, issues: List[str]) -> pd.DataFrame:
    """
    Returns per-player form with shares/efficiency proxies.

    Columns:
      player, team, position (blank), target_share, rush_share, rz_tgt_share, rz_carry_share,
      yprr_proxy, ypc, ypt, qb_ypa
    """
    def warn_or_raise(msg: str):
        if strict:
            raise RuntimeError(msg)
        print("WARN:", msg)
        issues.append(msg)

    if not _ok(pbp):
        warn_or_raise("[derive_player_from_pbp] PBP is empty — cannot compute player shares")
        return pd.DataFrame(columns=[
            "player","team","position","target_share","rush_share","rz_tgt_share","rz_carry_share",
            "yprr_proxy","ypc","ypt","qb_ypa"
        ])

    posteam = _pick(["posteam","offense_team"], pbp) or "posteam"

    # Generous aliases
    rec_name  = _pick([
        "receiver_player_name","receiver","receiver_name",
        "target_player_name","targeted_receiver","target_name"
    ], pbp)
    rush_name = _pick(["rusher_player_name","rusher","rusher_name"], pbp)
    pass_name = _pick(["passer_player_name","passer","passer_name","qb_player_name","quarterback_name"], pbp)
    yline     = _pick(["yardline_100","yardline","yardline_200"], pbp)
    yards_gained = _pick(["yards_gained","yards"], pbp)
    yac          = _pick(["yards_after_catch","yac"], pbp)

    missing_bits = []
    if not rec_name:  missing_bits.append("receiver name")
    if not rush_name: missing_bits.append("rusher name")
    if not yline:     missing_bits.append("yardline (needed for red-zone shares)")
    if missing_bits:
        warn_or_raise("[derive_player_from_pbp] Missing required PBP columns: " + ", ".join(missing_bits))

    pass_flag = _bool_col(pbp, ["pass","is_pass","qb_dropback"])
    rush_flag = _bool_col(pbp, ["rush","is_rush"]) & ~pass_flag

    # ---------- counting tables ----------
    targ_df = (pd.DataFrame() if not rec_name else
        pbp.loc[pass_flag & pbp[rec_name].notna() & pbp[posteam].notna(), [posteam, rec_name]]
           .assign(tgt=1)
           .groupby([posteam, rec_name], as_index=False)["tgt"].sum()
           .rename(columns={posteam:"team", rec_name:"player"})
    )

    rush_df = (pd.DataFrame() if not rush_name else
        pbp.loc[rush_flag & pbp[rush_name].notna() & pbp[posteam].notna(), [posteam, rush_name]]
           .assign(rush=1)
           .groupby([posteam, rush_name], as_index=False)["rush"].sum()
           .rename(columns={posteam:"team", rush_name:"player"})
    )

    rz_mask = (_safe_num(pbp[yline]) <= 20) if yline else pd.Series(False, index=pbp.index)

    rz_tgt_df = (pd.DataFrame() if not rec_name or not yline else
        pbp.loc[rz_mask & pass_flag & pbp[rec_name].notna() & pbp[posteam].notna(), [posteam, rec_name]]
           .assign(rz_tgt=1)
           .groupby([posteam, rec_name], as_index=False)["rz_tgt"].sum()
           .rename(columns={posteam:"team", rec_name:"player"})
    )

    rz_car_df = (pd.DataFrame() if not rush_name or not yline else
        pbp.loc[rz_mask & rush_flag & pbp[rush_name].notna() & pbp[posteam].notna(), [posteam, rush_name]]
           .assign(rz_car=1)
           .groupby([posteam, rush_name], as_index=False)["rz_car"].sum()
           .rename(columns={posteam:"team", rush_name:"player"})
    )

    if strict:
        feeder_issues = []
        if not _ok(targ_df):   feeder_issues.append("no receiver targets found")
        if not _ok(rush_df):   feeder_issues.append("no rusher attempts found")
        if not _ok(rz_tgt_df): feeder_issues.append("no receiver red-zone targets found")
        if not _ok(rz_car_df): feeder_issues.append("no rusher red-zone carries found")
        if feeder_issues:
            raise RuntimeError("[derive_player_from_pbp] Missing feeder data: " + "; ".join(feeder_issues))

    # ---------- efficiency tables ----------
    rec_yds_df = pd.DataFrame()
    if rec_name and yards_gained:
        rec_yds_df = (
            pbp.loc[pass_flag & pbp[rec_name].notna() & pbp[posteam].notna(), [posteam, rec_name, yards_gained]]
               .groupby([posteam, rec_name], as_index=False)[yards_gained].sum()
               .rename(columns={posteam:"team", rec_name:"player", yards_gained:"rec_yards"})
        )
    yac_df = pd.DataFrame()
    if rec_name and yac:
        yac_df = (
            pbp.loc[pass_flag & pbp[rec_name].notna() & pbp[posteam].notna(), [posteam, rec_name, yac]]
               .groupby([posteam, rec_name], as_index=False)[yac].sum()
               .rename(columns={posteam:"team", rec_name:"player", yac:"yac_sum"})
        )
    rush_yds_df = pd.DataFrame()
    if rush_name and yards_gained:
        rush_yds_df = (
            pbp.loc[rush_flag & pbp[rush_name].notna() & pbp[posteam].notna(), [posteam, rush_name, yards_gained]]
                .groupby([posteam, rush_name], as_index=False)[yards_gained].sum()
                .rename(columns={posteam: "team", rush_name: "player", yards_gained: "rush_yards"})
    )

    # ---------- merge per-player ----------
    pf = pd.DataFrame(columns=["player","team"])
    for base in [targ_df, rush_df, rz_tgt_df, rz_car_df, rec_yds_df, yac_df, rush_yds_df]:
        if _ok(base):
            pf = pf.merge(base, on=["team","player"], how="outer") if _ok(pf) else base.copy()
    if not _ok(pf):
        warn_or_raise("[derive_player_from_pbp] No player rows produced after merges; filling empty schema")
        return pd.DataFrame(columns=[
            "player","team","position","target_share","rush_share","rz_tgt_share","rz_carry_share",
            "yprr_proxy","ypc","ypt","qb_ypa"
        ])

    # ---------- final schema (always defined) ----------
    cols_out = [
        "target_share","rush_share","rz_tgt_share","rz_carry_share",
        "yprr_proxy","ypc","ypt","qb_ypa"
    ]

    pf_out = pf[["player","team"]].copy()
    pf_out["position"] = ""

    for c in cols_out:
        series = pf[c] if c in pf.columns else pd.Series(np.nan, index=pf.index)
        pf_out[c] = _safe_num(series).astype(float)

    if not strict:
        for c in cols_out:
            pf_out[c] = pf_out[c].fillna(0.0)

    return pf_out

# ============================
# ===== COMPOSERS ============
# ============================
def compose_team_form(season: int, cache: Dict[str, pd.DataFrame], strict: bool, issues: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns team_form and proe_week (for optional export)
    """
    pbp = _get_or_resolve("pbp", season, cache)
    require_nonempty(pbp, "pbp", strict, issues)

    tf_der, proe_week = derive_team_from_pbp(pbp, strict, issues)

    # Optional sources for box rates, etc.
    box   = _get_or_resolve("box_week", season, cache)
    proe  = _get_or_resolve("proe_week", season, cache)

    # --- Odds: compose consensus tables ---
    game_odds, props_odds = compose_odds(s)
    # also mirror to data/ for model convenience
    if _ok(game_odds):
        _write_csv(game_odds, DATA_MIRROR / "odds_game_consensus.csv")
    if _ok(props_odds):
        _write_csv(props_odds, DATA_MIRROR / "odds_props_consensus.csv")

    # integrate box rates if present
    if _ok(box) and {"team","def_light_box_rate","def_heavy_box_rate"} <= set(box.columns):
        b = box.groupby("team", as_index=False)[["def_light_box_rate","def_heavy_box_rate"]].mean()
        tf_der = tf_der.merge(b, on="team", how="left")
        tf_der["light_box_rate_z"] = _z(tf_der["def_light_box_rate"].fillna(0.0))
        tf_der["heavy_box_rate_z"] = _z(tf_der["def_heavy_box_rate"].fillna(0.0))

    # prefer addon PROE season if present (otherwise derived)
    if _ok(proe) and {"team","proe"} <= set(proe.columns):
        p = proe.groupby("team", as_index=False)["proe"].mean()
        tf_der = tf_der.drop(columns=["proe"], errors="ignore").merge(p, on="team", how="left")

    # ensure required cols exist
    cols = ["team",
            "def_pressure_rate_z","def_pass_epa_z","def_rush_epa_z","def_sack_rate_z",
            "pace_z","light_box_rate_z","heavy_box_rate_z","ay_per_att_z",
            "plays_est","proe","rz_rate"]

    for c in cols:
        if c not in tf_der.columns:
            tf_der[c] = 0.0 if c != "team" else ""

    # ✅ add season column the validator wants
    tf_der["season"] = int(season)

    return tf_der[cols + ["season"]], proe_week

def compose_player_form(season: int, cache: Dict[str, pd.DataFrame], strict: bool, issues: List[str]) -> pd.DataFrame:
    pbp = _get_or_resolve("pbp", season, cache)
    require_nonempty(pbp, "pbp", strict, issues)

    pf = derive_player_from_pbp(pbp, strict, issues)

    # Merge optional addons
    roles = _read_csv(ROOT / "outputs" / "roles" / f"roles_{season}.csv")
    rbm   = _read_csv(ROOT / "outputs" / "rb_metrics" / f"rb_metrics_{season}.csv")
    inj   = _get_or_resolve("injuries", season, cache)

    if _ok(roles):
        # where roles provide position & overrides for shares, prefer them
        for col, default in [
            ("player",""), ("team",""), ("position",""),
            ("target_share",np.nan), ("rush_share",np.nan),
            ("rz_tgt_share",np.nan), ("rz_carry_share",np.nan),
            ("yprr_proxy",np.nan), ("ypc",np.nan), ("ypt",np.nan), ("qb_ypa",np.nan),
        ]:
            if col not in roles.columns: roles[col] = default
        pf = pf.merge(roles[pf.columns], on=["player","team"], how="outer", suffixes=("", "_role"))
        # fill from roles if present
        for c in ["position","target_share","rush_share","rz_tgt_share","rz_carry_share","yprr_proxy","ypc","ypt","qb_ypa"]:
            rc = f"{c}_role"
            if rc in pf.columns:
                pf[c] = pf[c].where(pf[c].notna() & (pf[c] != 0), pf[rc])
        drop_cols = [c for c in pf.columns if c.endswith("_role")]
        pf = pf.drop(columns=drop_cols)

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

    # final NA cleanup
    pf["position"] = pf["position"].fillna("")
    for c in ["target_share","rush_share","rz_tgt_share","rz_carry_share","yprr_proxy","ypc","ypt","qb_ypa"]:
        if c not in pf.columns: pf[c] = 0.0
        pf[c] = pf[c].fillna(0.0)

    return pf[[
        "player","team","position",
        "target_share","rush_share","rz_tgt_share","rz_carry_share",
        "yprr_proxy","ypc","ypt","qb_ypa"
    ]]
def compose_odds(season: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ingests outputs from fetch_odds.py and computes bookmaker-consensus
    game-line and props tables. Returns (game_consensus, props_consensus).
    """
    out_dir = ROOT / "outputs"
    gl_path = out_dir / "game_lines.csv"
    pr_path = out_dir / "props_raw.csv"

    game = _read_csv_soft(gl_path)
    props = _read_csv_soft(pr_path)

    # ---------- GAME LINES ----------
    game_cons = pd.DataFrame()
    if _ok(game):
        # ensure typed
        for c in ["price", "point"]:
            if c in game.columns:
                game[c] = pd.to_numeric(game[c], errors="coerce")

        # implied probs by event+market+outcome+book
        if {"event_id","market","outcome","bookmaker","price"}.issubset(game.columns):
            key_cols = ["event_id","market","outcome","bookmaker"]
            game["impl"] = game["price"].map(_decimal_to_impl)
            # remove rows with no probability
            game = game.dropna(subset=["impl"])

            # build a pivot to de-vig within each (event_id, market, bookmaker)
            grp = game.groupby(["event_id","market","bookmaker"])
            # normalize outcomes per (event, market, book)
            game["impl_vigfree"] = grp["impl"].transform(lambda s: s / s.sum())

            # consensus per (event, market, outcome): average across books
            cons = (game.groupby(["event_id","market","outcome"], as_index=False)
                        .agg(cons_prob=("impl_vigfree","mean"),
                             avg_price=("price","mean"),
                             avg_point=("point","mean")))

            # also keep teams/time for joins
            meta = (game.groupby("event_id", as_index=False)
                        .agg(commence_time=("commence_time","first"),
                             home_team=("home_team","first"),
                             away_team=("away_team","first")))
            game_cons = cons.merge(meta, on="event_id", how="left")

    # ---------- PROPS ----------
    props_cons = pd.DataFrame()
    if _ok(props):
        for c in ["price","point"]:
            if c in props.columns:
                props[c] = pd.to_numeric(props[c], errors="coerce")

        if {"event_id","market","outcome","bookmaker","price"}.issubset(props.columns):
            props["impl"] = props["price"].map(_decimal_to_impl)
            props = props.dropna(subset=["impl"])

            grp = props.groupby(["event_id","market","label","player","bookmaker"])
            props["impl_vigfree"] = grp["impl"].transform(lambda s: s / s.sum())

            cons = (props.groupby(["event_id","market","label","player","outcome"], as_index=False)
                         .agg(cons_prob=("impl_vigfree","mean"),
                              avg_price=("price","mean"),
                              avg_point=("point","mean")))

            meta = (props.groupby("event_id", as_index=False)
                        .agg(commence_time=("commence_time","first")))
            props_cons = cons.merge(meta, on="event_id", how="left")

    # write outputs (for debugging/consumption)
    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)
    if _ok(game_cons):
        game_cons.to_csv(ROOT / "outputs" / "odds_game_consensus.csv", index=False)
    if _ok(props_cons):
        props_cons.to_csv(ROOT / "outputs" / "odds_props_consensus.csv", index=False)

    return game_cons, props_cons

# ============================
# ===== BUNDLE DRIVER ========
# ============================
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
    ap.add_argument("--strict", type=int, default=1, help="1=hard-fail on missing inputs, 0=fill & continue")
    args = ap.parse_args()

    _mkdirs()
    seasons = seasons_from_args(args)
    strict = bool(args.strict)
    data_issues: List[str] = []

    print(f"[make_all] seasons={seasons} strict={int(strict)}")

    fetch_bundle(seasons)

    try:
    for s in seasons:
        print(f"[compose] season {s}")
        cache: Dict[str, pd.DataFrame] = {}
        cache["pbp"] = _get_or_resolve("pbp", s, cache)  # early for proxies

        # 🆕 Fetch and compose odds consensus FIRST (before team/player forms)
        game_odds, props_odds = compose_odds(s)

        # 🆕 Write odds results so downstream steps can read them
        if _ok(game_odds):
            _write_csv(game_odds, DATA_MIRROR / "odds_game_consensus.csv")
        if _ok(props_odds):
            _write_csv(props_odds, DATA_MIRROR / "odds_props_consensus.csv")

        # Existing form builders
        team_form, proe_week = compose_team_form(s, cache, strict, data_issues)
        player_form = compose_player_form(s, cache, strict, data_issues)

        # Write outputs
        _write_csv(team_form, OUT_METRICS / "team_form.csv")
        _write_csv(player_form, OUT_METRICS / "player_form.csv")
        _write_csv(team_form, DATA_MIRROR / "team_form.csv")
        _write_csv(player_form, DATA_MIRROR / "player_form.csv")

        # Optional export of weekly proe (debug/inspection)
        if _ok(proe_week):
            out_proe = ROOT / "outputs" / "proe" / f"proe_week_{s}.csv"
            _write_csv(proe_week, out_proe)

        # write data quality issues when non-strict
        if data_issues and not strict:
            dq = pd.DataFrame({"issue": data_issues})
            _write_csv(dq, ROOT / "outputs" / "data_quality_issues.csv")

        print("✅ make_all completed.")
        return 0

    except DataQualityError as e:
        print(f"❌ DataQualityError: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

