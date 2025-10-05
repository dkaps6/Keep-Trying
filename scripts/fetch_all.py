# scripts/fetch_all.py
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

# --- Optional odds provider (we call whatever you already have if present) ---
def _try_import(module: str):
    try:
        return __import__(module, fromlist=["*"])
    except Exception:
        return None

_odds_api = _try_import("scripts.odds_api")  # your existing file
_nfl_data  = _try_import("nfl_data_py")      # if available

ROOT = Path(".").resolve()
OUT_DIR = ROOT / "outputs"
METRICS = ROOT / "metrics"
INPUTS = ROOT / "inputs"
METRICS.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
INPUTS.mkdir(parents=True, exist_ok=True)

# ------------------------- helpers -------------------------

def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")

def _write_csv(df: pd.DataFrame, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return int(len(df))

def _read_csv(path: Path) -> pd.DataFrame:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()

def _first_callable(mod, names: list[str]) -> Optional[Callable]:
    if not mod:
        return None
    for n in names:
        fn = getattr(mod, n, None)
        if callable(fn):
            return fn
    return None

# ------------------------- props fetch -------------------------

def _load_props(date: str, season: str) -> pd.DataFrame:
    """
    We try your odds module first; otherwise fall back to any local CSVs.
    Expected columns minimally: ['player','team','market','line','price'].
    """
    # Try to call into your odds module with any known names
    if _odds_api:
        for name in ["fetch_props", "get_props", "build_props_frame", "load_props"]:
            fn = getattr(_odds_api, name, None)
            if callable(fn):
                try:
                    df = fn(date=date, season=season) if "season" in fn.__code__.co_varnames else fn(date=date)
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        print(f"[fetch_all] props via scripts.odds_api.{name}: {len(df)} rows")
                        return df
                except Exception as e:
                    print(f"[fetch_all] odds_api.{name} failed: {e}")

    # Fallbacks: anything you may already write/read locally
    for candidate in [
        OUT_DIR / "props_raw.csv",
        ROOT / "data" / "odds_sample.csv",
        INPUTS / "props.csv",
    ]:
        df = _read_csv(candidate)
        if not df.empty:
            print(f"[fetch_all] props from {candidate}: {len(df)} rows")
            return df

    print("[fetch_all] WARNING: no props found anywhere")
    return pd.DataFrame()

# ------------------------- feature builders (real if available, else neutral) -------------------------

def _team_week_form_real(season: int) -> pd.DataFrame:
    if not _nfl_data:
        return pd.DataFrame()
    try:
        # This is deliberately minimal; swap in your real calls if you have them.
        # nfl_data_py often exposes weekly team summaries by season.
        df = _nfl_data.import_team_desc(season)  # may not exist for 2025 / adjust for your env
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def _neutral_team_week_form_from_props(props: pd.DataFrame) -> pd.DataFrame:
    if props.empty:
        return pd.DataFrame()
    cols = ["team", "opp", "week", "drive_pace_z", "ns_epa_pass_z", "ns_epa_rush_z",
            "blitz_rate_z", "man_coverage_z", "opp_pressure_rate_z", "ol_pressure_allowed_z",
            "redzone_pass_rate", "goal_line_rush_share"]
    # Build neutral rows for each team-week we see in props
    base = props[["team", "opp"]].dropna().drop_duplicates().copy()
    # If you carry 'week' in props, keep it; else set 0
    if "week" in props.columns:
        wk = props[["team", "week"]].dropna().drop_duplicates()
        base = base.merge(wk, on="team", how="left")
    else:
        base["week"] = 0
    for c in cols[2:]:
        base[c] = 0.0
    base["redzone_pass_rate"] = 0.5
    base["goal_line_rush_share"] = 0.55
    return base.rename(columns={"week": "week"})

def _player_form_real(season: int) -> pd.DataFrame:
    # If you have a player-level participation function, hook it here.
    # Otherwise return empty and we’ll derive neutral from props.
    return pd.DataFrame()

def _player_form_from_props(props: pd.DataFrame) -> pd.DataFrame:
    if props.empty:
        return pd.DataFrame()
    keep = props[["player", "team"]].drop_duplicates().copy()
    # If week present, preserve; else set 0
    if "week" in props.columns:
        wk = props[["player", "team", "week"]].drop_duplicates()
        keep = keep.merge(wk, on=["player", "team"], how="left")
    else:
        keep["week"] = 0
    keep["player_form"] = 1.0   # neutral participation
    # Optional hints if you have them in props
    for c in ("routes", "targets", "catch_rate", "rush_att", "ypc", "attempts", "ypa", "target_share"):
        if c not in keep.columns:
            keep[c] = pd.NA
    return keep

def _id_map_real(season: int) -> pd.DataFrame:
    # Plug your ESPN/NFL roster pull here if/when available.
    return pd.DataFrame()

def _id_map_from_props(props: pd.DataFrame) -> pd.DataFrame:
    if props.empty:
        return pd.DataFrame()
    m = props[["player", "team"]].dropna().drop_duplicates().copy()
    m["player_id"] = m["player"].apply(_slug) + "-" + m["team"].apply(_slug)
    return m

def _weather_from_inputs_or_empty() -> pd.DataFrame:
    # If you have inputs/stadiums.csv + an inference, load it; otherwise empty is fine.
    st = _read_csv(INPUTS / "stadiums.csv")
    if st.empty:
        return pd.DataFrame()
    # If you do have stadiums + schedule keyed by game_id, join to produce weather;
    # for now we return empty to avoid false joins.
    return pd.DataFrame()

# ------------------------- public API -------------------------

def build_props_frame(date: str = "today", season: str = "2025") -> pd.DataFrame:
    """Return props so the engine can run even if features are neutral."""
    return _load_props(date, season)

def main(date: str = "today", season: str = "2025") -> pd.DataFrame:
    """
    Build features and (optionally) props. Always writes a metrics/fetch_status.json
    and CSVs for: team_week_form.csv, player_form.csv, id_map.csv, weather.csv
    """
    season_int = int(season) if str(season).isdigit() else 0
    status: dict[str, Any] = {
        "season": season_int or season,
        "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
        "providers": {},
        "rows": {},
        "notes": [],
    }

    # 1) PROPS
    props = _load_props(date, season)
    status["providers"]["props"] = "odds" if not props.empty else "missing"
    status["rows"]["props"] = int(len(props))

    # 2) TEAM-WEEK FORM
    tdf = _team_week_form_real(season_int)
    if tdf.empty and not props.empty:
        status["providers"]["team_week_form.csv"] = "neutral-from-props"
        tdf = _neutral_team_week_form_from_props(props)
    else:
        status["providers"]["team_week_form.csv"] = "nflverse" if not tdf.empty else "nflverse:error"
    status["rows"]["team_week_form.csv"] = _write_csv(tdf, METRICS / "team_week_form.csv")

    # 3) PLAYER FORM
    pdf = _player_form_real(season_int)
    if pdf.empty and not props.empty:
        status["providers"]["player_form.csv"] = "props-derived"
        pdf = _player_form_from_props(props)
    else:
        status["providers"]["player_form.csv"] = "provider" if not pdf.empty else "pbp:error"
    status["rows"]["player_form.csv"] = _write_csv(pdf, METRICS / "player_form.csv")

    # 4) ID MAP
    idm = _id_map_real(season_int)
    if idm.empty and not props.empty:
        status["providers"]["id_map.csv"] = "props-derived"
        idm = _id_map_from_props(props)
    else:
        status["providers"]["id_map.csv"] = "espn" if not idm.empty else "espn:empty"
    status["rows"]["id_map.csv"] = _write_csv(idm, INPUTS / "player_id_cache.csv")

    # 5) WEATHER
    wdf = _weather_from_inputs_or_empty()
    status["providers"]["weather.csv"] = "inferred" if not wdf.empty else "placeholder"
    status["rows"]["weather.csv"] = _write_csv(wdf, METRICS / "weather.csv")

    # 6) Save status JSON
    (METRICS / "fetch_status.json").write_text(json.dumps(status, indent=2))
    print("Fetch complete:\n" + json.dumps(status, indent=2))

    # Return props for the engine (it will do robust merges + pricing)
    return props
