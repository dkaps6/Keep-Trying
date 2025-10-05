# scripts/fetch_all.py
from __future__ import annotations
import inspect, json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

# ---- safe imports -------------------------------------------------
def _try_import(module: str):
    try:
        return __import__(module, fromlist=["*"])
    except Exception:
        return None

_odds_api = _try_import("scripts.odds_api")
_espn = _try_import("scripts.providers.espn_free") or _try_import("scripts.espn_free")
_nflv = _try_import("scripts.providers.nflverse_free") or _try_import("scripts.nflverse_free")

ROOT = Path(".").resolve()
OUT_DIR = ROOT / "outputs"
METRICS = ROOT / "metrics"
INPUTS = ROOT / "inputs"
for p in (OUT_DIR, METRICS, INPUTS):
    p.mkdir(parents=True, exist_ok=True)

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

# ---- flexible caller for odds_api functions -----------------------
_ARG_ALIAS = {
    "date": ["date", "target_date", "slate_date"],
    "season": ["season", "season_year", "year"],
}

def _call_with_aliases(fn, date: str, season: str) -> Optional[pd.DataFrame]:
    sig = inspect.signature(fn)
    # try multiple kwarg combos, mapping our (date, season) into whatever the fn accepts
    combos: list[Dict[str, Any]] = []
    for d_key in _ARG_ALIAS["date"]:
        for s_key in _ARG_ALIAS["season"]:
            combos.append({d_key: date, s_key: season})
        combos.append({d_key: date})
    combos.append({})  # last resort: no kwargs

    for kw in combos:
        # keep only params the function actually accepts
        filtered = {k: v for k, v in kw.items() if k in sig.parameters}
        try:
            df = fn(**filtered)
            if isinstance(df, pd.DataFrame):
                used = ", ".join(f"{k}={filtered[k]!r}" for k in filtered)
                print(f"[fetch_all] odds_api.{fn.__name__}({used}) -> {len(df)} rows")
                return df
        except Exception as e:
            print(f"[fetch_all] odds_api.{fn.__name__} failed ({kw}): {e}")
    return None

# ---- props loader -------------------------------------------------
def _load_props(date: str, season: str) -> pd.DataFrame:
    # Try your odds_api functions in this order
    if _odds_api:
        for name in ("fetch_props", "get_props", "build_props_frame", "load_props"):
            fn = getattr(_odds_api, name, None)
            if callable(fn):
                df = _call_with_aliases(fn, date=date, season=season)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    _write_csv(df, OUT_DIR / "props_raw.csv")  # debug snapshot
                    return df

    # Fallback to local CSV (manual) if present
    for candidate in (OUT_DIR/"props_raw.csv", INPUTS/"props.csv", ROOT/"data"/"odds_sample.csv"):
        df = _read_csv(candidate)
        if not df.empty:
            print(f"[fetch_all] props from {candidate}: {len(df)} rows")
            return df

    print("[fetch_all] WARNING: no props found (API & local fallbacks empty)")
    return pd.DataFrame()

# ---- public API ---------------------------------------------------
def build_props_frame(date: str = "today", season: str = "2025") -> pd.DataFrame:
    return _load_props(date, season)

def main(date: str = "today", season: str = "2025") -> pd.DataFrame:
    status: dict[str, Any] = {
        "season": season,
        "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
        "providers": {},
        "rows": {},
        "notes": [],
    }

    # 1) PROPS
    props = _load_props(date, season)
    status["providers"]["props"] = "odds_api" if not props.empty else "missing"
    status["rows"]["props"] = int(len(props))

    # 2) TEAM-WEEK FORM (nflverse -> features)
    tdf = pd.DataFrame()
    if _nflv:
        try:
            pbp = _nflv.import_pbp_2025()
            if not pbp.empty:
                tdf = _nflv.team_week_form_from_pbp(pbp)
                status["providers"]["team_week_form.csv"] = "nflverse/pbp"
            else:
                status["providers"]["team_week_form.csv"] = "nflverse:empty"
        except Exception as e:
            status["providers"]["team_week_form.csv"] = f"nflverse:error:{e}"
    else:
        status["providers"]["team_week_form.csv"] = "nfl_data_py:missing"
    status["rows"]["team_week_form.csv"] = _write_csv(tdf, METRICS / "team_week_form.csv")

    # 3) PLAYER FORM (neutral from props for now)
    pdf = pd.DataFrame()
    if not props.empty:
        base = props[["player","team"]].drop_duplicates().copy()
        base["week"] = props["week"] if "week" in props.columns else 0
        base["player_form"] = 1.0
        pdf = base
        status["providers"]["player_form.csv"] = "props-derived"
    else:
        status["providers"]["player_form.csv"] = "missing"
    status["rows"]["player_form.csv"] = _write_csv(pdf, METRICS / "player_form.csv")

    # 4) ID MAP (ESPN roster)
    idm = pd.DataFrame()
    if _espn and hasattr(_espn, "build_id_map_from_espn"):
        try:
            idm = _espn.build_id_map_from_espn()
            status["providers"]["id_map.csv"] = "espn"
        except Exception as e:
            status["providers"]["id_map.csv"] = f"espn:error:{e}"
    else:
        status["providers"]["id_map.csv"] = "espn:module-missing"
    status["rows"]["id_map.csv"] = _write_csv(idm, INPUTS / "player_id_cache.csv")

    # 5) WEATHER placeholder
    wdf = pd.DataFrame()
    status["providers"]["weather.csv"] = "placeholder"
    status["rows"]["weather.csv"] = _write_csv(wdf, METRICS / "weather.csv")

    # 6) Save status
    (METRICS / "fetch_status.json").write_text(json.dumps(status, indent=2))
    print("Fetch complete:\n" + json.dumps(status, indent=2))

    return props
