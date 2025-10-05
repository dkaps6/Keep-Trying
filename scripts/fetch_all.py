# scripts/fetch_all.py
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

def _try_import(module: str):
    try:
        return __import__(module, fromlist=["*"])
    except Exception:
        return None

_odds = _try_import("scripts.odds_api") or _try_import("odds_api")
_dk   = _try_import("scripts.providers.draftkings_free") or _try_import("scripts.draftkings_free") or _try_import("draftkings_free")

ROOT = Path(".").resolve()
OUT_DIR, METRICS, INPUTS = ROOT/"outputs", ROOT/"metrics", ROOT/"inputs"
for p in (OUT_DIR, METRICS, INPUTS): p.mkdir(parents=True, exist_ok=True)

def _write(df: pd.DataFrame, p: Path) -> int:
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return int(len(df))

def _read(p: Path) -> pd.DataFrame:
    try:
        if p.exists(): return pd.read_csv(p)
    except Exception:
        pass
    return pd.DataFrame()

def _load_props(date: str, season: str) -> pd.DataFrame:
    # 1) Try Odds API ONCE (no alias loops)
    if _odds and hasattr(_odds, "fetch_props"):
        try:
            df = _odds.fetch_props(date=date, season=season)
            if isinstance(df, pd.DataFrame) and not df.empty:
                _write(df, OUT_DIR/"props_raw.csv")
                return df
        except RuntimeError as rte:
            if "ODDS_API_OUT_OF_CREDITS" in str(rte):
                print("[fetch_all] Odds API out of credits; skipping.")
            else:
                print(f"[fetch_all] odds_api.fetch_props error: {rte}")
        except Exception as e:
            print(f"[fetch_all] odds_api.fetch_props error: {e}")

    # 2) DK fallback (if available)
    if _dk and hasattr(_dk, "fetch_dk_props"):
        try:
            df = _dk.fetch_dk_props()
            if not df.empty:
                print("[fetch_all] using DraftKings fallback")
                _write(df, OUT_DIR/"props_raw.csv")
                return df
        except Exception as e:
            print(f"[fetch_all] draftkings_free.fetch_dk_props error: {e}")

    # 3) Manual local CSVs
    for c in (OUT_DIR/"props_raw.csv", INPUTS/"props.csv", ROOT/"data"/"odds_sample.csv"):
        df = _read(c)
        if not df.empty:
            print(f"[fetch_all] props from {c}: {len(df)} rows")
            return df

    print("[fetch_all] WARNING: no props found (API & DK & local fallbacks empty)")
    return pd.DataFrame()

def main(date: str = "today", season: str = "2025") -> pd.DataFrame:
    status: dict[str, Any] = {
        "season": season,
        "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
        "providers": {},
        "rows": {},
        "notes": [],
    }

    props = _load_props(date, season)
    status["providers"]["props"] = ("odds_api" if _odds else "none") + "+dk+manual"
    status["rows"]["props"] = int(len(props))

    # Minimal safe stubs so downstream runs
    _write(pd.DataFrame(), METRICS/"team_week_form.csv")
    _write(pd.DataFrame(), METRICS/"player_form.csv")
    _write(pd.DataFrame(), INPUTS/"player_id_cache.csv")
    _write(pd.DataFrame(), METRICS/"weather.csv")

    (METRICS/"fetch_status.json").write_text(json.dumps(status, indent=2))
    print("Fetch complete:\n" + json.dumps(status, indent=2))
    return props

# CLI entry
if __name__ == "__main__":
    main()

