# scripts/fetch_all.py
from __future__ import annotations
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

def _try_import(mod: str):
    try:
        return __import__(mod, fromlist=["*"])
    except Exception:
        return None

# Providers (Odds API + DraftKings free)
_odds = _try_import("scripts.odds_api") or _try_import("odds_api")
_dk   = _try_import("scripts.providers.draftkings_free") or _try_import("scripts.draftkings_free") or _try_import("draftkings_free")

ROOT = Path(".").resolve()
OUT  = ROOT / "outputs"
MET  = ROOT / "metrics"
DBG  = OUT / "debug"
for p in (OUT, MET, DBG):
    p.mkdir(parents=True, exist_ok=True)

def _write_csv(df: pd.DataFrame, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return int(len(df))

def _status_stub() -> Dict[str, Any]:
    return {
        "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
        "providers": {},
        "rows": {},
        "notes": [],
    }

def _save_status(status: Dict[str, Any]) -> None:
    (MET / "fetch_status.json").write_text(json.dumps(status, indent=2))

def _fetch_odds_api(date: str, season: str, status: Dict[str, Any]) -> Optional[pd.DataFrame]:
    if not _odds or not hasattr(_odds, "fetch_props"):
        status["providers"]["odds_api"] = "module-missing"
        return None
    try:
        df = _odds.fetch_props(date=date, season=season)
        n = 0 if df is None else int(len(df))
        status["providers"]["odds_api"] = "ok" if n > 0 else "ok-empty"
        status["rows"]["odds_api"] = n
        if n > 0:
            _write_csv(df, OUT / "props_raw.csv")
            return df
        return None
    except RuntimeError as rte:
        # Bubble up out-of-credits marker distinctly in the status
        msg = str(rte)
        status["providers"]["odds_api"] = f"error:{msg}"
        status["notes"].append(f"odds_api error: {msg}")
        return None
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        status["providers"]["odds_api"] = f"error:{msg}"
        status["notes"].append(f"odds_api error: {msg}")
        return None

def _fetch_dk(status: Dict[str, Any]) -> Optional[pd.DataFrame]:
    if not _dk or not hasattr(_dk, "fetch_dk_props"):
        status["providers"]["draftkings"] = "module-missing"
        return None
    try:
        df = _dk.fetch_dk_props()
        n = 0 if df is None else int(len(df))
        status["providers"]["draftkings"] = "ok" if n > 0 else "ok-empty"
        status["rows"]["draftkings"] = n
        if n > 0:
            _write_csv(df, OUT / "props_raw.csv")
            return df
        return None
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        status["providers"]["draftkings"] = f"error:{msg}"
        status["notes"].append(f"draftkings error: {msg}")
        return None

def build_props_frame(date: str = "today", season: str = "2025") -> pd.DataFrame:
    """
    Direct-from-provider fetch. No local CSV fallback.
    Order is controlled by env PREFER_SOURCE:
      - "odds"     -> Odds API only
      - "dk"       -> DraftKings only
      - "odds,dk"  -> try Odds API, then DK (default)
    If neither yields rows, we raise SystemExit with a clear reason.
    """
    prefer = os.getenv("PREFER_SOURCE", "odds,dk").replace(" ", "").lower().split(",")
    status = _status_stub()
    status["season"] = season
    status["prefer"] = prefer

    df: Optional[pd.DataFrame] = None
    reasons: list[str] = []

    for src in prefer:
        if src == "odds":
            d = _fetch_odds_api(date, season, status)
            if d is not None and not d.empty:
                df = d; break
            reasons.append(f"odds_api={status['providers'].get('odds_api','n/a')}")
        elif src == "dk":
            d = _fetch_dk(status)
            if d is not None and not d.empty:
                df = d; break
            reasons.append(f"draftkings={status['providers'].get('draftkings','n/a')}")
        else:
            status["notes"].append(f"unknown source in PREFER_SOURCE: {src}")

    _save_status(status)

    if df is None or df.empty:
        # Fail hard and tell you exactly why
        msg = "No props fetched from providers. Reasons: " + "; ".join(reasons or ["none"])
        # Also write an explicit flag so the workflow can surface it easily
        (OUT / "_NO_PROPS").write_text(msg)
        print("[fetch_all] " + msg)
        raise SystemExit(2)

    return df

def main(date: str = "today", season: str = "2025") -> pd.DataFrame:
    return build_props_frame(date=date, season=season)

if __name__ == "__main__":
    main()
