# scripts/fetch_all.py
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

def _try_import(mod: str):
    try: return __import__(mod, fromlist=["*"])
    except Exception: return None

_odds = _try_import("scripts.odds_api") or _try_import("odds_api")

ROOT = Path(".").resolve()
OUT  = ROOT / "outputs"
MET  = ROOT / "metrics"
for p in (OUT, MET): p.mkdir(parents=True, exist_ok=True)

def _write_csv(df: pd.DataFrame, p: Path) -> int:
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return int(len(df))

def _status() -> dict[str, Any]:
    return {"timestamp_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
            "providers": {}, "rows": {}, "notes": []}

def build_props_frame(date: str = "today", season: str = "2025") -> pd.DataFrame:
    s = _status(); s["season"] = season; s["prefer"] = ["odds"]
    if not _odds or not hasattr(_odds, "fetch_props"):
        s["providers"]["odds_api"] = "module-missing"
        (MET/"fetch_status.json").write_text(json.dumps(s, indent=2))
        raise SystemExit("odds_api module missing")

    try:
        df = _odds.fetch_props(date=date, season=season)
    except RuntimeError as rte:
        s["providers"]["odds_api"] = f"error:{rte}"
        (MET/"fetch_status.json").write_text(json.dumps(s, indent=2))
        (OUT/"_NO_PROPS").write_text(str(rte))
        raise SystemExit(2)
    except Exception as e:
        s["providers"]["odds_api"] = f"error:{type(e).__name__}: {e}"
        (MET/"fetch_status.json").write_text(json.dumps(s, indent=2))
        (OUT/"_NO_PROPS").write_text(str(e))
        raise SystemExit(2)

    n = 0 if df is None else int(len(df))
    s["providers"]["odds_api"] = "ok" if n > 0 else "ok-empty"
    s["rows"]["odds_api"] = n
    (MET/"fetch_status.json").write_text(json.dumps(s, indent=2))

    if n == 0:
        (OUT/"_NO_PROPS").write_text("odds_api returned 0 rows for this slate")
        raise SystemExit(2)

    _write_csv(df, OUT/"props_raw.csv")
    return df

def main(date: str = "today", season: str = "2025") -> pd.DataFrame:
    return build_props_frame(date=date, season=season)

if __name__ == "__main__":
    main()
