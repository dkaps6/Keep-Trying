# scripts/fetch_all.py
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

def _try(mod): 
    try: return __import__(mod, fromlist=["*"])
    except Exception: return None

_odds = _try("scripts.odds_api") or _try("odds_api")

ROOT = Path("."); OUT = ROOT/"outputs"; MET = ROOT/"metrics"
for p in (OUT, MET): p.mkdir(parents=True, exist_ok=True)

def _status() -> dict[str, Any]:
    return {"timestamp_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
            "providers": {}, "rows": {}, "notes": []}

def main(date: str="today", season: str="2025") -> pd.DataFrame:
    s = _status(); s["season"] = season; s["prefer"] = ["odds"]
    if not _odds or not hasattr(_odds, "fetch_props"):
        s["providers"]["odds_api"] = "module-missing"
        (MET/"fetch_status.json").write_text(json.dumps(s, indent=2))
        (OUT/"_NO_PROPS").write_text("odds_api module missing")
        raise SystemExit(2)

    try:
        df = _odds.fetch_props(date=date, season=season)
    except Exception as e:
        s["providers"]["odds_api"] = f"error:{type(e).__name__}: {e}"
        (MET/"fetch_status.json").write_text(json.dumps(s, indent=2))
        (OUT/"_NO_PROPS").write_text(str(e))
        raise SystemExit(2)

    n = 0 if df is None else len(df)
    s["providers"]["odds_api"] = "ok" if n>0 else "ok-empty"
    s["rows"]["odds_api"] = n
    (MET/"fetch_status.json").write_text(json.dumps(s, indent=2))

    if n == 0:
        (OUT/"_NO_PROPS").write_text("odds_api returned 0 rows")
        raise SystemExit(2)

    return df

if __name__ == "__main__":
    main()
