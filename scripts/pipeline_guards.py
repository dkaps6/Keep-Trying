# scripts/pipeline_guards.py
from __future__ import annotations
import pandas as pd

REQUIRED_PREPRICE_COLS = ["player", "market", "line", "price"]

def assert_preprice_ready(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_PREPRICE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns before pricing: "
            f"{missing}. Present: {list(df.columns)}"
        )
    if df.empty:
        raise ValueError(
            "No rows to price.\n"
            "- Most common cause: empty id_map or inner join dropped all rows.\n"
            "- See metrics/needs_mapping.csv and metrics/fetch_status.json"
        )

def write_xlsx_if_nonempty(df: pd.DataFrame, path: str) -> None:
    if df is None or df.empty:
        print("[info] Skipping XLSX export: no rows")
        return
    try:
        import pandas as pd  # noqa
        df.to_excel(path, index=False)
        print(f"[info] Wrote {path}")
    except Exception as e:
        print(f"[warn] Failed XLSX export ({path}): {e}")
