from __future__ import annotations
import pandas as pd
from pathlib import Path

def _read(path, cols=None):
    p=Path(path)
    if not p.exists() or p.stat().st_size==0: return pd.DataFrame(columns=cols or [])
    return pd.read_csv(p)

def main():
    Path("outputs").mkdir(exist_ok=True)
    sheets = {
        "team_form": _read("data/team_form.csv"),
        "player_form": _read("data/player_form.csv"),
        "metrics_ready": _read("data/metrics_ready.csv"),
        "props_raw": _read("outputs/props_raw.csv"),
        "odds_game": _read("outputs/odds_game.csv"),
        "predictions": _read("outputs/master_model_predictions.csv"),
        "sgp_candidates": _read("outputs/sgp_candidates.csv"),
    }
    out = Path("outputs/master_model_report.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
    print(f"[export] wrote {out} with sheets={list(sheets)}")

if __name__=="__main__":
    main()
