# scripts/calibration.py
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np

"""
Usage:
  python -m scripts.calibration --postgame metrics/postgame.csv

postgame.csv expected columns (minimal):
  player, market_key, line, result_value
  # You can compute hit/miss yourself or provide result_value (stat achieved)

This script computes per-market bias and writes metrics/calibration.json with a simple shrink parameter.
"""

def compute_shrink(df: pd.DataFrame) -> dict:
    out = {}
    for mk, g in df.groupby("market_key"):
        # crude bias: compare share of results > line
        if "hit" in g.columns:
            over_rate = g["hit"].mean()
        elif "result_value" in g.columns:
            over_rate = (g["result_value"] > g["line"]).mean()
        else:
            continue
        # target is 0.5; if model tends to overpredict overs, shrink more toward market
        # map |over_rate-0.5| ∈ [0,0.2] → shrink ∈ [1.0, 0.85]
        delta = abs(over_rate - 0.5)
        shrink = max(0.85, 1.0 - delta)  # simple rule
        out[mk] = {"shrink": round(float(shrink), 3)}
    return out

def main(postgame_path: str):
    df = pd.read_csv(postgame_path)
    cal = compute_shrink(df)
    Path("metrics").mkdir(exist_ok=True)
    Path("metrics/calibration.json").write_text(json.dumps(cal, indent=2))
    print("wrote metrics/calibration.json:", cal)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--postgame", required=True)
    args = p.parse_args()
    main(args.postgame)
