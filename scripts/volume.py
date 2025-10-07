# scripts/volume.py
# Volume estimator for player × market rows using team plays + PROE + player shares.

from __future__ import annotations
import pandas as pd
from typing import Any

DEF_PLAYS = 60.0    # conservative default team plays/gm
DEF_PROE  = 0.00    # neutral pass rate over expectation
DEF_RZ    = 0.20
DEF_TGT   = 0.18
DEF_RSH   = 0.40
DEF_YPT   = 7.8
DEF_YPC   = 4.2
DEF_YPA   = 7.1

def _f(x, default):
    try:
        if x is None: return float(default)
        v = float(x)
        if v != v: return float(default)  # NaN
        return v
    except Exception:
        return float(default)

def _cap(x, lo, hi):
    return max(lo, min(hi, x))

def estimate_volume(row: pd.Series) -> float:
    mk = str(row.get("market_internal", "")).strip()
    plays = _f(row.get("plays_est"), DEF_PLAYS)
    proe  = _f(row.get("proe"), DEF_PROE)
    rz    = _f(row.get("rz_rate"), DEF_RZ)

    base_pass = 0.56 + proe
    pass_rate = _cap(base_pass, 0.40, 0.70)
    rush_rate = 1.0 - pass_rate

    tgt_share  = _f(row.get("target_share"), DEF_TGT)
    rush_share = _f(row.get("rush_share"),  DEF_RSH)
    ypt = _f(row.get("ypt"), DEF_YPT)

    if mk in ("receptions","rec_yards","rush_rec_yards","pass_yards","pass_tds"):
        attempts = plays * pass_rate
        if mk == "receptions":
            # receptions ~= targets * catch%; catch% proxy from ypt/9
            catch_pct = _cap(ypt / 9.0, 0.45, 0.80)
            return attempts * tgt_share * catch_pct
        elif mk in ("rec_yards","rush_rec_yards"):
            # volume here is "targets"; efficiency applied in pricing
            return attempts * tgt_share
        elif mk == "pass_yards":
            return attempts
        elif mk == "pass_tds":
            return attempts * (rz * (0.30 + proe * 0.5))
    elif mk in ("rush_att","rush_yards"):
        carries = plays * rush_rate * rush_share
        return max(carries, 0.0)
    return 0.0

def add_volume(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df["volume_est"] = []
        return df
    df = df.copy()
    df["volume_est"] = df.apply(estimate_volume, axis=1)
    return df
