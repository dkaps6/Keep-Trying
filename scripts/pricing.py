# scripts/pricing.py
from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# default SDs (tune later / make dynamic via features)
SD_DEFAULTS = {
    "rec_yards": 26.0,
    "receptions": 1.8,
    "rush_yards": 22.0,
    "rush_att": 3.0,
    "pass_yards": 48.0,
    "pass_tds": 0.9,
    "rush_rec_yards": 30.0,
}

def american_to_prob(odds: float) -> float:
    if odds is None or np.isnan(odds): return np.nan
    if odds < 0: return (-odds) / ((-odds) + 100.0)
    return 100.0 / (odds + 100.0)

def prob_to_american(p: float) -> float:
    p = min(max(p, 1e-6), 1-1e-6)
    if p >= 0.5:
        return - (p / (1-p)) * 100.0
    return ( (1-p) / p ) * 100.0

def devig_two_way(p_over: float, p_under: float) -> float | None:
    if np.isnan(p_over) or np.isnan(p_under): return None
    s = p_over + p_under
    if s <= 0: return None
    return p_over / s

def norm_inv(p: float) -> float:
    # scipy-free probit (approx)
    return np.sqrt(2) * erfinv(2*p - 1.0)

def erfinv(x: float) -> float:
    # Winitzki approximation for erfinv
    a = 0.147
    ln = np.log(1.0 - x*x)
    s = (2/(np.pi*a) + ln/2.0)
    inner = s*s - ln/a
    return np.sign(x) * np.sqrt( np.sqrt(inner) - s )

def price_props(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Input columns: player, market_key, line, price_over, price_under, book, event_id, ...
    Output: projections + fair odds + edge
    """
    df = df_in.copy()

    # implied probs
    df["p_over_imp"] = df["price_over"].apply(american_to_prob)
    df["p_under_imp"] = df["price_under"].apply(american_to_prob)

    # devig if both sides present
    df["p_over_fair_market"] = np.vectorize(devig_two_way)(df["p_over_imp"], df["p_under_imp"])
    # if only one side, fall back to single-side implied (anchor)
    df["p_over_fair_market"] = df["p_over_fair_market"].fillna(df["p_over_imp"])

    # model probability at line using default SD heuristic
    def model_prob(row) -> float:
        mk = row["market_key"]
        sd = SD_DEFAULTS.get(mk, 25.0)
        p = row["p_over_fair_market"]
        if pd.isna(p) or sd <= 0:
            return np.nan
        z = norm_inv(p)
        # mean consistent with P(X>line)=p for Normal(μ,σ)
        mu = row["line"] + z * sd
        row["_mu"] = mu
        row["_sd"] = sd
        return p  # our baseline model uses market-implied p for now

    df["p_over_model"] = df.apply(model_prob, axis=1)

    # 65/35 blend
    df["p_over_blend"] = 0.65*df["p_over_model"] + 0.35*df["p_over_fair_market"]
    df["fair_over_odds"] = df["p_over_blend"].apply(prob_to_american)

    # edge vs book (only if we have price_over)
    df["edge_pct"] = df["p_over_blend"] - df["p_over_imp"]
    # clean projection columns
    df["model_mean"] = df.get("_mu", np.nan)
    df["model_sd"] = df.get("_sd", np.nan)

    # order columns
    keep = [
        "player","market_key","line","book","event_id","home_team","away_team","commence_time",
        "price_over","price_under","p_over_imp","p_under_imp","p_over_fair_market",
        "p_over_model","p_over_blend","fair_over_odds","edge_pct","model_mean","model_sd"
    ]
    out = df[keep].copy()

    # tiering
    def tier(e):
        if pd.isna(e): return "NA"
        if e >= 0.06: return "ELITE"
        if e >= 0.04: return "GREEN"
        if e >= 0.01: return "AMBER"
        return "RED"
    out["tier"] = out["edge_pct"].apply(tier)

    return out

def write_outputs(df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    csvp = outdir / "props_priced.csv"
    xlsx = outdir / "props_priced.xlsx"
    df.to_csv(csvp, index=False)
    try:
        df.to_excel(xlsx, index=False)  # requires openpyxl
    except Exception as e:
        print("[warn] xlsx export failed:", e)
