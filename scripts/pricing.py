# scripts/pricing.py
# Robust pricing helpers for props: convert odds, compute probs, fair price, edge, and Kelly.
# All functions are NaN-tolerant and safe for vectorized use.

from __future__ import annotations
import math
import numpy as np
import pandas as pd

# ---------- Odds / Probability conversions ----------

def american_to_decimal(american: float | int | str) -> float | float("nan"):
    """
    Convert American odds (e.g., -120, +115) to decimal odds.
    Returns np.nan on invalid input.
    """
    try:
        a = float(american)
    except Exception:
        return np.nan
    if a == 0:
        return np.nan
    if a > 0:
        return 1.0 + (a / 100.0)
    else:
        return 1.0 + (100.0 / abs(a))

def implied_prob_from_american(american: float | int | str) -> float | float("nan"):
    """
    Convert American odds to implied probability (book, not de-vigged).
    """
    d = american_to_decimal(american)
    if not np.isfinite(d) or d <= 1.0:
        return np.nan
    return 1.0 / d

def prob_to_american(p: float | int | str) -> float | float("nan"):
    """
    Convert probability in [0,1] to American odds.
    """
    try:
        p = float(p)
    except Exception:
        return np.nan
    if not (0.0 < p < 1.0):
        return np.nan
    # decimal odds first
    d = 1.0 / p
    # convert decimal to American style
    if d >= 2.0:
        # positive American
        return (d - 1.0) * 100.0
    else:
        # negative American
        return -100.0 / (d - 1.0)

# ---------- Kelly ----------

def kelly(prob: float | int | str, american_price: float | int | str, floor_zero: bool = True) -> float:
    """
    Fractional Kelly on American odds.
    If prob or price is invalid → returns 0.
    """
    try:
        p = float(prob)
    except Exception:
        return 0.0
    if not (0.0 <= p <= 1.0):
        return 0.0

    d = american_to_decimal(american_price)
    if not np.isfinite(d) or d <= 1.0:
        return 0.0

    b = d - 1.0  # net decimal payout
    f = (p * (b + 1.0) - 1.0) / b  # equivalent to (bp - q)/b with stake = 1
    if floor_zero:
        return float(max(0.0, f))
    return float(f)

# ---------- Pricing frame helpers ----------

def compute_market_prob(series: pd.Series) -> pd.Series:
    """
    series: American odds (string/number). Returns implied probabilities.
    """
    return series.apply(implied_prob_from_american)

def compute_fair_prob(model_prob: pd.Series) -> pd.Series:
    """
    For now, fair_prob = model_prob (you can adjust to incorporate sigma, correlation, etc.)
    """
    return model_prob.astype(float)

def compute_edge(fair_prob: pd.Series, market_prob: pd.Series) -> pd.Series:
    """
    Simple probability edge: fair - market. Returns NaN where either is NaN.
    """
    fair = pd.to_numeric(fair_prob, errors="coerce")
    mkt = pd.to_numeric(market_prob, errors="coerce")
    out = fair - mkt
    out[~np.isfinite(fair) | ~np.isfinite(mkt)] = np.nan
    return out

def compute_kelly_col(prob: pd.Series, american_price: pd.Series) -> pd.Series:
    return pd.Series([
        kelly(p, a) if (np.isfinite(p) and p >= 0.0 and p <= 1.0) else 0.0
        for p, a in zip(pd.to_numeric(prob, errors="coerce"), american_price)
    ], index=prob.index)

def attach_pricing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mutates and returns df with: market_prob, fair_prob, fair_american, edge, kelly.
    Required columns:
        - 'odds' (American)  → we’ll map from 'price' if needed in engine
        - 'model_prob'       → from rules (mu/sigma → prob)
    """
    if "odds" not in df.columns and "price" in df.columns:
        df["odds"] = df["price"]

    # numeric conversions
    if "model_prob" in df.columns:
        df["model_prob"] = pd.to_numeric(df["model_prob"], errors="coerce")
    if "odds" in df.columns:
        # keep as string/number; conversions happen per-row
        pass

    # market & fair
    df["market_prob"]   = compute_market_prob(df.get("odds", pd.Series(index=df.index)))
    df["fair_prob"]     = compute_fair_prob(df.get("model_prob", pd.Series(index=df.index)))

    # fair price in American
    df["fair_american"] = df["fair_prob"].apply(prob_to_american)

    # edge & kelly
    df["edge"]  = compute_edge(df["fair_prob"], df["market_prob"])
    df["kelly"] = compute_kelly_col(df["fair_prob"], df.get("odds", pd.Series(index=df.index)))

    return df
