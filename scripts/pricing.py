# scripts/pricing.py
# Pricing utilities used by engine.py
from __future__ import annotations

import math
from typing import Iterable, Union

import numpy as np
import pandas as pd

ProbArray = Union[np.ndarray, pd.Series, Iterable[float]]

# ----------------------------
# 1) Model-side probability
# ----------------------------
def _model_prob(mu: ProbArray, sigma: ProbArray) -> np.ndarray:
    """
    Convert strength (mu / sigma) into a probability via a clipped logistic.
    This does not require a market threshold, so it works even when lines are missing.

    p = sigmoid( clip(mu/sigma, -6, 6) )

    Returns np.ndarray in [0.01, 0.99] to avoid exact 0/1.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    z = mu / np.maximum(sigma, 1e-6)
    z = np.clip(z, -6.0, 6.0)
    p = 1.0 / (1.0 + np.exp(-z))
    # soften extremes slightly
    return np.clip(p, 0.01, 0.99)


# ----------------------------
# 2) Market fair probability
# ----------------------------
def _american_to_prob(price: Union[float, int]) -> float:
    """Implied probability from American odds (no vig)."""
    try:
        price = float(price)
    except Exception:
        return np.nan
    if price > 0:
        return 100.0 / (price + 100.0)
    else:
        return (-price) / ((-price) + 100.0)


def _devig_pair(p_over: float, p_under: float) -> tuple[float, float]:
    """
    Simple two-way de-vig: scale so p_over + p_under = 1 when both sides present.
    If either side is NaN, return inputs.
    """
    if np.isnan(p_over) or np.isnan(p_under):
        return p_over, p_under
    s = p_over + p_under
    if s <= 0:
        return p_over, p_under
    return p_over / s, p_under / s


def _market_fair_prob(df: pd.DataFrame) -> pd.Series:
    """
    Compute market fair probability from 'price' (American odds).
    If both sides are present within the same (event, market, player, line) group,
    de-vig them so they sum to 1. Otherwise fall back to single side implied.
    """
    if df is None or len(df) == 0:
        return pd.Series([], dtype=float)

    # Try to identify the natural grouping keys present in your odds dataframe.
    # We’ll include only keys that actually exist to stay robust.
    candidate_keys = ["event_id", "market", "player", "team", "line"]
    keys = [k for k in candidate_keys if k in df.columns]
    if not keys:
        # no grouping info; just single-side implied
        return df.get("price", pd.Series(np.nan, index=df.index)).map(_american_to_prob)

    work = df.copy()
    work["side"] = work.get("side") if "side" in work.columns else ""

    # implied single-side
    work["imp"] = work["price"].map(_american_to_prob)

    # Two-way de-vig when we can
    def devig_grp(g: pd.DataFrame) -> pd.Series:
        over_mask = g["side"].astype(str).str.lower().str.contains("over")
        under_mask = g["side"].astype(str).str.lower().str.contains("under")
        if over_mask.any() and under_mask.any():
            p_over = g.loc[over_mask, "imp"].iloc[0]
            p_under = g.loc[under_mask, "imp"].iloc[0]
            p_over_fair, p_under_fair = _devig_pair(p_over, p_under)
            fair = g["imp"].copy()
            fair.loc[over_mask] = p_over_fair
            fair.loc[under_mask] = p_under_fair
            return fair
        return g["imp"]

    fair = work.groupby(keys, group_keys=False).apply(devig_grp)
    fair.index = df.index  # align
    return fair


# ----------------------------
# 3) Blend, edge, conversions
# ----------------------------
def _blend(model_prob: ProbArray, fair_prob: ProbArray, w: float = 0.5) -> np.ndarray:
    """
    Weighted blend of model and market (default 50/50).
    """
    mp = np.asarray(model_prob, dtype=float)
    fp = np.asarray(fair_prob, dtype=float)
    return np.clip(w * mp + (1.0 - w) * fp, 0.0, 1.0)


def prob_to_american(p: ProbArray) -> np.ndarray:
    """
    Convert probability to American odds.
    """
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    out = np.where(
        p >= 0.5,
        - (p / (1.0 - p)) * 100.0,       # favorites (negative)
        ((1.0 - p) / p) * 100.0          # underdogs (positive)
    )
    return np.round(out, 0)


def _edge(model_prob: ProbArray, fair_prob: ProbArray) -> np.ndarray:
    """
    Edge as difference in probabilities (model - market fair).
    Positive => model more bullish than market.
    """
    return np.asarray(model_prob, dtype=float) - np.asarray(fair_prob, dtype=float)


# ----------------------------
# 4) Kelly criterion (American odds)
# ----------------------------
def _american_to_decimal(price: Union[float, int]) -> float:
    try:
        price = float(price)
    except Exception:
        return np.nan
    if price > 0:
        return 1.0 + price / 100.0
    else:
        return 1.0 + 100.0 / (-price)


def kelly(prob: ProbArray, price: ProbArray) -> np.ndarray:
    """
    Kelly fraction using probability 'prob' and American odds 'price'.
    Returns fraction of bankroll to wager (clipped to [0, 1]).
    If inputs are missing/invalid, returns 0.
    """
    p = np.asarray(prob, dtype=float)
    dec = np.asarray([_american_to_decimal(v) for v in np.asarray(price)], dtype=float)
    b = dec - 1.0
    q = 1.0 - p
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (b * p - q) / b
    f = np.where(~np.isfinite(f), 0.0, f)
    return np.clip(f, 0.0, 1.0)


__all__ = [
    "_model_prob",
    "_market_fair_prob",
    "_blend",
    "prob_to_american",
    "_edge",
    "kelly",
]

