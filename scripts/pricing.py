# scripts/pricing.py
"""
Pricing utilities:
- Convert American odds <-> probability
- De-vig two-way markets
- Compute model probability at a line from (mu, sigma)
- Blend model vs market to get fair odds
- Compute edge and Kelly stake
"""

from __future__ import annotations
from typing import Optional, Tuple
import math

import numpy as np
from scipy.stats import norm


# ------------------------
# American odds converters
# ------------------------

def american_to_prob(odds: Optional[float]) -> Optional[float]:
    """Convert American odds to implied probability (includes vig)."""
    if odds is None:
        return None
    o = float(odds)
    if o == 0:
        return None
    if o < 0:
        x = abs(o)
        return x / (x + 100.0)
    else:
        return 100.0 / (o + 100.0)


def prob_to_american(p: Optional[float]) -> Optional[int]:
    """Convert probability (0,1) to American odds. Returns None if invalid."""
    if p is None or p <= 0.0 or p >= 1.0:
        return None
    if p > 0.5:
        # negative odds
        return -int(round(100.0 * p / (1.0 - p)))
    else:
        # positive odds
        return int(round(100.0 * (1.0 - p) / p))


def devig_american_prob(
    over_odds: Optional[float],
    under_odds: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    """
    De-vig two-way market quoted in American odds.

    If both sides exist:
      fair Over   = p_over / (p_over + p_under)
      fair Under  = p_under / (p_over + p_under)

    If only one side exists, return that side's implied prob (anchor) and None for the other.
    """
    p_o = american_to_prob(over_odds)
    p_u = american_to_prob(under_odds)
    if p_o is None and p_u is None:
        return (None, None)
    if p_o is None or p_u is None:
        return (p_o, p_u)
    s = p_o + p_u
    if s <= 0:
        return (None, None)
    return (p_o / s, p_u / s)


# ------------------------
# Model probability at line
# ------------------------

def normal_over_prob(mu: float, sigma: float, line: float) -> float:
    """Pr(X > line) for X~N(mu, sigma^2). Robust to sigma<=0."""
    s = max(float(sigma), 1e-6)
    z = (line - float(mu)) / s
    return float(1.0 - norm.cdf(z))


def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ------------------------
# Per-row pricing helpers
# ------------------------

def _default_sigma_for_market(market: str) -> float:
    m = (market or "").lower()
    if "reception" in m and "yard" not in m:
        return 1.8
    if "receiving" in m:
        return 26.0
    if "rushing" in m and "attempt" not in m:
        return 22.0
    if "attempt" in m:
        return 4.0
    if "passing yard" in m:
        return 48.0
    if "passing td" in m:
        return 0.9
    if "rush+rec" in m:
        return 28.0
    return 20.0


def _model_prob(row) -> Optional[float]:
    """
    Compute model probability for the quoted side at the quoted line.
    Continuous stats → Normal(mu, sigma).
    Anytime TD → Bernoulli with 'td_prob' if present; otherwise soft logistic from expected_tds.
    """
    market = str(row.get("market", "")).lower()
    side = str(row.get("outcome", row.get("side", "over"))).lower()
    line = row.get("point", None)

    # Binary markets (Anytime TD / Yes-No)
    if "anytime" in market or ("td" in market and "anytime" in market or "yes" in side or "no" in side):
        # Prefer explicit td_prob if available
        p = row.get("td_prob", None)
        if p is None:
            # crude proxy from expected_tds or team rate if present
            lam = float(row.get("expected_tds", 0.0) or 0.0)
            # squashed into (0,1)
            p = max(0.01, min(0.99, logistic(0.9 * lam - 1.1)))
        if side in ("yes", "over"):
            return float(p)
        else:
            return float(1.0 - p)

    # Continuous (yards, receptions, attempts, etc.)
    mu = row.get("mu", row.get("expected_yards", None))
    sigma = row.get("sigma", None)
    if mu is None:
        return None
    mu = float(mu)
    if sigma is None or float(sigma) <= 0.0:
        sigma = _default_sigma_for_market(market)

    # If no book line, can't compute over/under
    if line is None or (isinstance(line, float) and np.isnan(line)):
        return None

    line = float(line)
    p_over = normal_over_prob(mu, float(sigma), line)
    return float(p_over if side in ("over", "yes") else (1.0 - p_over))


def _market_fair_prob(row) -> Optional[float]:
    """Return de-vigged market probability for the quoted side if we have both sides; else anchor."""
    price = row.get("price", None)
    # If we only have one side’s price, return its implied prob (anchor)
    p = american_to_prob(price)
    return None if p is None else float(p)


def _blend(model_p: Optional[float], market_p: Optional[float], w_model: float = 0.65) -> Optional[float]:
    if model_p is None and market_p is None:
        return None
    if model_p is None:
        return float(market_p)
    if market_p is None:
        return float(model_p)
    w = float(w_model)
    return float(w * model_p + (1.0 - w) * market_p)


def _edge(blended_p: Optional[float], market_p: Optional[float]) -> Optional[float]:
    if blended_p is None or market_p is None:
        return None
    return float(blended_p - market_p)


def kelly_fraction(blended_p: Optional[float], price: Optional[float], cap: float = 0.05) -> Optional[float]:
    """
    Kelly for American odds; cap position (e.g., 5%).
    """
    if blended_p is None or price is None:
        return None
    # Decimal odds:
    if price >= 0:
        dec = 1.0 + (price / 100.0)
    else:
        dec = 1.0 + (100.0 / abs(price))
    b = dec - 1.0
    p = blended_p
    q = 1.0 - p
    k = (b * p - q) / b
    return float(max(0.0, min(cap, k)))
