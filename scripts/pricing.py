# scripts/pricing.py
# Utilities for converting American odds <-> probabilities and removing vig
# for 2-way markets.

from typing import Optional, Tuple


def american_to_prob(odds: Optional[float]) -> Optional[float]:
    """
    Convert American odds to implied probability (includes vig if present).
    Returns None for None/zero.
    +150  -> 100 / (150 + 100) = 0.4000
    -150  -> 150 / (150 + 100) = 0.6000
    """
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
    """
    Convert a fair probability to American odds. Returns None if p invalid.
    """
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
    De-vig a 2-way market quoted in American odds.

    If both sides are provided:
        p_over_implied = american_to_prob(over_odds)
        p_under_implied = american_to_prob(under_odds)
        fair Over   = p_over_implied / (p_over_implied + p_under_implied)
        fair Under  = p_under_implied / (p_over_implied + p_under_implied)

    If only one side is provided, we cannot remove vig; we return the quoted
    implied probability for that side and None for the other. Callers treat
    this as a market anchor (common on props where one side is missing).
    """
    p_o = american_to_prob(over_odds)
    p_u = american_to_prob(under_odds)

    if p_o is None and p_u is None:
        return (None, None)
    if p_o is None or p_u is None:
        # One-sided quote — return the anchor as-is.
        return (p_o, p_u)

    s = p_o + p_u
    if s <= 0:
        return (None, None)

    fair_o = p_o / s
    fair_u = p_u / s
    return (fair_o, fair_u)


# Convenience aliases some parts of the code may import.
def implied_prob_from_american(odds: float) -> float:
    return american_to_prob(odds)  # type: ignore[return-value]


def fair_prob_from_two_way(over_odds: float, under_odds: float) -> Tuple[float, float]:
    return devig_american_prob(over_odds, under_odds)  # type: ignore[return-value]
