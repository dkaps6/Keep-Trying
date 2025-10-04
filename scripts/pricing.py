# scripts/pricing.py
import math

def american_to_prob(odds: float) -> float:
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return (-odds) / ((-odds) + 100.0)

def prob_to_american(p: float) -> float:
    p = max(min(float(p), 0.999999), 1e-6)
    if p >= 0.5:
        return - (p * 100.0) / (1.0 - p)
    else:
        return (100.0 * (1.0 - p)) / p

def devig_two_way(p_over_raw: float | None, p_under_raw: float | None):
    """Return de-vigged fair probs for both sides when at least one is present."""
    if p_over_raw is None and p_under_raw is None:
        return (0.5, 0.5)
    if p_over_raw is None:
        return (1.0 - p_under_raw, p_under_raw)
    if p_under_raw is None:
        return (p_over_raw, 1.0 - p_over_raw)
    s = p_over_raw + p_under_raw
    if s <= 0: return (0.5, 0.5)
    return (p_over_raw / s, p_under_raw / s)

def blend(p_model: float, p_market_fair: float, w_model: float = 0.65) -> float:
    return max(min(w_model * p_model + (1.0 - w_model) * p_market_fair, 1.0), 0.0)

def edge_pct(p_blend: float, p_market_fair: float) -> float:
    """
    Edge in probability points (−1..+1). If you prefer percent points, multiply by 100 in caller.
    """
    return float(p_blend) - float(p_market_fair)

def kelly_fraction(p_blend: float, american_price: float, cap: float = 0.05) -> float:
    """Fractional Kelly vs posted price."""
    b = abs(american_price) / 100.0 if american_price < 0 else 100.0 / american_price
    q = 1.0 - p_blend
    f = (p_blend * (b + 1.0) - 1.0) / b
    return max(0.0, min(f, cap))

def tier(edge_pp: float) -> str:
    """Classify by edge in probability points (e.g., 0.06 = 6pp)."""
    if edge_pp >= 0.06: return "ELITE"
    if edge_pp >= 0.04: return "GREEN"
    if edge_pp >= 0.01: return "AMBER"
    return "RED"
