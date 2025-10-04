# scripts/td_model.py
import math

def implied_team_totals(total_points: float, spread_home_minus_away: float, home_is_favorite: bool):
    """
    Split total into home/away team totals. Rough linear mapping: ~0.45 total swing per point of spread.
    """
    k = 0.45
    home_total = total_points/2 - k * spread_home_minus_away
    away_total = total_points - home_total
    return max(home_total, 0.0), max(away_total, 0.0)

def total_to_td_lambda(team_points: float) -> float:
    """Approximate expected TDs from team points."""
    return max(team_points / 7.0, 0.0)

def player_td_probability(team_td_lambda: float, redzone_share: float, goal_line_bias: float = 1.0) -> float:
    lam = max(team_td_lambda * max(redzone_share, 0.0) * max(goal_line_bias, 0.0), 0.0)
    return 1.0 - math.exp(-lam)

def qb_pass_tds_lambda(team_td_lambda: float, pass_rate: float, gadget_td_share: float = 0.03) -> float:
    lam = max(team_td_lambda * max(pass_rate, 0.0) * (1.0 - gadget_td_share), 0.0)
    return lam

def yes_prob_from_lambda(lam: float) -> float:
    return 1.0 - math.exp(-max(lam, 0.0))
