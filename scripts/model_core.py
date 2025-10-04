import numpy as np
from scipy.stats import norm
from .elite_rules import (pressure_qb_adjust, sack_to_attempts, funnel_multiplier,
                          injury_redistribution, coverage_penalty, airy_cap,
                          boxcount_ypp_mod, script_escalators, pace_smoothing,
                          volatility_widen)

DEFAULT_SD = {
    "player_rec_yds": 26.0, "player_receptions": 1.8,
    "player_rush_yds": 23.0, "player_rush_attempts": 3.0,
    "player_pass_yds": 48.0, "player_pass_tds": 0.9
}

def base_sigma(market): return DEFAULT_SD.get(market, 20.0)

def over_prob_normal(L, mu, sigma):
    sigma = max(float(sigma), 1e-6)
    z = (L - mu) / sigma
    return 1 - norm.cdf(z)

# -------- Volume model (simplified but real) --------
def _safe_div(a, b, default=0.0):
    return default if b in (0, None) else (a / b)

def estimate_team_rates(team_form_row):
    # pass rate ~ dropbacks / plays ; rush rate ~ rushes / plays
    plays = team_form_row.get("off_plays_l4") or team_form_row.get("off_plays") or 60
    drop = team_form_row.get("off_dropbacks_l4") or team_form_row.get("off_dropbacks") or 35
    rush = team_form_row.get("off_rushes_l4") or team_form_row.get("off_rushes") or 25
    pr = _safe_div(drop, plays, 0.55)
    rr = _safe_div(rush, plays, 0.45)
    return plays, pr, rr

def player_shares(player_row, team_rows_last4):
    # target share / rush share / catch rate / YPR / YPC
    tgt = player_row.get("tgt_l4", 0.0); rec = player_row.get("rec_l4", 0.0)
    ryd = player_row.get("rec_yds_l4", 0.0)
    ra  = player_row.get("ra_l4", 0.0);  ryd_rush = player_row.get("ry_l4", 0.0)
    team_tgts = max(1.0, team_rows_last4.get("tgt_team_l4", 30.0))
    team_att  = max(1.0, team_rows_last4.get("rush_att_team_l4", 25.0))
    tgt_share  = _safe_div(tgt, team_tgts, 0.17)
    rush_share = _safe_div(ra,  team_att,  0.35)
    catch_rate = _safe_div(rec, max(tgt, 1.0), 0.68)
    ypr        = _safe_div(ryd, max(rec, 1.0), 11.0)
    ypc        = _safe_div(ryd_rush, max(ra, 1.0), 4.2)
    return tgt_share, rush_share, catch_rate, ypr, ypc

def mu_receptions(plays, pass_rate, tgt_share, catch_rate):
    team_targets = plays * pass_rate * 1.0  # 1 target per dropback approx
    return team_targets * tgt_share * catch_rate

def mu_rec_yards(mu_rec, ypr):
    return mu_rec * max(ypr, 0.1)

def mu_rush_atts(plays, rush_rate, rush_share):
    team_rushes = plays * rush_rate
    return team_rushes * rush_share

def mu_rush_yards(atts, ypc): return atts * max(ypc, 0.1)

def mu_pass_yards(team_dropbacks, qb_ypa, z_opp_pressure=0.0, z_opp_pass_epa=0.0):
    base = team_dropbacks * max(qb_ypa, 3.0)
    return pressure_qb_adjust(base, z_opp_pressure, z_opp_pass_epa)

