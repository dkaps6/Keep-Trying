# scripts/rules_engine.py
from __future__ import annotations
import math
from typing import Dict, Tuple

# We reuse the atomic knobs you already have in elite_rules.py
from .elite_rules import (
    pressure_qb_adjust, sack_to_attempts, funnel_multiplier,
    injury_redistribution, coverage_penalty, airy_cap,
    boxcount_ypp_mod, script_escalators, pace_smoothing, volatility_widen
)

def _nz(v, d=0.0):
    try:
        return float(v) if v is not None and not math.isnan(float(v)) else d
    except Exception:
        return d

def _bool(v): return bool(v) if v is not None else False

def apply_rules(
    market: str,
    side: str,
    mu: float,
    sigma: float,
    *,
    # minimal context (safe defaults if empty)
    player: Dict,          # last-week player_form row as dict
    team_ctx: Dict,        # {"tgt_team_l4":..., "rush_att_team_l4":...}
    opp_pressure_z: float = 0.0,
    opp_pass_epa_z: float = 0.0,
    run_funnel: bool = False,
    pass_funnel: bool = False,
    alpha_limited: bool = False,
    tough_shadow: bool = False,
    heavy_man: bool = False,
    heavy_zone: bool = False,
    team_ay_att_z: float = 0.0,
    light_box_share: float | None = None,
    heavy_box_share: float | None = None,
    win_prob: float = 0.5,
    qb_inconsistent: bool = False,
    pressure_mismatch: bool = False,
) -> Tuple[float, float, str]:
    """
    Returns (mu_adj, sigma_adj, notes)
    """
    notes = []

    # 1) Funnels (small efficiency/volume multipliers)
    if market in ("player_receptions", "player_reception_yds", "player_pass_yds", "player_pass_tds"):
        mult = funnel_multiplier(True, def_rush_epa_z=( -1.0 if run_funnel else 0.0 ),
                                      def_pass_epa_z=(  1.0 if run_funnel else 0.0 ))
        if mult != 1.0:
            mu *= mult; notes.append(f"pass_funnel_mult={mult:.2f}")
    if market in ("player_rush_attempts", "player_rush_yds"):
        mult = funnel_multiplier(False, def_rush_epa_z=( 1.0 if run_funnel else 0.0 ),
                                       def_pass_epa_z=( -1.0 if run_funnel else 0.0 ))
        if mult != 1.0:
            mu *= mult; notes.append(f"run_funnel_mult={mult:.2f}")

    # 2) Coverage / shadow penalties or zone boost (WR/TE receiving)
    if market in ("player_receptions", "player_reception_yds"):
        ypt = _nz(player.get("rec_yds_l4", 0)) / max(_nz(player.get("rec_l4", 0), 1.0), 1.0)
        ts  = 0.17  # default if we don’t have team shares handy here
        ypt2, ts2 = coverage_penalty(ypt, ts, tough_shadow=tough_shadow, heavy_man=heavy_man, heavy_zone=heavy_zone)
        if abs(ypt2-ypt) > 1e-6 or abs(ts2-ts) > 1e-6:
            # scale mu proportionally by ypt change (ts change already flowed into baseline via shares)
            if ypt > 0:
                mu *= (ypt2 / ypt)
                notes.append("coverage_adj")

    # 3) Air-yards sanity cap for deep WR optimism
    if market == "player_reception_yds":
        ypr_proxy = _nz(player.get("rec_yds_l4", 0)) / max(_nz(player.get("rec_l4", 0), 1.0), 1.0)
        ypr_capped = airy_cap(ypr_proxy, team_ay_att_z, cap_pct=0.80, ay_threshold_z=-0.8)
        if ypr_capped < ypr_proxy and ypr_proxy > 0:
            mu *= (ypr_capped / ypr_proxy); notes.append("air_yards_cap")

    # 4) Box count leverage for RB YPC / attempts
    if market == "player_rush_yds":
        ypc_proxy = _nz(player.get("ry_l4", 0)) / max(_nz(player.get("ra_l4", 0), 1.0), 1.0)
        ypc2 = boxcount_ypp_mod(ypc_proxy, light_box_share=light_box_share, heavy_box_share=heavy_box_share)
        if ypc_proxy > 0 and abs(ypc2 - ypc_proxy) > 1e-6:
            mu *= (ypc2 / ypc_proxy); notes.append("boxcount_ypp_mod")

    # 5) Sack-to-attempt elasticity for pass attempts based markets
    if market in ("player_pass_yds", "player_pass_tds", "player_receptions", "player_reception_yds"):
        # treat “volume” part as indirectly affected
        atts_mult = sack_to_attempts(1.0, sack_rate_above_avg=0.0)  # placeholder hook for live sack rate
        if atts_mult != 1.0:
            mu *= atts_mult; notes.append("sack_elasticity")

    # 6) Pressure-adjusted QB baseline
    if market in ("player_pass_yds", "player_pass_tds"):
        before = mu
        mu = pressure_qb_adjust(mu, z_opp_pressure=opp_pressure_z, z_opp_pass_epa=opp_pass_epa_z)
        if mu != before:
            notes.append(f"pressure_qb_adj")

    # 7) Script escalators (rb atts up / qb scrambles down when favored)
    if market in ("player_rush_attempts", "player_rush_yds"):
        rb_atts, qb_scr = script_escalators(mu if market=="player_rush_attempts" else 0.0, 0.0, win_prob=win_prob)
        if market == "player_rush_attempts" and rb_atts != mu:
            mu = rb_atts; notes.append("script_escalator")

    # 8) Pace smoothing (already applied upstream for plays; here mild extra safety)
    mu *= pace_smoothing(1.0, 0.0, 0.0)

    # 9) Volatility widening (tails)
    if market in ("player_pass_yds", "player_reception_yds") and (pressure_mismatch or qb_inconsistent):
        sigma = volatility_widen(sigma, pressure_mismatch=pressure_mismatch, qb_inconsistent=qb_inconsistent)
        notes.append("volatility_widen")

    return mu, sigma, ";".join(notes)
