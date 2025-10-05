# scripts/rules_engine.py
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List

def apply_rules(*args, **kwargs) -> Tuple[float, float, str]:
    # NEW style: first arg = features dict
    if args and isinstance(args[0], dict):
        features: Dict[str, Any] = args[0] or {}
        side   = kwargs.get("side",   features.get("side"))
        mu     = kwargs.get("mu",     features.get("mu_pred") or features.get("mu") or 0.0)
        sigma  = kwargs.get("sigma",  features.get("sigma_pred") or features.get("sigma") or 0.25)
        player = kwargs.get("player", features.get("player"))
        team_ctx = kwargs.get("team_ctx", features.get("team_ctx"))
    else:
        # OLD style: side, mu, sigma as positionals; features via kw
        if len(args) < 3:
            raise TypeError("apply_rules() requires either (features_dict) or (side, mu, sigma, ...)")
        side, mu, sigma = args[:3]
        features: Dict[str, Any] = kwargs.get("features", {}) or {}
        player = kwargs.get("player")
        team_ctx = kwargs.get("team_ctx")

    mu_adj, sigma_adj, notes = _apply_rules_core(
        side=side,
        mu=float(mu),
        sigma=float(sigma),
        player=player,
        team_ctx=team_ctx,
        features=features,
    )
    return mu_adj, sigma_adj, notes


def _apply_rules_core(
    *,
    side: Optional[str],
    mu: float,
    sigma: float,
    player: Optional[Dict[str, Any]],
    team_ctx: Optional[Dict[str, Any]],
    features: Dict[str, Any],
) -> Tuple[float, float, str]:
    """
    Starter "elite rules" (safe by default). Uses external features if present.
    """
    notes: List[str] = []

    # Sigma floor
    if sigma < 0.05:
        sigma = 0.05
        notes.append("sigma_floor_0.05")

    # Pressure-adjusted QB baseline (if features present)
    opp_press_z = features.get("opp_team_pressure_z") or features.get("opp_pressure_z") or 0.0
    opp_pass_epa_z = features.get("opp_pass_epa_z") or 0.0
    if "passing" in str(features.get("market", "")).lower():
        adj = (1.0 - 0.35 * float(opp_press_z)) * (1.0 - 0.25 * float(opp_pass_epa_z))
        if adj != 1.0:
            mu *= max(0.80, min(1.20, adj))
            notes.append("qb_pressure_epa_adj")

    # Funnel nudges (if present)
    run_funnel = features.get("opp_run_funnel", 0)
    pass_funnel = features.get("opp_pass_funnel", 0)
    if run_funnel and "rushing" in str(features.get("market", "")).lower():
        mu *= 1.03
        notes.append("run_funnel_+3%")
    if pass_funnel and ("receiv" in str(features.get("market", "")).lower() or "passing" in str(features.get("market", "")).lower()):
        mu *= 1.03
        notes.append("pass_funnel_+3%")

    # Volatility widening flags (if present)
    if features.get("qb_volatility_flag") or features.get("protection_mismatch"):
        sigma *= 1.15
        notes.append("vol_widen_+15%")

    # Cap absurd mu if piping weird data
    if not (-1e6 < mu < 1e6):
        mu = 0.0
        notes.append("mu_reset_out_of_bounds")

    return mu, sigma, "|".join(notes)
