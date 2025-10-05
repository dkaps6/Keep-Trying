# scripts/rules_engine.py
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List

"""
Compatibility wrapper for rules application.

Supported call styles:

NEW:
    apply_rules(features_dict)
    apply_rules(features_dict, side='over', mu=0.5, sigma=0.25, player=..., team_ctx=...)

OLD (positional):
    apply_rules(side, mu, sigma, player=..., team_ctx=...)

In all cases, returns: (mu, sigma, notes)
"""

def apply_rules(*args, **kwargs) -> Tuple[float, float, str]:
    # Detect NEW style: first arg is a dict of features
    if args and isinstance(args[0], dict):
        features: Dict[str, Any] = args[0] or {}
        side   = kwargs.get("side",   features.get("side"))
        mu     = kwargs.get("mu",     features.get("mu_pred") or features.get("mu"))
        sigma  = kwargs.get("sigma",  features.get("sigma_pred") or features.get("sigma") or 0.25)
        player = kwargs.get("player", features.get("player"))
        team_ctx = kwargs.get("team_ctx", features.get("team_ctx"))
    else:
        # OLD style: side, mu, sigma as positionals; features may be provided via kw
        if len(args) < 3:
            raise TypeError(
                "apply_rules() requires either (features_dict) or (side, mu, sigma, ...)"
            )
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
    Put your real “elite” rules here. This starter keeps behavior stable but
    also protects against bad inputs so the pipeline won’t crash.
    """
    notes: List[str] = []

    # — Examples of gentle, safe adjustments —
    # floor sigma to avoid over-confidence explosions
    if sigma < 0.05:
        sigma = 0.05
        notes.append("sigma_floor_0.05")

    # optional: cap or backstop mu if missing
    if mu is None:
        mu = 0.0
        notes.append("mu_none_to_0")

    # Example: light bump if team/form feature says “hot”
    hot_form = features.get("team_form_hot") or features.get("form_hot")
    if hot_form:
        mu += 0.01
        notes.append("bump_hot_form")

    # Make sure outputs are clean floats
    try:
        mu = float(mu)
    except Exception:
        mu = 0.0
        notes.append("mu_cast_fail_to_0")

    try:
        sigma = float(sigma)
    except Exception:
        sigma = 0.25
        notes.append("sigma_cast_fail_to_0.25")

    return mu, sigma, "|".join(notes)
