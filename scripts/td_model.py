# scripts/td_model.py (STRICT)

from __future__ import annotations
import numpy as np

def total_to_td_lambda(team_total_points: float) -> float:
    lam = float(team_total_points) / 7.0
    if lam <= 0:
        raise ValueError("[STRICT] team_total_points must be > 0 for Anytime TD modeling.")
    return lam

def player_td_probability(lam_team: float, rz_share: float, role_bias: float = 1.0) -> float:
    lam = float(lam_team)
    rz  = float(rz_share)
    rb  = float(role_bias)
    if lam <= 0 or rz <= 0:
        raise ValueError("[STRICT] lam_team and rz_share must be > 0 for Anytime TD modeling.")
    cap = min(0.9, max(0.0, rz * rb))
    p = 1.0 - np.exp(-lam * cap)
    if p <= 0 or p >= 1:
        # guard catastrophic inputs (should not happen with sane rz and lam)
        raise ValueError(f"[STRICT] Computed TD probability out of (0,1): {p}")
    return float(p)
