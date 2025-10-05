# scripts/rules_engine.py
from __future__ import annotations

from typing import Dict, Tuple, List, Any

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _getz(d: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        v = float(d.get(key, default))
        if v != v:  # NaN guard
            return default
        return v
    except Exception:
        return default

def _lower(s: Any) -> str:
    return str(s or "").lower()

# ---------------------------------------------------------------------
# Core public API (new style)
# ---------------------------------------------------------------------
def apply_rules(features: Dict[str, Any]) -> Tuple[float, float, List[str]]:
    """
    Primary entrypoint. Expects a dict that *may* include:
      - 'mu', 'sigma', 'side', 'prop_type', 'position'
      - 'neutral_pass_rate_def_z', 'rz_trips_def_z',
        'explosive_pass_z', 'explosive_rush_z'
      - (optionally any of your older signals: 'opp_pressure_z', etc.)

    Returns (mu, sigma, notes)
    """
    notes: List[str] = []

    mu = float(features.get("mu", 0.0))
    sigma = float(features.get("sigma", max(1e-6, abs(mu) * 0.10)))  # safe default

    prop = _lower(features.get("prop_type"))
    pos  = _lower(features.get("position"))

    # ----------------------------
    # 1) Neutral pass rate (defense)
    # ----------------------------
    # Tilt pass-volume / receiving props when defense *faces* more passes in neutral situations.
    npr_z = _getz(features, "neutral_pass_rate_def_z", 0.0)
    if abs(npr_z) > 0 and (
        any(k in prop for k in ["rec", "reception", "pass att", "attempt"]) or
        pos in ["wr", "te", "qb"]
    ):
        # +3% mu per z (capped ±10%) — gentle, multiplicative
        scale = 1.0 + _clip(0.03 * npr_z, -0.10, 0.10)
        mu *= scale
        notes.append(f"neutral_pass_rate_def_z {npr_z:+.2f} → mu x{scale:.3f}")

    # ----------------------------
    # 2) Red-zone proxies (defense)
    # ----------------------------
    # If opponent allows more scoring/drive → slightly boost anytime TD style props.
    rz_def_z = _getz(features, "rz_trips_def_z", 0.0)
    if abs(rz_def_z) > 0 and any(k in prop for k in ["anytime", "td", "touchdown"]):
        scale = 1.0 + _clip(0.05 * rz_def_z, -0.15, 0.15)  # ±15% cap
        mu *= scale
        notes.append(f"rz_trips_def_z {rz_def_z:+.2f} → mu x{scale:.3f}")

    # ----------------------------
    # 3) Explosive splits → volatility (sigma)
    # ----------------------------
    exp_pass_z = _getz(features, "explosive_pass_z", 0.0)
    exp_rush_z = _getz(features, "explosive_rush_z", 0.0)

    # Heuristic classification
    is_receivingish = any(k in prop for k in ["rec", "reception", "receiving", "pass", "longest rec"])
    is_rushingish   = any(k in prop for k in ["rush", "rushing", "longest rush"])

    if abs(exp_pass_z) > 0 and (is_receivingish or pos in ["wr", "te", "qb"]):
        vol_scale = 1.0 + _clip(0.05 * exp_pass_z, -0.20, 0.20)  # ±20% cap on sigma
        sigma *= max(0.30, vol_scale)  # never let sigma collapse too far
        notes.append(f"explosive_pass_z {exp_pass_z:+.2f} → sigma x{vol_scale:.3f}")

    if abs(exp_rush_z) > 0 and (is_rushingish or pos in ["rb"]):
        vol_scale = 1.0 + _clip(0.05 * exp_rush_z, -0.20, 0.20)
        sigma *= max(0.30, vol_scale)
        notes.append(f"explosive_rush_z {exp_rush_z:+.2f} → sigma x{vol_scale:.3f}")

    # ----------------------------
    # 4) (Optional) QB pressure / pass-epa stubs remain for future use
    # ----------------------------
    # If you already feed these in features, you can uncomment the nudges.
    # opp_press = _getz(features, "opp_pressure_z", 0.0)
    # opp_pass_epa = _getz(features, "opp_pass_epa_z", 0.0)
    # if abs(opp_press) > 0 and any(k in prop for k in ["pass", "attempt", "passing"]):
    #     scale = 1.0 + _clip(-0.02 * opp_press, -0.08, 0.08)
    #     mu *= scale
    #     notes.append(f"opp_pressure_z {opp_press:+.2f} → mu x{scale:.3f}")

    # Safety floor for sigma
    sigma = max(sigma, max(1e-6, abs(mu) * 0.02))
    return mu, sigma, notes


# ---------------------------------------------------------------------
# Backward compatibility wrapper
# Your engine has called this shape in some runs
# ---------------------------------------------------------------------
def apply_rules_compat(fdct: Dict[str, Any],
                       side: str,
                       base_mu: float,
                       base_sigma: float) -> Tuple[float, float, List[str]]:
    """
    Legacy compatibility wrapper:
      returns (mu, sigma, notes)
    """
    # Merge base values into a new features dict for the new apply_rules
    features = dict(fdct or {})
    features.setdefault("mu", base_mu)
    features.setdefault("sigma", base_sigma)
    features.setdefault("side", side)
    return apply_rules(features)
