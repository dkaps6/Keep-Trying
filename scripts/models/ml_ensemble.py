# scripts/models/ml_ensemble.py
from . import Leg, LegResult
def run(leg: Leg) -> LegResult:
    # Expect a pre-trained model per market loaded elsewhere.
    p_ml = leg.features.get("p_ml")  # if present from saved model
    if p_ml is None: return LegResult(p_model=0.5, mu=None, sigma=None, notes="ML fallback 0.5")
    return LegResult(p_model=float(p_ml), mu=None, sigma=None, notes="ML ensemble")

