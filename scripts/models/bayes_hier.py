# scripts/models/bayes_hier.py
from . import Leg, LegResult
def run(leg: Leg) -> LegResult:
    # Placeholder: pull posterior mean/var from cached fit,
    # or fall back to priors blended with market.
    mu_b = leg.features.get("bayes_mu", leg.features.get("mu"))
    sd_b = leg.features.get("bayes_sd", leg.features.get("sd"))
    # Compute P(X>line) assuming Normal; later: draw from posterior predictive.
    from math import erf, sqrt
    z = (leg.line - mu_b)/max(1e-6,sd_b)
    p_over = 1 - 0.5*(1+erf(z/sqrt(2)))
    return LegResult(p_model=p_over, mu=mu_b, sigma=sd_b, notes="Bayes pooled")

