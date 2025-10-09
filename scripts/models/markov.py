# scripts/models/markov.py
from . import Leg, LegResult
def run(leg: Leg) -> LegResult:
    # Use pace, PROE, spread, funnels to adjust attempts/targets,
    # then map to yards via efficiency priors.
    att = leg.features.get("adj_attempts", None)
    eff = leg.features.get("eff_mu", None)   # yds/att or yds/tgt
    sd  = leg.features.get("eff_sd", None)
    if att and eff and sd:
        mu = att * eff
        # variance: att*sd^2 + att*(1-att/n)*eff^2 (approx) -> keep simple:
        sigma = (att**0.5)*sd
        # price at line:
        from math import erf, sqrt
        z = (leg.line - mu)/max(1e-6,sigma)
        p_over = 1 - 0.5*(1+erf(z/sqrt(2)))
        return LegResult(p_model=p_over, mu=mu, sigma=sigma, notes="Markov volume-adjusted")
    return LegResult(p_model=0.5, mu=None, sigma=None, notes="Markov fallback 0.5")

