# scripts/models/monte_carlo.py
import numpy as np
from . import Leg, LegResult
from math import erf, sqrt

def normal_cdf(x): return 0.5*(1+erf(x/sqrt(2)))

def run(leg: Leg, n=20000) -> LegResult:
    mu = float(leg.features.get("mu", 0.0))
    sd = max(1e-6, float(leg.features.get("sd", 1.0)))
    # volatility widening
    widen = float(leg.features.get("sd_widen", 1.0))
    sd *= widen
    # direct CDF is faster & exact for Normal:
    z = (leg.line - mu)/sd
    p_over = 1.0 - normal_cdf(z)
    return LegResult(p_model=p_over, mu=mu, sigma=sd, notes=f"MC Normal μ={mu:.1f}, σ={sd:.1f}")

