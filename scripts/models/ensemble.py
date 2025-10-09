# scripts/models/ensemble.py
from . import Leg, LegResult
from . import monte_carlo, bayes_hier, markov, ml_ensemble

def blend(leg: Leg, context: dict) -> dict:
    res = {}
    r_mc  = monte_carlo.run(leg)
    r_b   = bayes_hier.run(leg)
    r_mk  = markov.run(leg)
    r_ml  = ml_ensemble.run(leg)

    # dynamic weights (can be config-driven)
    w_mc = context.get("w_mc", 0.25)
    w_b  = context.get("w_bayes", 0.25)
    w_mk = context.get("w_markov", 0.25)
    w_ml = context.get("w_ml", 0.25)

    p_blend = w_mc*r_mc.p_model + w_b*r_b.p_model + w_mk*r_mk.p_model + w_ml*r_ml.p_model
    p_mkt   = leg.features.get("p_market_fair", 0.5)
    p_final = 0.65*p_blend + 0.35*p_mkt

    return {
        "p_mc": r_mc.p_model, "p_bayes": r_b.p_model, "p_markov": r_mk.p_model, "p_ml": r_ml.p_model,
        "p_blend": p_blend, "p_market": p_mkt, "p_final": p_final,
        "mu": r_mc.mu or r_b.mu or r_mk.mu, "sigma": r_mc.sigma or r_b.sigma or r_mk.sigma,
        "notes": " | ".join([r_mc.notes, r_b.notes, r_mk.notes, r_ml.notes]),
    }

