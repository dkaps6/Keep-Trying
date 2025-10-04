import numpy as np
from scipy.stats import norm

# default pairwise correlations
DEFAULT_R = {
    ("QB_pass_yds","WR_rec_yds"): 0.60,
    ("RB_rush_yds","QB_pass_yds"): -0.35,
    ("WR1_rec_yds","WR2_rec_yds"): 0.20,
}

def _rho(a, b, overrides=None):
    if overrides and (a,b) in overrides: return overrides[(a,b)]
    if overrides and (b,a) in overrides: return overrides[(b,a)]
    return DEFAULT_R.get((a,b), DEFAULT_R.get((b,a), 0.0))

def parlay_prob_monte_carlo(legs, n=5000, overrides=None, seed=42):
    """
    legs: list of dicts with keys:
      - 'name': unique leg label
      - 'p': marginal hit probability (blended)
      - 'tag': market tag for correlation lookup e.g., 'QB_pass_yds'
    """
    rng = np.random.default_rng(seed)
    k = len(legs)
    # Build correlation matrix for normals
    R = np.eye(k)
    for i in range(k):
        for j in range(i+1, k):
            R[i,j] = R[j,i] = _rho(legs[i]["tag"], legs[j]["tag"], overrides)

    # Cholesky (ensure PSD)
    eps = 1e-6
    try:
        L = np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        # fallback: jitter diagonal
        L = np.linalg.cholesky(R + eps*np.eye(k))

    # thresholds for marginals
    z = norm.ppf([leg["p"] for leg in legs])

    hits = 0
    for _ in range(n):
        z0 = rng.standard_normal(k)
        zc = L @ z0
        # leg is hit if zc[i] <= z[i] for Bernoulli via probit
        ok = True
        for i in range(k):
            if not (zc[i] <= z[i]):
                ok = False; break
        hits += 1 if ok else 0
    return hits / n
