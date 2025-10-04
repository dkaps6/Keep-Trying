import numpy as np
from scipy.stats import norm

def brier(y_true, p_pred):
    p = np.clip(p_pred, 1e-6, 1-1e-6)
    return float(np.mean((p - y_true)**2))

def crps_normal(y, mu, sigma):
    # Approx; for exact closed form we can add later
    from numpy import sqrt, pi
    z = (y - mu)/sigma
    return float(sigma*( z*(2*norm.cdf(z)-1) + 2*norm.pdf(z) - 1/np.sqrt(pi) ))

def shrink_mu(mu, mu_mkt, alpha=0.1):
    return 0.9*mu + alpha*mu_mkt
