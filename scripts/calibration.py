import numpy as np
from scipy.stats import norm
import pandas as pd
import json, os

CAL_FILE = "inputs/calibration.json"

def brier(y_true, p_pred):
    p = np.clip(p_pred, 1e-6, 1-1e-6)
    return float(np.mean((p - y_true)**2))

def crps_normal(y, mu, sigma):
    # approximate closed form
    sigma = max(sigma, 1e-6)
    z = (y - mu)/sigma
    return float(sigma*( z*(2*norm.cdf(z)-1) + 2*norm.pdf(z) - 1/np.sqrt(np.pi) ))

def load_cal():
    if os.path.exists(CAL_FILE):
        with open(CAL_FILE, "r") as f:
            return json.load(f)
    return {"mu_shrink": 0.10}  # default α=0.10

def save_cal(d):
    with open(CAL_FILE, "w") as f:
        json.dump(d, f, indent=2)

def shrink_mu(mu, mu_mkt, alpha=None):
    cfg = load_cal()
    a = cfg.get("mu_shrink", 0.10) if alpha is None else alpha
    return 0.9*mu + a*mu_mkt

def postgame_update(results_csv: str, priced_csv: str):
    """
    results_csv columns: event_id, player_name_raw, market, outcome (Over/Under/Yes/No),
                         line, result_value
    priced_csv: the props_priced.csv you produced pre-game
    """
    pred = pd.read_csv(priced_csv)
    res  = pd.read_csv(results_csv)
    df = pred.merge(res, on=["event_id","player_name_raw","market","outcome","point"], how="inner")
    # binary hit
    df["hit"] = ((df["outcome"].isin(["Over","Yes"]) & (df["result_value"] > df["point"])) |
                 (df["outcome"].isin(["Under","No"]) & (df["result_value"] < df["point"]))).astype(int)
    b = brier(df["hit"].values, df["blend_prob"].values)
    # simple rule: if Brier worse than 0.22, increase market anchoring (raise shrink α)
    cal = load_cal()
    if b > 0.22:
        cal["mu_shrink"] = min(0.25, cal.get("mu_shrink", 0.10) + 0.02)
    else:
        cal["mu_shrink"] = max(0.05, cal.get("mu_shrink", 0.10) - 0.01)
    save_cal(cal)
    return {"brier": b, "new_mu_shrink": cal["mu_shrink"], "n": len(df)}
