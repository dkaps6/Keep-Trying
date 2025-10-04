# scripts/calibration.py
import json, os
import numpy as np

CAL_PATH = "metrics/calibration.json"

def load_cal():
    if os.path.exists(CAL_PATH):
        with open(CAL_PATH,"r") as f: return json.load(f)
    return {}

def save_cal(obj):
    os.makedirs("metrics", exist_ok=True)
    with open(CAL_PATH,"w") as f: json.dump(obj, f, indent=2)

def apply_shrinkage(market: str, mu: float):
    cal = load_cal()
    alpha = float(cal.get(market, 1.0))
    return 0.9 * mu + 0.1 * alpha * mu

def update_after_week(market: str, model_mean: float, actual: float):
    cal = load_cal()
    alpha = float(cal.get(market, 1.0))
    err = actual - model_mean
    alpha *= (0.99 if err < 0 else 1.01)  # shrink if overshoot
    cal[market] = float(np.clip(alpha, 0.85, 1.15))
    save_cal(cal)

