# scripts/pricing.py (STRICT)
# - NO silent fallbacks. Hard-fails if required inputs are missing or NaN.
# - Devig + 65/35 blend
# - Weather multipliers (μ, σ)
# - Plays/PROE volume μ blended into pre-weather μ (0.35 weight)
# - Anytime TD as Bernoulli + 65/35 blend (requires team_total + rz_share + role)
# - Calibration shrink on μ (uses metrics/calibration.json)
# - Kelly & tiering + CSV/XLSX outputs

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from scripts.td_model import total_to_td_lambda, player_td_probability
from scripts.volume import team_volume_estimates


# ---------------------------
# Strict validators
# ---------------------------

GLOBAL_REQUIRED: List[str] = [
    # core
    "player", "team", "opponent", "market", "line",
    # book odds to devig
    "over_odds", "under_odds",
    # core model fields
    "mu0", "mu_mult_nowx", "sig_mult_nowx", "sigma",
    # game context for volume/script
    "team_wp", "is_home",
    # weather (explicitly required)
    "wind_mph", "precip", "temp_f",
]

# Per-market columns that MUST be present and non-null
PER_MARKET_REQUIRED: Dict[str, List[str]] = {
    "receptions":     ["plays_base", "proe", "pace_z", "target_share", "route_rate", "catch_rate"],
    "rec_yards":      ["plays_base", "proe", "pace_z", "route_rate", "yprr"],
    "rush_yards":     ["plays_base", "proe", "pace_z", "rush_share", "ypc"],
    "rush_att":       ["plays_base", "proe", "pace_z", "rush_share"],
    "pass_yards":     ["plays_base", "proe", "pace_z", "ypa_qb"],
    "rush_rec_yards": ["plays_base", "proe", "pace_z", "route_rate", "yprr", "rush_share", "ypc"],
    "anytime_td":     ["team_total", "rz_share", "role"],  # explicit requirement
}

# Optional helpful columns (not strictly required here):
#   'home_spread', 'total', 'mu_market'
# If you prefer requiring them too, add to GLOBAL_REQUIRED.

def _fail_if_missing_columns(df: pd.DataFrame, cols: List[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"[STRICT] Missing required columns in {label}: {missing}\n"
            f"Present columns: {list(df.columns)}"
        )

def _fail_if_nulls(df: pd.DataFrame, cols: List[str], label: str) -> None:
    bad = {c: df.index[df[c].isna()].tolist() for c in cols if c in df.columns and df[c].isna().any()}
    if bad:
        preview = {c: idxs[:10] for c, idxs in bad.items()}
        raise ValueError(
            f"[STRICT] Null/NaN values found in {label} for required columns (showing up to 10 row indices each): {preview}"
        )


# ---------------------------
# Utilities
# ---------------------------

def american_to_prob(odds: float) -> float:
    o = float(odds)
    if o < 0:
        return (-o) / ((-o) + 100.0)
    return 100.0 / (o + 100.0)

def devig_two_way(p_over_imp: float, p_under_imp: float) -> Tuple[float, float]:
    denom = p_over_imp + p_under_imp
    if denom <= 0:
        raise ValueError("[STRICT] Devig failure: implied probs sum <= 0.")
    return p_over_imp / denom, p_under_imp / denom

def fair_odds_from_p(p: float) -> float:
    if p <= 0 or p >= 1:
        raise ValueError("[STRICT] fair_odds_from_p received invalid p outside (0,1).")
    return (-100.0 * p / (1.0 - p)) if p >= 0.5 else (100.0 * (1.0 - p) / p)

def decimal_from_american(odds: float) -> float:
    o = float(odds)
    return 1.0 + (o / 100.0) if o > 0 else 1.0 + (100.0 / (-o))

def kelly_fraction(p: float, b: float, cap: float = 0.05) -> float:
    if b <= 0 or p <= 0 or p >= 1:
        return 0.0
    f = (p * (b + 1) - 1) / b
    return float(max(0.0, min(cap, f)))

def _role_goal_line_bias(role: str) -> float:
    r = (role or "").upper()
    if r.startswith("RB"):
        return 1.25
    if r.startswith("TE"):
        return 1.10
    if r.startswith("WR"):
        return 1.00
    return 1.00

def _load_calibration() -> dict:
    p = Path("metrics/calibration.json")
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f) or {}


# ---------------------------
# Weather multipliers (μ, σ)
# ---------------------------

def weather_mu_sigma(wind_mph: float, precip: str, market: str) -> Tuple[float, float]:
    mu, sig = 1.0, 1.0
    w = float(wind_mph)
    p = str(precip).lower()

    if w >= 15.0:
        if market in {"pass_yards", "rec_yards", "rush_rec_yards"}:
            mu *= 0.94
            sig *= 1.04
    if p in {"rain", "snow"}:
        if market in {"rec_yards", "rush_rec_yards"}:
            mu *= 0.97
            sig *= 1.03
        if market in {"rush_yards", "rush_att"}:
            mu *= 1.02
    return mu, sig


# ---------------------------
# Normal tail probability
# ---------------------------

def p_over_normal(line: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 1.0 if mu > line else 0.0
    z = (line - mu) / sigma
    return float(1.0 - norm.cdf(z))


# ---------------------------
# Main pricing routine (STRICT)
# ---------------------------

def price_props(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # 1) Global required columns present & non-null
    _fail_if_missing_columns(df, GLOBAL_REQUIRED, "global")
    _fail_if_nulls(df, GLOBAL_REQUIRED, "global")

    # 2) Per-market required columns present & non-null
    #    We validate each subset (mask) separately so you know exactly which rows are wrong.
    for mk, req_cols in PER_MARKET_REQUIRED.items():
        mask = (df["market"] == mk)
        if mask.any():
            _fail_if_missing_columns(df.loc[mask], req_cols, f"market={mk}")
            _fail_if_nulls(df.loc[mask], req_cols, f"market={mk}")

    # 3) De-vig the book
    df["p_over_imp"] = df["over_odds"].apply(american_to_prob)
    df["p_under_imp"] = df["under_odds"].apply(american_to_prob)
    pov, pun = [], []
    for po, pu in zip(df["p_over_imp"], df["p_under_imp"]):
        fov, fun = devig_two_way(float(po), float(pu))
        pov.append(fov); pun.append(fun)
    df["p_market_over"] = pov
    df["p_market_under"] = pun

    # 4) Weather multipliers (strict: all must be present & numeric)
    df[["mu_mult_wx", "sig_mult_wx"]] = df.apply(
        lambda r: pd.Series(weather_mu_sigma(float(r["wind_mph"]), str(r["precip"]), str(r["market"]))),
        axis=1
    )

    # 5) Combine non-weather multipliers (given) + weather
    df["mu_mult"] = df["mu_mult_nowx"] * df["mu_mult_wx"]
    df["sig_mult"] = df["sig_mult_nowx"] * df["sig_mult_wx"]

    # 6) Pre-weather μ
    df["mu_model_nowx"] = df["mu0"] * df["mu_mult_nowx"]

    # 7) Plays/PROE volume μ (STRICT: fails if volume module raises)
    def _volume_mu_for_row(r):
        # Volume module expects real values; we rely on earlier strict checks
        plays, pass_rate, rush_rate, _ = team_volume_estimates(
            {
                "event_id": r.get("event_id"),
                "home_spread": r.get("home_spread"),
                "total": r.get("total"),
                "team_wp": r.get("team_wp"),
            },
            bool(r.get("is_home")),
            {
                "plays_base": r.get("plays_base"),
                "proe": r.get("proe"),
                "pace_z": r.get("pace_z"),
            },
            (r.get("wind_mph"), r.get("precip"), r.get("temp_f")),
        )
        mk = str(r["market"])
        tgt_sh = float(r["target_share"]) if "target_share" in r else np.nan
        route_rate = float(r["route_rate"]) if "route_rate" in r else np.nan
        rush_sh = float(r["rush_share"]) if "rush_share" in r else np.nan
        yprr = float(r["yprr"]) if "yprr" in r else np.nan
        ypc  = float(r["ypc"]) if "ypc" in r else np.nan
        ypa_qb = float(r["ypa_qb"]) if "ypa_qb" in r else np.nan
        catch = float(r["catch_rate"]) if "catch_rate" in r else np.nan

        team_dropbacks = plays * pass_rate
        team_rushes    = plays * rush_rate

        if mk == "receptions":
            targets = team_dropbacks * tgt_sh
            return float(targets * catch)
        if mk == "rec_yards":
            routes = team_dropbacks * route_rate
            return float(routes * yprr)
        if mk == "rush_yards":
            return float(team_rushes * rush_sh * ypc)
        if mk == "rush_att":
            return float(team_rushes * rush_sh)
        if mk == "pass_yards":
            return float(team_dropbacks * ypa_qb)
        if mk == "rush_rec_yards":
            routes = team_dropbacks * route_rate
            rec_y = routes * yprr
            rush_y = team_rushes * rush_sh * ypc
            return float(rec_y + rush_y)
        return np.nan

    df["mu_vol"] = df.apply(_volume_mu_for_row, axis=1)

    # 8) Blend volume μ into pre-weather μ, then apply weather
    VOL_W = 0.35
    df["mu_model_nowx"] = (1 - VOL_W) * df["mu_model_nowx"] + VOL_W * df["mu_vol"]
    df["mu_model"] = df["mu_model_nowx"] * df["mu_mult_wx"]
    df["sigma_model"] = df["sigma"] * df["sig_mult"]

    # 9) Calibration shrink (optional file; if empty, no-op)
    cal = _load_calibration()
    if isinstance(cal, dict) and len(cal):
        def _apply_shrink_row(r):
            a = float(cal.get(str(r["market"]), 0.0))
            mu_mkt = r["mu_market"] if "mu_market" in r and pd.notna(r["mu_market"]) else r["line"]
            return (1 - a) * float(r["mu_model"]) + a * float(mu_mkt)
        df["mu_model"] = df.apply(_apply_shrink_row, axis=1)

    # 10) Model P(Over) for continuous markets
    def _model_p_over_row(r):
        mk = str(r["market"])
        if mk == "anytime_td":
            return np.nan
        return p_over_normal(float(r["line"]), float(r["mu_model"]), float(r["sigma_model"]))
    df["p_model_over"] = df.apply(_model_p_over_row, axis=1)

    # 11) Anytime TD (STRICT: requires team_total, rz_share, role per PER_MARKET_REQUIRED)
    any_mask = (df["market"] == "anytime_td")
    if any_mask.any():
        ttot = df.loc[any_mask, "team_total"].astype(float)
        rz   = df.loc[any_mask, "rz_share"].astype(float)
        roles= df.loc[any_mask, "role"].astype(str)

        lam_team = ttot.apply(total_to_td_lambda).astype(float)
        role_bias = roles.apply(_role_goal_line_bias).astype(float)

        # Player p via td_model helper (no clipping beyond helper’s bounds)
        p_any_model = []
        for lam, rz_sh, rb in zip(lam_team.tolist(), rz.tolist(), role_bias.tolist()):
            p_any_model.append(player_td_probability(lam, rz_sh, rb))
        p_any_model = pd.Series(p_any_model, index=ttot.index).astype(float)

        df.loc[any_mask, "p_model_over"] = p_any_model

    # 12) Blend with market (65/35). Strict: p_market_over must exist & finite
    if df["p_market_over"].isna().any():
        bad = df.index[df["p_market_over"].isna()].tolist()
        raise ValueError(f"[STRICT] Missing p_market_over on rows {bad}. Ensure both over_odds and under_odds present.")

    df["p_blend_over"] = 0.65 * df["p_model_over"].astype(float) + 0.35 * df["p_market_over"].astype(float)

    # 13) Fair odds, edge, Kelly, tiers
    df["fair_over_odds"] = df["p_blend_over"].apply(fair_odds_from_p)
    df["edge_abs"] = df["p_blend_over"] - df["p_market_over"]

    df["b_over"] = df["over_odds"].apply(decimal_from_american) - 1.0
    df["kelly_over"] = [
        kelly_fraction(p, b, cap=(0.025 if (mk in {"rush_rec_yards"} or (isinstance(sig, float) and sig >= 55.0)) else 0.05))
        for p, b, mk, sig in zip(df["p_blend_over"], df["b_over"], df["market"], df["sigma_model"])
    ]

    def _tier(e):
        if e >= 0.06:
            return "ELITE"
        if e >= 0.04:
            return "GREEN"
        if e >= 0.01:
            return "AMBER"
        return "RED"

    df["tier"] = df["edge_abs"].apply(_tier)

    # Final safety: ensure no NaNs in critical outputs
    critical = ["p_model_over", "p_blend_over", "fair_over_odds", "edge_abs", "kelly_over", "tier"]
    _fail_if_nulls(df, critical, "outputs")

    return df


# ---------------------------
# Writing outputs
# ---------------------------

def write_outputs(df: pd.DataFrame, out_dir: str, basename: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(out_dir) / f"{basename}.csv"
    df.to_csv(csv_path, index=False)
    print(f"[pricing.write_outputs] wrote {csv_path}")

    try:
        import openpyxl  # noqa: F401
        xlsx_path = Path(out_dir) / f"{basename}.xlsx"
        df.to_excel(xlsx_path, index=False)
        print(f"[pricing.write_outputs] wrote {xlsx_path}")
    except Exception as e:
        print(f"[pricing.write_outputs] XLSX skipped ({e})")

