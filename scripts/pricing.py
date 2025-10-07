# scripts/pricing.py
from __future__ import annotations
import os, math
import numpy as np
import pandas as pd
from pathlib import Path

# Import your volume function
from .volume import add_volume

PBLEND_MODEL = 0.65  # model vs market probability blend (unchanged)
MU_BLEND_WEIGHT = float(os.getenv("MU_BLEND_WEIGHT", "0.35"))  # volume×efficiency μ weight

# ---- odds/prob helpers ----

def american_to_prob(american: float) -> float:
    if pd.isna(american): return np.nan
    o = float(american)
    return 100.0 / (o + 100.0) if o > 0 else (-o) / ((-o) + 100.0)

def prob_to_american(p: float) -> float:
    if p <= 0 or p >= 1 or pd.isna(p): return np.nan
    return -100.0 * p / (1.0 - p) if p > 0.5 else 100.0 * (1.0 - p) / p

def devig_two_way(p_over: float, p_under: float):
    if any(pd.isna(x) for x in (p_over, p_under)): return (np.nan, np.nan)
    denom = (p_over + p_under)
    if denom <= 0: return (np.nan, np.nan)
    return (p_over/denom, p_under/denom)

# ---- μ path: volume × efficiency ----

def _safe_float(x, default):
    try:
        if x is None: return float(default)
        v = float(x)
        if v != v: return float(default)
        return v
    except Exception:
        return float(default)

def mu_volume_efficiency(row) -> float:
    mk = row.get("market_internal")
    vol = row.get("volume_est", np.nan)
    if pd.isna(vol): return np.nan

    yprr = _safe_float(row.get("yprr_proxy"), 1.6)
    ypc  = _safe_float(row.get("ypc"), 4.2)
    ypa  = _safe_float(row.get("qb_ypa"), 7.1)
    ypt  = _safe_float(row.get("ypt"), 7.8)

    if mk == "receptions":
        return vol
    if mk == "rec_yards":
        return vol * ypt
    if mk == "rush_yards":
        return vol * ypc
    if mk == "rush_rec_yards":
        return 0.5 * vol * ypt + 0.5 * vol * ypc
    if mk == "pass_yards":
        return vol * ypa
    if mk == "pass_tds":
        return vol * 0.07
    return np.nan

# ---- Weather multipliers (quick patch style) ----

def weather_multiplier(wind, precip, mk):
    m = 1.0
    try:
        wind = float(wind)
    except Exception:
        wind = np.nan

    if not pd.isna(wind) and wind >= 15:
        if mk in {"pass_yards","rec_yards","rush_rec_yards"}:
            m *= 0.94
    if str(precip).lower() in {"rain","snow"}:
        if mk in {"rec_yards","rush_rec_yards"}:
            m *= 0.97
        if mk in {"rush_yards","rush_att"}:
            m *= 1.02
    return m

# ---- σ defaults by market (then volatility widening happens elsewhere if needed) ----
SIGMA_DEFAULT = {
    "rec_yards": 26.0, "receptions": 1.8,
    "rush_yards": 23.0, "rush_att": 3.5,
    "pass_yards": 48.0, "pass_tds": 0.9,
    "rush_rec_yards": 30.0,
}

# ---- main pricing ----

def _line_key(row):
    # group rows to find Over/Under on same book+line
    return (row["event_id"], row["bookmaker_key"], row["market_internal"], row["player"], row.get("line"))

def compute_market_fair(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-book per-line fair probs from vigged odds found in props_raw."""
    if df.empty:
        return pd.DataFrame(columns=[
            "event_id","market_internal","player","line",
            "bookmaker_key","bookmaker_title",
            "market_fair_over","market_fair_under","market_fair_yes","market_fair_no"
        ])
    d = df.copy()
    d["side_norm"] = d["side"].str.title()
    d["p_vig"] = d["price"].apply(american_to_prob)
    keys = ["event_id","market_internal","player","line","bookmaker_key","bookmaker_title"]
    piv = d.pivot_table(index=keys, columns="side_norm", values="p_vig", aggfunc="first").reset_index()
    # devig
    def _dev(r):
        fo=fu=fy=fn=np.nan
        if "Over" in r and "Under" in r:
            fo, fu = devig_two_way(r["Over"], r["Under"])
        if "Yes" in r and "No" in r:
            fy, fn = devig_two_way(r["Yes"], r["No"])
        return pd.Series({"market_fair_over":fo,"market_fair_under":fu,"market_fair_yes":fy,"market_fair_no":fn})
    fair = piv.apply(_dev, axis=1)
    out = pd.concat([piv, fair], axis=1)
    return out

def price_props(props_raw: pd.DataFrame) -> pd.DataFrame:
    if props_raw.empty:
        return pd.DataFrame()

    # Enrich
    from .features_external import enrich_props
    df = enrich_props(props_raw)

    # Add volume
    df = add_volume(df)

    # μ_market anchored proxy:
    # If you already compute this elsewhere, keep it; here we build a simple anchor using line and book odds.
    # When Over & Under exist, μ_market ≈ line (for continuous). We'll rely on your existing code if present.
    df["mu_market"] = df["line"]  # safe default; your rules will then modulate with weather/pressure/etc.

    # μ volume×efficiency
    df["mu_vol_eff"] = df.apply(mu_volume_efficiency, axis=1)

    # Blend μ paths
    def _blend_mu(r):
        mkt = r["mu_market"]
        vol = r["mu_vol_eff"]
        if pd.isna(vol):
            return mkt
        try:
            return MU_BLEND_WEIGHT * vol + (1.0 - MU_BLEND_WEIGHT) * mkt
        except Exception:
            return mkt
    df["model_proj"] = df.apply(_blend_mu, axis=1)

    # Weather multiplicative tweak
    def _apply_weather(r):
        mk = r.get("market_internal")
        mult = weather_multiplier(r.get("wind_mph"), r.get("precip"), mk)
        return r["model_proj"] * mult
    df["model_proj"] = df.apply(_apply_weather, axis=1)

    # σ default (you can widen later per your volatility flags)
    df["model_sd"] = df["market_internal"].map(SIGMA_DEFAULT).fillna(25.0)

    # Model Over probability at quoted line
    # p_model_over = 1 - Phi((L - μ)/σ)
    from math import erf, sqrt
    def _norm_cdf(x):  # Φ
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))
    def _p_over(r):
        L = r.get("line")
        mu = r.get("model_proj")
        sd = r.get("model_sd")
        if pd.isna(L) or pd.isna(mu) or sd <= 0:
            return np.nan
        z = (L - mu) / sd
        return 1.0 - _norm_cdf(z)
    df["model_over_pct"] = df.apply(_p_over, axis=1)

    # Market fair prob (per book/line)
    fair = compute_market_fair(df.rename(columns={"american_odds":"price"}) if "price" not in df.columns else df)
    # Merge the fair over prob (or yes prob for TD market)
    df = df.merge(fair[[
        "event_id","market_internal","player","line","bookmaker_key",
        "market_fair_over","market_fair_yes"
    ]], on=["event_id","market_internal","player","line","bookmaker_key"], how="left")

    # Choose market fair ref per market
    def _market_ref_prob(r):
        mk = r["market_internal"]
        if mk == "anytime_td":
            return r.get("market_fair_yes")
        return r.get("market_fair_over")
    df["market_fair_prob"] = df.apply(_market_ref_prob, axis=1)

    # 65/35 prob blend
    def _pblend(r):
        pm = r.get("model_over_pct")
        mkt = r.get("market_fair_prob")
        if pd.isna(pm) and pd.isna(mkt): return np.nan
        if pd.isna(pm): return mkt
        if pd.isna(mkt): return pm
        return PBLEND_MODEL * pm + (1.0 - PBLEND_MODEL) * mkt
    df["p_blend"] = df.apply(_pblend, axis=1)

    # Fair odds from blended prob
    df["fair_over_american"] = df["p_blend"].apply(prob_to_american)

    # Edge% vs market fair
    df["edge_abs"] = df["p_blend"] - df["market_fair_prob"]

    # Kelly sizing (cap per your spec)
    def _kelly(p, b):
        # b = net odds in decimal from American
        if pd.isna(p) or pd.isna(b): return np.nan
        return (p*(b+1) - (1-p)) / b
    def _b_from_american(o):
        if pd.isna(o): return np.nan
        o = float(o)
        return (o/100.0) if o > 0 else (100.0/(-o))
    df["book_odds_b"] = df["price"].apply(_b_from_american)
    df["kelly_raw"] = df.apply(lambda r: _kelly(r["p_blend"], r["book_odds_b"]), axis=1)

    # Caps
    def _cap_kelly(r):
        mk = r["market_internal"]
        base_cap = 0.05
        if mk.endswith("_alternate"):
            base_cap = 0.025
        # volatility flags could halve cap (wire later)
        return max(0.0, min(base_cap, r["kelly_raw"] if not pd.isna(r["kelly_raw"]) else 0.0))
    df["kelly_cap"] = df.apply(_cap_kelly, axis=1)

    # Tiering
    def _tier(edge):
        if pd.isna(edge): return "RED"
        if edge >= 0.06: return "ELITE"
        if edge >= 0.04: return "GREEN"
        if edge >= 0.01: return "AMBER"
        return "RED"
    df["tier"] = df["edge_abs"].apply(_tier)

    # Friendly columns
    keep = [
        "event_id","commence_time","home_team","away_team",
        "bookmaker_key","bookmaker_title",
        "market_api","market_internal","player","side","line","price",
        "market_fair_prob","model_proj","model_sd","model_over_pct","p_blend",
        "fair_over_american","edge_abs","kelly_cap","tier",
        # traceability
        "plays_est","proe","rz_rate","target_share","rush_share",
        "yprr_proxy","ypc","ypt","qb_ypa",
        "wind_mph","temp_f","precip",
    ]
    out = df[keep].copy()

    Path("outputs").mkdir(parents=True, exist_ok=True)
    out.to_csv("outputs/props_priced_clean.csv", index=False)
    print(f"[pricing] wrote {len(out)} rows → outputs/props_priced_clean.csv")
    return out

# optional CLI: price outputs/props_enriched.csv or props_raw.csv
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Price enriched props → outputs/props_priced_clean.csv")
    ap.add_argument("--in", dest="inp", default="outputs/props_raw.csv")
    ap.add_argument("--out", default="outputs/props_priced_clean.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.inp) if Path(args.inp).exists() else pd.DataFrame()
    out = price_props(df)
    if args.out != "outputs/props_priced_clean.csv":
        out.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()

