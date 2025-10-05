# scripts/pricing.py
from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# ---------- Default SDs (can be tuned or replaced with per-player priors) ----------
SD_DEFAULTS = {
    "rec_yards": 26.0,
    "receptions": 1.8,
    "rush_yards": 22.0,
    "rush_att": 3.0,
    "pass_yards": 48.0,
    "pass_tds": 0.9,
    "rush_rec_yards": 30.0,
    "anytime_td": 1.0,  # not used as Normal; left for completeness
}

# ---------- Helpers ----------
def american_to_prob(odds: float) -> float:
    if odds is None or pd.isna(odds): return np.nan
    o = float(odds)
    if o < 0: return (-o)/((-o)+100.0)
    return 100.0/(o+100.0)

def prob_to_american(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1-1e-6))
    if p >= 0.5:  # favorite
        return - (p/(1-p)) * 100.0
    return ((1-p)/p) * 100.0

def devig_two_way(p_over: float, p_under: float) -> float | None:
    if np.isnan(p_over) or np.isnan(p_under): return None
    s = p_over + p_under
    if s <= 0: return None
    return p_over/s

def erfinv(x: float) -> float:
    # Winitzki approx
    a = 0.147
    ln = np.log(1.0 - x*x)
    s = (2/(np.pi*a) + ln/2.0)
    inner = s*s - ln/a
    return np.sign(x) * np.sqrt(np.sqrt(inner) - s)

def norm_inv(p: float) -> float:
    return np.sqrt(2.0) * erfinv(2.0*p - 1.0)

def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z/np.sqrt(2.0)))

# ---------- Context loaders ----------
def _load_game_lines() -> pd.DataFrame:
    p = Path("outputs/game_lines.csv")
    if p.exists():
        df = pd.read_csv(p)
        # event_id, home_team, away_team, home_wp, away_wp
        return df
    return pd.DataFrame()

def _load_team_metrics() -> pd.DataFrame:
    """
    Optional: metrics/team_form.csv with *defensive* fields; we z-score them.
    Expect (best effort): team, def_pressure_rate, def_pass_epa, def_rush_epa,
                          light_box_rate, heavy_box_rate, pace
    Missing columns are safely ignored.
    """
    p = Path("metrics/team_form.csv")
    if not p.exists(): 
        return pd.DataFrame()
    df = pd.read_csv(p)
    # z-score available numeric columns
    for col in ["def_pressure_rate","def_pass_epa","def_rush_epa","pace","light_box_rate","heavy_box_rate","def_sack_rate"]:
        if col in df.columns:
            s = df[col].astype(float)
            df[f"{col}_z"] = (s - s.mean()) / (s.std(ddof=0)+1e-9)
    return df

# ---------- Post-mortem adjustment rules (functions return multipliers / sd bumps) ----------
def qb_pressure_multiplier(opp_pressure_z: float, opp_pass_epa_z: float) -> float:
    # 1) Pressure-adjusted QB baseline
    m = (1.0 - 0.35*opp_pressure_z) * (1.0 - 0.25*opp_pass_epa_z)
    return float(np.clip(m, 0.70, 1.10))

def sack_to_attempt_multiplier(opp_sack_rate_above_avg: float) -> float:
    # 2) Sack-to-attempt elasticity (volume)
    m = 1.0 - 0.15*opp_sack_rate_above_avg
    return float(np.clip(m, 0.80, 1.05))

def funnel_multiplier(def_pass_epa_z: float, def_rush_epa_z: float, market_key: str) -> float:
    # 3) Run/pass funnels
    if pd.isna(def_pass_epa_z) or pd.isna(def_rush_epa_z):
        return 1.0
    run_funnel = (def_rush_epa_z >= 0.25) and (def_pass_epa_z <= -0.25)  # approx 60th/40th pct
    pass_funnel = (def_pass_epa_z >= 0.25) and (def_rush_epa_z <= -0.25)
    if run_funnel:
        if market_key in {"rush_yards","rush_att","rush_rec_yards"}: return 1.05
        if market_key in {"pass_yards","receptions","rec_yards","pass_tds"}: return 0.96
    if pass_funnel:
        if market_key in {"rush_yards","rush_att","rush_rec_yards"}: return 0.96
        if market_key in {"pass_yards","receptions","rec_yards","pass_tds"}: return 1.04
    return 1.0

def boxcount_multiplier(light_box_rate: float, heavy_box_rate: float, market_key: str) -> float:
    # 7) Box-count leverage (RB YPC mod → affects rush yards more than attempts)
    if market_key != "rush_yards": 
        return 1.0
    if not pd.isna(light_box_rate) and light_box_rate >= 0.60: 
        return 1.07
    if not pd.isna(heavy_box_rate) and heavy_box_rate >= 0.60:
        return 0.94
    return 1.0

def script_escalator(win_prob: float, market_key: str) -> float:
    # 8) Script escalators (favorite RB bump, modest QB/YAC downshift)
    if pd.isna(win_prob): 
        return 1.0
    if market_key in {"rush_att","rush_yards"} and win_prob >= 0.55:
        # +2–4 attempts equivalent → ~6–10% volume bump
        return 1.0 + min(0.10, (win_prob - 0.55) * 0.40 / 0.10)  # scale
    if market_key in {"pass_yards","receptions","rec_yards"} and win_prob >= 0.60:
        return 0.98  # tiny nudge down when heavily favored
    return 1.0

def pace_smoothing_multiplier(off_pace_z: float, def_pace_z: float) -> float:
    # 9) Pace smoothing (keep small; we don't have drive-based plays here)
    zsum = 0.5 * ((off_pace_z or 0.0) + (def_pace_z or 0.0))
    return float(1.0 + 0.03 * zsum)

def volatility_widening_factor(opp_pressure_z: float, qb_inconsistent: bool) -> float:
    # 10) widen SD if pressure mismatch high or QB inconsistency flagged
    bump = 0.0
    if not pd.isna(opp_pressure_z) and opp_pressure_z >= 1.0:
        bump += 0.15
    if qb_inconsistent:
        bump += 0.10
    return 1.0 + min(0.25, bump)

# ---------- Main pricing ----------
def price_props(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Input columns (from normalize):
        player, market_key, line, book, event_id, home_team, away_team, commence_time,
        price_over, price_under
    Optional (from engine id_map merge):
        player_id, team
    Context files optionally read here:
        outputs/game_lines.csv
        metrics/team_form.csv
    """
    df = df_in.copy()

    # implied probs (book)
    df["p_over_imp"] = df["price_over"].apply(american_to_prob)
    df["p_under_imp"] = df["price_under"].apply(american_to_prob)
    df["p_over_fair_market"] = np.vectorize(devig_two_way)(df["p_over_imp"], df["p_under_imp"])
    df["p_over_fair_market"] = df["p_over_fair_market"].fillna(df["p_over_imp"])

    # load context
    gl = _load_game_lines()           # event-level win probs
    tm = _load_team_metrics()         # team-level defensive metrics

    # attach win prob for the player's team (needs 'team' from id_map to work perfectly)
    if "team" in df.columns and not gl.empty:
        gl_small = gl[["event_id","home_team","away_team","home_wp","away_wp"]].drop_duplicates("event_id")
        df = df.merge(gl_small, on="event_id", how="left")
        def _pick_wp(row):
            if pd.isna(row.get("home_wp")): return np.nan
            if row.get("team") == row.get("home_team"): return row.get("home_wp")
            if row.get("team") == row.get("away_team"): return row.get("away_wp")
            # if we don't know team, split the difference to stay conservative
            return 0.5
        df["win_prob"] = df.apply(_pick_wp, axis=1)
    else:
        df["win_prob"] = np.nan

    # opponent map for team metrics
    def _opp(row):
        if "team" not in row or pd.isna(row["team"]): return None
        if row["team"] == row["home_team"]: return row["away_team"]
        if row["team"] == row["away_team"]: return row["home_team"]
        return None
    df["opp_team"] = df.apply(_opp, axis=1)

    # attach defensive z-scores (pressure, EPA splits, boxes, pace)
    if not tm.empty and "opp_team" in df.columns:
        mcols = ["def_pressure_rate_z","def_pass_epa_z","def_rush_epa_z",
                 "light_box_rate_z","heavy_box_rate_z","pace_z","def_sack_rate_z"]
        keep = ["team"] + mcols
        tm2 = tm.rename(columns={"team":"opp_team"})
        for c in mcols:
            if c not in tm2.columns: tm2[c] = np.nan
        df = df.merge(tm2[["opp_team"]+mcols], on="opp_team", how="left")
    else:
        for c in ["def_pressure_rate_z","def_pass_epa_z","def_rush_epa_z",
                  "light_box_rate_z","heavy_box_rate_z","pace_z","def_sack_rate_z"]:
            df[c] = np.nan

    # --- Baseline model: set μ so that P(X>line)=p (Normal with default σ) ---
    def base_mu(row) -> float:
        mk = row["market_key"]
        sd = SD_DEFAULTS.get(mk, 25.0)
        p  = row["p_over_fair_market"]
        if pd.isna(p) or sd <= 0: 
            return np.nan
        z = norm_inv(p)
        return row["line"] + z * sd
    df["_mu0"] = df.apply(base_mu, axis=1)
    df["_sd0"] = df["market_key"].map(SD_DEFAULTS).fillna(25.0)

    # --- Apply post-mortem rules (multipliers on μ; widening factor on σ) ---
    def row_multipliers(row) -> Tuple[float,float]:
        mk = row["market_key"]
        # 1) QB pressure + pass EPA (yards/TDs)
        m_qb = 1.0
        if mk in {"pass_yards","pass_tds"}:
            m_qb = qb_pressure_multiplier(row["def_pressure_rate_z"], row["def_pass_epa_z"])

        # 2) Sack -> attempts elasticity (pass volume proxy)
        m_sack = 1.0
        if mk in {"pass_yards","receptions","rec_yards"}:
            m_sack = sack_to_attempt_multiplier((row.get("def_sack_rate_z") or 0.0))

        # 3) Run/Pass funnels
        m_funnel = funnel_multiplier(row["def_pass_epa_z"], row["def_rush_epa_z"], mk)

        # 7) Box-count leverage (RB YPC)
        m_box = boxcount_multiplier(row.get("light_box_rate_z"), row.get("heavy_box_rate_z"), mk)

        # 8) Script escalators from win prob
        m_script = script_escalator(row.get("win_prob"), mk)

        # 9) Pace smoothing (very gentle)
        m_pace = pace_smoothing_multiplier(row.get("pace_z"), row.get("pace_z"))

        # 10) Volatility widening (σ multiplier)
        sd_mult = volatility_widening_factor(row.get("def_pressure_rate_z"), qb_inconsistent=False)

        mu_mult = m_qb * m_sack * m_funnel * m_box * m_script * m_pace
        return mu_mult, sd_mult

    mu_mults, sd_mults = zip(*df.apply(row_multipliers, axis=1))
    df["_mu_adj"] = df["_mu0"] * pd.Series(mu_mults).astype(float)
    df["_sd_adj"] = df["_sd0"] * pd.Series(sd_mults).astype(float)

    # --- Model probability at the *book line* with adjusted μ,σ ---
    def p_model_over(row) -> float:
        mu = row["_mu_adj"]; sd = row["_sd_adj"]; L = row["line"]
        if sd <= 0 or pd.isna(mu) or pd.isna(sd) or pd.isna(L): 
            return np.nan
        z = (L - mu) / sd
        return 1.0 - norm_cdf(z)
    df["p_over_model"] = df.apply(p_model_over, axis=1)

    # --- 65/35 blend w/ market anchor ---
    df["p_over_blend"] = 0.65*df["p_over_model"].fillna(df["p_over_fair_market"]) + 0.35*df["p_over_fair_market"]
    df["fair_over_odds"] = df["p_over_blend"].apply(prob_to_american)

    # edge vs the posted Over price
    df["edge_pct"] = df["p_over_blend"] - df["p_over_imp"]

    # outputs
    df["model_mean"] = df["_mu_adj"]
    df["model_sd"]   = df["_sd_adj"]

    # tidy
    keep = [
        "player","team","market_key","line","book","event_id","home_team","away_team","commence_time",
        "price_over","price_under","p_over_imp","p_under_imp","p_over_fair_market",
        "p_over_model","p_over_blend","fair_over_odds","edge_pct","model_mean","model_sd",
        "win_prob","def_pressure_rate_z","def_pass_epa_z","def_rush_epa_z","pace_z"
    ]
    out = df[keep].copy()

    # Tiering
    def tier(e):
        if pd.isna(e): return "NA"
        if e >= 0.06: return "ELITE"
        if e >= 0.04: return "GREEN"
        if e >= 0.01: return "AMBER"
        return "RED"
    out["tier"] = out["edge_pct"].apply(tier)
    return out

def write_outputs(df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    csvp = outdir / "props_priced.csv"
    xlsx = outdir / "props_priced.xlsx"
    df.to_csv(csvp, index=False)
    try:
        df.to_excel(xlsx, index=False)
    except Exception as e:
        print("[warn] xlsx export failed:", e)
