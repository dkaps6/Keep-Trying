# scripts/pricing.py
from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

# ---------------------------
# small utils
# ---------------------------

def _maybe_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

def american_to_prob(odds: Optional[float]) -> Optional[float]:
    if odds is None or (isinstance(odds, float) and np.isnan(odds)):
        return None
    try:
        o = float(odds)
    except Exception:
        return None
    if o > 0:
        return 100.0 / (o + 100.0)
    elif o < 0:
        return (-o) / ((-o) + 100.0)
    return None

def prob_to_american(p: float) -> Optional[int]:
    if p <= 0 or p >= 1:
        return None
    if p > 0.5:
        # negative
        return int(round(-100 * p / (1 - p)))
    else:
        # positive
        return int(round(100 * (1 - p) / p))

def devig_two_way(over_odds: Optional[float], under_odds: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    """
    Return (p_over_fair, p_under_fair). If one side missing, return its implied prob and None for the other.
    """
    p_o = american_to_prob(over_odds)
    p_u = american_to_prob(under_odds)
    if p_o is None and p_u is None:
        return None, None
    if p_o is None:
        return p_u, None  # anchor one side
    if p_u is None:
        return p_o, None
    s = p_o + p_u
    if s <= 0:
        return None, None
    return p_o / s, p_u / s

def inverse_mu_given_p_sigma(line: float, p_over: float, sigma: float) -> float:
    """
    Given P(X>line)=p_over for Normal(mu, sigma^2), solve for mu.
    """
    if sigma <= 0 or p_over is None or p_over <= 0 or p_over >= 1:
        return line
    z = norm.ppf(1 - p_over)
    # p = 1 - Φ((L - μ)/σ) -> (L-μ)/σ = z  -> μ = L - zσ
    return line - z * sigma

# ---------------------------
# loading externals
# ---------------------------

def _load_team_form() -> pd.DataFrame:
    return _maybe_csv("metrics/team_form.csv")

def _load_roles() -> pd.DataFrame:
    return _maybe_csv("data/roles.csv")

def _load_injuries() -> pd.DataFrame:
    return _maybe_csv("data/injuries.csv")

def _load_coverage() -> pd.DataFrame:
    # tags like top_shadow, heavy_man, heavy_zone
    return _maybe_csv("data/coverage.csv")

def _load_cb_assign() -> pd.DataFrame:
    # columns: defense_team, receiver, cb, penalty (0..0.25) or quality text
    return _maybe_csv("data/cb_assignments.csv")

def _load_game_lines() -> pd.DataFrame:
    return _maybe_csv("outputs/game_lines.csv")

def _load_weather() -> pd.DataFrame:
    # data/weather.csv -> event_id, wind_mph, temp_f, precip
    return _maybe_csv("data/weather.csv")

def _load_calibration() -> Dict[str, float]:
    p = Path("data/calibration.json")
    if not p.exists():
        p = Path("metrics/calibration.json")
        if not p.exists():
            return {}
    try:
        import json
        return json.loads(p.read_text())
    except Exception:
        return {}

# ---------------------------
# z-scores for team_form
# ---------------------------

ZCOLS = [
    "def_pressure_rate",
    "def_pass_epa",
    "def_rush_epa",
    "def_sack_rate",
    "pace",
    "ay_per_att",
    "neutral_pass_rate",
    "proe",
]

def _add_zscores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ZCOLS:
        if c in out.columns:
            col = out[c].astype(float)
            mu = col.mean(skipna=True)
            sd = col.std(skipna=True)
            if sd and sd > 0:
                out[f"{c}_z"] = (col - mu) / sd
            else:
                out[f"{c}_z"] = 0.0
        else:
            out[f"{c}_z"] = 0.0
    return out

# ---------------------------
# default SDs per market
# ---------------------------

DEFAULT_SIGMA = {
    "rec_yards": 26.0,
    "receptions": 1.8,
    "rush_yards": 23.0,
    "rush_att": 3.8,
    "pass_yards": 48.0,
    "rush_rec_yards": 32.0,
}

# ---------------------------
# weather multipliers
# ---------------------------

def weather_mu_sigma(market: str,
                     wind_mph: Optional[float],
                     precip: Optional[str],
                     temp_f: Optional[float]) -> Tuple[float, float]:
    """
    Conservative, policy-friendly weather effects:
      - Wind >=15 mph: pass/rec/rush_rec down a bit, slightly wider tails
      - Rain/snow: small YAC downshift (hurts rec_yards) + run tendency bump
      - Temp extremes: small pass/rec nudge
    """
    mu = 1.0
    sig = 1.0
    try:
        w = float(wind_mph) if wind_mph is not None else np.nan
    except Exception:
        w = np.nan
    pcp = (str(precip or "")).lower()
    try:
        t = float(temp_f) if temp_f is not None else np.nan
    except Exception:
        t = np.nan

    if not np.isnan(w) and w >= 15:
        if market in {"pass_yards", "rec_yards", "rush_rec_yards"}:
            mu *= 0.94
            sig *= 1.05

    if pcp in {"rain", "snow"}:
        if market in {"rec_yards", "rush_rec_yards"}:
            mu *= 0.97
        if market in {"rush_yards", "rush_att"}:
            mu *= 1.02  # run rate up a touch

    if not np.isnan(t) and (t <= 20 or t >= 90):
        if market in {"pass_yards", "rec_yards"}:
            mu *= 0.98

    return mu, sig

# ---------------------------
# core multipliers (non-weather)
# ---------------------------

def base_sigma(market: str) -> float:
    return float(DEFAULT_SIGMA.get(market, 25.0))

def row_multipliers(row: pd.Series, market: str) -> Tuple[float, float]:
    """
    return (mu_mult, sigma_mult) combining all contextual rules *except* weather
    (weather applied separately so we can unit-test it easily)
    """
    mu_mult = 1.0
    sig_mult = 1.0

    opp_press_z = row.get("def_pressure_rate_z", 0.0)
    opp_pass_epa_z = row.get("def_pass_epa_z", 0.0)
    opp_sack_z = row.get("def_sack_rate_z", 0.0)
    pace_z = row.get("pace_z", 0.0)
    ay_z = row.get("ay_per_att_z", 0.0)

    # Pressure-adjusted QB baseline
    if market == "pass_yards":
        mu_mult *= (1.0 - 0.35 * float(opp_press_z))
        mu_mult *= (1.0 - 0.25 * float(opp_pass_epa_z))

    # Sack-to-attempt elasticity (lower pass volume under heavy sacks)
    if market == "pass_yards" and opp_sack_z > 0:
        mu_mult *= (1.0 - 0.15 * float(opp_sack_z))

    # Funnels via EPA split (heuristic)
    if row.get("def_rush_epa_z", 0.0) >= 0.25 and opp_pass_epa_z <= -0.25:
        # run funnel
        if market == "rush_yards":
            mu_mult *= 1.03
        if market == "pass_yards":
            mu_mult *= 0.98
    elif row.get("def_rush_epa_z", 0.0) <= -0.25 and opp_pass_epa_z >= 0.25:
        # pass funnel
        if market == "pass_yards":
            mu_mult *= 1.03
        if market == "rush_yards":
            mu_mult *= 0.98

    # Pace smoothing (small)
    mu_mult *= (1.0 + 0.02 * float(pace_z))

    # Air-yards sanity (cap WR deep optimism when scheme doesn't support it)
    if market in {"rec_yards", "rush_rec_yards"} and ay_z <= -0.84:  # ~20th pct
        mu_mult *= 0.92

    # Coverage and CB shadow (optional files; if empty -> no-op)
    tag = str(row.get("coverage_tag", "")).lower()
    if market in {"rec_yards", "receptions"}:
        if "top_shadow" in tag or "heavy_man" in tag:
            mu_mult *= 0.94
        if "heavy_zone" in tag:
            mu_mult *= 1.04
    # explicit CB penalty if present (e.g., 0.08)
    cb_pen = row.get("cb_penalty", np.nan)
    if market in {"rec_yards", "receptions"} and not pd.isna(cb_pen):
        mu_mult *= max(0.0, 1.0 - float(cb_pen))

    # Script escalators from win prob (if present)
    wp = row.get("favored_wp", np.nan)
    if not pd.isna(wp) and wp >= 0.55:
        if market == "rush_yards":
            mu_mult *= 1.04
        if market == "pass_yards":
            mu_mult *= 0.985

    # Volatility widening in tough pass matchups
    if market == "pass_yards" and (opp_press_z > 0.75 or opp_sack_z > 0.75):
        sig_mult *= 1.12

    return mu_mult, sig_mult

# ---------------------------
# main pricing
# ---------------------------

def _merge_context(df: pd.DataFrame) -> pd.DataFrame:
    tf = _add_zscores(_load_team_form())
    roles = _load_roles()
    inj = _load_injuries()
    cov = _load_coverage()
    cb  = _load_cb_assign()
    gl  = _load_game_lines()
    wx  = _load_weather()

    out = df.copy()

    # best-effort opponent key
    # prefer 'defense_team' -> else 'opponent' -> else 'opp_team'
    def infer_opp(row):
        for k in ("defense_team", "opponent", "opp_team"):
            if k in row and pd.notna(row[k]):
                return row[k]
        return np.nan

    out["defense_team"] = out.apply(infer_opp, axis=1)

    # merge team_form on opponent defense
    if not tf.empty:
        out = out.merge(tf.add_prefix("opp_"), left_on="defense_team", right_on="opp_team", how="left")

    # roles join (player/team keyed)
    if not roles.empty:
        roles.columns = [c.lower() for c in roles.columns]
        out["player_lc"] = out["player"].str.lower()
        roles["player_lc"] = roles["player"].str.lower()
        out = out.merge(roles[["player_lc","role"]], on="player_lc", how="left")

    # injuries join (down-weights alpha WR, etc.)
    if not inj.empty:
        inj.columns = [c.lower() for c in inj.columns]
        inj["player_lc"] = inj["player"].str.lower()
        out = out.merge(inj[["player_lc","status"]], on="player_lc", how="left")

    # coverage tag by defense team
    if not cov.empty:
        cov.columns = [c.lower() for c in cov.columns]
        cov = cov.groupby("defense_team")["tag"].apply(lambda s: "|".join(sorted(set(s)))).reset_index()
        cov.rename(columns={"tag":"coverage_tag"}, inplace=True)
        out = out.merge(cov, on="defense_team", how="left")
    else:
        out["coverage_tag"] = ""

    # CB assignment (per receiver)
    if not cb.empty:
        cb.columns = [c.lower() for c in cb.columns]
        cb["receiver_lc"] = cb.get("receiver","").str.lower()
        cb["defense_team"] = cb["defense_team"].str.upper()
        merged = out.merge(cb, left_on=["player_lc","defense_team"], right_on=["receiver_lc","defense_team"], how="left")
        # penalty resolve
        if "penalty" in merged.columns:
            merged["cb_penalty"] = merged["penalty"]
        else:
            merged["cb_penalty"] = np.nan
        out = merged
    else:
        out["cb_penalty"] = np.nan

    # win prob join
    if not gl.empty:
        # compute favored_wp for row's player's team
        gl = gl.copy()
        for c in ["home_wp","away_wp"]:
            if c in gl.columns:
                gl[c] = pd.to_numeric(gl[c], errors="coerce")
        team_is_home = out.get("team").str.upper() == out.get("home_team","").astype(str).str.upper() if "home_team" in out.columns else False
        out = out.merge(gl[["event_id","home_team","away_team","home_wp","away_wp"]], on="event_id", how="left")
        def pick_wp(r):
            if pd.isna(r.get("home_wp")) or pd.isna(r.get("away_wp")):
                return np.nan
            t = str(r.get("team","")).upper()
            if t == str(r.get("home_team","")).upper():
                return float(r["home_wp"])
            if t == str(r.get("away_team","")).upper():
                return float(r["away_wp"])
            # if we can't match, use max
            return max(float(r["home_wp"]), float(r["away_wp"]))
        out["favored_wp"] = out.apply(pick_wp, axis=1)
    else:
        out["favored_wp"] = np.nan

    # weather join
    if not wx.empty and "event_id" in out.columns:
        out = out.merge(wx[["event_id","wind_mph","temp_f","precip"]], on="event_id", how="left")
    else:
        out["wind_mph"] = np.nan
        out["temp_f"] = np.nan
        out["precip"] = np.nan

    return out

def _pick_market(row: pd.Series) -> str:
    mk = str(row.get("market","")).lower()
    # normalize a few common variants
    if mk in {"receptions","player_receptions"}:
        return "receptions"
    if mk in {"receiving_yards","player_receiving_yds","rec_yards"}:
        return "rec_yards"
    if mk in {"rushing_yards","player_rushing_yds","rush_yards"}:
        return "rush_yards"
    if mk in {"rushing_attempts","rush_att","player_rush_att"}:
        return "rush_att"
    if mk in {"passing_yards","player_pass_yds","pass_yards"}:
        return "pass_yards"
    if mk in {"rush_rec_yards","player_rush_rec_yds"}:
        return "rush_rec_yards"
    return mk

def _color_and_tier(edge: float) -> Tuple[str, str]:
    if edge >= 0.06:
        return "GREEN", "ELITE"
    if edge >= 0.04:
        return "GREEN", "GREEN"
    if edge >= 0.01:
        return "AMBER", "AMBER"
    return "RED", "RED"

def price_props(df_props: pd.DataFrame) -> pd.DataFrame:
    """
    df_props expected minimal columns:
      event_id, player, team, defense_team/opponent, market, line, over_odds, under_odds
    Returns enriched dataframe with model projections, probabilities, edges, color tiering.
    """
    if df_props.empty:
        return df_props

    df = df_props.copy()
    df["market"] = df.apply(_pick_market, axis=1)
    df["line"] = pd.to_numeric(df["line"], errors="coerce")

    # devig
    dev = df.apply(lambda r: devig_two_way(r.get("over_odds"), r.get("under_odds")), axis=1, result_type="expand")
    df[["p_market_over", "p_market_under"]] = dev

    # defaults for sigma
    df["sigma"] = df["market"].map(DEFAULT_SIGMA).fillna(25.0)

    # merge context (team_form z-scores, injuries/roles, coverage/CB, win prob, weather)
    df = _merge_context(df)

    # starting mu from market (invert with current sigma)
    df["mu0"] = df.apply(lambda r: inverse_mu_given_p_sigma(r["line"], r["p_market_over"], r["sigma"])
                         if pd.notna(r["line"]) and pd.notna(r["p_market_over"]) else r["line"], axis=1)

    # apply non-weather multipliers
    mults = df.apply(lambda r: row_multipliers(r, r["market"]), axis=1, result_type="expand")
    df[["mu_mult_nowx","sig_mult_nowx"]] = mults

    # apply weather multipliers
    wx = df.apply(lambda r: weather_mu_sigma(r["market"], r.get("wind_mph"), r.get("precip"), r.get("temp_f")),
                  axis=1, result_type="expand")
    df[["mu_mult_wx","sig_mult_wx"]] = wx

    df["mu_mult"]  = df["mu_mult_nowx"] * df["mu_mult_wx"]
    df["sig_mult"] = df["sig_mult_nowx"] * df["sig_mult_wx"]

    df["mu_model"] = df["mu0"] * df["mu_mult"]
    df["sigma_model"] = df["sigma"] * df["sig_mult"]

    # model probability over
    def model_p_over(r):
        if pd.isna(r["line"]) or pd.isna(r["mu_model"]) or pd.isna(r["sigma_model"]) or r["sigma_model"] <= 0:
            return np.nan
        return 1.0 - norm.cdf((r["line"] - r["mu_model"]) / r["sigma_model"])

    df["p_model_over"] = df.apply(model_p_over, axis=1)

    # blend 65/35 with devig market
    def blend(r):
        pm = r["p_model_over"]
        mk = r["p_market_over"]
        if pd.isna(pm) and pd.isna(mk):
            return np.nan
        if pd.isna(mk):
            return pm
        if pd.isna(pm):
            return mk
        return 0.65 * pm + 0.35 * mk

    df["p_over_blend"] = df.apply(blend, axis=1)

    # fair odds and edge
    df["fair_over_odds"] = df["p_over_blend"].apply(lambda p: prob_to_american(p) if pd.notna(p) else None)
    df["edge_over"] = df["p_over_blend"] - df["p_market_over"]

    # choose side with higher blended edge vs devig
    def bet_side(r):
        eo = r["edge_over"]
        # compute under from over
        po = r["p_over_blend"]
        pmk = r["p_market_over"]
        if pd.isna(po) or pd.isna(pmk):
            return "", np.nan
        # under edge is symmetrical
        eu = (1 - po) - (1 - pmk)
        if eo >= eu:
            return "Over", eo
        else:
            return "Under", eu

    side_edge = df.apply(lambda r: bet_side(r), axis=1, result_type="expand")
    df[["bet_side","edge_abs"]] = side_edge

    df["color"], df["tier"] = zip(*df["edge_abs"].apply(lambda e: _color_and_tier(float(e)) if pd.notna(e) else ("RED","RED")))

    # clean view with the exact fields you asked for
    keep = [
        "event_id","bookmaker","player","team","defense_team","market","line",
        "over_odds","under_odds",
        # vegas de-vig view
        "p_market_over","fair_over_odds",
        # model view
        "mu_model","sigma_model","p_model_over","p_over_blend",
        # weather shown for transparency
        "wind_mph","temp_f","precip",
        # decision
        "bet_side","edge_abs","color","tier"
    ]
    # backfill bookmaker if missing
    if "bookmaker" not in df.columns:
        df["bookmaker"] = ""

    out = df[keep].copy()

    # also write a more verbose raw table if caller wants
    Path("outputs").mkdir(parents=True, exist_ok=True)
    out.to_csv("outputs/props_priced_clean.csv", index=False)
    df.to_csv("outputs/props_priced.csv", index=False)

    return out
