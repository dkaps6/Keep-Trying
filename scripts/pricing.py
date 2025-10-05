# scripts/pricing.py
from __future__ import annotations
import json, math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# ---------- Default SDs ----------
SD_DEFAULTS = {
    "rec_yards": 26.0,
    "receptions": 1.8,
    "rush_yards": 22.0,
    "rush_att": 3.0,
    "pass_yards": 48.0,
    "pass_tds": 0.9,
    "rush_rec_yards": 30.0,
    "anytime_td": 1.0,  # placeholder
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
def _maybe_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

def _load_game_lines() -> pd.DataFrame:
    return _maybe_csv("outputs/game_lines.csv")

def _load_team_metrics() -> pd.DataFrame:
    df = _maybe_csv("metrics/team_form.csv")
    if df.empty: return df
    for col in ["def_pressure_rate","def_pass_epa","def_rush_epa","pace","light_box_rate","heavy_box_rate","def_sack_rate","ay_per_att"]:
        if col in df.columns:
            s = df[col].astype(float)
            df[f"{col}_z"] = (s - s.mean()) / (s.std(ddof=0)+1e-9)
    return df

def _load_id_map() -> pd.DataFrame:
    df = _maybe_csv("data/id_map.csv")
    if not df.empty:
        # normalize headers
        ren = {}
        for want in ["player_name","team","position","role"]:
            for c in df.columns:
                if c.lower() == want:
                    ren[c] = want
        df = df.rename(columns=ren)
    return df

def _load_injuries() -> pd.DataFrame:
    df = _maybe_csv("data/injuries.csv")
    if not df.empty:
        df["status"] = df["status"].astype(str).str.strip().str.title()
    return df

def _load_roles() -> pd.DataFrame:
    df = _maybe_csv("data/roles.csv")
    if not df.empty:
        df["role"] = df["role"].astype(str).str.upper()
    return df

def _load_coverage() -> pd.DataFrame:
    df = _maybe_csv("data/coverage.csv")
    if not df.empty:
        df["tag"] = df["tag"].astype(str).lower()
    return df

# NEW: CB assignment loader
def _load_cb_assignments() -> pd.DataFrame:
    df = _maybe_csv("data/cb_assignments.csv")
    if df.empty: return df
    # normalize headers
    ren = {}
    for want in ["defense_team","receiver","cb","quality","penalty"]:
        for c in df.columns:
            if c.lower() == want:
                ren[c] = want
    df = df.rename(columns=ren)
    # canonical columns
    if "penalty" not in df.columns: df["penalty"] = np.nan
    if "quality" not in df.columns:  df["quality"]  = ""
    # backfill penalty from quality when missing
    qual_map = {"elite":0.08, "good":0.05, "avg":0.03}
    df["penalty"] = df["penalty"].astype(float)
    df.loc[df["penalty"].isna(), "penalty"] = df["quality"].astype(str).str.lower().map(qual_map)
    # left-join key aliases
    df = df.rename(columns={"defense_team":"opp_team","receiver":"player"})
    # clip sanity
    df["penalty"] = df["penalty"].fillna(0.06).clip(0.0, 0.25)
    return df

def _load_calibration() -> Dict[str, Dict[str, float]]:
    p = Path("metrics/calibration.json")
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}

# ---------- Post-mortem rule components ----------
def qb_pressure_multiplier(opp_pressure_z: float, opp_pass_epa_z: float) -> float:
    m = (1.0 - 0.35*(opp_pressure_z or 0.0)) * (1.0 - 0.25*(opp_pass_epa_z or 0.0))
    return float(np.clip(m, 0.70, 1.10))

def sack_to_attempt_multiplier(opp_sack_rate_z: float) -> float:
    above = max(0.0, (opp_sack_rate_z or 0.0))
    m = 1.0 - 0.15*above
    return float(np.clip(m, 0.80, 1.05))

def funnel_multiplier(def_pass_epa_z: float, def_rush_epa_z: float, market_key: str) -> float:
    run_funnel  = (def_rush_epa_z or 0.0) >= 0.25 and (def_pass_epa_z or 0.0) <= -0.25
    pass_funnel = (def_pass_epa_z or 0.0) >= 0.25 and (def_rush_epa_z or 0.0) <= -0.25
    if run_funnel:
        if market_key in {"rush_yards","rush_att","rush_rec_yards"}: return 1.05
        if market_key in {"pass_yards","receptions","rec_yards","pass_tds"}: return 0.96
    if pass_funnel:
        if market_key in {"rush_yards","rush_att","rush_rec_yards"}: return 0.96
        if market_key in {"pass_yards","receptions","rec_yards","pass_tds"}: return 1.04
    return 1.0

def boxcount_multiplier(light_box_rate_z: float, heavy_box_rate_z: float, market_key: str) -> float:
    if market_key != "rush_yards": return 1.0
    if (light_box_rate_z or 0.0) >= 0.6: return 1.07
    if (heavy_box_rate_z or 0.0) >= 0.6: return 0.94
    return 1.0

def script_escalator(win_prob: float, market_key: str) -> float:
    if pd.isna(win_prob): return 1.0
    if market_key in {"rush_att","rush_yards"} and win_prob >= 0.55:
        return 1.0 + min(0.10, (win_prob - 0.55) * 0.40 / 0.10)
    if market_key in {"pass_yards","receptions","rec_yards"} and win_prob >= 0.60:
        return 0.98
    return 1.0

def pace_smoothing_multiplier(off_pace_z: float, def_pace_z: float) -> float:
    zsum = 0.5 * ((off_pace_z or 0.0) + (def_pace_z or 0.0))
    return float(1.0 + 0.03 * zsum)

def volatility_widening_factor(opp_pressure_z: float, qb_inconsistent: bool) -> float:
    bump = 0.0
    if (opp_pressure_z or 0.0) >= 1.0: bump += 0.15
    if qb_inconsistent: bump += 0.10
    return 1.0 + min(0.25, bump)

# (4) Alpha-WR injury elasticity
def injury_multiplier(status: str, role: str, market_key: str) -> float:
    if not status: return 1.0
    s = str(status).lower()
    if role in {"WR1","WR2","SLOT","TE"} and market_key in {"receptions","rec_yards","rush_rec_yards"}:
        if s in {"out","doubtful"}:          return 0.78
        if s in {"questionable","limited"}:  return 0.90
        if s in {"probable"}:                return 0.97
    return 1.0

# (5) Generic coverage tags
def coverage_multiplier(tags: set[str], role: str, market_key: str) -> float:
    if not tags: return 1.0
    t = set([x.lower() for x in tags])
    if ("top_shadow" in t or "heavy_man" in t) and role in {"WR1"} and market_key in {"receptions","rec_yards"}:
        return 0.92
    if "heavy_zone" in t and role in {"SLOT","TE"} and market_key in {"receptions","rec_yards"}:
        return 1.05
    return 1.0

# (6) Air-yards sanity cap
def air_yards_sanity_cap(role: str, market_key: str, ay_per_att_z: float) -> float:
    if role == "WR1" and market_key in {"rec_yards"} and (ay_per_att_z or 0.0) <= -0.84:
        return 0.80
    return 1.0

# NEW: CB assignment multiplier (player-specific)
def cb_assignment_multiplier(cb_penalty: float, market_key: str) -> float:
    if pd.isna(cb_penalty): return 1.0
    if market_key in {"receptions","rec_yards"}:
        return float(1.0 - np.clip(cb_penalty, 0.0, 0.25))
    return 1.0

# ---------- Main pricing ----------
def price_props(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()

    # implied & devigged market probs
    df["p_over_imp"] = df["price_over"].apply(american_to_prob)
    df["p_under_imp"] = df["price_under"].apply(american_to_prob)
    df["p_over_fair_market"] = np.vectorize(devig_two_way)(df["p_over_imp"], df["p_under_imp"])
    df["p_over_fair_market"] = df["p_over_fair_market"].fillna(df["p_over_imp"])

    # ---- Join ID map (team/position/role) if needed ----
    if "team" not in df.columns or "role" not in df.columns or "position" not in df.columns:
        idm = _load_id_map()
        if not idm.empty:
            idm = idm.rename(columns={"player_name":"player"})
            cols = ["player","team"] + [c for c in ["position","role"] if c in idm.columns]
            df = df.merge(idm[cols], on="player", how="left")

    # ---- Roles override (optional) ----
    roles = _load_roles()
    if not roles.empty:
        roles = roles.rename(columns={"player":"player","team":"team","role":"role"})
        df = df.merge(roles, on=["player","team"], how="left", suffixes=("","_role2"))
        df["role"] = df["role_role2"].combine_first(df["role"])
        df.drop(columns=[c for c in df.columns if c.endswith("_role2")], inplace=True)

    # ---- Win probabilities ----
    gl = _load_game_lines()
    if not gl.empty:
        gl_small = gl[["event_id","home_team","away_team","home_wp","away_wp"]].drop_duplicates("event_id")
        df = df.merge(gl_small, on="event_id", how="left")
        def _wp(row):
            if pd.isna(row.get("home_wp")): return np.nan
            if row.get("team") == row.get("home_team"): return row.get("home_wp")
            if row.get("team") == row.get("away_team"): return row.get("away_wp")
            return 0.5
        df["win_prob"] = df.apply(_wp, axis=1)
    else:
        df["win_prob"] = np.nan

    # ---- Team metrics ----
    tm = _load_team_metrics()
    if not tm.empty:
        def _opp(row):
            if pd.isna(row.get("team")): return None
            if row["team"] == row["home_team"]: return row["away_team"]
            if row["team"] == row["away_team"]: return row["home_team"]
            return None
        df["opp_team"] = df.apply(_opp, axis=1)
        keep = ["team","def_pressure_rate_z","def_pass_epa_z","def_rush_epa_z","pace_z",
                "light_box_rate_z","heavy_box_rate_z","def_sack_rate_z","ay_per_att_z"]
        tm2 = tm.rename(columns={"team":"opp_team"})
        for c in keep:
            if c not in tm2.columns: tm2[c] = np.nan
        df = df.merge(tm2[["opp_team"] + keep[1:]], on="opp_team", how="left")
    else:
        for c in ["def_pressure_rate_z","def_pass_epa_z","def_rush_epa_z","pace_z","light_box_rate_z","heavy_box_rate_z","def_sack_rate_z","ay_per_att_z"]:
            df[c] = np.nan

    # ---- Injuries ----
    inj = _load_injuries()
    if not inj.empty:
        inj = inj.rename(columns={"player":"inj_player","team":"inj_team","status":"inj_status"})
        df = df.merge(inj, left_on=["player","team"], right_on=["inj_player","inj_team"], how="left")
    else:
        df["inj_status"] = np.nan

    # ---- Coverage tags (generic) ----
    cov = _load_coverage()
    if not cov.empty:
        cov = cov.rename(columns={"defense_team":"opp_team"})
        cov_grp = cov.groupby("opp_team")["tag"].apply(lambda s: set(s.str.lower().tolist())).reset_index()
        df = df.merge(cov_grp, on="opp_team", how="left")
        df["tag"] = df["tag"].apply(lambda x: x if isinstance(x,set) else set())
    else:
        df["tag"] = [set()]*len(df)

    # ---- CB assignments (player-specific) ----
    cb = _load_cb_assignments()
    if not cb.empty:
        df = df.merge(cb[["opp_team","player","cb","penalty"]], on=["opp_team","player"], how="left")
        df.rename(columns={"penalty":"cb_penalty","cb":"cb_name"}, inplace=True)
    else:
        df["cb_penalty"] = np.nan
        df["cb_name"] = np.nan

    # --- Baseline μ so that P(X>line)=p ---
    def base_mu(row) -> float:
        mk = row["market_key"]
        sd = SD_DEFAULTS.get(mk, 25.0)
        p  = row["p_over_fair_market"]
        if pd.isna(p) or sd <= 0: return np.nan
        return row["line"] + norm_inv(p) * sd
    df["_mu0"] = df.apply(base_mu, axis=1)
    df["_sd0"] = df["market_key"].map(SD_DEFAULTS).fillna(25.0)

    # --- Multipliers for μ; widening for σ ---
    def row_multipliers(row) -> Tuple[float,float]:
        mk   = row["market_key"]
        role = (row.get("role") or "").upper()

        m_qb     = qb_pressure_multiplier(row.get("def_pressure_rate_z"), row.get("def_pass_epa_z")) if mk in {"pass_yards","pass_tds"} else 1.0
        m_sack   = sack_to_attempt_multiplier(row.get("def_sack_rate_z")) if mk in {"pass_yards","receptions","rec_yards"} else 1.0
        m_funnel = funnel_multiplier(row.get("def_pass_epa_z"), row.get("def_rush_epa_z"), mk)
        m_box    = boxcount_multiplier(row.get("light_box_rate_z"), row.get("heavy_box_rate_z"), mk)
        m_script = script_escalator(row.get("win_prob"), mk)
        m_pace   = pace_smoothing_multiplier(row.get("pace_z"), row.get("pace_z"))

        # injuries / coverage / AY sanity
        m_inj    = injury_multiplier(row.get("inj_status"), role, mk)

        # CB assignment (player-specific) — overrides generic coverage if present
        m_cb     = cb_assignment_multiplier(row.get("cb_penalty"), mk)
        m_cov    = 1.0 if not pd.isna(row.get("cb_penalty")) else coverage_multiplier(row.get("tag"), role, mk)

        m_aycap  = air_yards_sanity_cap(role, mk, row.get("ay_per_att_z"))

        sd_mult  = volatility_widening_factor(row.get("def_pressure_rate_z"), qb_inconsistent=False)

        mu_mult = m_qb * m_sack * m_funnel * m_box * m_script * m_pace * m_inj * m_cov * m_cb * m_aycap
        return mu_mult, sd_mult

    mu_mults, sd_mults = zip(*df.apply(row_multipliers, axis=1))
    df["_mu_adj"] = df["_mu0"] * pd.Series(mu_mults).astype(float)
    df["_sd_adj"] = df["_sd0"] * pd.Series(sd_mults).astype(float)

    # --- Model probability at the line with adjusted μ,σ ---
    def p_model_over(row) -> float:
        mu = row["_mu_adj"]; sd = row["_sd_adj"]; L = row["line"]
        if sd <= 0 or pd.isna(mu) or pd.isna(sd) or pd.isna(L): return np.nan
        return 1.0 - norm_cdf((L - mu) / sd)
    df["p_over_model"] = df.apply(p_model_over, axis=1)

    # --- Calibration shrinkage (optional) ---
    cal = _load_calibration()
    if cal:
        def _shrink(row):
            s = cal.get(row["market_key"], {}).get("shrink")
            if not s: return row["p_over_model"]
            return float(s)*row["p_over_model"] + (1.0-float(s))*row["p_over_fair_market"]
        df["p_over_model"] = df.apply(_shrink, axis=1)

    # --- 65/35 blend + fair odds + edge ---
    df["p_over_blend"] = 0.65*df["p_over_model"].fillna(df["p_over_fair_market"]) + 0.35*df["p_over_fair_market"]
    df["fair_over_odds"] = df["p_over_blend"].apply(prob_to_american)
    df["edge_pct"] = df["p_over_blend"] - df["p_over_imp"]

    # outputs
    df["model_mean"] = df["_mu_adj"]
    df["model_sd"]   = df["_sd_adj"]

    keep = [
        "player","team","role","market_key","line","book","event_id","home_team","away_team","commence_time",
        "price_over","price_under","p_over_imp","p_under_imp","p_over_fair_market",
        "p_over_model","p_over_blend","fair_over_odds","edge_pct","model_mean","model_sd",
        "win_prob","def_pressure_rate_z","def_pass_epa_z","def_rush_epa_z","pace_z","inj_status","cb_name","cb_penalty"
    ]
    out = df[keep].copy()

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
    (outdir / "props_priced.csv").write_text(df.to_csv(index=False))
    try:
        df.to_excel(outdir / "props_priced.xlsx", index=False)
    except Exception as e:
        print("[warn] xlsx export failed:", e)
