# engine.py
# Never-blank pipeline with live weather override + weather-aware mu/sigma tweaks.

from __future__ import annotations

import os
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

from scripts.odds_api import fetch_props_all_events
from scripts.rules_engine import apply_rules
from scripts.pricing import (
    _model_prob,
    _market_fair_prob,
    _blend,
    prob_to_american,
    _edge,
    kelly,
)

METRICS_DIR = "metrics"
INPUTS_DIR = "inputs"
OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ---------- Safe readers ------------------------------------------------------

def _read_csv_safe(path: str, cols: list[str] | None = None) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=cols or [])
    try:
        df = pd.read_csv(path)
        if cols:
            for c in cols:
                if c not in df.columns:
                    df[c] = pd.NA
        return df
    except Exception:
        return pd.DataFrame(columns=cols or [])

def _normalize_team(s: Any) -> str:
    try:
        return str(s).strip().upper()
    except Exception:
        return ""

TEAM_WEEK_COLS = [
    "season","team","week","pass_plays","rush_plays","neutral_pass_rate",
    "exp_pass_rate","exp_rush_rate","rz_rate","total_plays"
]
PLAYER_FORM_COLS = [
    "season","team","week","player","targets","rush_att","tgt_share","rush_share"
]
ID_MAP_COLS = ["player","team","position","espn_player_id"]
WEATHER_BASE_COLS = ["team","abbr","is_dome","surface","altitude_ft","temp_f","wind_mph","precip_in"]
WEATHER_LIVE_COLS = ["team","abbr","date","kickoff_local","lat","lon","temp_f","wind_mph","precip_mm","precip_prob","is_dome"]

# ---------- Feature building --------------------------------------------------

def _merge_weather() -> pd.DataFrame:
    """
    Merge baseline stadium weather (inferred) with live weather (if available).
    Live weather overrides baseline for the same team.
    """
    base = _read_csv_safe(os.path.join(INPUTS_DIR, "weather.csv"), WEATHER_BASE_COLS).copy()
    if base.empty:
        base["team"] = []
    base["team_norm"] = (base.get("abbr") or base.get("team")).astype(str).str.upper() if not base.empty else []

    live = _read_csv_safe(os.path.join(INPUTS_DIR, "weather_live.csv"), WEATHER_LIVE_COLS).copy()
    if not live.empty:
        live["team_norm"] = live.get("abbr", live.get("team")).astype(str).str.upper()

        # Left join base -> live; prefer live values where present
        merged = base.merge(
            live[["team_norm","temp_f","wind_mph","precip_mm","precip_prob","is_dome"]],
            on="team_norm", how="left", suffixes=("","_live")
        )

        # override numeric where live available
        for col in ("temp_f","wind_mph"):
            merged[col] = merged[f"{col}_live"].combine_first(merged[col])
        # precip: base uses "precip_in" while live gives "precip_mm" and "precip_prob"
        # we keep both so rules can choose what they prefer; also convert mm→in as aux
        merged["precip_mm"] = merged["precip_mm"].fillna(np.nan)
        merged["precip_prob"] = merged["precip_prob"].fillna(np.nan)
        merged["precip_in"] = merged["precip_in"].astype(float).fillna(merged["precip_mm"].astype(float) / 25.4)

        # dome flag: if base says dome, keep dome; else accept live is_dome if present
        merged["is_dome"] = np.where(merged["is_dome"].isna(), merged["is_dome_live"], merged["is_dome"])
        merged.drop(columns=[c for c in merged.columns if c.endswith("_live")], inplace=True)
        return merged

    # no live → just base
    return base

def _merge_features(props_df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    """
    Returns (features_df, source_tag, note)
    source_tag in {"team_only","team_player"}
    """
    team_form = _read_csv_safe(os.path.join(METRICS_DIR, "team_form.csv"))
    team_week = _read_csv_safe(os.path.join(METRICS_DIR, "team_week_form.csv"), TEAM_WEEK_COLS)
    player_form = _read_csv_safe(os.path.join(METRICS_DIR, "player_form.csv"), PLAYER_FORM_COLS)
    id_map = _read_csv_safe(os.path.join(INPUTS_DIR, "id_map.csv"), ID_MAP_COLS)
    weather = _merge_weather()

    if props_df.empty:
        return props_df.assign(source="none", note="no_props"), "none", "no_props"

    # Normalize keys
    props_df["team_norm"] = props_df["team"].map(_normalize_team)
    team_form["team_norm"] = team_form["team"].map(_normalize_team)
    team_week["team_norm"] = team_week["team"].map(_normalize_team)
    if not weather.empty:
        wx_key = "abbr" if "abbr" in weather.columns else "team"
        weather["team_norm"] = weather[wx_key].map(_normalize_team)

    # Merge team form + weekly rates
    base = props_df.merge(
        team_form.drop_duplicates(subset=["team_norm"]),
        on="team_norm", how="left", suffixes=("","_tf")
    ).merge(
        team_week[["team_norm","neutral_pass_rate","exp_pass_rate","exp_rush_rate","rz_rate"]]
            .drop_duplicates("team_norm"),
        on="team_norm", how="left"
    )

    # Merge stadium/weather (prefer live)
    if not weather.empty:
        base = base.merge(
            weather.drop_duplicates("team_norm"),
            on="team_norm", how="left", suffixes=("","_wx")
        )

    enriched = base.copy()
    can_player = not player_form.empty and not id_map.empty and ("player" in props_df.columns)

    if can_player:
        id_map["player_key"] = id_map["player"].astype(str).str.strip().str.upper()
        id_map["team_norm"] = id_map["team"].map(_normalize_team)

        enriched["player_key"] = enriched["player"].astype(str).str.strip().str.upper()
        enriched = enriched.merge(
            id_map[["player_key","team_norm","position","espn_player_id"]],
            on=["player_key","team_norm"], how="left"
        )

        pf = player_form.copy()
        pf["player_key"] = pf["player"].astype(str).str.strip().str.upper()
        pf_mean = (
            pf.groupby(["player_key","team"], dropna=False)[["tgt_share","rush_share"]]
              .mean()
              .reset_index()
        )
        pf_mean["team_norm"] = pf_mean["team"].map(_normalize_team)
        enriched = enriched.merge(
            pf_mean[["player_key","team_norm","tgt_share","rush_share"]],
            on=["player_key","team_norm"], how="left"
        )
        source, note = "team_player", ""
    else:
        source, note = "team_only", "player_form/id_map missing; team-only scoring"

    return enriched, source, note

# ---------- Rules safety wrapper ---------------------------------------------

def _rules_safe(feature_row: pd.Series) -> Tuple[float, float, str]:
    try:
        feats: Dict[str, Any] = feature_row.to_dict()
        out = apply_rules(feats)
        if isinstance(out, tuple) and len(out) >= 2:
            mu, sigma = float(out[0]), float(out[1])
            note = out[2] if len(out) > 2 else ""
            return mu, sigma, str(note)
        if isinstance(out, dict):
            mu = float(out.get("mu", 0.0))
            sigma = float(out.get("sigma", 1.0))
            note = str(out.get("note", ""))
            return mu, sigma, note
        return 0.0, 1.0, "rules_return_unknown"
    except Exception as e:
        return 0.0, 1.0, f"rules_error:{type(e).__name__}"

# ---------- Weather-aware adjustments -----------------------------------------

# Tunable conservative coefficients
WEATHER_ADJ = {
    "wind_mph_threshold": 15.0,   # start penalizing deep/air yards
    "wind_mu_penalty":    0.07,   # 7% mean reduction for pass/rec at threshold (scales up to ~12% @ 25 mph)
    "wind_sigma_widen":   0.05,   # widen sigma 5%

    "precip_mm_heavy":    1.0,    # heavy precip threshold per hour
    "precip_prob_high":   0.70,   # or high precipitation probability
    "precip_mu_penalty":  0.05,   # 5% mean reduction (rec YAC / catch issues)
    "precip_sigma_widen": 0.05,

    "temp_cold":          25.0,   # F — below this, small pass penalty
    "temp_hot":           85.0,   # F — above this, tiny rush fatigue nudge (can widen sigma slightly)
    "temp_mu_penalty":    0.02,   # 2% adjustment
}

def _market_type(row: pd.Series) -> str:
    m = str(row.get("market") or row.get("stat") or "").lower()
    # normalize to buckets
    if "pass" in m and "yard" in m:
        return "pass_yds"
    if "receiv" in m and "yard" in m:
        return "rec_yds"
    if "reception" in m:
        return "recs"
    if "rush" in m and "yard" in m:
        return "rush_yds"
    if "rush" in m and "attempt" in m:
        return "rush_att"
    if "pass" in m and "td" in m:
        return "pass_tds"
    if "field goal" in m or "fg" in m:
        return "fg"
    return "other"

def _apply_weather_to_mu_sigma(row: pd.Series, mu: float, sigma: float) -> Tuple[float, float, str]:
    # If dome, do nothing
    if str(row.get("is_dome")).upper() in ("TRUE","1","YES","Y"):
        return mu, sigma, ""

    wind = row.get("wind_mph")
    temp = row.get("temp_f")
    precip_mm = row.get("precip_mm")
    precip_prob = row.get("precip_prob")

    mk = _market_type(row)
    note_bits = []

    # Wind: penalize pass/receiving markets
    if pd.notna(wind) and mk in ("pass_yds","rec_yds","recs","fg"):
        if wind >= WEATHER_ADJ["wind_mph_threshold"]:
            # scale penalty linearly 15→25 mph adds up to ~1.7x penalty
            extra = max(0.0, (wind - WEATHER_ADJ["wind_mph_threshold"]) / 10.0)
            wind_pen = WEATHER_ADJ["wind_mu_penalty"] * (1.0 + 0.7 * extra)
            mu *= (1.0 - wind_pen)
            sigma *= (1.0 + WEATHER_ADJ["wind_sigma_widen"])
            note_bits.append(f"wind_penalty_{wind:.0f}mph")

    # Precipitation: handling & YAC impacts
    heavy_precip = (pd.notna(precip_mm) and precip_mm >= WEATHER_ADJ["precip_mm_heavy"]) or \
                   (pd.notna(precip_prob) and precip_prob >= WEATHER_ADJ["precip_prob_high"])
    if heavy_precip and mk in ("rec_yds","recs","pass_yds"):
        mu *= (1.0 - WEATHER_ADJ["precip_mu_penalty"])
        sigma *= (1.0 + WEATHER_ADJ["precip_sigma_widen"])
        note_bits.append("precip_penalty")

    # Temperature: small effects
    if pd.notna(temp):
        if temp <= WEATHER_ADJ["temp_cold"] and mk in ("pass_yds","rec_yds","recs","fg"):
            mu *= (1.0 - WEATHER_ADJ["temp_mu_penalty"])
            note_bits.append("cold_penalty")
        elif temp >= WEATHER_ADJ["temp_hot"] and mk in ("rush_yds","rush_att"):
            # hot → slight fatigue / rotation uncertainty => widen sigma a hair
            sigma *= (1.0 + WEATHER_ADJ["temp_sigma_widen"] if "temp_sigma_widen" in WEATHER_ADJ else 0.02)
            note_bits.append("heat_sigma")

    return mu, sigma, ";".join(note_bits)

# ---------- Pipeline ----------------------------------------------------------

def run_pipeline(target_date: str, season: int, out_dir: str | None = None):
    out_dir = out_dir or OUTPUTS_DIR

    # 1) Lines and props
    props_df: pd.DataFrame = fetch_props_all_events()
    if props_df is None or props_df.empty:
        pd.DataFrame().to_csv(os.path.join(out_dir, "props_priced.csv"), index=False)
        print("No props returned from odds API.")
        return

    # 2) Merge features (team-only guaranteed; player optional; weather merged)
    feats_df, source, note = _merge_features(props_df)

    # 3) Apply rules row-wise
    mu_list, sigma_list, notes = [], [], []
    for _, row in feats_df.iterrows():
        mu, sigma, n = _rules_safe(row)
        # Weather-aware adjustments layered on top
        mu, sigma, wx_note = _apply_weather_to_mu_sigma(row, mu, sigma)
        mu_list.append(mu)
        sigma_list.append(sigma)
        notes.append((n or "") + ((";" + wx_note) if wx_note else ""))

    feats_df["mu"] = mu_list
    feats_df["sigma"] = sigma_list
    feats_df["rule_note"] = notes

    # 4) Probabilities / edges
    feats_df["model_prob"] = _model_prob(feats_df["mu"], feats_df["sigma"])
    feats_df["fair_prob"] = _market_fair_prob(feats_df)
    feats_df["blend_prob"] = _blend(feats_df["model_prob"], feats_df["fair_prob"])
    feats_df["model_american"] = prob_to_american(feats_df["model_prob"])
    feats_df["fair_american"] = prob_to_american(feats_df["fair_prob"])
    feats_df["edge"] = _edge(feats_df["model_prob"], feats_df["fair_prob"])
    feats_df["kelly"] = kelly(feats_df["blend_prob"], feats_df.get("price", pd.Series(dtype=float)))

    # 5) Traceability
    feats_df["source"] = source
    feats_df["note"] = note

    # 6) Write
    out_path = os.path.join(out_dir, "props_priced.csv")
    feats_df.to_csv(out_path, index=False)
    print(f"Wrote {len(feats_df)} rows → {out_path}")

# rules wrapper from earlier version (kept here)
def _rules_safe(feature_row: pd.Series) -> Tuple[float, float, str]:
    try:
        feats: Dict[str, Any] = feature_row.to_dict()
        out = apply_rules(feats)
        if isinstance(out, tuple) and len(out) >= 2:
            mu, sigma = float(out[0]), float(out[1])
            note = out[2] if len(out) > 2 else ""
            return mu, sigma, str(note)
        if isinstance(out, dict):
            mu = float(out.get("mu", 0.0))
            sigma = float(out.get("sigma", 1.0))
            note = str(out.get("note", ""))
            return mu, sigma, note
        return 0.0, 1.0, "rules_return_unknown"
    except Exception as e:
        return 0.0, 1.0, f"rules_error:{type(e).__name__}"

# For run_model.py import
def run_pipeline_entry(target_date: str, season: int, out_dir: str | None = None):
    run_pipeline(target_date, season, out_dir)
