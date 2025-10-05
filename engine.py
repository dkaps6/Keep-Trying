# engine.py
# End-to-end pipeline:
#  1) fetch/load props
#  2) merge contextual features (team/player form, weather, id map)
#  3) apply rules to compute (mu, sigma) and model probability
#  4) attach pricing columns (market_prob, fair_prob, fair_american, edge, kelly)
#  5) write outputs

from __future__ import annotations
import os
import json
from math import erf, sqrt
from typing import Dict, Tuple, Any, Optional

import pandas as pd
import numpy as np

# --- Optional imports guarded so engine never breaks if they’re missing ---
try:
    # Your rules entry point must exist
    from scripts.rules_engine import apply_rules
except Exception as e:  # pragma: no cover
    apply_rules = None

try:
    # Optional: external feature merge (weather, injuries, extra metrics)
    from scripts.features_external import merge_external_features
except Exception:
    merge_external_features = None

try:
    # Optional: odds API fetchers
    from scripts.odds_api import fetch_props_all_events
except Exception:
    fetch_props_all_events = None

# Pricing helpers (shipped earlier)
from scripts.pricing import attach_pricing_columns

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

ROOT = os.getcwd()
DIR_OUTPUTS = os.path.join(ROOT, "outputs")
DIR_METRICS = os.path.join(ROOT, "metrics")
DIR_INPUTS  = os.path.join(ROOT, "inputs")

os.makedirs(DIR_OUTPUTS, exist_ok=True)
os.makedirs(DIR_METRICS, exist_ok=True)
os.makedirs(DIR_INPUTS,  exist_ok=True)


def _read_csv_safe(path: str, **kw) -> pd.DataFrame:
    """Read CSV if it exists; otherwise return empty DF with no error."""
    try:
        if os.path.exists(path):
            return pd.read_csv(path, **kw)
    except Exception:
        pass
    return pd.DataFrame()


def _write_csv_safe(df: pd.DataFrame, path: str) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
    except Exception:
        # best-effort write; don't crash the pipeline on I/O
        pass


def _norm_cdf(z: float) -> float:
    """Standard normal CDF via erf (no SciPy dependency)."""
    if z is None or not np.isfinite(z):
        return np.nan
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def _normalize_team(abbr: Any) -> str:
    """Uppercase simple normalizer; add mapping here if you need."""
    if isinstance(abbr, str):
        return abbr.strip().upper()
    return ""


def _load_context_tables() -> Dict[str, pd.DataFrame]:
    """Load metrics and inputs tables; empty DataFrames if missing."""
    team_form       = _read_csv_safe(os.path.join(DIR_METRICS, "team_form.csv"))
    team_week_form  = _read_csv_safe(os.path.join(DIR_METRICS, "team_week_form.csv"))
    player_form     = _read_csv_safe(os.path.join(DIR_METRICS, "player_form.csv"))
    id_map          = _read_csv_safe(os.path.join(DIR_INPUTS,  "id_map.csv"))
    weather         = _read_csv_safe(os.path.join(DIR_INPUTS,  "weather.csv"))

    # Basic normalization for merge keys
    for df in (team_form, team_week_form):
        if "team" in df.columns:
            df["team"] = df["team"].astype(str).str.upper()

    return {
        "team_form": team_form,
        "team_week_form": team_week_form,
        "player_form": player_form,
        "id_map": id_map,
        "weather": weather,
    }


def _fetch_props(date: Optional[str], season: Optional[int]) -> pd.DataFrame:
    """
    Fetch props from your odds source if available.
    Fallbacks:
      - a local 'inputs/props.csv' if you keep one around for tests
      - otherwise return empty DF
    """
    # Preferred: live odds API if configured
    if callable(fetch_props_all_events):
        try:
            return fetch_props_all_events(date=date, season=season)
        except Exception:
            pass

    # Fallback for local testing
    local = os.path.join(DIR_INPUTS, "props.csv")
    df = _read_csv_safe(local)
    if not df.empty:
        return df

    # Last resort: empty
    return pd.DataFrame(columns=["event_id", "market", "prop", "player", "team", "opp_team", "price"])


def _coalesce_team_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure 'team' and 'opp_team' exist; coalesce from common alternatives if needed."""
    if "team" not in df.columns:
        for cand in ("home_team", "participant", "team_abbr", "team_name"):
            if cand in df.columns:
                df["team"] = df[cand]
                break
    if "opp_team" not in df.columns:
        for cand in ("away_team", "opponent", "opp_abbr", "opp_name"):
            if cand in df.columns:
                df["opp_team"] = df[cand]
                break

    # Normalize to uppercase strings
    if "team" in df.columns:
        df["team"] = df["team"].astype(str).str.upper()
    else:
        df["team"] = ""

    if "opp_team" in df.columns:
        df["opp_team"] = df["opp_team"].astype(str).str.upper()
    else:
        df["opp_team"] = ""

    return df


def _merge_features(props_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Merge contextual tables with raw props.
    Returns (merged_df, source_note)
    """
    note_chunks = []

    if props_df is None or props_df.empty:
        return pd.DataFrame(), "props:empty"

    df = props_df.copy()
    df = _coalesce_team_columns(df)

    ctx = _load_context_tables()
    team_form      = ctx["team_form"]
    team_week_form = ctx["team_week_form"]
    player_form    = ctx["player_form"]
    weather        = ctx["weather"]

    # Left joins on team (and week if you have one)
    if not team_form.empty and "team" in team_form.columns:
        df = df.merge(team_form.add_prefix("team_"), left_on="team", right_on="team_team", how="left")
        note_chunks.append("team_form")
    else:
        note_chunks.append("team_form:empty")

    if not team_week_form.empty and "team" in team_week_form.columns:
        df = df.merge(team_week_form.add_prefix("tw_"), left_on="team", right_on="tw_team", how="left")
        note_chunks.append("team_week_form")
    else:
        note_chunks.append("team_week_form:empty")

    if not player_form.empty:
        # Heuristic join on player name, if present
        key = "player" if "player" in df.columns else None
        if key and "player" in player_form.columns:
            df = df.merge(player_form.add_prefix("pf_"), left_on=key, right_on="pf_player", how="left")
            note_chunks.append("player_form")
        else:
            note_chunks.append("player_form:nojoin")
    else:
        note_chunks.append("player_form:empty")

    # Optional external merge
    if callable(merge_external_features):
        try:
            df_ext = merge_external_features(df)
            if isinstance(df_ext, pd.DataFrame) and len(df_ext) == len(df):
                df = df_ext
                note_chunks.append("external_features")
            else:
                note_chunks.append("external_features:skipped")
        except Exception:
            note_chunks.append("external_features:err")
    else:
        note_chunks.append("external_features:none")

    # Optional weather merge: if event_id key exists
    if not weather.empty and "event_id" in df.columns and "event_id" in weather.columns:
        df = df.merge(weather.add_prefix("w_"), on="event_id", how="left")
        note_chunks.append("weather")
    else:
        note_chunks.append("weather:skip")

    return df, ";".join(note_chunks)


def _make_features_row(row: pd.Series, merge_note: str) -> Dict[str, Any]:
    """
    Produce a features dict per row to feed the rules engine.
    Put *everything* you might need here; rules_engine can pick/ignore.
    """
    out = {
        "player": row.get("player"),
        "market": row.get("market"),
        "prop":   row.get("prop"),
        "team":   _normalize_team(row.get("team")),
        "opp_team": _normalize_team(row.get("opp_team")),
        "price":  row.get("price"),
        "odds":   row.get("odds", row.get("price")),
        "merge_note": merge_note,
        # examples from merges (safe .get)
        "team_pressure_rate": row.get("team_pressure_rate"),
        "pass_epa_allowed":   row.get("team_pass_epa_allowed", row.get("pass_epa_allowed")),
        "pace":               row.get("tw_pace", row.get("pace")),
        "pf_usage":           row.get("pf_usage"),
        "w_temp":             row.get("w_temp"),
        "w_wind":             row.get("w_wind"),
    }
    return out


def _apply_rules_compat(fdict: Dict[str, Any]) -> Tuple[float, float, str]:
    """
    Call rules_engine.apply_rules(features) and normalize its return.
    Supported returns:
      - (mu, sigma, note)
      - dict with keys: mu, sigma, note or notes
    """
    if not callable(apply_rules):
        # Fallback: neutral model → p=0.5
        return 0.0, 1e-6, "rules:missing"

    try:
        res = apply_rules(fdict)
    except Exception as e:
        return 0.0, 1e-6, f"rules:error:{e}"

    # tuple-like
    if isinstance(res, (list, tuple)) and len(res) >= 2:
        mu = float(res[0])
        sigma = float(res[1]) if np.isfinite(res[1]) and res[1] != 0 else 1e-6
        note = str(res[2]) if len(res) > 2 else ""
        return mu, sigma, note

    # dict-like
    if isinstance(res, dict):
        mu = float(res.get("mu", 0.0))
        sigma = res.get("sigma", 1e-6)
        try:
            sigma = float(sigma)
            if not np.isfinite(sigma) or sigma == 0:
                sigma = 1e-6
        except Exception:
            sigma = 1e-6
        note = str(res.get("note", res.get("notes", "")))
        return mu, sigma, note

    # unknown fallback
    return 0.0, 1e-6, "rules:unknown_return"


def _price_frame(priced_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize columns and attach pricing metrics (market_prob, fair_prob, fair_american, edge, kelly).
    - Accepts 'price' as American odds; maps to 'odds'
    - Requires 'model_prob' (produced below); if missing, defaults to 0.5
    """
    if priced_df is None or len(priced_df) == 0:
        return priced_df

    df = priced_df.copy()

    if "player" not in df.columns and "player_name_raw" in df.columns:
        df["player"] = df["player_name_raw"]

    if "odds" not in df.columns and "price" in df.columns:
        df["odds"] = df["price"]

    if "model_prob" not in df.columns:
        df["model_prob"] = 0.5
    else:
        df["model_prob"] = pd.to_numeric(df["model_prob"], errors="coerce").fillna(0.5)

    df = attach_pricing_columns(df)

    if "rule_notes" in df.columns:
        df["rule_notes"] = df["rule_notes"].apply(lambda x: x if isinstance(x, str) else str(x))

    return df


# -------------------------------------------------------------------------
# Public entrypoint
# -------------------------------------------------------------------------

def run_pipeline(date: Optional[str] = None,
                 season: Optional[int] = None,
                 write_outputs: bool = True) -> pd.DataFrame:
    """
    Orchestrates the full run. Returns the final priced DataFrame.
    """
    # 1) Fetch props
    props_df = _fetch_props(date=date, season=season)

    # 2) Merge features
    merged_df, merge_note = _merge_features(props_df)

    # 3) Apply rules → mu, sigma, model_prob
    rows = []
    for _, row in merged_df.iterrows():
        fdict = _make_features_row(row, merge_note)
        mu, sigma, note = _apply_rules_compat(fdict)
        # model probability from z-score (mu/sigma)
        z = mu / sigma if sigma not in (0, None) else np.nan
        model_prob = _norm_cdf(z) if np.isfinite(z) else 0.5

        rows.append({
            **row.to_dict(),
            "mu": mu,
            "sigma": sigma,
            "model_prob": model_prob,
            "rule_notes": note
        })

    if rows:
        priced_input = pd.DataFrame(rows)
    else:
        priced_input = merged_df.copy()

    # 4) Pricing
    priced = _price_frame(priced_input)

    # 5) Write outputs
    if write_outputs:
        _write_csv_safe(priced, os.path.join(DIR_OUTPUTS, "props_priced.csv"))

    return priced
