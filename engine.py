# engine.py
# ---------------------------------------------------------------------
# Core pipeline for: fetch lines/props -> build features -> apply rules
# -> (optionally) merge external features -> compute output CSVs.
#
# Safe with new team_form schema (posteam vs team) and string accessors.
# ---------------------------------------------------------------------

from __future__ import annotations

import os
import sys
import math
import json
from datetime import datetime
from typing import Dict, Tuple, Optional

import pandas as pd
import numpy as np

# ----------------------------
# Internal modules (yours)
# ----------------------------
# Odds / lines
try:
    from scripts.odds_api import fetch_props_all_events, fetch_game_lines
except Exception:
    # Allow engine to load without odds (pipeline can still run parts)
    def fetch_props_all_events() -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_game_lines() -> pd.DataFrame:
        return pd.DataFrame()

# Rules (your business logic)
from scripts.rules_engine import apply_rules

# Optional external feature merger (signature may vary across versions)
try:
    from scripts.features_external import merge_external_features as _merge_ext_raw

    def _merge_ext(df: pd.DataFrame, ext: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Wrap the external merge to tolerate either:
          merge_external_features(df, ext)
        or:
          merge_external_features(df)
        """
        try:
            return _merge_ext_raw(df, ext)  # new signature
        except TypeError:
            return _merge_ext_raw(df)       # old signature
except Exception:
    _merge_ext_raw = None

    def _merge_ext(df: pd.DataFrame, ext: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        # No-op if the module/signature isn't available
        return df

# Optional: pricing helpers (if present in your repo)
try:
    from scripts.pricing import (
        _model_prob,
        _market_fair_prob,
        _blend,
        prob_to_american,
        _edge,
        kelly,
    )
    _HAS_PRICING = True
except Exception:
    _HAS_PRICING = False


# =====================================================================
# Helpers
# =====================================================================

def _load_csv_safe(path: str, **kw) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kw)
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def _load_team_form(path: str = "metrics/team_form.csv") -> pd.DataFrame:
    """
    Load team_form and normalize schema so downstream code can safely use:
      - a 'team' column (uppercase strings)
      - other columns present as produced by fetch_all
    Works whether upstream produced 'posteam' or 'team'.
    """
    df = _load_csv_safe(path)

    if df.empty:
        return pd.DataFrame(columns=["team", "plays", "pass_rate", "rush_rate"])

    # Ensure there is a single canonical 'team' column
    if "team" not in df.columns:
        for cand in ("posteam", "team_abbr", "team_code"):
            if cand in df.columns:
                df = df.rename(columns={cand: "team"})
                break

    if "team" not in df.columns:
        df["team"] = ""

    # Force uppercase strings so we never need .str again later
    df["team"] = df["team"].fillna("").astype(str).str.upper()

    # Provide defaults for common numeric columns if missing
    for c in ("plays", "pass_rate", "rush_rate"):
        if c not in df.columns:
            df[c] = 0

    return df


def _build_external(season: Optional[int]) -> Dict[str, pd.DataFrame]:
    """
    Collects any external feature frames the pipeline may use.
    Extend as needed; keep keys stable.
    """
    ext: Dict[str, pd.DataFrame] = {}
    ext["team_form"] = _load_team_form("metrics/team_form.csv")
    # If you add more external CSVs later, load & normalize here and store in ext.
    return ext


def _coerce_team(val) -> str:
    return ("" if pd.isna(val) else str(val)).upper()


def _opp_from_row(row: pd.Series) -> str:
    # Try multiple field names to be tolerant of different sources
    for k in ("opp", "opponent", "opponent_team", "opponent_abbr", "opp_team"):
        if k in row and pd.notna(row[k]):
            return _coerce_team(row[k])
    # Try to derive from matchup like "DAL @ PHI"
    for k in ("matchup", "game", "label"):
        if k in row and isinstance(row[k], str) and "@" in row[k]:
            parts = [p.strip().upper() for p in row[k].replace("  ", " ").split("@")]
            if len(parts) == 2:
                # If we have a team field, decide away/home. Otherwise just return right side.
                if "team" in row and isinstance(row["team"], str):
                    t = _coerce_team(row["team"])
                    if t == parts[0]:  # team is away; opponent is home
                        return parts[1]
                    if t == parts[1]:  # team is home; opponent is away
                        return parts[0]
                return parts[1]
    return ""


def _make_features_row(row: pd.Series, ext: Dict[str, pd.DataFrame]) -> pd.Series:
    """
    Row-wise features built from external frames.
    Safe w.r.t types: never calls .str accessors on non-strings.
    """
    out = {}

    team_form = ext.get("team_form", pd.DataFrame())
    if not team_form.empty:
        opp = _opp_from_row(row)
        if opp:
            tf = team_form.loc[team_form["team"] == opp]
            if not tf.empty:
                # pull a few aggregates (extend as you wish)
                out["opp_pass_rate"] = float(tf["pass_rate"].mean())
                out["opp_rush_rate"] = float(tf["rush_rate"].mean())
                out["opp_plays"] = float(tf["plays"].mean())
            else:
                out["opp_pass_rate"] = 0.0
                out["opp_rush_rate"] = 0.0
                out["opp_plays"] = 0.0
        else:
            out["opp_pass_rate"] = 0.0
            out["opp_rush_rate"] = 0.0
            out["opp_plays"] = 0.0
    else:
        out["opp_pass_rate"] = 0.0
        out["opp_rush_rate"] = 0.0
        out["opp_plays"] = 0.0

    return pd.Series(out)


def _apply_rules_compat(features: pd.DataFrame):
    """
    Support both signatures of your rules engine:
      1) apply_rules(features) -> (mu, sigma, notes)
      2) apply_rules(side, mu, sigma, features=...) -> (mu, sigma, notes)
    We first try the new one; if it fails, fall back to old.
    """
    try:
        mu, sigma, notes = apply_rules(features)
        return mu, sigma, notes
    except TypeError:
        # Expect the caller to provide base side/mu/sigma if you use the old signature
        # We derive neutral placeholders here to remain backward compatible.
        side = np.where(features.get("market_side", pd.Series(dtype=float)).fillna(0).values >= 0, 1, -1)
        base_mu = np.zeros(len(features), dtype=float)
        base_sigma = np.ones(len(features), dtype=float)
        mu, sigma, notes = apply_rules(side, base_mu, base_sigma, features=features)
        return mu, sigma, notes


def _price_block(priced_df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional: compute blended probabilities, American fair odds, edges, Kelly.
    Only runs if scripts.pricing is available.
    """
    if not _HAS_PRICING or priced_df.empty:
        return priced_df

    # Expect model mean/sigma columns from rules
    if "mu" not in priced_df.columns or "sigma" not in priced_df.columns:
        return priced_df

    # Model probability of going over (toy example: normal CDF at 0 centered at mu/sigma).
    # Replace with your repo's _model_prob definition if different.
    try:
        priced_df["model_prob"] = _model_prob(priced_df["mu"], priced_df["sigma"])
    except Exception:
        # Fallback: 50/50 if model helper not compatible
        priced_df["model_prob"] = 0.5

    # Market fair probability (from price columns if present)
    if {"american", "price"}.intersection(priced_df.columns):
        price_col = "american" if "american" in priced_df.columns else "price"
        try:
            priced_df["market_fair_prob"] = _market_fair_prob(priced_df[price_col])
        except Exception:
            priced_df["market_fair_prob"] = np.nan
    else:
        priced_df["market_fair_prob"] = np.nan

    # Blend (if you want a blend of model & market)
    try:
        priced_df["blend_prob"] = _blend(priced_df["model_prob"], priced_df["market_fair_prob"])
    except Exception:
        priced_df["blend_prob"] = priced_df["model_prob"]

    # Edge & Kelly
    try:
        priced_df["edge"] = _edge(priced_df["blend_prob"], priced_df.get("market_implied_prob", np.nan))
    except Exception:
        priced_df["edge"] = np.nan

    try:
        priced_df["kelly"] = kelly(priced_df["blend_prob"], priced_df.get("market_implied_prob", np.nan))
    except Exception:
        priced_df["kelly"] = np.nan

    # Optional: fair American odds from blend
    try:
        priced_df["fair_american"] = prob_to_american(priced_df["blend_prob"])
    except Exception:
        pass

    return priced_df


# =====================================================================
# Pipeline
# =====================================================================

def run_pipeline(target_date: Optional[str] = None,
                 season: Optional[int] = None,
                 out_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Main entrypoint used by run_model.py
    """
    print("Loading engine …")
    ext = _build_external(season)

    # -----------------------------------------------------------------
    # 1) Fetch market data
    # -----------------------------------------------------------------
    print("Fetching game lines …")
    try:
        game_lines = fetch_game_lines()
    except Exception as e:
        print("Game lines fetch failed:", e)
        game_lines = pd.DataFrame()

    print("Fetching all props for events …")
    try:
        props_raw = fetch_props_all_events()
    except Exception as e:
        print("Props fetch failed:", e)
        props_raw = pd.DataFrame()

    # If we have nothing, return early with an empty output
    if props_raw.empty:
        print("No props returned; writing empty outputs.")
        priced_df = pd.DataFrame()
        if out_dir:
            os.makedirs("outputs", exist_ok=True)
            priced_df.to_csv(os.path.join("outputs", "props_priced.csv"), index=False)
        return priced_df

    # -----------------------------------------------------------------
    # 2) Build features (row-wise using external frames)
    # -----------------------------------------------------------------
    print("Applying feature builders …")
    feat_rows = props_raw.apply(lambda r: _make_features_row(r, ext), axis=1)
    features = pd.concat([props_raw.reset_index(drop=True), feat_rows.reset_index(drop=True)], axis=1)

    # Optionally merge any “big” external frame(s) using your helper if available
    try:
        features = _merge_ext(features, ext)
    except Exception as e:
        print("External merge skipped (non-fatal):", e)

    # -----------------------------------------------------------------
    # 3) Apply rules (your model logic) to produce mu, sigma, notes
    # -----------------------------------------------------------------
    print("Applying elite rules …")
    mu, sigma, notes = _apply_rules_compat(features)
    priced_df = features.copy()
    priced_df["mu"] = mu
    priced_df["sigma"] = sigma
    priced_df["notes"] = notes

    # -----------------------------------------------------------------
    # 4) Optional pricing block (probabilities, edges, Kelly, fair odds)
    # -----------------------------------------------------------------
    print("Pricing / edges …")
    priced_df = _price_block(priced_df)

    # -----------------------------------------------------------------
    # 5) Write outputs
    # -----------------------------------------------------------------
    out_dir = out_dir or "outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "props_priced.csv")
    priced_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(priced_df):,} rows")

    return priced_df


# Allow direct CLI execution for quick testing if needed:
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="today")
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--write", default="outputs")
    args = ap.parse_args()

    run_pipeline(target_date=args.date, season=args.season, out_dir=args.write)
