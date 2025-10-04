# engine.py
from __future__ import annotations

import os
import sys
from typing import Tuple, List

import numpy as np
import pandas as pd

# Local modules
from scripts.engine_helpers import make_team_last4_from_player_form
from scripts.rules_engine import apply_rules as rules_apply  # your existing rules engine
from scripts.model_core import price_props_for_events          # your pricing core (if different, adjust)
from scripts.odds_api import fetch_game_lines, fetch_props_all_events
from scripts.pricing import devig_american_prob               # if needed inside pricing core
from scripts.features_external import merge_external_features  # optional; if you have it

# --------------------------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def safe_read_csv(path: str, **kw) -> pd.DataFrame:
    try:
        if not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_csv(path, **kw)
    except Exception as e:
        print(f"⚠️  Failed to read {path}: {e}")
        return pd.DataFrame()


def _normalize_team_form(team_form: pd.DataFrame) -> pd.DataFrame:
    """Normalize/patch external team features so downstream joins never fail."""
    tf = team_form.copy()

    # Standardize team column
    if "team" not in tf.columns:
        if "defteam" in tf.columns:
            tf = tf.rename(columns={"defteam": "team"})
        elif "posteam" in tf.columns:
            tf = tf.rename(columns={"posteam": "team"})

    if "team" not in tf.columns:
        # give up but keep schema
        tf["team"] = "UNK"

    # Map proe_proxy -> proe if needed
    if "proe" not in tf.columns and "proe_proxy" in tf.columns:
        tf["proe"] = tf["proe_proxy"]
    elif "proe" not in tf.columns:
        tf["proe"] = 0.0

    # Optional columns (coverage & box shares) — fill safe defaults
    for col in ["man_rate_z", "zone_rate_z", "heavy_box_share", "light_box_share"]:
        if col not in tf.columns:
            tf[col] = 0.0

    # Required numeric columns — if missing, create safe defaults
    for col in ["pressure_rate", "pressure_z", "pass_epa_allowed", "pass_epa_z"]:
        if col not in tf.columns:
            tf[col] = 0.0

    keep = ["team", "pressure_rate", "pressure_z",
            "pass_epa_allowed", "pass_epa_z",
            "proe", "man_rate_z", "zone_rate_z",
            "heavy_box_share", "light_box_share"]
    tf = tf[[c for c in keep if c in tf.columns]].drop_duplicates(subset=["team"])
    return tf.reset_index(drop=True)


def _downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    for c in df.select_dtypes(include=["int64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    return df


# --------------------------------------------------------------------------------------
# External features loader (resilient)
# --------------------------------------------------------------------------------------

def load_external_features(season: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      team_form (normalized),
      player_form (raw),
      id_map
    """
    team_form = safe_read_csv("metrics/team_form.csv")
    if team_form.empty:
        print("⚠️  team_form.csv is empty.")
    else:
        # display any missing optional columns
        opt = ['heavy_box_share', 'light_box_share', 'man_rate_z', 'zone_rate_z', 'proe', 'proe_proxy']
        missing = [c for c in opt if c not in team_form.columns]
        if missing:
            print(f"ℹ️  team_form.csv missing columns: {missing}")
    team_form = _normalize_team_form(team_form)

    player_form = safe_read_csv("metrics/player_form.csv")
    if player_form.empty:
        print("ℹ️  player_form.csv is empty.")

    id_map = safe_read_csv("metrics/id_map.csv")
    if id_map.empty:
        print("ℹ️  id_map.csv is empty.")

    return team_form, player_form, id_map


# --------------------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------------------

def run_pipeline(target_date: str, season: int, out_dir: str = "outputs") -> None:
    """
    Full slate pipeline:
      1) Fetch odds (games + props)
      2) Load external features (team_form, player_form, id_map)
      3) Build team aggregates from player form (resilient)
      4) Merge features
      5) Apply rules (elite rules engine)
      6) Price props & write outputs
    """
    _ensure_dir(out_dir)

    print(f"{season} done.")
    print("Downcasting floats.")

    # ----------------------------------------------------------------------------------
    # 1) Fetch odds
    # ----------------------------------------------------------------------------------
    print("Fetching game lines …")
    try:
        game_df = fetch_game_lines()
    except Exception as e:
        print(f"⚠️  fetch_game_lines() failed: {e}")
        game_df = pd.DataFrame()

    print("Fetching all props for events …")
    try:
        props_df = fetch_props_all_events()
    except Exception as e:
        print(f"⚠️  fetch_props_all_events() failed: {e}")
        props_df = pd.DataFrame()

    # ----------------------------------------------------------------------------------
    # 2) External features
    # ----------------------------------------------------------------------------------
    team_form, player_form, id_map = load_external_features(season)

    # ----------------------------------------------------------------------------------
    # 3) Team aggregates from player_form (resilient)
    # ----------------------------------------------------------------------------------
    team_l4 = make_team_last4_from_player_form(player_form)

    # ----------------------------------------------------------------------------------
    # 4) Merge features (best-effort)
    # ----------------------------------------------------------------------------------
    # Example: if you have a helper to merge: merge_external_features()
    # Otherwise, do light merges here.
    try:
        features = merge_external_features(game_df, team_form, team_l4, id_map)
    except Exception as e:
        print(f"ℹ️  merge_external_features unavailable or failed ({e}); using minimal features merge.")
        # Minimal merge on team if possible
        features = game_df.copy()
        for side in ("home_team", "away_team"):
            if side in features.columns and "team" in team_form.columns:
                tf = team_form.add_prefix(f"{side}_")
                features = features.merge(
                    tf.rename(columns={f"{side}_team": side}),
                    on=side, how="left"
                )

    # ----------------------------------------------------------------------------------
    # 5) Apply elite rules (works on features/props)
    # ----------------------------------------------------------------------------------
    try:
        mu, sigma, notes = rules_apply(features)  # Your rules_engine.apply_rules signature
    except TypeError:
        # Older signature fallback: rules_apply(features) -> (mu, sigma)
        tmp = rules_apply(features)
        if isinstance(tmp, tuple) and len(tmp) == 3:
            mu, sigma, notes = tmp
        else:
            mu, sigma = tmp
            notes = pd.Series([""] * len(features))

    # Attach model parameters
    features["model_mu"] = mu
    features["model_sigma"] = sigma
    features["notes"] = notes

    # ----------------------------------------------------------------------------------
    # 6) Price props & write outputs
    # ----------------------------------------------------------------------------------
    try:
        priced = price_props_for_events(features, props_df)
    except Exception as e:
        print(f"⚠️  price_props_for_events failed: {e}")
        priced = pd.DataFrame()

    priced = _downcast_numeric(priced)
    out_path = os.path.join(out_dir, "props_priced.csv")
    priced.to_csv(out_path, index=False)
    print(f"✅ Wrote {out_path} with {len(priced)} rows.")
