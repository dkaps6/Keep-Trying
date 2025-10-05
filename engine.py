# engine.py
# Turn-key pipeline driver for pricing + rules + outputs
# Robust to optional modules and signature changes.

from __future__ import annotations

import os
import sys
import math
import importlib
from typing import Dict, Tuple, Any, Optional

import pandas as pd


# -----------------------------
# Optional helpers / fallbacks
# -----------------------------
def _print(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# --- optional helpers (safe to miss) ---
try:
    from scripts.engine_helpers import (
        make_team_last4_from_player_form,
        safe_divide,
    )
except Exception:
    def make_team_last4_from_player_form(_: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame()

    def safe_divide(a, b, default=0.0):
        try:
            return a / b if b else default
        except Exception:
            return default


# --------------------------------
# Odds API (required in our flow)
# --------------------------------
from scripts.odds_api import (
    fetch_game_lines,
    fetch_props_all_events,
)

# --------------------------------
# Pricing core (required)
# --------------------------------
from scripts.model_core import price_props_for_events

# --------------------------------
# External features (optional)
# --------------------------------
try:
    from scripts.features_external import merge_external_features
    _HAS_MERGE_EXTERNAL = True
except Exception:
    _HAS_MERGE_EXTERNAL = False


# --------------------------------
# Rules engine – add compatibility
# --------------------------------
from scripts.rules_engine import apply_rules as _apply_rules

def _apply_rules_compat(features: Dict[str, Any],
                        side: str,
                        mu: float,
                        sigma: float) -> Tuple[float, float, str]:
    """
    Call scripts.rules_engine.apply_rules regardless of signature.
    Supports:
      - apply_rules(features) -> (mu, sigma, notes)
      - apply_rules(side, mu, sigma, features) -> (mu, sigma, notes)
      - apply_rules(side=..., mu=..., sigma=..., features=...) -> (mu, sigma, notes)
    """
    try:
        return _apply_rules(features)  # new style
    except TypeError:
        try:
            return _apply_rules(side, mu, sigma, features)  # old positional
        except TypeError:
            return _apply_rules(side=side, mu=mu, sigma=sigma, features=features)  # keyword


# -------------------------------------------------
# Utility: gentle downcast to keep CSV sizes small
# -------------------------------------------------
def _downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


# -------------------------------------------------
# Core pipeline
# -------------------------------------------------
def run_pipeline(target_date: str,
                 season: int,
                 out_dir: str = "outputs") -> None:
    """
    End-to-end run:
      1) Fetch game lines
      2) Fetch all props/odds for events
      3) Price with model_core
      4) Merge external features (if available)
      5) Apply elite rules (signature-agnostic)
      6) Write CSV outputs
    """
    # Log where we import this module from (helps in Actions logs)
    _print(f"Loaded engine module from: {os.path.abspath(__file__)}")

    # Make dirs
    ensure_dir(out_dir)
    ensure_dir("metrics")
    ensure_dir("inputs")

    # -------- external caches present? (informational) --------
    team_form_path = os.path.join("metrics", "team_form.csv")
    player_form_path = os.path.join("metrics", "player_form.csv")
    id_map_path = os.path.join("metrics", "id_map.csv")

    # Fetch game lines
    _print("Fetching game lines …")
    game_df = fetch_game_lines()
    if game_df is None or len(game_df) == 0:
        _print("WARNING: game_df is empty – continuing but props may fail.")

    # Fetch props for all events (Odds API)
    _print("Fetching all props for events …")
    props_df = fetch_props_all_events()
    if props_df is None or len(props_df) == 0:
        _print("WARNING: props_df is empty. Nothing to price.")
        # Still write empty outputs to avoid failing downstream users
        empty = pd.DataFrame()
        empty.to_csv(os.path.join(out_dir, "props_priced.csv"), index=False)
        return

    # Price with the model core
    priced_df = price_props_for_events(props_df)
    if priced_df is None or len(priced_df) == 0:
        _print("WARNING: pricing produced no rows.")
        priced_df = pd.DataFrame()

    # Merge team/player external features (optional)
    try:
        if _HAS_MERGE_EXTERNAL:
            priced_df = merge_external_features(priced_df)
        else:
            _print("merge_external_features unavailable or failed; using minimal features merge.")
    except Exception as e:
        _print(f"merge_external_features raised {type(e).__name__}: {e}; using minimal features merge.")

    # Info messages (to mirror your previous logs)
    if not os.path.exists(team_form_path) or os.path.getsize(team_form_path) == 0:
        _print("🛈  team_form.csv is empty.")
    if not os.path.exists(player_form_path) or os.path.getsize(player_form_path) == 0:
        _print("🛈  player_form.csv is empty.")
    if not os.path.exists(id_map_path) or os.path.getsize(id_map_path) == 0:
        _print("🛈  id_map.csv is empty.")

    # Apply rules per row with signature-compatible wrapper
    if len(priced_df) > 0:
        _print("Applying elite rules …")
        adj_mu = []
        adj_sigma = []
        notes = []

        # Build features dict per row; keep whatever columns exist
        feature_like_cols = set(priced_df.columns)

        for _, row in priced_df.iterrows():
            base_mu = float(row.get("mu", 0.0))
            base_sigma = float(row.get("sigma", 0.0))
            side = row.get("side", row.get("bet_side", "over"))

            # You can curate which keys to send; for now send everything
            fdict = {k: row[k] for k in feature_like_cols if k in row}

            # Also include canonical fields
            fdict.update({
                "mu": base_mu,
                "sigma": base_sigma,
                "side": side,
            })

            mu1, sig1, note = _apply_rules_compat(fdict, side, base_mu, base_sigma)
            adj_mu.append(mu1)
            adj_sigma.append(sig1)
            notes.append(note)

        priced_df["mu"] = adj_mu
        priced_df["sigma"] = adj_sigma
        priced_df["rules_notes"] = notes

    # Final housekeeping
    _print("Downcasting floats.")
    priced_df = _downcast_numeric(priced_df)

    # Write outputs
    ensure_dir(out_dir)
    priced_path = os.path.join(out_dir, "props_priced.csv")
    priced_df.to_csv(priced_path, index=False)
    _print(f"Wrote {len(priced_df)} rows to {priced_path}")


# -------------------------------------------------
# Dev entry (local runs)
# -------------------------------------------------
if __name__ == "__main__":
    # Defaults for quick local testing:
    # python engine.py 2025-10-04 2025 outputs
    date_arg = sys.argv[1] if len(sys.argv) > 1 else "today"
    season_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 2025
    out_arg = sys.argv[3] if len(sys.argv) > 3 else "outputs"
    run_pipeline(target_date=date_arg, season=season_arg, out_dir=out_arg)
