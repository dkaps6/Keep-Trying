# engine.py
# Turn-key pipeline driver for pricing + rules + outputs
# Robust to optional modules, tuple returns, and rules signature changes.

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Tuple

import pandas as pd


# -----------------------------
# Small utils
# -----------------------------
def _print(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def _as_df(obj: Any, label: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Normalize function results that may be either:
      - DataFrame
      - (DataFrame, meta_dict)
      - None
    Returns (df, meta).
    """
    meta: Dict[str, Any] = {}
    if obj is None:
        return pd.DataFrame(), meta
    if isinstance(obj, pd.DataFrame):
        return obj, meta
    if isinstance(obj, tuple):
        head = obj[0] if len(obj) > 0 else None
        if isinstance(head, pd.DataFrame):
            meta = obj[1] if len(obj) > 1 and isinstance(obj[1], dict) else {}
            return head, meta
    # last resort: empty df and a note
    _print(f"WARNING: {label} returned unexpected type {type(obj).__name__}; using empty DataFrame.")
    return pd.DataFrame(), meta


# --------------------------------
# Optional helpers
# --------------------------------
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
# Rules engine – compatibility
# --------------------------------
from scripts.rules_engine import apply_rules as _apply_rules

def _apply_rules_compat(features: Dict[str, Any],
                        side: str,
                        mu: float,
                        sigma: float) -> Tuple[float, float, str]:
    """
    Call scripts.rules_engine.apply_rules regardless of signature.
    Supports:
      - apply_rules(features)
      - apply_rules(side, mu, sigma, features)
      - apply_rules(side=..., mu=..., sigma=..., features=...)
    """
    try:
        return _apply_rules(features)  # new style
    except TypeError:
        try:
            return _apply_rules(side, mu, sigma, features)  # old positional
        except TypeError:
            return _apply_rules(side=side, mu=mu, sigma=sigma, features=features)  # keyword


# -------------------------------------------------
# Core pipeline
# -------------------------------------------------
def run_pipeline(target_date: str,
                 season: int,
                 out_dir: str = "outputs") -> None:
    _print(f"Loaded engine module from: {os.path.abspath(__file__)}")

    ensure_dir(out_dir)
    ensure_dir("metrics")
    ensure_dir("inputs")

    team_form_path = os.path.join("metrics", "team_form.csv")
    player_form_path = os.path.join("metrics", "player_form.csv")
    id_map_path = os.path.join("metrics", "id_map.csv")

    # 1) Lines
    _print("Fetching game lines …")
    game_raw = fetch_game_lines()
    game_df, _ = _as_df(game_raw, "fetch_game_lines")
    if game_df.empty:
        _print("WARNING: game_df is empty – continuing but props may fail.")

    # 2) Props/odds
    _print("Fetching all props for events …")
    props_raw = fetch_props_all_events()
    props_df, _ = _as_df(props_raw, "fetch_props_all_events")
    if props_df.empty:
        _print("WARNING: props_df is empty. Nothing to price.")
        pd.DataFrame().to_csv(os.path.join(out_dir, "props_priced.csv"), index=False)
        return

    # 3) Pricing
    priced_raw = price_props_for_events(props_df)
    priced_df, priced_meta = _as_df(priced_raw, "price_props_for_events")
    if priced_df.empty:
        _print("WARNING: pricing produced no rows.")
        pd.DataFrame().to_csv(os.path.join(out_dir, "props_priced.csv"), index=False)
        return

    # 4) External features (optional)
    try:
        if _HAS_MERGE_EXTERNAL:
            merged_raw = merge_external_features(priced_df)
            priced_df, _ = _as_df(merged_raw, "merge_external_features")
        else:
            _print("merge_external_features unavailable or failed; using minimal features merge.")
    except Exception as e:
        _print(f"merge_external_features raised {type(e).__name__}: {e}; using minimal features merge.")

    # Info messages to mirror earlier logs
    if not os.path.exists(team_form_path) or os.path.getsize(team_form_path) == 0:
        _print("🛈  team_form.csv is empty.")
    if not os.path.exists(player_form_path) or os.path.getsize(player_form_path) == 0:
        _print("🛈  player_form.csv is empty.")
    if not os.path.exists(id_map_path) or os.path.getsize(id_map_path) == 0:
        _print("🛈  id_map.csv is empty.")

    # 5) Apply rules
    _print("Applying elite rules …")
    if not priced_df.empty:
        adj_mu, adj_sigma, notes = [], [], []
        feature_like_cols = set(priced_df.columns)

        for _, row in priced_df.iterrows():
            base_mu = float(row.get("mu", 0.0))
            base_sigma = float(row.get("sigma", 0.0))
            side = row.get("side", row.get("bet_side", "over"))

            fdict: Dict[str, Any] = {k: row[k] for k in feature_like_cols if k in row}
            fdict.update({"mu": base_mu, "sigma": base_sigma, "side": side})

            mu1, sig1, note = _apply_rules_compat(fdict, side, base_mu, base_sigma)
            adj_mu.append(mu1)
            adj_sigma.append(sig1)
            notes.append(note)

        priced_df["mu"] = adj_mu
        priced_df["sigma"] = adj_sigma
        priced_df["rules_notes"] = notes

    # 6) Save
    _print("Downcasting floats.")
    priced_df = _downcast_numeric(priced_df)

    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "props_priced.csv")
    priced_df.to_csv(out_path, index=False)
    _print(f"Wrote {len(priced_df)} rows to {out_path}")


# -------------------------------------------------
# Dev entry (local runs)
# -------------------------------------------------
if __name__ == "__main__":
    date_arg = sys.argv[1] if len(sys.argv) > 1 else "today"
    season_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 2025
    out_arg = sys.argv[3] if len(sys.argv) > 3 else "outputs"
    run_pipeline(target_date=date_arg, season=season_arg, out_dir=out_arg)
