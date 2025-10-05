# engine.py
# ---------------------------------------------------------------------
# Core pipeline for: fetch lines/props -> build features -> apply rules
# -> (optionally) merge external features -> compute output CSVs.
#
# Safe with new team_form schema and rules signature variations.
# ---------------------------------------------------------------------

from __future__ import annotations

import os
import math
from typing import Dict, Optional

import pandas as pd
import numpy as np

# ----------------------------
# Internal modules (yours)
# ----------------------------
try:
    from scripts.odds_api import fetch_props_all_events, fetch_game_lines
except Exception:
    def fetch_props_all_events() -> pd.DataFrame:
        return pd.DataFrame()
    def fetch_game_lines() -> pd.DataFrame:
        return pd.DataFrame()

from scripts.rules_engine import apply_rules

# Optional external feature merger (signature may vary)
try:
    from scripts.features_external import merge_external_features as _merge_ext_raw
    def _merge_ext(df: pd.DataFrame, ext: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        try:
            return _merge_ext_raw(df, ext)  # new signature
        except TypeError:
            return _merge_ext_raw(df)       # old signature
except Exception:
    _merge_ext_raw = None
    def _merge_ext(df: pd.DataFrame, ext: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        return df

# Optional pricing helpers
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
    except Exception:
        return pd.DataFrame()

def _load_team_form(path: str = "metrics/team_form.csv") -> pd.DataFrame:
    """
    Normalize to a canonical 'team' (uppercase string) and ensure a few
    numeric columns exist, regardless of upstream schema.
    """
    df = _load_csv_safe(path)
    if df.empty:
        return pd.DataFrame(columns=["team", "plays", "pass_rate", "rush_rate"])

    if "team" not in df.columns:
        for cand in ("posteam", "team_abbr", "team_code"):
            if cand in df.columns:
                df = df.rename(columns={cand: "team"})
                break
    if "team" not in df.columns:
        df["team"] = ""

    df["team"] = df["team"].fillna("").astype(str).str.upper()
    for c in ("plays", "pass_rate", "rush_rate"):
        if c not in df.columns:
            df[c] = 0.0
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df

def _build_external(season: Optional[int]) -> Dict[str, pd.DataFrame]:
    ext: Dict[str, pd.DataFrame] = {}
    ext["team_form"] = _load_team_form("metrics/team_form.csv")
    return ext

def _coerce_team(val) -> str:
    return ("" if pd.isna(val) else str(val)).upper()

def _opp_from_row(row: pd.Series) -> str:
    for k in ("opp", "opponent", "opponent_team", "opponent_abbr", "opp_team"):
        if k in row and pd.notna(row[k]):
            return _coerce_team(row[k])
    for k in ("matchup", "game", "label"):
        if k in row and isinstance(row[k], str) and "@" in row[k]:
            parts = [p.strip().upper() for p in row[k].replace("  ", " ").split("@")]
            if len(parts) == 2:
                if "team" in row and isinstance(row["team"], str):
                    t = _coerce_team(row["team"])
                    if t == parts[0]:
                        return parts[1]  # away vs home
                    if t == parts[1]:
                        return parts[0]
                return parts[1]
    return ""

def _make_features_row(row: pd.Series, ext: Dict[str, pd.DataFrame]) -> pd.Series:
    out = {}
    team_form = ext.get("team_form", pd.DataFrame())
    if not team_form.empty:
        opp = _opp_from_row(row)
        if opp:
            tf = team_form.loc[team_form["team"] == opp]
            if not tf.empty:
                out["opp_pass_rate"] = float(tf["pass_rate"].mean())
                out["opp_rush_rate"] = float(tf["rush_rate"].mean())
                out["opp_plays"]     = float(tf["plays"].mean())
            else:
                out["opp_pass_rate"] = 0.0
                out["opp_rush_rate"] = 0.0
                out["opp_plays"]     = 0.0
        else:
            out["opp_pass_rate"] = 0.0
            out["opp_rush_rate"] = 0.0
            out["opp_plays"]     = 0.0
    else:
        out["opp_pass_rate"] = 0.0
        out["opp_rush_rate"] = 0.0
        out["opp_plays"]     = 0.0

    return pd.Series(out)

def _apply_rules_compat(features: pd.DataFrame):
    """
    Robustly invoke your rules function regardless of signature.
    Order:
      (1) keyword-only:   apply_rules(features=features)
      (2) single-positional: apply_rules(features)
      (3) legacy: apply_rules(side, base_mu, base_sigma, features=features)
    """
    # 1) keyword-only
    try:
        return apply_rules(features=features)
    except TypeError:
        pass
    except Exception as e:
        print("apply_rules(features=...) raised:", e)

    # 2) positional single argument
    try:
        return apply_rules(features)
    except TypeError:
        pass
    except Exception as e:
        print("apply_rules(features) raised:", e)

    # 3) legacy: need side/mu/sigma scaffolding
    side = np.where(features.get("market_side", pd.Series(dtype=float)).fillna(0).values >= 0, 1, -1)
    base_mu = np.zeros(len(features), dtype=float)
    base_sigma = np.ones(len(features), dtype=float)
    return apply_rules(side, base_mu, base_sigma, features=features)

def _price_block(priced_df: pd.DataFrame) -> pd.DataFrame:
    if not _HAS_PRICING or priced_df.empty:
        return priced_df
    if "mu" not in priced_df.columns or "sigma" not in priced_df.columns:
        return priced_df

    try:
        priced_df["model_prob"] = _model_prob(priced_df["mu"], priced_df["sigma"])
    except Exception:
        priced_df["model_prob"] = 0.5

    if {"american", "price"}.intersection(priced_df.columns):
        price_col = "american" if "american" in priced_df.columns else "price"
        try:
            priced_df["market_fair_prob"] = _market_fair_prob(priced_df[price_col])
        except Exception:
            priced_df["market_fair_prob"] = np.nan
    else:
        priced_df["market_fair_prob"] = np.nan

    try:
        priced_df["blend_prob"] = _blend(priced_df["model_prob"], priced_df["market_fair_prob"])
    except Exception:
        priced_df["blend_prob"] = priced_df["model_prob"]

    try:
        priced_df["edge"] = _edge(priced_df["blend_prob"], priced_df.get("market_implied_prob", np.nan))
    except Exception:
        priced_df["edge"] = np.nan

    try:
        priced_df["kelly"] = kelly(priced_df["blend_prob"], priced_df.get("market_implied_prob", np.nan))
    except Exception:
        priced_df["kelly"] = np.nan

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
    print("Loading engine …")
    ext = _build_external(season)

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

    if props_raw.empty:
        print("No props returned; writing empty outputs.")
        priced_df = pd.DataFrame()
        out_dir = out_dir or "outputs"
        os.makedirs(out_dir, exist_ok=True)
        priced_df.to_csv(os.path.join(out_dir, "props_priced.csv"), index=False)
        return priced_df

    print("Applying feature builders …")
    feat_rows = props_raw.apply(lambda r: _make_features_row(r, ext), axis=1)
    features = pd.concat([props_raw.reset_index(drop=True), feat_rows.reset_index(drop=True)], axis=1)

    try:
        features = _merge_ext(features, ext)
    except Exception as e:
        print("External merge skipped (non-fatal):", e)

    print("Applying elite rules …")
    mu, sigma, notes = _apply_rules_compat(features)
    priced_df = features.copy()
    priced_df["mu"] = mu
    priced_df["sigma"] = sigma
    priced_df["notes"] = notes

    print("Pricing / edges …")
    priced_df = _price_block(priced_df)

    out_dir = out_dir or "outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "props_priced.csv")
    priced_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(priced_df):,} rows")

    return priced_df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="today")
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--write", default="outputs")
    args = ap.parse_args()
    run_pipeline(target_date=args.date, season=args.season, out_dir=args.write)
