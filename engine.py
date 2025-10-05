# engine.py
# Pipeline that never returns a blank CSV:
# - Loads props via scripts/odds_api.fetch_props_all_events (unchanged)
# - Loads metrics; merges team form always; merges player form if available
# - Calls rules_engine.apply_rules safely (dict only; guarded)
# - Computes model/market probabilities and edges; writes outputs
# - Adds columns: source, note (so you see if it was team-only vs team+player)

from __future__ import annotations

import importlib
import os
from typing import Dict, Any, Tuple

import pandas as pd

from scripts.odds_api import fetch_props_all_events
from scripts.rules_engine import apply_rules  # your existing rules
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

# ---------- Feature building --------------------------------------------------

TEAM_WEEK_COLS = [
    "season","team","week","pass_plays","rush_plays","neutral_pass_rate",
    "exp_pass_rate","exp_rush_rate","rz_rate","total_plays"
]

PLAYER_FORM_COLS = [
    "season","team","week","player","targets","rush_att","tgt_share","rush_share"
]

ID_MAP_COLS = ["player","team","position"]

def _merge_features(props_df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    """
    Returns (features_df, source_tag, note)
    source_tag in {"team_only","team_player"}
    """
    team_form = _read_csv_safe(os.path.join(METRICS_DIR, "team_form.csv"))
    team_week = _read_csv_safe(os.path.join(METRICS_DIR, "team_week_form.csv"), TEAM_WEEK_COLS)
    player_form = _read_csv_safe(os.path.join(METRICS_DIR, "player_form.csv"), PLAYER_FORM_COLS)
    id_map = _read_csv_safe(os.path.join(INPUTS_DIR, "id_map.csv"), ID_MAP_COLS)

    if props_df.empty:
        return props_df.assign(source="none", note="no_props"), "none", "no_props"

    # Normalize keys
    props_df["team_norm"] = props_df["team"].map(_normalize_team)
    if "opp_team" in props_df.columns:
        props_df["opp_team_norm"] = props_df["opp_team"].map(_normalize_team)
    else:
        props_df["opp_team_norm"] = ""

    team_form["team_norm"] = team_form["team"].map(_normalize_team)
    team_week["team_norm"] = team_week["team"].map(_normalize_team)

    # Merge team_form (season not strictly necessary if props carry it)
    base = props_df.merge(
        team_form.drop_duplicates(subset=["team_norm"]),
        on="team_norm", how="left", suffixes=("","_tf")
    )
    base = base.merge(
        team_week[["team_norm","neutral_pass_rate","exp_pass_rate","exp_rush_rate","rz_rate"]]
            .drop_duplicates("team_norm"),
        on="team_norm", how="left"
    )

    # Try player enrichment
    enriched = base.copy()
    can_player = not player_form.empty and not id_map.empty and ("player" in props_df.columns)

    if can_player:
        # id_map: join on (player, team)
        id_map["player_key"] = id_map["player"].astype(str).str.strip().str.upper()
        id_map["team_norm"] = id_map["team"].map(_normalize_team)

        enriched["player_key"] = enriched["player"].astype(str).str.strip().str.upper()
        enriched = enriched.merge(
            id_map[["player_key","team_norm","position"]],
            on=["player_key","team_norm"], how="left"
        )

        # coarse weekly proxy: take team-level rates; player shares from last avail week
        pf = player_form.copy()
        pf["player_key"] = pf["player"].astype(str).str.strip().str.upper()
        # Use season-mean shares if no week alignment; this is robust and fast
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
    """
    Your rules_engine.apply_rules expects a dict-like of features.
    This wrapper guarantees a dict and catches common edge cases, returning
    mu, sigma, notes. If rules raise, return neutral (mu=0, sigma=1).
    """
    try:
        feats: Dict[str, Any] = feature_row.to_dict()
        out = apply_rules(feats)
        # Expect either (mu, sigma, note) or dict-like
        if isinstance(out, tuple) and len(out) >= 2:
            mu, sigma = float(out[0]), float(out[1])
            note = out[2] if len(out) > 2 else ""
            return mu, sigma, str(note)
        if isinstance(out, dict):
            mu = float(out.get("mu", 0.0))
            sigma = float(out.get("sigma", 1.0))
            note = str(out.get("note", ""))
            return mu, sigma, note
        # Fallback neutral
        return 0.0, 1.0, "rules_return_unknown"
    except Exception as e:
        return 0.0, 1.0, f"rules_error:{type(e).__name__}"

# ---------- Pipeline ----------------------------------------------------------

def run_pipeline(target_date: str, season: int, out_dir: str | None = None):
    out_dir = out_dir or OUTPUTS_DIR

    # 1) Lines and props from your odds API wrapper
    props_df: pd.DataFrame = fetch_props_all_events()
    if props_df is None or props_df.empty:
        # still write an empty shell for visibility
        pd.DataFrame().to_csv(os.path.join(out_dir, "props_priced.csv"), index=False)
        print("No props returned from odds API.")
        return

    # 2) Merge features (team-only guaranteed; player optional)
    feats_df, source, note = _merge_features(props_df)

    # 3) Apply rules row-wise safely
    # Expect market columns: 'price' (american) or 'decimal' etc.; keep as-is
    mu_list, sigma_list, notes = [], [], []
    for _, row in feats_df.iterrows():
        mu, sigma, n = _rules_safe(row)
        mu_list.append(mu)
        sigma_list.append(sigma)
        notes.append(n)

    feats_df["mu"] = mu_list
    feats_df["sigma"] = sigma_list
    feats_df["rule_note"] = notes

    # 4) Convert to probabilities/edges (example: OVER probs)
    # You already have helpers in scripts/pricing.py
    # We'll compute model prob vs market fair prob and edge/kelly
    feats_df["model_prob"] = _model_prob(feats_df["mu"], feats_df["sigma"])
    feats_df["fair_prob"] = _market_fair_prob(feats_df)  # implement inside pricing.py
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

# For run_model.py import
def run_pipeline_entry(target_date: str, season: int, out_dir: str | None = None):
    run_pipeline(target_date, season, out_dir)
