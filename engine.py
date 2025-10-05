# engine.py
from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

# --- Odds fetchers (these already exist in your repo) ---
from scripts.odds_api import fetch_game_lines, fetch_props_all_events

# --- Rules (new flexible signature we shipped) ---
from scripts.rules_engine import apply_rules

# --- Pricing helpers (this is the “tiny nudge” import) ---
from scripts.pricing import (
    _model_prob,
    _market_fair_prob,
    _blend,
    prob_to_american,
    _edge,
    kelly_fraction,
)

# Optional external merge (OK if missing)
try:
    from scripts.features_external import merge_external_features  # optional
    HAS_MERGE_EXTERNAL = True
except Exception:
    HAS_MERGE_EXTERNAL = False


# -------------------------
# Utility / safe operations
# -------------------------
def _norm(s: Any) -> str:
    if s is None:
        return ""
    return str(s).strip()

def _safe_float(x: Any, default: float = np.nan) -> float:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return default
        return float(x)
    except Exception:
        return default


# -------------------------------------
# External files (optional, non-fatal)
# -------------------------------------
def build_external(season: int) -> Dict[str, pd.DataFrame]:
    """
    Tries to read optional external CSVs (team_form, player_form, id_map, weather).
    Returns empty/tiny frames if not found so the pipeline keeps running.
    """
    base = Path("metrics")
    inputs = Path("inputs")
    base.mkdir(parents=True, exist_ok=True)
    inputs.mkdir(parents=True, exist_ok=True)

    def _read_csv(p: Path, cols: Optional[List[str]] = None) -> pd.DataFrame:
        if p.exists():
            try:
                df = pd.read_csv(p)
                if cols:
                    for c in cols:
                        if c not in df.columns:
                            df[c] = np.nan
                return df
            except Exception:
                pass
        # minimal empty frame
        return pd.DataFrame(columns=cols or [])

    team_form = _read_csv(
        base / "team_form.csv",
        cols=[
            "team", "week",
            "pressure_z", "pass_epa_z",
            "run_funnel", "pass_funnel"
        ],
    )
    player_form = _read_csv(
        base / "player_form.csv",
        cols=[
            "gsis_id", "week",
            "usage_routes", "usage_targets", "usage_carries",
            "mu_pred", "sigma_pred"
        ],
    )
    id_map = _read_csv(
        inputs / "id_map.csv",
        cols=["player_name", "gsis_id", "recent_team", "position"],
    )
    weather = _read_csv(
        inputs / "weather.csv",
        cols=["game_id", "wind_mph", "temp_f"],
    )

    return dict(
        team_form=team_form,
        player_form=player_form,
        id_map=id_map,
        weather=weather,
    )


# --------------------
# Feature construction
# --------------------
def _sigma_default_for_market(market: str) -> float:
    m = market.lower()
    if "reception" in m and "yard" not in m:
        return 1.8
    if "receiving" in m:
        return 26.0
    if "rushing" in m and "attempt" not in m:
        return 22.0
    if "attempt" in m:
        return 4.0
    if "passing yard" in m:
        return 48.0
    if "passing td" in m:
        return 0.9
    if "rush+rec" in m:
        return 28.0
    return 20.0


def _make_features_row(row: pd.Series, ext: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Build a features dict that the rules engine can use.
    Works fine if external data are empty — everything defaults gracefully.
    """
    market = _norm(row.get("market"))
    side = _norm(row.get("outcome") or row.get("side") or "over")
    point = row.get("point", np.nan)

    # map ids if available (non-fatal)
    id_map = ext.get("id_map", pd.DataFrame())
    player_name = _norm(row.get("player"))
    gsis_id = None
    if not id_map.empty and player_name:
        m = id_map[id_map["player_name"].str.lower() == player_name.lower()]
        if not m.empty:
            gsis_id = m.iloc[0].get("gsis_id")

    # pull a few team/defense features if available
    team_form = ext.get("team_form", pd.DataFrame())
    opp_team = _norm(row.get("away_team" if _norm(row.get("team")) == _norm(row.get("home_team")) else "home_team"))
    tf_row = pd.Series({})
    if not team_form.empty and opp_team:
        tf = team_form[team_form["team"].str.upper() == opp_team.upper()]
        if not tf.empty:
            tf_row = tf.iloc[-1]  # latest

    return dict(
        market=market,
        side=side,
        point=_safe_float(point, np.nan),
        player=player_name,
        player_id=gsis_id,
        team_ctx=dict(
            opp_team=opp_team,
        ),
        # optional metrics (rules will use when present)
        opp_team_pressure_z=_safe_float(tf_row.get("pressure_z"), 0.0),
        opp_pass_epa_z=_safe_float(tf_row.get("pass_epa_z"), 0.0),
        opp_run_funnel=int(_safe_float(tf_row.get("run_funnel"), 0.0) > 0.5),
        opp_pass_funnel=int(_safe_float(tf_row.get("pass_funnel"), 0.0) > 0.5),
    )


def _initial_mu_sigma(row: pd.Series) -> Tuple[float, float]:
    """
    Neutral starting point if no player-form projection:
      - mu starts at the book line (so raw model P≈0.5)
      - sigma is market-based default
    Rules then nudge mu/sigma from features/context.
    """
    market = _norm(row.get("market"))
    mu0 = _safe_float(row.get("point"), 0.0)
    sigma0 = _sigma_default_for_market(market)
    return mu0, sigma0


# --------------
# Main pipeline
# --------------
def run_pipeline(
    target_date: str = "today",
    season: int = 2025,
    out_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    1) Fetch game lines + player props
    2) Optionally merge external features
    3) Apply rules → mu, sigma, notes
    4) Price: model prob, fair odds, edge, Kelly
    5) Write outputs (props_priced.csv) if out_dir provided
    """
    if out_dir is None:
        out_dir = "outputs"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # 1) Fetch lines/props
    print("Fetching game lines …")
    _ = fetch_game_lines()  # not used yet in pricing, but kept for completeness

    print("Fetching all props for events …")
    props_raw = fetch_props_all_events()
    if props_raw is None or len(props_raw) == 0:
        print("No props returned from odds API.")
        props_raw = []

    props_df = pd.DataFrame(props_raw)
    if props_df.empty:
        print("No props available after normalization.")
        priced_df = pd.DataFrame(columns=[
            "event_id","player","market","outcome","point","price",
            "mu","sigma","notes",
            "model_prob","market_fair_prob","blended_prob","fair_odds","edge","kelly_cap_5pct"
        ])
        priced_df.to_csv(Path(out_dir) / "props_priced.csv", index=False)
        return priced_df

    # Ensure canonical columns exist
    for c in ["event_id","player","market","outcome","side","point","price","home_team","away_team","team"]:
        if c not in props_df.columns:
            props_df[c] = np.nan

    # 2) External metrics (optional)
    ext = build_external(season)
    if HAS_MERGE_EXTERNAL:
        try:
            props_df = merge_external_features(props_df, ext)  # non-fatal if it raises, we catch below
        except Exception as e:
            print(f"merge_external_features unavailable or failed: {e}; using minimal features.")
    else:
        print("merge_external_features not present; using minimal features.")

    # 3) Apply rules → mu, sigma, notes
    mu_list, sigma_list, notes_list = [], [], []
    for _, row in props_df.iterrows():
        # baseline
        mu0, sigma0 = _initial_mu_sigma(row)
        features = _make_features_row(row, ext)
        # Allow rules to adjust baseline (new style signature)
        mu1, sig1, note = apply_rules(dict(
            side=features["side"],
            mu=mu0,
            sigma=sigma0,
            player=row.get("player"),
            team_ctx=features.get("team_ctx"),
            market=features.get("market"),
            opp_pressure_z=features.get("opp_team_pressure_z"),
            opp_pass_epa_z=features.get("opp_pass_epa_z"),
            opp_run_funnel=features.get("opp_run_funnel"),
            opp_pass_funnel=features.get("opp_pass_funnel"),
        ))
        mu_list.append(mu1)
        sigma_list.append(sig1)
        notes_list.append(note)

    priced_df = props_df.copy()
    priced_df["mu"] = mu_list
    priced_df["sigma"] = sigma_list
    priced_df["notes"] = notes_list

    # 4) Pricing block (the “nudge” to write meaningful numbers)
    model_p   = priced_df.apply(_model_prob, axis=1)
    market_p  = priced_df.apply(_market_fair_prob, axis=1)
    blend_p   = [_blend(mp, bp, 0.65) for mp, bp in zip(model_p, market_p)]
    fair_odds = [prob_to_american(p) for p in blend_p]
    edge_val  = [_edge(bp, mp) for bp, mp in zip(blend_p, market_p)]
    kelly     = [kelly_fraction(bp, pr) for bp, pr in zip(blend_p, priced_df["price"])]

    priced_df["model_prob"] = model_p
    priced_df["market_fair_prob"] = market_p
    priced_df["blended_prob"] = blend_p
    priced_df["fair_odds"] = fair_odds
    priced_df["edge"] = edge_val
    priced_df["kelly_cap_5pct"] = kelly

    # 5) Write outputs
    out_path = Path(out_dir) / "props_priced.csv"
    priced_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} rows={len(priced_df)}")

    return priced_df


# Allow CLI execution like: python engine.py
if __name__ == "__main__":
    run_pipeline(target_date="today", season=2025, out_dir="outputs")
