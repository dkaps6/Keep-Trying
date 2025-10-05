# engine.py
# Drop-in replacement. Provides robust team handling and end-to-end pipeline.

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Tuple, Dict, Any

import pandas as pd
import numpy as np

# ---- Optional imports guarded for safety ------------------------------------
# odds + props
try:
    from scripts.odds_api import fetch_props_all_events, fetch_game_lines
except Exception:
    fetch_props_all_events = None
    fetch_game_lines = None

# rules
try:
    from scripts.rules_engine import apply_rules  # expects a dict-like "features"
except Exception:
    def apply_rules(features: Dict[str, Any]) -> Tuple[float, float, str]:
        # ultra-safe fallback: zero mean, unit variance, and a note
        return 0.0, 1.0, "rules_fallback"

# pricing helpers (import what exists; fallback implemented below if missing)
try:
    from scripts.pricing import (
        _model_prob,       # model mean->prob transform
        _market_fair_prob, # market American -> prob
        _blend,            # blend model + market prob
        prob_to_american,  # prob -> American
        _edge,             # edge % vs fair
        kelly,             # Kelly sizing
    )
except Exception:
    _model_prob = None
    _market_fair_prob = None
    _blend = None
    prob_to_american = None
    _edge = None
    kelly = None

# -----------------------------------------------------------------------------
#                            Helpers & Normalization
# -----------------------------------------------------------------------------

_TEAM_ALIAS = {
    "ARIZONA": "ARI", "CARDINALS": "ARI", "ARI": "ARI",
    "ATLANTA": "ATL", "FALCONS": "ATL", "ATL": "ATL",
    "BALTIMORE": "BAL", "RAVENS": "BAL", "BAL": "BAL",
    "BUFFALO": "BUF", "BILLS": "BUF", "BUF": "BUF",
    "CAROLINA": "CAR", "PANTHERS": "CAR", "CAR": "CAR",
    "CHICAGO": "CHI", "BEARS": "CHI", "CHI": "CHI",
    "CINCINNATI": "CIN", "BENGALS": "CIN", "CIN": "CIN",
    "CLEVELAND": "CLE", "BROWNS": "CLE", "CLE": "CLE",
    "DALLAS": "DAL", "COWBOYS": "DAL", "DAL": "DAL",
    "DENVER": "DEN", "BRONCOS": "DEN", "DEN": "DEN",
    "DETROIT": "DET", "LIONS": "DET", "DET": "DET",
    "GREEN BAY": "GB", "GREENBAY": "GB", "PACKERS": "GB", "GB": "GB",
    "HOUSTON": "HOU", "TEXANS": "HOU", "HOU": "HOU",
    "INDIANAPOLIS": "IND", "COLTS": "IND", "IND": "IND",
    "JACKSONVILLE": "JAX", "JACKSONVILE": "JAX", "JAGUARS": "JAX", "JAGS": "JAX", "JAX": "JAX",
    "KANSAS CITY": "KC", "KANSASCITY": "KC", "CHIEFS": "KC", "KC": "KC",
    "LAS VEGAS": "LV", "LASVEGAS": "LV", "RAIDERS": "LV", "LV": "LV",
    "LOS ANGELES RAMS": "LAR", "LA RAMS": "LAR", "RAMS": "LAR", "LAR": "LAR",
    "LOS ANGELES CHARGERS": "LAC", "LA CHARGERS": "LAC", "CHARGERS": "LAC", "LAC": "LAC",
    "MIAMI": "MIA", "DOLPHINS": "MIA", "MIA": "MIA",
    "MINNESOTA": "MIN", "VIKINGS": "MIN", "MIN": "MIN",
    "NEW ENGLAND": "NE", "PATRIOTS": "NE", "NE": "NE",
    "NEW ORLEANS": "NO", "SAINTS": "NO", "NO": "NO",
    "NEW YORK GIANTS": "NYG", "GIANTS": "NYG", "NYG": "NYG",
    "NEW YORK JETS": "NYJ", "JETS": "NYJ", "NYJ": "NYJ",
    "PHILADELPHIA": "PHI", "EAGLES": "PHI", "PHI": "PHI",
    "PITTSBURGH": "PIT", "STEELERS": "PIT", "PIT": "PIT",
    "SAN FRANCISCO": "SF", "SANFRANCISCO": "SF", "49ERS": "SF", "NINERS": "SF", "SF": "SF",
    "SEATTLE": "SEA", "SEAHAWKS": "SEA", "SEA": "SEA",
    "TAMPA BAY": "TB", "TAMPABAY": "TB", "BUCCANEERS": "TB", "BUCS": "TB", "TB": "TB",
    "TENNESSEE": "TEN", "TITANS": "TEN", "TEN": "TEN",
    "WASHINGTON": "WAS", "WASHINGTON COMMANDERS": "WAS", "COMMANDERS": "WAS", "WSH": "WAS", "WAS": "WAS",
}

_VALID_ABBRS = {
    "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU","IND","JAX","KC","LV",
    "LAR","LAC","MIA","MIN","NE","NO","NYG","NYJ","PHI","PIT","SF","SEA","TB","TEN","WAS"
}

def _normalize_team(val: object) -> str:
    if val is None:
        return "UNK"
    x = str(val).strip().upper()
    if x in _TEAM_ALIAS:
        return _TEAM_ALIAS[x]
    x2 = x.replace(".", "").replace("-", " ").replace("_", " ")
    x2 = " ".join(x2.split())
    if x in _VALID_ABBRS:
        return x
    return _TEAM_ALIAS.get(x2, "UNK")

def _first_present_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _read_csv(path: str | Path) -> pd.DataFrame:
    try:
        if Path(path).exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()

# -----------------------------------------------------------------------------
#                                Merging
# -----------------------------------------------------------------------------

def _merge_features(props_df: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    """
    Ensure a reliable 'team' and 'team_norm' exist in props_df.
    If team-like column is missing, try to infer from id_map via 'player'.
    Return (df_augmented, source, note).
    """
    note_bits: list[str] = []
    df = props_df.copy()

    team_col = _first_present_col(
        df, ["team", "player_team", "team_abbr", "team_name", "Team", "club", "franchise"]
    )

    if team_col is None:
        id_map = _read_csv("inputs/id_map.csv")
        if "player" in df.columns and {"player", "team"} <= set(id_map.columns):
            df = df.merge(id_map[["player", "team"]], on="player", how="left", suffixes=("", "_from_idmap"))
            team_col = "team"
            note_bits.append("team_from_id_map")
        else:
            note_bits.append("team_missing")

    if team_col is None:
        df["team"] = "UNK"
        team_col = "team"
    elif team_col != "team":
        df["team"] = df[team_col]

    df["team_norm"] = df["team"].astype(str).map(_normalize_team)

    source = "basic"
    note = "; ".join(note_bits) if note_bits else ""
    return df, source, note

# -----------------------------------------------------------------------------
#                    Rules compatibility & feature construction
# -----------------------------------------------------------------------------

def _apply_rules_compat(features: Dict[str, Any]) -> Tuple[float, float, str]:
    """
    Keeps your rules_engine flexible. Expects a dict of features.
    Returns (mu, sigma, notes).
    """
    try:
        mu, sigma, notes = apply_rules(features)
        return float(mu), float(sigma), str(notes)
    except TypeError:
        # if someone changed rules signature, handle gracefully
        out = apply_rules(features)  # any 2- or 3-tuple variant
        if isinstance(out, tuple) and len(out) == 2:
            mu, sigma = out
            return float(mu), float(sigma), ""
        elif isinstance(out, tuple) and len(out) >= 3:
            mu, sigma, notes = out[0], out[1], out[2]
            return float(mu), float(sigma), str(notes)
        return 0.0, 1.0, "rules_signature_unexpected"

def _make_features_row(row: pd.Series, team_form: pd.DataFrame, player_form: pd.DataFrame) -> Dict[str, Any]:
    """
    Build the dict of features your rules expect. Extend as needed.
    """
    d: Dict[str, Any] = row.to_dict()

    # Attach team aggregate features if available
    if not team_form.empty and {"team_norm"} <= set(team_form.columns):
        tf = team_form.loc[team_form["team_norm"] == row.get("team_norm")].head(1)
        if not tf.empty:
            for c in tf.columns:
                if c not in {"team", "team_norm"}:
                    d[f"team_{c}"] = tf.iloc[0][c]

    # Attach player-level features if available (match on player)
    if not player_form.empty and {"player"} <= set(player_form.columns) and "player" in row:
        pf = player_form.loc[player_form["player"] == row["player"]].head(1)
        if not pf.empty:
            for c in pf.columns:
                if c not in {"player"}:
                    d[f"player_{c}"] = pf.iloc[0][c]

    return d

# -----------------------------------------------------------------------------
#                      Pricing fallbacks (if pricing.py missing)
# -----------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + float(np.math.erf(x / np.sqrt(2.0))))

def _fallback_model_prob(mu: float, sigma: float) -> float:
    # z=0 threshold => prob(mu>0). Tweak as your threshold dictates.
    z = (mu - 0.0) / max(sigma, 1e-6)
    return float(_norm_cdf(z))

def _fallback_edge(model_prob: float, market_prob: float) -> float:
    # simple difference
    return float(model_prob - market_prob) if market_prob is not None else float(model_prob)

def _prob_to_american(p: float) -> float:
    p = max(min(p, 0.999999), 1e-6)
    if p >= 0.5:
        return -100.0 * p / (1.0 - p)
    return 100.0 * (1.0 - p) / p

def _fallback_kelly(edge: float, price_prob: float) -> float:
    # Standard Kelly on probability edge, clipped
    if price_prob is None:
        return 0.0
    b = (1.0 / max(price_prob, 1e-6)) - 1.0
    f = (b * edge) / max(b, 1e-6)
    return float(np.clip(f, -1.0, 1.0))

# -----------------------------------------------------------------------------
#                               Pricing wrapper
# -----------------------------------------------------------------------------

def _price_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given df with mu/sigma and market American odds in column 'odds',
    compute probabilities, fair odds, edge, Kelly, etc.
    """
    out = df.copy()

    # Model probability
    if _model_prob is not None:
        out["model_prob"] = out.apply(lambda r: _model_prob(r["mu"], r["sigma"]), axis=1)
    else:
        out["model_prob"] = out.apply(lambda r: _fallback_model_prob(r["mu"], r["sigma"]), axis=1)

    # Market prob (if odds present)
    if "odds" in out.columns and out["odds"].notna().any():
        if _market_fair_prob is not None:
            out["market_prob"] = out["odds"].apply(lambda o: _market_fair_prob(o))
        else:
            # American -> implied prob
            def _imp_prob(o):
                try:
                    o = float(o)
                except Exception:
                    return np.nan
                if o < 0:
                    return abs(o) / (abs(o) + 100.0)
                return 100.0 / (o + 100.0)
            out["market_prob"] = out["odds"].apply(_imp_prob)
    else:
        out["market_prob"] = np.nan

    # Blend (optional)
    if _blend is not None:
        out["fair_prob"] = out.apply(lambda r: _blend(r["model_prob"], r.get("market_prob", np.nan)), axis=1)
    else:
        out["fair_prob"] = out["model_prob"].copy()

    # Fair American
    if prob_to_american is not None:
        out["fair_american"] = out["fair_prob"].apply(prob_to_american)
    else:
        out["fair_american"] = out["fair_prob"].apply(_prob_to_american)

    # Edge
    if _edge is not None:
        out["edge"] = out.apply(lambda r: _edge(r["fair_prob"], r.get("market_prob", np.nan)), axis=1)
    else:
        out["edge"] = out.apply(lambda r: _fallback_edge(r["fair_prob"], r.get("market_prob", np.nan)), axis=1)

    # Kelly
    if kelly is not None:
        out["kelly"] = out.apply(lambda r: kelly(r["edge"], r.get("market_prob", np.nan)), axis=1)
    else:
        out["kelly"] = out.apply(lambda r: _fallback_kelly(r["edge"], r.get("market_prob", np.nan)), axis=1)

    return out

# -----------------------------------------------------------------------------
#                                Pipeline
# -----------------------------------------------------------------------------

def _load_external_metrics() -> dict[str, pd.DataFrame]:
    team_form = _read_csv("metrics/team_form.csv")
    team_week_form = _read_csv("metrics/team_week_form.csv")
    player_form = _read_csv("metrics/player_form.csv")

    # normalize keys where helpful
    if not team_form.empty:
        if "team" in team_form.columns and "team_norm" not in team_form.columns:
            team_form["team_norm"] = team_form["team"].astype(str).map(_normalize_team)
    if not team_week_form.empty:
        if "team" in team_week_form.columns and "team_norm" not in team_week_form.columns:
            team_week_form["team_norm"] = team_week_form["team"].astype(str).map(_normalize_team)

    return {
        "team_form": team_form,
        "team_week_form": team_week_form,
        "player_form": player_form,
    }

def _safe_fetch_props() -> pd.DataFrame:
    if fetch_props_all_events is None:
        return pd.DataFrame()
    try:
        return fetch_props_all_events()
    except Exception:
        return pd.DataFrame()

def run_pipeline(target_date: str = "today", season: int = 2025, out_dir: bool = True) -> pd.DataFrame:
    """
    End-to-end: fetch props, add robust team fields, build features,
    run rules, price, and write outputs/props_priced.csv
    """
    print(f"Loaded engine module from: {__file__}")
    print("Loading engine …")

    # 1) Fetch props (or empty frame if fetcher not available)
    props_raw = _safe_fetch_props()
    if props_raw.empty:
        print("⚠ props fetch returned empty; proceeding with empty frame.")
        props_raw = pd.DataFrame(columns=["player", "market", "line", "odds", "team"])

    # 2) Robust team fields
    props_aug, source, note = _merge_features(props_raw)

    # 3) External metrics (optional)
    ext = _load_external_metrics()
    team_form = ext["team_form"]
    player_form = ext["player_form"]

    # 4) Build features row-by-row and apply rules
    rows = []
    for _, r in props_aug.iterrows():
        features = _make_features_row(r, team_form=team_form, player_form=player_form)
        mu, sigma, notes = _apply_rules_compat(features)
        # accumulate
        rr = dict(r)
        rr["mu"] = mu
        rr["sigma"] = sigma
        rr["rule_notes"] = notes
        rows.append(rr)

    priced_df = pd.DataFrame(rows)

    # 5) Price the props (probabilities, fair, edge, kelly)
    if not priced_df.empty and {"mu", "sigma"} <= set(priced_df.columns):
        priced_df = _price_frame(priced_df)
    else:
        # ensure columns exist
        for col in ["model_prob", "market_prob", "fair_prob", "fair_american", "edge", "kelly"]:
            if col not in priced_df.columns:
                priced_df[col] = np.nan

    # 6) Write outputs
    Path("outputs").mkdir(parents=True, exist_ok=True)
    out_path = Path("outputs/props_priced.csv")
    priced_df.to_csv(out_path, index=False)
    print(f"Wrote {len(priced_df):,} rows to {out_path}")

    return priced_df


if __name__ == "__main__":
    # quick local smoke-test
    run_pipeline(target_date="today", season=2025, out_dir=True)
