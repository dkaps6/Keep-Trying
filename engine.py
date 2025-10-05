# engine.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ---- Local modules (import defensively) --------------------------------------
try:
    from scripts import rules_engine as rules
except Exception as e:
    raise RuntimeError(f"Failed to import rules_engine: {e}")

try:
    from scripts import pricing as pr
except Exception as e:
    raise RuntimeError(f"Failed to import pricing: {e}")

# optional fetchers; tolerate absence
try:
    from scripts.odds_api import fetch_props_all_events, fetch_game_lines
except Exception:
    fetch_props_all_events = None
    fetch_game_lines = None


# ---- Small helpers -----------------------------------------------------------
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _read_csv(path: str, **kwargs) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return pd.DataFrame()


def _american_to_decimal(american: Any) -> Optional[float]:
    try:
        a = float(american)
    except Exception:
        return None
    if a > 0:
        return 1.0 + a / 100.0
    if a < 0:
        return 1.0 + 100.0 / abs(a)
    return None


def _decimal_to_prob(decimal_price: Optional[float]) -> Optional[float]:
    if decimal_price and decimal_price > 1.0:
        return 1.0 / decimal_price
    return None


TEAM_ABBR_MAP = {
    "JAX": "JAC",
    "LA": "LAR",
    "WSH": "WAS",
}


def _norm_team(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip().upper()
    return TEAM_ABBR_MAP.get(s, s)


# ---- Rules adapter (new and legacy signatures) -------------------------------
def _apply_rules_compat(row: Dict[str, Any]) -> Tuple[float, float, str]:
    """
    Works with either:
      - new: apply_rules(features: dict) -> (mu, sigma, notes) or dict
      - old: apply_rules(side, mu, sigma, features={..., player=..., team_ctx=...})
    """
    team_ctx = row.get("team", "") or row.get("team_norm", "")
    player = row.get("player", "")

    features = dict(row)
    features["team_ctx"] = team_ctx
    features["player"] = player

    try:
        out = rules.apply_rules(features)  # new style?
        if isinstance(out, dict):
            mu = float(out.get("mu", 0.0))
            sigma = float(out.get("sigma", 1.0))
            notes = str(out.get("notes", ""))
            return mu, sigma, notes
        if isinstance(out, (list, tuple)) and len(out) >= 2:
            mu = float(out[0])
            sigma = float(out[1])
            notes = "" if len(out) < 3 else str(out[2])
            return mu, sigma, notes
    except TypeError:
        # legacy signature
        try:
            mu, sigma, notes = rules.apply_rules(
                features.get("side", "over"),
                features.get("mu", 0.0),
                features.get("sigma", 1.0),
                features=features,
            )
            return float(mu), float(sigma), str(notes)
        except Exception:
            pass
    except Exception:
        pass

    return 0.0, 1.0, "rules_default"


# ---- Pricing adapters --------------------------------------------------------
def _call(fn_name: str, *args, **kwargs):
    fn = getattr(pr, fn_name, None)
    if callable(fn):
        return fn(*args, **kwargs)
    return None


def _model_prob(mu: float, sigma: float, side: str, line: float) -> Optional[float]:
    for name in ("_model_prob", "model_prob", "model_probability"):
        out = _call(name, mu, sigma, side, line)
        if out is not None:
            return float(out)
    try:
        from math import erf, sqrt
        z = (line - mu) / (sigma if sigma > 0 else 1.0)
        phi = 0.5 * (1.0 + erf(-z / sqrt(2.0)))
        return 1.0 - phi if str(side).lower() == "over" else phi
    except Exception:
        return None


def _fair_prob_blend(model_prob: Optional[float], market_prob: Optional[float]) -> Optional[float]:
    for name in ("_blend", "blend_prob", "blend"):
        out = _call(name, model_prob, market_prob)
        if out is not None:
            return float(out)
    if model_prob is not None and market_prob is not None:
        return 0.5 * model_prob + 0.5 * market_prob
    return model_prob if model_prob is not None else market_prob


def _edge(model_prob: Optional[float], market_prob: Optional[float]) -> Optional[float]:
    for name in ("_edge", "edge"):
        out = _call(name, model_prob, market_prob)
        if out is not None:
            return float(out)
    if model_prob is None or market_prob is None:
        return None
    return model_prob - market_prob


def _prob_to_american(prob: Optional[float]) -> Optional[float]:
    for name in ("prob_to_american", "prob2american", "prob_to_price"):
        out = _call(name, prob)
        if out is not None:
            return float(out)
    if prob is None or prob <= 0 or prob >= 1:
        return None
    if prob >= 0.5:
        return -100.0 * prob / (1.0 - prob)
    else:
        return 100.0 * (1.0 - prob) / prob


def _kelly(edge: Optional[float], market_prob: Optional[float]) -> Optional[float]:
    out = _call("kelly", edge, market_prob)
    if out is not None:
        try:
            return float(out)
        except Exception:
            pass
    if edge is None or market_prob is None:
        return None
    return max(0.0, edge) * 0.5


# ---- Core pipeline -----------------------------------------------------------
@dataclass
class PipelineResult:
    priced: pd.DataFrame
    features_source: str
    note: str = ""


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with duplicate column labels removed (keep-first)."""
    if df.empty:
        return df
    return df.loc[:, ~df.columns.duplicated()].copy()


def _load_metrics() -> Dict[str, pd.DataFrame]:
    mets = {
        "team_form": _read_csv("metrics/team_form.csv"),
        "team_week_form": _read_csv("metrics/team_week_form.csv"),
        "player_form": _read_csv("metrics/player_form.csv"),
        "id_map": _read_csv("inputs/id_map.csv"),
        "weather": _read_csv("inputs/weather.csv"),
    }
    # de-duplicate and normalize team columns
    for k in ("team_form", "team_week_form"):
        df = mets[k]
        if df.empty:
            continue
        df = _dedupe_columns(df)
        if "team" in df.columns and "team_norm" not in df.columns:
            df["team_norm"] = df["team"].map(_norm_team)
        elif "team_norm" in df.columns:
            df["team_norm"] = df["team_norm"].map(_norm_team)
        mets[k] = df
    return mets


def _fetch_props(odds_key: Optional[str]) -> pd.DataFrame:
    if fetch_props_all_events is not None:
        try:
            return fetch_props_all_events(api_key=odds_key)
        except TypeError:
            try:
                return fetch_props_all_events(ODDS_API_KEY=odds_key)
            except Exception:
                pass
        except Exception:
            pass

    for p in ("inputs/props.csv", "inputs/props_raw.csv"):
        if os.path.exists(p):
            df = _read_csv(p)
            if not df.empty:
                return df

    return pd.DataFrame(
        columns=[
            "event_id", "player", "team", "market", "line", "price", "side"
        ]
    )


def _normalize_props(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()

    rename = {
        "american_odds": "price",
        "odds_american": "price",
        "odds": "price",
        "participant": "player",
        "book": "bookmaker",
        "team_abbr": "team",
    }
    for a, b in rename.items():
        if a in out.columns and b not in out.columns:
            out = out.rename(columns={a: b})

    if "prob" in out.columns:
        out["market_prob"] = pd.to_numeric(out["prob"], errors="coerce")
    elif "price" in out.columns:
        dec = out["price"].map(_american_to_decimal)
        out["market_prob"] = dec.map(_decimal_to_prob)
    else:
        out["market_prob"] = np.nan

    if "team" in out.columns:
        out["team_norm"] = out["team"].map(_norm_team)
    else:
        out["team_norm"] = ""

    if "side" not in out.columns:
        out["side"] = "over"

    if "line" in out.columns:
        with np.errstate(all="ignore"):
            out["line"] = pd.to_numeric(out["line"], errors="coerce")

    out = out[out["line"].notna()].copy()

    REQUIRED = ["event_id", "player", "market", "line", "price"]
    missing = [c for c in REQUIRED if c not in out.columns]
    if missing:
        out["__schema_missing__"] = ",".join(missing)

    return out


def _merge_features(props: pd.DataFrame, mets: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, str]:
    """Left-join any metrics we have on team. De-dup join key to avoid
    'column label team_norm is not unique' errors."""
    src = []
    out = props.copy()

    for k in ("team_form", "team_week_form"):
        df = mets[k]
        if df.empty:
            continue

        # de-duplicate any repeated headers from source CSVs
        df = _dedupe_columns(df)

        left_key = "team_norm" if "team_norm" in out.columns else "team"
        right_key = "team_norm" if "team_norm" in df.columns else "team"

        # ensure the join key appears exactly once on the right
        base_cols = [c for c in df.columns if c not in ("team", "team_norm")]
        df_sub = df.loc[:, base_cols + [right_key]].copy()

        out = out.merge(
            df_sub,
            how="left",
            left_on=left_key,
            right_on=right_key,
            suffixes=("", f"_{k}"),
        )
        src.append(k)

    return out, ("+".join(src) if src else "none")


def _price_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    records = []
    for _, r in df.iterrows():
        features = r.to_dict()
        mu, sigma, notes = _apply_rules_compat(features)

        model_prob = _model_prob(mu, sigma, str(r.get("side", "over")).lower(), float(r.get("line", 0.0)))
        market_prob = r.get("market_prob", None)
        fair_prob = _fair_prob_blend(model_prob, market_prob)
        edge = _edge(model_prob, market_prob)
        fair_american = _prob_to_american(fair_prob)
        kelly = _kelly(edge, market_prob)

        rec = dict(r)
        rec.update({
            "mu": mu,
            "sigma": sigma,
            "model_prob": model_prob,
            "market_prob": market_prob,
            "fair_prob": fair_prob,
            "edge": edge,
            "fair_american": fair_american,
            "kelly": kelly,
            "rule_notes": notes,
        })
        records.append(rec)

    out = pd.DataFrame.from_records(records)
    return out


# ------------------------------------------------------------------------------
def run_pipeline(
    season: int = 2025,
    date: Optional[str] = None,
    *,
    target_date: Optional[str] = None,   # accepted for CLI compatibility
    write_outputs: bool = True,
    odds_api_key: Optional[str] = None,
    ODDS_API_KEY: Optional[str] = None,
    **kwargs,
) -> PipelineResult:
    """
    Main entry point used by run_model.py.
    """
    odds_key = odds_api_key or ODDS_API_KEY

    # 1) Fetch props (or load inputs/props.csv if fetcher not available)
    props_raw = _fetch_props(odds_key)
    props_norm = _normalize_props(props_raw)

    # 2) Merge external metrics (best-effort)
    mets = _load_metrics()
    feats_df, feats_src = _merge_features(props_norm, mets)

    # 3) Apply rules + pricing
    priced_df = _price_frame(feats_df)

    # 4) Write outputs (optional)
    if write_outputs:
        _ensure_dir("outputs")
        out_path = os.path.join("outputs", "props_priced.csv")
        priced_df.to_csv(out_path, index=False)

    return PipelineResult(priced=priced_df, features_source=feats_src, note="ok")


if __name__ == "__main__":
    res = run_pipeline(write_outputs=True)
    print(f"Wrote {len(res.priced)} rows (features: {res.features_source})")
