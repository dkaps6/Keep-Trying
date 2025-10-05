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
    # prefer attribute-access over fragile "from x import y"
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
        # keep pipeline alive; upstream sources might be empty
        return pd.DataFrame()


def _american_to_decimal(american: Any) -> Optional[float]:
    """Convert American odds to decimal price; returns None on failure."""
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
    # keep this tiny and non-opinionated; rules/metrics can own the full map
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
    Adapts to either:
      - new style: apply_rules(features: dict) -> (mu, sigma, notes) or dict
      - old style: apply_rules(side, mu, sigma, features={..., player=..., team_ctx=...})
    """
    # Most models use the team of the player (if present)
    team_ctx = row.get("team", "") or row.get("team_norm", "")
    player = row.get("player", "")

    # Prefer a single-features dict
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

    # Fallback: neutral, uninformative
    return 0.0, 1.0, "rules_default"


# ---- Pricing adapters --------------------------------------------------------
def _call(fn_name: str, *args, **kwargs):
    """Call a function in scripts.pricing by name if present, else None."""
    fn = getattr(pr, fn_name, None)
    if callable(fn):
        return fn(*args, **kwargs)
    return None


def _model_prob(mu: float, sigma: float, side: str, line: float) -> Optional[float]:
    # try most common internal helper names
    for name in ("_model_prob", "model_prob", "model_probability"):
        out = _call(name, mu, sigma, side, line)
        if out is not None:
            return float(out)
    # very conservative default if nothing provided
    # assuming normal model: P(X >= line) for OVER
    try:
        from math import erf, sqrt

        z = (line - mu) / (sigma if sigma > 0 else 1.0)
        # Φ(z)
        phi = 0.5 * (1.0 + erf(-z / sqrt(2.0)))
        return 1.0 - phi if str(side).lower() == "over" else phi
    except Exception:
        return None


def _fair_prob_blend(model_prob: Optional[float], market_prob: Optional[float]) -> Optional[float]:
    for name in ("_blend", "blend_prob", "blend"):
        out = _call(name, model_prob, market_prob)
        if out is not None:
            return float(out)
    # default: if both available, 50/50 blend; else whichever exists
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
    # convert to American
    if prob >= 0.5:
        return -100.0 * prob / (1.0 - prob)
    else:
        return 100.0 * (1.0 - prob) / prob


def _kelly(edge: Optional[float], market_prob: Optional[float]) -> Optional[float]:
    # Preferred: pricing.kelly(american_or_prob, market_prob) or similar
    out = _call("kelly", edge, market_prob)
    if out is not None:
        try:
            return float(out)
        except Exception:
            pass
    # simple Kelly fraction on an even unit bet if we only have prob/edge
    if edge is None or market_prob is None:
        return None
    # Kelly ~ edge / variance proxy (avoid negative explosions)
    return max(0.0, edge) * 0.5


# ---- Core pipeline -----------------------------------------------------------
@dataclass
class PipelineResult:
    priced: pd.DataFrame
    features_source: str
    note: str = ""


def _load_metrics() -> Dict[str, pd.DataFrame]:
    mets = {
        "team_form": _read_csv("metrics/team_form.csv"),
        "team_week_form": _read_csv("metrics/team_week_form.csv"),
        "player_form": _read_csv("metrics/player_form.csv"),
        "id_map": _read_csv("inputs/id_map.csv"),
        "weather": _read_csv("inputs/weather.csv"),
    }
    # normalize basic team column if present
    for k in ("team_form", "team_week_form"):
        if not mets[k].empty and "team" in mets[k].columns:
            mets[k]["team_norm"] = mets[k]["team"].map(_norm_team)
    return mets


def _fetch_props(odds_key: Optional[str]) -> pd.DataFrame:
    # 1) live from Odds API if available
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

    # 2) if you’ve stashed a CSV, use it
    for p in ("inputs/props.csv", "inputs/props_raw.csv"):
        if os.path.exists(p):
            df = _read_csv(p)
            if not df.empty:
                return df

    # 3) otherwise empty frame with expected schema
    return pd.DataFrame(
        columns=[
            "event_id", "player", "team", "market", "line", "price",
            "side"  # optional; default to OVER
        ]
    )


def _normalize_props(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    # common column names from Odds API providers
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

    # derive prob from American odds if only price present
    if "prob" not in out.columns and "price" in out.columns:
        dec = out["price"].map(_american_to_decimal)
        out["market_prob"] = dec.map(_decimal_to_prob)
    elif "prob" in out.columns:
        out["market_prob"] = out["prob"].astype(float)

    # team normalization (best-effort)
    if "team" in out.columns:
        out["team_norm"] = out["team"].map(_norm_team)
    else:
        out["team_norm"] = ""

    if "side" not in out.columns:
        out["side"] = "over"

    # line numeric
    if "line" in out.columns:
        with np.errstate(all="ignore"):
            out["line"] = pd.to_numeric(out["line"], errors="coerce")

    # keep only props with a line
    out = out[out["line"].notna()].copy()

    REQUIRED = ["event_id", "player", "market", "line", "price"]
    missing = [c for c in REQUIRED if c not in out.columns]
    if missing:
        # don’t hard-fail; keep pipeline usable but indicate missing schema
        out["__schema_missing__"] = ",".join(missing)

    return out


def _merge_features(props: pd.DataFrame, mets: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, str]:
    """Left-join any metrics we have on team."""
    src = []
    out = props.copy()

    for k in ("team_form", "team_week_form"):
        df = mets[k]
        if df.empty:
            continue
        use_cols = [c for c in df.columns if c not in ("team",)]
        left_key = "team_norm" if "team_norm" in out.columns else "team"
        right_key = "team_norm" if "team_norm" in df.columns else "team"

        out = out.merge(
            df[use_cols + [right_key]],
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

    Parameters
    ----------
    season : int
        Season tag you’re running against.
    date : str | None
        Optional date selector (not required).
    target_date : str | None
        Accepted but unused (compatibility with older run_model.py).
    write_outputs : bool
        Write outputs/props_priced.csv if True.
    odds_api_key / ODDS_API_KEY : str | None
        Odds API key (either spelling is accepted).
    """
    # Make the API key flexible
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


# Convenience for ad-hoc runs
if __name__ == "__main__":
    res = run_pipeline(write_outputs=True)
    print(f"Wrote {len(res.priced)} rows (features: {res.features_source})")
