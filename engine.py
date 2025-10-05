# engine.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

def _try_import(mod: str):
    try:
        return __import__(mod, fromlist=["*"])
    except Exception:
        return None

_pricing = _try_import("scripts.pricing")
_model_core = _try_import("scripts.model_core")
_rules_engine = _try_import("scripts.rules_engine")
_fetch_all = _try_import("scripts.fetch_all")

from scripts.robust_merges import safe_merge_id_map, safe_merge_weather
from scripts.pipeline_guards import assert_preprice_ready, write_xlsx_if_nonempty

# ---------- generic helpers & fallbacks ----------

def _cdf_over_at_line(mu: float, sigma: float, line: float) -> float:
    try:
        from math import erf, sqrt
        if sigma <= 0:
            return float(mu > line)
        z = (line - mu) / max(sigma, 1e-9)
        return float(max(0.0, min(1.0, 0.5 * (1.0 + erf(-z / sqrt(2))))))
    except Exception:
        return 0.5

def _american_to_prob(american: Optional[float]) -> Optional[float]:
    if american is None or (isinstance(american, float) and np.isnan(american)):
        return None
    try:
        a = float(american)
    except Exception:
        return None
    if a == 0:
        return None
    if a > 0:
        return 100.0 / (a + 100.0)
    return (-a) / ((-a) + 100.0)

def _prob_to_american(p: float) -> int:
    p = float(max(1e-6, min(1 - 1e-6, p)))
    if p >= 0.5:
        return int(round(-100.0 * p / (1 - p)))
    return int(round(100.0 * (1 - p) / p))

def _blend_65_35(model_prob: Optional[float], market_prob: Optional[float], w: float = 0.65) -> Optional[float]:
    if model_prob is None and market_prob is None:
        return None
    if model_prob is None:
        return float(max(0.0, min(1.0, market_prob)))
    if market_prob is None:
        return float(max(0.0, min(1.0, model_prob)))
    mp = float(max(0.0, min(1.0, model_prob)))
    bp = float(max(0.0, min(1.0, market_prob)))
    w = float(w)
    return w * mp + (1.0 - w) * bp

def _kelly(prob: float, american_odds: Optional[float], cap: float = 0.05) -> float:
    if american_odds is None or (isinstance(american_odds, float) and np.isnan(american_odds)):
        return 0.0
    p = float(max(1e-6, min(1 - 1e-6, prob)))
    a = float(american_odds)
    b = (a / 100.0) if a > 0 else (100.0 / -a)
    q = 1.0 - p
    k = ((b * p) - q) / b
    return float(max(0.0, min(cap, k)))

def _get_func(mod, names: list[str], default: Optional[Callable] = None) -> Optional[Callable]:
    if not mod:
        return default
    for n in names:
        f = getattr(mod, n, None)
        if callable(f):
            return f
    return default

_MODEL_PROB = _get_func(_pricing, ["_model_prob"], None)
_MARKET_FAIR_PROB = _get_func(_pricing, ["_market_fair_prob", "market_prob"], None)
_BLEND = _get_func(_pricing, ["_blend", "blend_prob", "blend"], _blend_65_35)
_EDGE = _get_func(_pricing, ["_edge", "edge"], lambda p_blend, p_mkt: (None if p_blend is None or p_mkt is None else (p_blend - p_mkt)))
_PROB_TO_AMERICAN = _get_func(_pricing, ["prob_to_american"], _prob_to_american)
_KELLY = _get_func(_pricing, ["kelly", "kelly_fraction"], _kelly)
_COMPUTE_MU_SIGMA = _get_func(_model_core, ["compute_mu_sigma"], None)
_APPLY_RULES = _get_func(_rules_engine, ["apply_rules"], None)

# ---------- small I/O helpers ----------

def _read_csv(path: Path) -> pd.DataFrame:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception as e:
        print(f"[warn] failed reading {path}: {e}")
    return pd.DataFrame()

def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[info] wrote {path} ({len(df)} rows)")

# ---------- internal compute helpers ----------

def _compute_model_prob(row) -> float:
    mu = float(row.get("model_mu", row.get("mu", row.get("baseline_mu", row.get("line", 0.0)))))
    sigma = float(row.get("model_sigma", row.get("sigma", 0.0)) or 0.0)
    line = float(row.get("line", 0.0))
    if _MODEL_PROB:
        try:
            return float(max(0.0, min(1.0, _MODEL_PROB(mu, sigma, line))))
        except Exception:
            pass
    return _cdf_over_at_line(mu, sigma, line)

def _compute_market_prob(row) -> Optional[float]:
    if _MARKET_FAIR_PROB:
        try:
            v = _MARKET_FAIR_PROB(row)
            return None if v is None else float(max(0.0, min(1.0, v)))
        except Exception:
            pass
    price = row.get("price") or row.get("odds_over") or row.get("american_odds")
    return _american_to_prob(price)

def _apply_rules_rowwise(df: pd.DataFrame) -> pd.DataFrame:
    if not _APPLY_RULES:
        return df
    out = df.copy()
    mus, sigs, notes = [], [], []
    for _, r in out.iterrows():
        try:
            res = _APPLY_RULES(r)
            if isinstance(res, tuple) and len(res) >= 2:
                mu, sg = float(res[0]), float(res[1])
                note = res[2] if len(res) > 2 else ""
            else:
                mu, sg, note = r.get("model_mu", r.get("line")), r.get("model_sigma", 0.0), ""
        except Exception as e:
            mu, sg, note = r.get("model_mu", r.get("line")), r.get("model_sigma", 0.0), f"rules_error:{e}"
        mus.append(mu); sigs.append(sg); notes.append(note)
    out["model_mu"] = mus
    out["model_sigma"] = sigs
    if "notes" in out.columns:
        out["notes"] = out["notes"].fillna("").astype(str) + np.where(pd.Series(notes).astype(str) != "", " | " + pd.Series(notes).astype(str), "")
    else:
        out["notes"] = notes
    return out

# ---------- PUBLIC ENTRYPOINT ----------

def run_pipeline(date: str = "today", season: str = "2025", write: str = "outputs", **kwargs) -> dict:
    """
    Orchestrates: fetch/assemble -> features -> rules -> price -> outputs.

    Accepts legacy/alias kwargs so older callers don't crash:
      - target_date / slate_date -> date
      - season_year            -> season
      - out_dir / output       -> write
      (any other kwargs are ignored)
    """
    # ---- alias handling ----
    date = kwargs.get("target_date", kwargs.get("slate_date", date))
    season = kwargs.get("season_year", season)
    write = kwargs.get("out_dir", kwargs.get("output", write))

    root = Path(".").resolve()
    out_dir = root / write
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 1) FETCH / ASSEMBLE PROPS
    props = pd.DataFrame()
    try:
        if _fetch_all and hasattr(_fetch_all, "build_props_frame"):
            props = _fetch_all.build_props_frame(date=date, season=season)
        elif _fetch_all and hasattr(_fetch_all, "main"):
            props = _fetch_all.main(date=date, season=season) or pd.DataFrame()
    except Exception as e:
        print(f"[warn] fetch_all failed: {e}")

    if props.empty:
        for candidate in [
            root / "outputs" / "props_raw.csv",
            root / "data" / "odds_sample.csv",
            root / "inputs" / "props.csv",
        ]:
            props = _read_csv(candidate)
            if not props.empty:
                print(f"[info] props loaded from {candidate}")
                break

    if props.empty:
        print("[error] No props available to price (fetch produced nothing and no local CSVs found).")
        _write_csv(pd.DataFrame(), out_dir / "props_priced.csv")
        return {"props": 0, "priced": 0}

    # 2) FEATURES / CONTEXT
    id_map = _read_csv(root / "inputs" / "player_id_cache.csv")
    team_week = _read_csv(root / "metrics" / "team_week_form.csv")
    player_form = _read_csv(root / "metrics" / "player_form.csv")
    weather = _read_csv(root / "metrics" / "weather.csv")

    props = safe_merge_id_map(props, id_map)

    if not team_week.empty:
        keys = [k for k in ("team", "opp", "week") if k in props.columns and k in team_week.columns]
        if keys:
            props = props.merge(team_week, on=keys, how="left")
    if not player_form.empty:
        keys = [k for k in ("player", "team", "week") if k in props.columns and k in player_form.columns]
        if keys:
            props = props.merge(player_form, on=keys, how="left")
    props = safe_merge_weather(props, weather, key_cols=("game_id",))

    # 3) MODEL μ/σ
    if _COMPUTE_MU_SIGMA:
        try:
            props = _COMPUTE_MU_SIGMA(props)
        except Exception as e:
            print(f"[warn] compute_mu_sigma failed; falling back to neutral line-based μ: {e}")
            props["model_mu"] = props.get("line")
            props["model_sigma"] = props.get("model_sigma").fillna(0.0) if "model_sigma" in props else 0.0
    else:
        props["model_mu"] = props.get("line")
        props["model_sigma"] = props.get("model_sigma").fillna(0.0) if "model_sigma" in props else 0.0

    props = _apply_rules_rowwise(props)

    # 4) PRICE
    preprice = props.copy()
    assert_preprice_ready(preprice)

    if "price" not in preprice.columns and "odds_over" in preprice.columns:
        preprice = preprice.rename(columns={"odds_over": "price"})

    model_probs, market_probs, blend_probs, fair_odds, edges, kellys = [], [], [], [], [], []

    for _, row in preprice.iterrows():
        p_model = _compute_model_prob(row)
        p_market = _compute_market_prob(row)
        p_blend = _BLEND(p_model, p_market)
        fair = _PROB_TO_AMERICAN(p_blend if p_blend is not None else p_model)
        ed = _EDGE(p_blend if p_blend is not None else p_model,
                   p_market if p_market is not None else p_model)
        price = row.get("price", None)
        k = _KELLY(p_blend if p_blend is not None else p_model, price)

        model_probs.append(p_model)
        market_probs.append(p_market)
        blend_probs.append(p_blend)
        fair_odds.append(fair)
        edges.append(ed)
        kellys.append(k)

    priced = preprice.copy()
    priced["model_prob"] = model_probs
    priced["market_prob"] = market_probs
    priced["blend_prob"] = blend_probs
    priced["fair_odds"] = fair_odds
    priced["edge"] = edges
    priced["kelly"] = kellys

    def _tier(e):
        if e is None:
            return "N/A"
        if e >= 0.06:
            return "ELITE"
        if e >= 0.04:
            return "GREEN"
        if e >= 0.01:
            return "AMBER"
        return "RED"

    priced["tier"] = [ _tier(e) for e in edges ]

    _write_csv(priced, (Path(write) if os.path.isabs(write) else Path(".") / write) / "props_priced.csv")
    write_xlsx_if_nonempty(priced, str((Path(write) if os.path.isabs(write) else Path(".") / write) / "props_priced.xlsx"))

    games_csv = Path("outputs") / "game_lines.csv"
    if not games_csv.exists():
        games = priced[["game_id", "team"]].dropna().drop_duplicates() if {"game_id", "team"}.issubset(priced.columns) else pd.DataFrame({"game_id": [], "team": []})
        _write_csv(games, games_csv)

    return {"props": int(len(props)), "priced": int(len(priced))}
