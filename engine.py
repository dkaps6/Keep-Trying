# engine.py
# Orchestrates fetching sportsbook props, normalizing, pricing, and writing outputs.
# - No team filter by default (fetch ALL teams in the window)
# - Robust imports for odds fetcher and normalizer
# - Supports optional specific event IDs via --events

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import importlib
import inspect


# ---------- Helpers to parse inputs ----------

def _to_list_if_csv(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    if isinstance(s, list):
        return s
    s = str(s).strip()
    if not s or s.lower() in {"all", "none", "null"}:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]

def _coerce_args(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(kwargs)
    out.setdefault("season", None)
    out.setdefault("date", "today")
    out.setdefault("window", "168h")
    out.setdefault("cap", 0)
    out.setdefault("markets", None)
    out.setdefault("books", "dk")
    out.setdefault("order", "odds")
    out.setdefault("teams", None)       # default: no team filter
    out.setdefault("selection", None)   # optional selection filter
    out.setdefault("events", None)      # optional event IDs filter
    out.setdefault("write_dir", "outputs")
    out.setdefault("basename", None)
    return out

def _log(msg: str) -> None:
    print(f"[engine] {msg}")


# ---------- Robust import of odds fetcher ----------

def _import_odds_fetcher():
    err = None
    try:
        from scripts.odds_api import get_props as fn  # type: ignore
        return fn
    except Exception as e:
        err = e
    try:
        from scripts.odds_api import fetch_props as fn  # type: ignore
        return fn
    except Exception as e:
        err = e
    try:
        from scripts.odds_api import get_props_df as fn  # type: ignore
        return fn
    except Exception as e:
        err = e
    # last resort: scan module
    try:
        mod = importlib.import_module("scripts.odds_api")
        for name in ("get_props", "fetch_props", "get_props_df"):
            fn = getattr(mod, name, None)
            if callable(fn):
                return fn
    except Exception as e:
        err = e
    raise ImportError(
        "Could not import a props fetcher from scripts.odds_api "
        "(tried get_props, fetch_props, get_props_df)"
    ) from err

def _call_odds_fetcher(fn, **kwargs):
    """
    Call the fetcher with only parameters it accepts, mapping common aliases.
    Supports event ID filters via any of: events, event_ids, ids.
    """
    sig = inspect.signature(fn)
    params = {}

    canonical = {
        "date": kwargs.get("date"),
        "season": kwargs.get("season"),
        "window": kwargs.get("window"),
        "cap": kwargs.get("cap"),
        "markets": kwargs.get("markets"),
        "order": kwargs.get("order"),
        "books": kwargs.get("books"),
        "team_filter": kwargs.get("team_filter"),
        "selection_filter": kwargs.get("selection"),
        "event_ids": kwargs.get("event_ids"),   # already list[str]
    }

    for k, v in canonical.items():
        if k in sig.parameters:
            params[k] = v

    # map team_filter <-> teams
    if "team_filter" not in sig.parameters and "teams" in sig.parameters:
        params["teams"] = canonical["team_filter"]
    if "teams" not in sig.parameters and "team_filter" in sig.parameters:
        params["team_filter"] = canonical["team_filter"]

    # map selection_filter <-> selection
    if "selection_filter" not in sig.parameters and "selection" in sig.parameters:
        params["selection"] = canonical["selection_filter"]
    if "selection" not in sig.parameters and "selection_filter" in sig.parameters:
        params["selection_filter"] = canonical["selection_filter"]

    # map event_ids to whatever the fetcher expects
    if "event_ids" not in sig.parameters:
        for alt in ("events", "ids"):
            if alt in sig.parameters and "event_ids" in canonical and canonical["event_ids"] is not None:
                params[alt] = canonical["event_ids"]

    return fn(**params)


# ---------- Robust import of normalizer ----------

def _import_normalizer():
    err = None
    try:
        from scripts.normalize_props import normalize_props as fn  # type: ignore
        return fn
    except Exception as e:
        err = e
    try:
        from scripts.normalize_props import normalize as fn  # type: ignore
        return fn
    except Exception as e:
        err = e
    try:
        from scripts.normalize_props import normalize_df as fn  # type: ignore
        return fn
    except Exception as e:
        err = e
    try:
        from scripts.normalize_props import to_model_schema as fn  # type: ignore
        return fn
    except Exception as e:
        err = e

    # scan module
    try:
        mod = importlib.import_module("scripts.normalize_props")
        for name in ("normalize_props", "normalize", "normalize_df", "to_model_schema"):
            fn = getattr(mod, name, None)
            if callable(fn):
                return fn
    except Exception as e:
        err = e

    raise ImportError(
        "Could not import a normalizer from scripts.normalize_props "
        "(tried normalize_props, normalize, normalize_df, to_model_schema)"
    ) from err


# ---------- Pricing & writer ----------

from scripts.pricing import price_props, write_outputs


# ---------- Main pipeline ----------

def run_pipeline(**kwargs) -> int:
    args = _coerce_args(kwargs)

    season: Optional[int] = args["season"]
    date: str = args["date"]
    window: str = args["window"]
    cap: int = int(args["cap"])
    books: str = str(args["books"])
    order: str = str(args["order"])
    selection: Optional[str] = args["selection"]
    write_dir: str = str(args["write_dir"])
    basename: Optional[str] = args["basename"]

    # markets
    markets = args["markets"]
    if isinstance(markets, str) and markets.strip():
        markets = [m.strip() for m in markets.split(",") if m.strip()]
    elif not markets:
        markets = None

    # team filter (default: None)
    team_filter = _to_list_if_csv(args.get("teams"))
    if team_filter is None:
        _log("team filter: None (ALL teams in the date window)")
    else:
        _log(f"team filter: {team_filter}")

    # event IDs (optional)
    event_ids = _to_list_if_csv(args.get("events"))
    if event_ids:
        _log(f"event IDs filter: {event_ids}")

    if not basename:
        basename = f"props_priced_{date}"

    try:
        _log(f"fetching props… date={date} season={season}")
        _log(f"window={window} cap={cap}")
        _log(f"markets={','.join(markets) if markets else 'default'} order={order} books={books}")

        fetch_fn = _import_odds_fetcher()
        norm_fn  = _import_normalizer()

        # 1) Fetch
        df_props = _call_odds_fetcher(
            fetch_fn,
            date=date,
            season=season,
            window=window,
            cap=cap,
            markets=markets,
            order=order,
            books=books,
            team_filter=team_filter,
            selection=selection,
            event_ids=event_ids,
        )

        if df_props is None or (isinstance(df_props, (list, tuple)) and len(df_props) == 0):
            _log("No props available to price (props fetch returned 0 rows).")
            return 1
        if not isinstance(df_props, pd.DataFrame):
            df_props = pd.DataFrame(df_props)
        if df_props.empty:
            _log("No props available to price (props fetch returned empty DataFrame).")
            return 1

        _log(f"fetched {len(df_props)} raw props from odds_api")

        # 2) Normalize
        df_norm = norm_fn(df_props)
        if df_norm is None or (hasattr(df_norm, "empty") and df_norm.empty):
            _log("Normalization produced 0 rows. Check provider mapping / markets.")
            return 1
        _log(f"normalized {len(df_norm)} rows")

        # 3) Price
        df_priced = price_props(df_norm)
        if df_priced is None or df_priced.empty:
            _log("Pricing produced 0 rows. Check required inputs & strict validators.")
            return 1
        _log(f"priced {len(df_priced)} rows")

        # 4) Write
        Path(write_dir).mkdir(parents=True, exist_ok=True)
        write_outputs(df_priced, write_dir, basename)
        _log("pipeline complete.")
        return 0

    except Exception as e:
        _log(f"EXCEPTION: {e}")
        traceback.print_exc()
        return 1

