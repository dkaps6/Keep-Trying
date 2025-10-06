# engine.py
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import importlib
import inspect

# ---------- helpers ----------

def _to_list_if_csv(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    if isinstance(s, list):
        return s
    s = str(s).strip()
    if not s or s.lower() in {"all", "none", "null"}:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]

def _none_if_blank(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return None if s == "" else s

def _parse_window_to_hours(w: Optional[str | int]) -> Optional[int]:
    if w is None:
        return None
    if isinstance(w, int):
        return w
    s = str(w).strip().lower()
    if s.endswith("h"):
        s = s[:-1]
    try:
        return int(float(s))
    except Exception:
        return None

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
    out.setdefault("selection", None)   # we will coerce blank -> None
    out.setdefault("events", None)      # optional Odds API event IDs
    out.setdefault("write_dir", "outputs")
    out.setdefault("basename", None)
    return out

def _log(msg: str) -> None:
    print(f"[engine] {msg}")

# ---------- robust imports ----------

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
    try:
        mod = importlib.import_module("scripts.odds_api")
        for name in ("get_props", "fetch_props", "get_props_df"):
            fn = getattr(mod, name, None)
            if callable(fn):
                return fn
    except Exception as e:
        err = e
    raise ImportError("Could not import a props fetcher from scripts.odds_api") from err

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
    try:
        mod = importlib.import_module("scripts.normalize_props")
        for name in ("normalize_props", "normalize", "normalize_df", "to_model_schema"):
            fn = getattr(mod, name, None)
            if callable(fn):
                return fn
    except Exception as e:
        err = e
    raise ImportError("Could not import a normalizer from scripts.normalize_props") from err

from scripts.pricing import price_props, write_outputs

# ---------- call fetcher with adaptive params ----------

def _call_odds_fetcher(fn, **kwargs):
    """
    Pass only accepted params. Map:
      - window -> window|hours|lookahead (int hours)
      - team_filter <-> teams
      - selection: blank -> None
      - event_ids -> event_ids|events|ids
    """
    sig = inspect.signature(fn)
    params: Dict[str, Any] = {}

    selection = _none_if_blank(kwargs.get("selection"))
    hours = _parse_window_to_hours(kwargs.get("window"))

    # canonical pool
    canonical = {
        "date": kwargs.get("date"),
        "season": kwargs.get("season"),
        "cap": kwargs.get("cap"),
        "markets": kwargs.get("markets"),
        "order": kwargs.get("order"),
        "books": kwargs.get("books"),
        "team_filter": kwargs.get("team_filter"),
        "selection_filter": selection,
        "event_ids": kwargs.get("event_ids"),
        "window": hours,
        "hours": hours,
        "lookahead": hours,
    }

    # pass through only what the fetcher accepts (and only if not None)
    for k, v in list(canonical.items()):
        if k in sig.parameters and v is not None:
            params[k] = v

    # team_filter <-> teams
    if "team_filter" not in sig.parameters and "teams" in sig.parameters:
        params["teams"] = canonical["team_filter"]
    if "teams" not in sig.parameters and "team_filter" in sig.parameters:
        params["team_filter"] = canonical["team_filter"]

    # selection_filter <-> selection
    if "selection_filter" not in sig.parameters and "selection" in sig.parameters:
        params["selection"] = canonical["selection_filter"]
    if "selection" not in sig.parameters and "selection_filter" in sig.parameters:
        params["selection_filter"] = canonical["selection_filter"]

    # event_ids mapping
    if "event_ids" not in sig.parameters and kwargs.get("event_ids") is not None:
        for alt in ("events", "ids"):
            if alt in sig.parameters:
                params[alt] = kwargs["event_ids"]

    return fn(**params)

# ---------- main pipeline ----------

def run_pipeline(**kwargs) -> int:
    args = _coerce_args(kwargs)

    # HARD-CLEAR env knobs that might force selection/event filters inside odds_api
    for k in ("SELECTION_FILTER", "ODDS_SELECTION", "SELECTION", "EVENTS", "EVENT_IDS"):
        if os.environ.get(k) is not None:
            os.environ.pop(k, None)

    season: Optional[int] = args["season"]
    date: str = args["date"]
    window: str | int | None = args["window"]
    cap: int = int(args["cap"])
    books: str = str(args["books"])
    order: str = str(args["order"])
    selection: Optional[str] = _none_if_blank(args["selection"])  # force None if blank
    write_dir: str = str(args["write_dir"])
    basename: Optional[str] = args["basename"]

    # markets
    markets = args["markets"]
    if isinstance(markets, str) and markets.strip():
        markets = [m.strip() for m in markets.split(",") if m.strip()]
    elif not markets:
        markets = None

    # teams (NO FILTER by default)
    team_filter = _to_list_if_csv(args.get("teams"))
    if team_filter is None:
        _log("team filter: None (ALL teams in the date window)")
    else:
        _log(f"team filter: {team_filter}")

    # events (optional)
    event_ids = _to_list_if_csv(args.get("events"))
    if event_ids:
        _log(f"event IDs filter: {event_ids}")

    if not basename:
        basename = f"props_priced_{date}"

    try:
        _log(f"fetching props… date={date} season={season}")
        _log(f"window={window} cap={cap}")
        _log(f"markets={','.join(markets) if markets else 'default'} order={order} books={books}")
        _log(f"selection={selection}")

        fetch_fn = _import_odds_fetcher()
        norm_fn  = _import_normalizer()

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
            selection=selection,  # now definitely None if blank
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

        df_norm = norm_fn(df_props)
        if df_norm is None or (hasattr(df_norm, "empty") and df_norm.empty):
            _log("Normalization produced 0 rows. Check provider mapping / markets.")
            return 1
        _log(f"normalized {len(df_norm)} rows")

        df_priced = price_props(df_norm)
        if df_priced is None or df_priced.empty:
            _log("Pricing produced 0 rows. Check required inputs & strict validators.")
            return 1
        _log(f"priced {len(df_priced)} rows")

        Path(write_dir).mkdir(parents=True, exist_ok=True)
        write_outputs(df_priced, write_dir, basename)
        _log("pipeline complete.")
        return 0

    except Exception as e:
        _log(f"EXCEPTION: {e}")
        traceback.print_exc()
        return 1
