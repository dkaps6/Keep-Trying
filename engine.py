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
    out.setdefault("selection", None)   # blank -> None
    out.setdefault("events", None)      # Odds API event IDs
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

def _import_odds_module():
    return importlib.import_module("scripts.odds_api")

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
    mod = importlib.import_module("scripts.normalize_props")
    for name in ("normalize_props", "normalize", "normalize_df", "to_model_schema"):
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    raise ImportError("Could not import a normalizer from scripts.normalize_props") from err

from scripts.pricing import price_props, write_outputs

# ---------- module override shim (CRITICAL) ----------

def _override_odds_api_defaults(mod, *, hours: Optional[int], selection: Optional[str],
                                team_filter: Optional[List[str]], event_ids: Optional[List[str]]):
    """
    Some repos read module-level variables/env inside odds_api instead of kwargs.
    We proactively set common names so the fetcher can't ignore our choices.
    """
    # hard-clear env that could force filters
    for k in ("SELECTION_FILTER", "ODDS_SELECTION", "SELECTION", "EVENTS", "EVENT_IDS",
              "WINDOW_HOURS", "LOOKAHEAD_HOURS", "WINDOW"):
        if os.environ.get(k) is not None:
            os.environ.pop(k, None)

    def _set(name, value):
        if hasattr(mod, name):
            setattr(mod, name, value)

    # selection-like names
    for name in dir(mod):
        lname = name.lower()
        if "select" in lname and ("filter" in lname or lname.endswith("selection")):
            _set(name, selection)

    # window/hours-like names
    for name in dir(mod):
        lname = name.lower()
        if any(tok in lname for tok in ("window", "hour", "lookahead")) and not callable(getattr(mod, name)):
            if hours is not None:
                _set(name, hours)

    # team filter-like names
    for name in dir(mod):
        lname = name.lower()
        if "team" in lname and "filter" in lname:
            _set(name, team_filter)

    # event id-like names
    for name in dir(mod):
        lname = name.lower()
        if ("event" in lname and ("ids" in lname or "filter" in lname)) or lname in {"events", "event_ids"}:
            _set(name, event_ids)

# ---------- call fetcher with adaptive params ----------

def _call_odds_fetcher(fn, odds_mod, **kwargs):
    """
    Pass only accepted params. Also monkeypatch module-level defaults so internal code can't
    override our args with env/config.
    """
    selection = _none_if_blank(kwargs.get("selection"))
    hours = _parse_window_to_hours(kwargs.get("window"))
    team_filter = kwargs.get("team_filter")
    event_ids = kwargs.get("event_ids")

    # Set module-level knobs first (covers fetchers that ignore kwargs)
    _override_odds_api_defaults(
        odds_mod,
        hours=hours,
        selection=selection,
        team_filter=team_filter,
        event_ids=event_ids,
    )

    # Build kwargs respecting the fetcher signature
    sig = inspect.signature(fn)
    params: Dict[str, Any] = {}

    canonical = {
        "date": kwargs.get("date"),
        "season": kwargs.get("season"),
        "cap": kwargs.get("cap"),
        "markets": kwargs.get("markets"),
        "order": kwargs.get("order"),
        "books": kwargs.get("books"),
        "team_filter": team_filter,
        "selection_filter": selection,
        "event_ids": event_ids,
        "window": hours,
        "hours": hours,
        "lookahead": hours,
    }

    for k, v in list(canonical.items()):
        if k in sig.parameters and v is not None:
            params[k] = v

    # team_filter <-> teams
    if "team_filter" not in sig.parameters and "teams" in sig.parameters:
        params["teams"] = team_filter
    if "teams" not in sig.parameters and "team_filter" in sig.parameters:
        params["team_filter"] = team_filter

    # selection_filter <-> selection
    if "selection_filter" not in sig.parameters and "selection" in sig.parameters:
        params["selection"] = selection
    if "selection" not in sig.parameters and "selection_filter" in sig.parameters:
        params["selection_filter"] = selection

    # event_ids mapping
    if "event_ids" not in sig.parameters and event_ids is not None:
        for alt in ("events", "ids"):
            if alt in sig.parameters:
                params[alt] = event_ids

    return fn(**params)

# ---------- main pipeline ----------

def run_pipeline(**kwargs) -> int:
    args = _coerce_args(kwargs)

    season: Optional[int] = args["season"]
    date: str = args["date"]
    window: str | int | None = args["window"]
    cap: int = int(args["cap"])
    books: str = str(args["books"])
    order: str = str(args["order"])
    selection: Optional[str] = _none_if_blank(args["selection"])
    write_dir: str = str(args["write_dir"])
    basename: Optional[str] = args["basename"]

    markets = args["markets"]
    if isinstance(markets, str) and markets.strip():
        markets = [m.strip() for m in markets.split(",") if m.strip()]
    elif not markets:
        markets = None

    team_filter = _to_list_if_csv(args.get("teams"))
    if team_filter is None:
        _log("team filter: None (ALL teams in the date window)")
    else:
        _log(f"team filter: {team_filter}")

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
        odds_mod = _import_odds_module()
        norm_fn  = _import_normalizer()

        df_props = _call_odds_fetcher(
            fetch_fn,
            odds_mod,
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
