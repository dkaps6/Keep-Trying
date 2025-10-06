from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional
import importlib
import inspect
import pandas as pd

# =====================================================
# IMPORTS
# =====================================================
from scripts.odds_api_v4 import get_props as odds_get_props


# =====================================================
# HELPERS
# =====================================================
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
    out.setdefault("teams", None)
    out.setdefault("selection", None)
    out.setdefault("events", None)
    out.setdefault("write_dir", "outputs")
    out.setdefault("basename", None)
    return out


def _log(msg: str) -> None:
    print(f"[engine] {msg}")


# =====================================================
# DYNAMIC IMPORTS
# =====================================================
def _import_odds_fetcher():
    from scripts.props_hybrid import get_props as fn  # new hybrid odds fetcher
    return fn


def _import_normalizer():
    err = None
    try:
        from scripts.normalize_props import normalize_props as fn
        return fn
    except Exception as e:
        err = e
    try:
        from scripts.normalize_props import normalize as fn
        return fn
    except Exception as e:
        err = e
    try:
        from scripts.normalize_props import normalize_df as fn
        return fn
    except Exception as e:
        err = e
    try:
        from scripts.normalize_props import to_model_schema as fn
        return fn
    except Exception as e:
        err = e
    mod = importlib.import_module("scripts.normalize_props")
    for name in ("normalize_props", "normalize", "normalize_df", "to_model_schema"):
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    raise ImportError("Could not import a normalizer from scripts.normalize_props") from err


# =====================================================
# UPDATED FETCHER ADAPTER
# =====================================================
def _call_odds_fetcher(
    date: str,
    season: str | None,
    window: int,
    cap: int,
    books: list[str] | str | None,
    markets: list[str] | str | None,
    order: str | None,
    selection: str | None,
    teams: list[str] | str | None,
    events: list[str] | str | None,
):
    """
    Adapter that calls scripts.props_hybrid.get_props and returns a DataFrame.
    Injects THE_ODDS_API_KEY automatically from environment.
    """
    fn = _import_odds_fetcher()

    api_key = (
        os.getenv("THE_ODDS_API_KEY")
        or os.getenv("ODDS_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "THE_ODDS_API_KEY is not set. "
            "Add it to your GitHub Secrets (or your local env)."
        )

    # normalize args
    books_arg = books
    markets_arg = markets

    _log(f"Calling get_props() with API key and cap={cap}")
    results = fn(
        api_key=api_key,
        books=books_arg,
        markets=markets_arg,
        limit=cap if isinstance(cap, int) and cap > 0 else 0,
    )

    if isinstance(results, pd.DataFrame):
        return results

    rows = []
    for item in results or []:
        ev = item.get("event", {})
        props = item.get("props", [])
        rows.append({
            "event_id": ev.get("id"),
            "commence_time": ev.get("commence_time"),
            "home_team": ev.get("home_team"),
            "away_team": ev.get("away_team"),
            "props_payload": props,
            "books": books_arg,
            "markets": markets_arg,
        })
    return pd.DataFrame(rows)


# =====================================================
# MAIN PIPELINE
# =====================================================
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
        norm_fn = _import_normalizer()

        df_props = _call_odds_fetcher(
            date=date,
            season=season,
            window=window,
            cap=cap,
            markets=markets,
            order=order,
            books=books,
            teams=team_filter,
            selection=selection,
            events=event_ids,
        )

        if df_props is None or (isinstance(df_props, (list, tuple)) and len(df_props) == 0):
            _log("No props available to price (props fetch returned 0 rows).")
            return 1
        if not isinstance(df_props, pd.DataFrame):
            df_props = pd.DataFrame(df_props)
        if df_props.empty:
            _log("No props available to price (props fetch returned empty DataFrame).")
            return 1

        _log(f"fetched {len(df_props)} raw props from props_hybrid")

        df_norm = norm_fn(df_props)
        if df_norm is None or (hasattr(df_norm, "empty") and df_norm.empty):
            _log("Normalization produced 0 rows. Check provider mapping / markets.")
            return 1
        _log(f"normalized {len(df_norm)} rows")

        # --- pricing logic placeholder ---
        df_priced = df_norm  # temporary; replace with actual pricing call
        _log(f"priced {len(df_priced)} rows")

        Path(write_dir).mkdir(parents=True, exist_ok=True)
        outfile = Path(write_dir) / f"{basename}.csv"
        df_priced.to_csv(outfile, index=False)
        _log(f"✅ pipeline complete: wrote {outfile}")
        return 0

    except Exception as e:
        _log(f"EXCEPTION: {e}")
        traceback.print_exc()
        return 1
