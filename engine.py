# engine.py
from __future__ import annotations

import inspect
import importlib
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pathlib import Path

# local data folder
REPO = Path(__file__).resolve().parent
DATA = REPO / "data"

def _read_soft(p: Path) -> pd.DataFrame:
    """Gracefully read a CSV file or return an empty DataFrame."""
    try:
        if p.exists() and p.stat().st_size > 0:
            return pd.read_csv(p)
    except Exception:
        pass
    return pd.DataFrame()

# -----------------------------
# Logging
# -----------------------------
def _log(msg: str) -> None:
    print(f"[engine] {msg}")


# -----------------------------
# Small helpers
# -----------------------------
def _to_list_if_csv(x: Optional[Union[str, List[str]]]) -> Optional[List[str]]:
    """Turn 'a,b,c' into ['a','b','c']; keep list as-is; normalize blanks to None."""
    if x is None:
        return None
    if isinstance(x, list):
        vals = [str(v).strip() for v in x if str(v).strip()]
        return vals or None
    s = str(x).strip()
    if not s or s.lower() in {"all", "none", "null"}:
        return None
    vals = [part.strip() for part in s.split(",") if part.strip()]
    return vals or None


def _none_if_blank(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return None if s == "" else s


def _parse_window_to_hours(w: Optional[Union[str, int]]) -> Optional[int]:
    """
    Accept '36' / '36h' / 36 -> 36. Accept '0' -> 0 ; None -> None
    """
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
    out.setdefault("date", "today")
    out.setdefault("season", None)
    out.setdefault("window", "168h")  # hours lookahead (string or int)
    out.setdefault("hours", None)     # ALSO supported now; runner may pass this
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


# -----------------------------
# Robust dynamic imports
# -----------------------------
def _import_odds_fetcher():
    try:
        from scripts.props_hybrid import get_props as fn  # type: ignore
        return fn
    except Exception as e:
        raise ImportError(
            "Unable to import props fetcher from scripts.props_hybrid.get_props. "
            "Make sure scripts/props_hybrid.py exists and exports get_props()."
        ) from e


def _import_normalizer():
    candidates = (
        ("scripts.normalize_props", "normalize_props"),
        ("scripts.normalize_props", "normalize"),
        ("scripts.normalize_props", "normalize_df"),
        ("scripts.normalize_props", "to_model_schema"),
    )
    last_err = None
    for mod_name, fn_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                return fn
        except Exception as e:
            last_err = e
    raise ImportError(
        "Could not import a normalizer from scripts.normalize_props. "
        "Expected one of: normalize_props, normalize, normalize_df, to_model_schema."
    ) from last_err


def _import_pricer():
    candidates = (
        ("scripts.price_props", "price_props"),
        ("scripts.pricing", "price_props"),
        ("scripts.pricer", "price_props"),
    )
    for mod_name, fn_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                return fn
        except Exception:
            pass

    def _passthrough(df: pd.DataFrame) -> pd.DataFrame:
        _log("WARNING: no pricing function found; passing rows through unchanged.")
        return df

    return _passthrough


def _import_writer():
    candidates = (
        ("scripts.write_outputs", "write_outputs"),
        ("scripts.outputs", "write_outputs"),
        ("scripts.io_utils", "write_outputs"),
    )
    for mod_name, fn_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                return fn
        except Exception:
            pass

    def _fallback_writer(df: pd.DataFrame, out_dir: str, base: str) -> None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_csv = Path(out_dir) / f"{base or 'props'}.csv"
        df.to_csv(out_csv, index=False)
        _log(f"wrote CSV fallback -> {out_csv}")

    return _fallback_writer


# -----------------------------
# Safe caller (pass only accepted params)
# -----------------------------
def _call_fetcher_safely(fn, **kwargs):
    """
    Inspect the fetcher’s signature and only pass supported parameters.
    Also adapts window/hours → hours and cleans blanks.
    """
    sig = inspect.signature(fn)

    selection = _none_if_blank(kwargs.get("selection"))
    # normalize a possible window/hours pair into one integer
    hours = kwargs.get("hours")
    if hours is None:
        hours = _parse_window_to_hours(kwargs.get("window"))
    else:
        hours = _parse_window_to_hours(hours)

    canonical = {
        "date": kwargs.get("date"),
        "season": kwargs.get("season"),
        "cap": kwargs.get("cap"),
        "markets": kwargs.get("markets"),
        "order": kwargs.get("order"),
        "books": kwargs.get("books"),
        "team_filter": kwargs.get("team_filter"),
        "selection": selection,
        "event_ids": kwargs.get("event_ids"),
        # windows/hours variants
        "window": hours,   # if fetcher expects 'window'
        "hours": hours,    # if fetcher expects 'hours'
        "lookahead": hours # if fetcher expects 'lookahead'
    }

    params: Dict[str, Any] = {}
    for k, v in canonical.items():
        if v is None:
            continue
        if k in sig.parameters:
            params[k] = v

    return fn(**params)


# -----------------------------
# Main pipeline
# -----------------------------
def run_pipeline(
    *,
    date: str = "today",
    season: Optional[int] = None,
    window: Optional[Union[str, int]] = "168h",
    hours: Optional[Union[str, int]] = None,  # <— NEW: accept hours directly
    cap: int = 0,
    markets: Optional[str] = None,
    books: str = "draftkings,fanduel",
    order: str = "odds",
    teams: Optional[str] = None,
    selection: Optional[str] = None,
    events: Optional[str] = None,
    write_dir: str = "outputs",
    basename: Optional[str] = None,
    odds_game_df: Optional[pd.DataFrame] = None,   
    odds_props_df: Optional[pd.DataFrame] = None,
) -> int:
    """
    Orchestrates the end-to-end pricing run.
    """
    try:
        # Normalize/parse inputs
        team_filter = _to_list_if_csv(teams)
        event_ids = _to_list_if_csv(events)
        markets_list = _to_list_if_csv(markets)

        # Resolve effective hours for logging (CLI may provide hours or window)
        eff_hours = _parse_window_to_hours(hours if hours is not None else window)

        _log(f"fetching props… date={date} season={season}")
        _log(f"markets={','.join(markets_list) if markets_list else 'default'} order={order} books={books}")
        _log(f"selection={_none_if_blank(selection)} window={window} hours={eff_hours} cap={cap}")
        # Odds consensus fallbacks (safe hybrid)
        if odds_game_df is None:
            odds_game_df = _read_soft(DATA / "odds_game_consensus.csv")
        if odds_props_df is None:
            odds_props_df = _read_soft(DATA / "odds_props_consensus.csv")

        _log(f"odds_game rows={len(odds_game_df)}; odds_props rows={len(odds_props_df)}")

        # Resolve modules
        fetch_fn = _import_odds_fetcher()
        norm_fn = _import_normalizer()
        price_fn = _import_pricer()
        writer_fn = _import_writer()

        # Fetch (pass both window and hours; safe-caller will adapt to fetcher signature)
        raw_df = _call_fetcher_safely(
            fetch_fn,
            date=date,
            season=season,
            window=window,
            hours=eff_hours,
            cap=cap,
            markets=markets_list,
            order=order,
            books=books,
            team_filter=team_filter,
            selection=selection,
            event_ids=event_ids,
        )

        if raw_df is None:
            _log("No props returned (None).")
            return 1
        if not isinstance(raw_df, pd.DataFrame):
            raw_df = pd.DataFrame(raw_df)
        if raw_df.empty:
            _log("No props available to price (props fetch returned empty DataFrame).")
            return 1

        _log(f"fetched {len(raw_df)} raw rows")

        # Normalize
        df_norm = norm_fn(raw_df)
        if df_norm is None or (hasattr(df_norm, "empty") and df_norm.empty):
            _log("Normalization produced 0 rows. Check provider mapping / markets.")
            return 1
        _log(f"normalized {len(df_norm)} rows")

        # Price
        df_priced = price_fn(df_norm)
        if df_priced is None or (hasattr(df_priced, "empty") and df_priced.empty):
            _log("Pricing produced 0 rows. Check required inputs & strict validators.")
            return 1
        _log(f"priced {len(df_priced)} rows")

        # Output
        if not basename:
            basename = f"props_priced_{date}"
        Path(write_dir).mkdir(parents=True, exist_ok=True)
        writer_fn(df_priced, write_dir, basename)
        _log("pipeline complete.")
        return 0

    except Exception as e:
        _log(f"EXCEPTION: {e}")
        traceback.print_exc()
        return 1

