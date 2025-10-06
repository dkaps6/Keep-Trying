# scripts/odds_api_safe.py
# Safe wrapper around your existing scripts.odds_api provider.
# - Normalizes window to hours
# - Applies selection/team/event-id filters once (blank == no filter)
# - Delegates fetch & transform to your current scripts.odds_api (so we don't break anything)

from __future__ import annotations

import re
import inspect
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

# We will call into your existing odds_api module for provider functions.
import importlib
_odds_mod = importlib.import_module("scripts.odds_api")


# ------------------------------------------------------------
# Window parser
# ------------------------------------------------------------

def _to_hours(hours=None, window=None, lookahead=None, default: int = 24) -> int:
    """
    Accepts values like 36, '36', '36h', '168h', returns integer hours.
    Falls back to `default` if nothing usable is provided.
    """
    for v in (hours, window, lookahead):
        if v is None:
            continue
        s = str(v).strip().lower()
        if s.endswith("h"):
            s = s[:-1]
        try:
            return int(float(s))
        except Exception:
            continue
    return int(default)


# ------------------------------------------------------------
# Filters (selection / team / event-ids)
# ------------------------------------------------------------

def _apply_selection_filters(
    events_list: Iterable[Any],
    teams: Optional[Iterable[str]] = None,
    events: Optional[Iterable[str]] = None,
    selection: Optional[str] = None,
):
    """
    Apply optional filters to a list of events.

    - teams: list[str] of substrings to match in home/away/team names.
             None/[]/"all" -> no team filter
    - events: list[str] of Odds API event IDs.
             None/[]/"all" -> no event-id filter
    - selection: string/regex to match event names.
             None/"" -> no selection filter
    Returns the filtered list.
    """
    # 1) selection
    sel_regex = None
    if selection is not None:
        s = str(selection).strip()
        if s != "":
            try:
                sel_regex = re.compile(s, re.IGNORECASE)
            except Exception:
                # treat as plain substring if regex fails
                sel_regex = re.compile(re.escape(s), re.IGNORECASE)

    def _sel_ok(ev):
        if sel_regex is None:
            return True
        name = str(getattr(ev, "name", "") or getattr(ev, "title", "") or "")
        return bool(sel_regex.search(name))

    # 2) team filter
    team_terms = None
    if teams is not None:
        if isinstance(teams, str):
            teams = [t.strip() for t in teams.split(",") if t.strip()]
        if isinstance(teams, (list, tuple)) and len(teams) > 0 and str(teams).lower() not in ("all",):
            team_terms = [t.lower() for t in teams]

    def _team_ok(ev):
        if team_terms is None:
            return True
        # Try to get home/away or team names safely
        home = str(getattr(ev, "home_team", "")).lower()
        away = str(getattr(ev, "away_team", "")).lower()
        name = str(getattr(ev, "name", "") or getattr(ev, "title", "") or "").lower()
        bucket = f"{home} {away} {name}"
        return any(term in bucket for term in team_terms)

    # 3) event-id filter
    event_ids = None
    if events is not None:
        if isinstance(events, str):
            events = [e.strip() for e in events.split(",") if e.strip()]
        if isinstance(events, (list, tuple)) and len(events) > 0 and str(events).lower() not in ("all",):
            event_ids = set(events)

    def _id_ok(ev):
        if event_ids is None:
            return True
        eid = getattr(ev, "id", None) or getattr(ev, "event_id", None) or getattr(ev, "key", None)
        return (eid in event_ids)

    filtered = [ev for ev in events_list if _sel_ok(ev) and _team_ok(ev) and _id_ok(ev)]
    return filtered


# ------------------------------------------------------------
# Delegate discovery (we call your existing provider/transformers)
# ------------------------------------------------------------

def _discover_provider_fetch():
    """
    Find a function that fetches raw events/props in *your* odds_api module.
    The function SHOULD accept (date, season, hours, books, markets, order, **kwargs)
    and return an iterable of events.
    """
    # search inside your existing scripts.odds_api first
    for name in (
        "_fetch_events_from_provider",
        "fetch_events_from_provider",
        "fetch_events",
        "_fetch_events",
        "raw_fetch_events",
    ):
        fn = getattr(_odds_mod, name, None)
        if callable(fn):
            return fn

    # last resort: if your module exposes fetch_props_raw, we’ll use it
    for name in ("fetch_props_raw", "get_raw_events"):
        fn = getattr(_odds_mod, name, None)
        if callable(fn):
            return fn

    raise RuntimeError(
        "No provider fetcher found in scripts.odds_api. "
        "Add a callable like `_fetch_events_from_provider(date, season, hours, books, markets, order, **kwargs)` "
        "that returns a list of events."
    )


def _discover_events_to_df():
    """
    Find a function that transforms events list -> pandas DataFrame of props in *your* odds_api module.
    """
    for name in (
        "events_to_dataframe",
        "_events_to_dataframe",
        "events_to_df",
        "_events_to_df",
        "to_dataframe",
        "to_df",
    ):
        fn = getattr(_odds_mod, name, None)
        if callable(fn):
            return fn

    raise RuntimeError(
        "No transformer found in scripts.odds_api. "
        "Provide `events_to_dataframe(events)` that returns a pandas.DataFrame."
    )


# ------------------------------------------------------------
# Public API (mirrors your old odds_api)
# ------------------------------------------------------------

def fetch_props(
    date: Optional[str] = None,
    season: Optional[int] = None,
    window: Optional[str | int] = None,   # may be '36h' etc.
    hours: Optional[int] = None,          # alternate
    lookahead: Optional[int] = None,      # alternate
    cap: int = 0,
    markets: Optional[List[str] | str] = None,
    order: str = "odds",
    books: str = "dk",
    team_filter: Optional[List[str] | str] = None,
    selection: Optional[str] = None,
    event_ids: Optional[List[str] | str] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Fetch raw events via your provider (in scripts.odds_api), apply filters safely, convert to DataFrame.
    """
    hrs = _to_hours(hours=hours, window=window, lookahead=lookahead, default=24)

    # Pretty log (kept to match your previous output style)
    print(f"[odds_api] window={hrs}h cap={cap}")
    if markets:
        mstr = ",".join(markets) if isinstance(markets, (list, tuple)) else str(markets)
    else:
        mstr = "player_receptions,player_receiving_yds,player_rush_yds,player_rush_attempts,player_pass_yds,player_pass_tds,player_anytime_td"
    print(f"markets={mstr}\norder={order},{books}")

    # ---- FETCH (from your existing odds_api) ----
    provider_fetch = _discover_provider_fetch()
    sig = inspect.signature(provider_fetch)
    pkwargs: Dict[str, Any] = {}
    for k, v in {
        "date": date,
        "season": season,
        "hours": hrs,
        "books": books,
        "markets": markets,
        "order": order,
    }.items():
        if k in sig.parameters:
            pkwargs[k] = v
    for k, v in kwargs.items():
        if k in sig.parameters:
            pkwargs[k] = v

    evs = provider_fetch(**pkwargs) or []
    before = len(evs) if hasattr(evs, "__len__") else -1

    # ---- FILTER (once) ----
    evs = _apply_selection_filters(
        events_list=evs,
        teams=team_filter,
        events=event_ids,
        selection=selection,
    )
    after = len(evs) if hasattr(evs, "__len__") else -1

    # optional log
    if (team_filter and (isinstance(team_filter, (list, tuple)) and len(team_filter) > 0)) \
       or (event_ids and (isinstance(event_ids, (list, tuple)) and len(event_ids) > 0)) \
       or (selection and str(selection).strip() != ""):
        print(f"[odds_api] selection filter -> {after} events kept")
    else:
        print("[odds_api] selection filter -> no filter applied")

    # ---- CAP ----
    if cap and cap > 0 and isinstance(evs, list):
        evs = evs[:cap]

    # ---- TO DATAFRAME (via your existing odds_api) ----
    to_df = _discover_events_to_df()
    df = to_df(evs)

    if df is None:
        return pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            return pd.DataFrame()

    return df


# Convenience aliases expected by some callers
def get_props(*args, **kwargs) -> pd.DataFrame:
    return fetch_props(*args, **kwargs)

def get_props_df(*args, **kwargs) -> pd.DataFrame:
    return fetch_props(*args, **kwargs)
