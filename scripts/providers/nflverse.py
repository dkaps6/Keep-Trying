#!/usr/bin/env python3
from __future__ import annotations
import pandas as pd

# Prefer nflreadpy (newer), fallback to nfl_data_py (older)
_loader = None
_loader_name = "none"
try:
    import nflreadpy as _loader  # type: ignore
    _loader_name = "nflreadpy"
except Exception as e:
    try:
        import nfl_data_py as _loader  # type: ignore
        _loader_name = "nfl_data_py"
    except Exception as e:
        _loader = None

def _fn(name: str):
    if _loader is None:
        return None
    f = getattr(_loader, name, None)
    return f if callable(f) else None

def _safe_call(func, **kwargs):
    if func is None:
        return None
    # handle old signatures (no scope/file_type)
    try:
        return func(**kwargs, file_type="csv")
    except TypeError:
        try:
            kwargs.pop("file_type", None)
            return func(**kwargs)
        except Exception as e:
            return None
    except Exception as e:
        return None

# ------------- basic loaders -------------
def pbp(season: int) -> pd.DataFrame | None:
    f = _fn("load_pbp") or _fn("import_pbp")
    return _safe_call(f, seasons=[season])

def schedules(season: int) -> pd.DataFrame | None:
    f = _fn("load_schedules") or _fn("import_schedules")
    return _safe_call(f, seasons=[season])

def rosters(season: int) -> pd.DataFrame | None:
    f = _fn("load_rosters")
    return _safe_call(f, seasons=[season])

def rosters_weekly(season: int) -> pd.DataFrame | None:
    f = _fn("load_rosters_weekly")
    return _safe_call(f, seasons=[season])

def depth_charts(season: int) -> pd.DataFrame | None:
    f = _fn("load_depth_charts")
    return _safe_call(f, seasons=[season])

def snap_counts(season: int) -> pd.DataFrame | None:
    f = _fn("load_snap_counts")
    return _safe_call(f, seasons=[season])

def participation(season: int) -> pd.DataFrame | None:
    f = _fn("load_participation")
    return _safe_call(f, seasons=[season])

def injuries(season: int) -> pd.DataFrame | None:
    f = _fn("load_injuries")
    return _safe_call(f, seasons=[season])

def team_stats_week(season: int) -> pd.DataFrame | None:
    f = _fn("load_team_stats")
    out = _safe_call(f, seasons=[season], scope="week")
    return out if isinstance(out, pd.DataFrame) and not out.empty else _safe_call(f, seasons=[season])

def team_stats_reg(season: int) -> pd.DataFrame | None:
    f = _fn("load_team_stats")
    out = _safe_call(f, seasons=[season], scope="reg")
    return out if isinstance(out, pd.DataFrame) and not out.empty else _safe_call(f, seasons=[season])

def player_stats_week(season: int) -> pd.DataFrame | None:
    f = _fn("load_player_stats")
    out = _safe_call(f, seasons=[season], scope="week")
    return out if isinstance(out, pd.DataFrame) and not out.empty else _safe_call(f, seasons=[season])

def player_stats_reg(season: int) -> pd.DataFrame | None:
    f = _fn("load_player_stats")
    out = _safe_call(f, seasons=[season], scope="reg")
    return out if isinstance(out, pd.DataFrame) and not out.empty else _safe_call(f, seasons=[season])
