# scripts/volume.py (STRICT)
# No defaults: will raise if inputs missing/invalid.

from __future__ import annotations
from typing import Tuple

def _as_pos_float(name: str, x) -> float:
    try:
        v = float(x)
    except Exception:
        raise ValueError(f"[STRICT] Missing/invalid float for '{name}': {x}")
    return v

def team_volume_estimates(
    event_row: dict,
    is_home: bool,
    team_form_row: dict,
    weather_triplet: tuple | None,
) -> Tuple[float, float, float, float]:
    """
    Returns (plays, pass_rate, rush_rate, win_prob) with strict checks.
    Required:
      event_row: team_wp, total (used by your system), home_spread optional here.
      team_form_row: plays_base, proe, pace_z
      weather_triplet: (wind_mph, precip, temp_f) required upstream; we don't re-validate precip type here.
    """
    win_prob = _as_pos_float("team_wp", event_row.get("team_wp"))
    plays_base = _as_pos_float("plays_base", team_form_row.get("plays_base"))
    pace_z = float(team_form_row.get("pace_z"))
    proe = float(team_form_row.get("proe"))

    # plays with pace smoothing (no defaults)
    plays = plays_base * (1.0 + 0.5 * pace_z)

    # script nudge (kept very small, but relies on real wp)
    plays *= (1.0 + 0.02 * (win_prob - 0.5) * 2.0)

    # pass rate (no neutral default): you must supply proe
    pass_rate = max(0.0, min(1.0, 0.57 + proe))
    rush_rate = 1.0 - pass_rate

    return float(plays), float(pass_rate), float(rush_rate), float(win_prob)
