# scripts/features_external.py
import os
import pandas as pd

def _safe_read(path: str) -> pd.DataFrame:
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

def build_external(season: int):
    """
    Returns dict with:
      - ids:         id map (player_name -> gsis_id/team/pos)
      - player_form: rolling player features (L4, red-zone shares)
      - team_form:   opponent z-scores and shares (index by team)
      - weather:     optional per-event weather
    """
    ids = _safe_read("metrics/id_map.csv")
    player_form = _safe_read("metrics/player_form.csv")
    team_form = _safe_read("metrics/team_form.csv")
    weather = _safe_read("inputs/weather.csv")
    if not team_form.empty and "team" in team_form.columns:
        team_form = team_form.set_index("team")

    return {
        "ids": ids,
        "player_form": player_form,
        "team_form": team_form,
        "weather": weather,
    }
