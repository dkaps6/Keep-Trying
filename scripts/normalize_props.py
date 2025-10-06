"""
normalize_props.py
Map the raw V4 rows into a consistent schema used by the rest of the pipeline.
"""

from __future__ import annotations

import pandas as pd

# Human labels (optional) for readability / downstream grouping
MARKET_LABEL = {
    # Passing
    "player_pass_yds": "Pass Yds",
    "player_pass_tds": "Pass TDs",
    "player_pass_attempts": "Pass Att",
    "player_pass_completions": "Pass Comp",
    "player_pass_interceptions": "Pass INTs",
    "player_pass_longest_completion": "Longest Completion",
    # Rushing
    "player_rush_yds": "Rush Yds",
    "player_rush_attempts": "Rush Att",
    "player_rush_longest": "Longest Rush",
    # Receiving
    "player_reception_yds": "Recv Yds",
    "player_receptions": "Receptions",
    "player_reception_longest": "Longest Reception",
    "player_reception_tds": "Reception TDs",
    # Combo / specials
    "player_anytime_td": "Anytime TD",
    "player_pass_rush_reception_tds": "Pass+Rush+Recv TDs",
    "player_pass_rush_reception_yds": "Pass+Rush+Recv Yds",
    # Defense / kicking
    "player_sacks": "Sacks",
    "player_solo_tackles": "Solo Tackles",
    "player_tackles_assists": "Tackles+Assists",
    "player_field_goals": "Field Goals",
    "player_pats": "PATs",
}

def normalize_props(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input columns from props_hybrid.get_props():
      event_id, commence_time, home_team, away_team,
      bookmaker, market_key, name, description, price, point, team, player

    Output schema:
      event_id, commence_time, bookmaker, market_key, market_label,
      player, team, side, line, price, selection
    """
    if df is None or df.empty:
        return df

    d = df.copy()

    # The “side” (Over/Under/Yes/No) is in `name`; protect if missing
    d["side"] = d["name"].fillna("")

    # A single numeric line for O/U markets comes as `point`
    d["line"] = d["point"]

    # Player: prefer 'player' column (alias of description) then description
    d["player"] = d["player"].fillna(d.get("description"))

    # Human-friendly label for BI
    d["market_label"] = d["market_key"].map(MARKET_LABEL).fillna(d["market_key"])

    # Keep a compact set of columns
    cols = [
        "event_id", "commence_time",
        "bookmaker",
        "market_key", "market_label",
        "player", "team",
        "side", "line", "price",
        "name", "description",  # keep originals for audit
        "home_team", "away_team",
    ]
    cols = [c for c in cols if c in d.columns]
    return d[cols]
