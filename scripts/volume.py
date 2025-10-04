# scripts/volume.py
from __future__ import annotations
import pandas as pd
import numpy as np

LEAGUE_PLAYS = 62.0   # baseline plays per team per game
BASE_PASS = 0.56      # baseline pass rate
BASE_RUSH = 1 - BASE_PASS

def _consensus_number(df: pd.DataFrame, market: str, name_filter=None) -> pd.DataFrame:
    d = df[df["market"].eq(market)].copy()
    if name_filter is not None:
        d = d[d["name"].str.contains(name_filter, na=False)]
    # average across books
    grp = d.groupby("event_id", as_index=False)["point"].mean().rename(columns={"point":"consensus"})
    return grp

def consensus_spread_total(game_df: pd.DataFrame) -> pd.DataFrame:
    # spreads: outcomes have names (home/away) and point is spread (negative for fav)
    sp = game_df[game_df["market"].eq("spreads")].copy()
    # choose home spread (book format varies); keep both & compute home_favored
    home_sp = sp[sp["name"].eq(sp["home"])].groupby("event_id", as_index=False)["point"].mean().rename(columns={"point":"home_spread"})
    tot = game_df[game_df["market"].eq("totals")].groupby("event_id", as_index=False)["point"].mean().rename(columns={"point":"total"})
    out = home_sp.merge(tot, on="event_id", how="outer")
    return out

def win_prob_from_spread(home_spread: float) -> float:
    """
    Approximate home win prob from spread using a probit-like mapping.
    Quick empirical: WP ~ Phi(spread / 13.45). Clamp [0.05,0.95].
    """
    if home_spread is None or pd.isna(home_spread):
        return 0.5
    from math import erf, sqrt
    z = home_spread / 13.45
    p = 0.5*(1+erf(z/sqrt(2)))
    return float(max(0.05, min(0.95, p)))

def team_volume_estimates(
    event_row: pd.Series,
    team_is_home: bool,
    team_form_row: pd.Series | None = None
):
    """
    Returns (plays, pass_rate, rush_rate, win_prob)
    """
    total = event_row.get("total", np.nan)
    home_spread = event_row.get("home_spread", np.nan)
    wp_home = win_prob_from_spread(home_spread)
    win_prob = wp_home if team_is_home else (1 - wp_home)

    # Plays: nudge by total (higher totals → more plays) small effect
    plays = LEAGUE_PLAYS * (1.0 + 0.12*((_nz(total:=total, d=44.0) - 44.0)/44.0))

    # Pass rate: start from recent team_form if present, else baseline
    if team_form_row is not None and "off_dropbacks_l4" in team_form_row and "off_plays_l4" in team_form_row:
        pr = float(team_form_row["off_dropbacks_l4"]) / max(float(team_form_row["off_plays_l4"]), 1.0)
        pass_rate = 0.8*pr + 0.2*BASE_PASS
    else:
        pass_rate = BASE_PASS

    # Script: when favored (win_prob high) slightly reduce pass_rate; when dog, increase
    pass_rate = pass_rate + (0.08 * ((0.5 - win_prob)))  # +-4% swing around even
    pass_rate = max(0.40, min(0.65, pass_rate))
    rush_rate = 1 - pass_rate
    return plays, pass_rate, rush_rate, win_prob

def _nz(v, d=0.0):
    try:
        return float(v) if v is not None and not np.isnan(float(v)) else d
    except Exception:
        return d
