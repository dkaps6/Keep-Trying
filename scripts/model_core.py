# scripts/model_core.py
"""
Core modeling and pricing functions for the NFL predictive pipeline.
"""

import pandas as pd
import numpy as np

from scripts.engine_helpers import safe_divide


def price_props_for_events(events_df, odds_df=None, player_form=None, team_form=None):
    """
    Core pricing engine for player and team props.

    Parameters
    ----------
    events_df : pd.DataFrame
        Event-level data for upcoming games (QB, WR, RB props, etc.)
    odds_df : pd.DataFrame, optional
        Odds from sportsbooks (Moneyline, Totals, Props) from odds_api.py
    player_form : pd.DataFrame, optional
        Player-level recent performance metrics (EPA, YAC, route %, etc.)
    team_form : pd.DataFrame, optional
        Team-level context metrics (pass_rate_over_exp, pace, neutral_script, etc.)

    Returns
    -------
    priced_df : pd.DataFrame
        Merged and priced dataset with model probabilities and fair odds
    """

    # --- Step 1: Basic merge of core sources
    df = events_df.copy()

    if odds_df is not None:
        df = df.merge(odds_df, on=["game_id", "player_id"], how="left")

    if player_form is not None:
        df = df.merge(
            player_form,
            on="player_id",
            how="left",
            suffixes=("", "_pform")
        )

    if team_form is not None:
        df = df.merge(
            team_form,
            on="team",
            how="left",
            suffixes=("", "_tform")
        )

    # --- Step 2: Core model scoring
    df["expected_yards"] = (
        0.35 * df.get("targets", 0)
        + 0.65 * df.get("receptions", 0)
        + 0.10 * df.get("yac_per_reception", 0)
        + 0.25 * df.get("air_yards_share", 0)
    )

    df["expected_tds"] = (
        0.04 * df.get("redzone_tgt_share", 0)
        + 0.03 * df.get("team_td_rate", 0)
        + 0.05 * df.get("deep_tgt_share", 0)
    )

    # --- Step 3: Fair odds and probability modeling
    df["proj_line"] = df["expected_yards"] + (df["expected_tds"] * 25)
    df["fair_prob"] = 1 / (1 + np.exp(-0.08 * (df["proj_line"] - df.get("prop_line", 0))))
    df["fair_odds"] = safe_divide(1, df["fair_prob"])

    # --- Step 4: Edge calculation
    if "book_odds" in df.columns:
        df["edge"] = (df["fair_odds"] - df["book_odds"]) / df["book_odds"]
    else:
        df["edge"] = np.nan

    # --- Step 5: Cleanup
    df["timestamp"] = pd.Timestamp.utcnow()

    return df


def simulate_event_outcomes(priced_df, n_sims=10000, random_seed=42):
    """
    Monte Carlo simulation for pricing and probability validation.
    """

    np.random.seed(random_seed)
    sims = []

    for _, row in priced_df.iterrows():
        mu = row.get("proj_line", 0)
        sigma = row.get("proj_line_std", 12)
        samples = np.random.normal(mu, sigma, n_sims)
        win_prob = np.mean(samples > row.get("prop_line", 0))
        sims.append(win_prob)

    priced_df["sim_win_prob"] = sims
    priced_df["sim_fair_odds"] = safe_divide(1, priced_df["sim_win_prob"])

    return priced_df
