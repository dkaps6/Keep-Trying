# engine.py
import os
import re
import pandas as pd

from scripts.odds_api import fetch_game_lines, fetch_props_all_events
from scripts.features_external import build_external
from scripts.id_map import map_players
from scripts.pricing import (
    american_to_prob, devig_two_way, blend, prob_to_american,
    edge_pct, kelly_fraction, tier
)
from scripts.model_core import (
    base_sigma, over_prob_normal,
    estimate_team_rates, player_shares,
    mu_receptions, mu_rec_yards, mu_rush_atts, mu_rush_yards, mu_pass_yards
)
from scripts.elite_rules import (
    funnel_multiplier, volatility_widen, pace_smoothing, sack_to_attempts
)
from scripts.engine_helpers import make_team_last4_from_player_form
from scripts.rules_engine import apply_rules
from scripts.volume import consensus_spread_total, team_volume_estimates


def _norm(val) -> str:
    """Normalize any input to a lowercase ascii-ish token string. Safe for None/NaN/numbers."""
    if val is None:
        return ""
    s = str(val)
    if s.strip().lower() in {"nan", "none", "null"}:
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def run_pipeline(target_date: str, season: int, out_dir: str = "outputs"):
    """
    1) Fetch lines/props
    2) Build external features
    3) Map players
    4) Volume + elite rules -> μ/σ
    5) Price lines, write CSVs
    """
    # 1) Odds
    game_df  = fetch_game_lines()
    props_df = fetch_props_all_events()

    # consensus spread/total for volume
    cons = consensus_spread_total(game_df)  # event_id, home_spread, total

    # 2) External features
    ext = build_external(season)
    ids = ext["ids"]
    team_form = ext["team_form"]
    pform = ext["player_form"]

    # team context (L4 totals)
    team_l4_map = make_team_last4_from_player_form(pform)

    # 3) map players
    props_df = map_players(props_df, ids)

    def get_team_context(team):
        return team_l4_map.get(team, {"tgt_team_l4": 30.0, "rush_att_team_l4": 25.0})

    rows = []
    for _, r in props_df.iterrows():
        market = r["market"]
        if market not in {
            "player_reception_yds", "player_receptions",
            "player_rush_yds", "player_rush_attempts",
            "player_pass_yds", "player_pass_tds",
            "player_anytime_td",
        }:
            continue

        price = r["price"]
        line  = r["point"]
        side  = r["outcome"]

        # implied prob & de-vi

