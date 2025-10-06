# scripts/market_keys.py
"""
Canonical, whitelisted Odds API v4 market keys for NFL + a small alias layer.
Anything not in VALID_MARKETS will be dropped before we hit the API.
"""

# ---- base (game) markets
GAME_MARKETS = {
    "h2h", "spreads", "totals",
    "alternate_spreads", "alternate_totals",
}

# ---- quarter/half variants
QH_MARKETS = {
    # h2h
    "h2h_q1", "h2h_q2", "h2h_q3", "h2h_q4",
    "h2h_h1", "h2h_h2",
    # spreads
    "spreads_q1", "spreads_q2", "spreads_q3", "spreads_q4",
    "spreads_h1", "spreads_h2",
    # totals
    "totals_q1", "totals_q2", "totals_q3", "totals_q4",
    "totals_h1", "totals_h2",
    # alternate spreads/totals by period
    "alternate_spreads_q1", "alternate_spreads_q2", "alternate_spreads_q3", "alternate_spreads_q4",
    "alternate_spreads_h1", "alternate_spreads_h2",
    "alternate_totals_q1", "alternate_totals_q2", "alternate_totals_q3", "alternate_totals_q4",
    "alternate_totals_h1", "alternate_totals_h2",
}

# ---- player props
PLAYER_MARKETS = {
    "player_receptions",
    "player_reception_yds",
    "player_reception_longest",
    "player_reception_tds",

    "player_rush_attempts",
    "player_rush_yds",
    "player_rush_longest",
    "player_rush_tds",

    "player_sacks",
    "player_solo_tackles",
    "player_tackles_assists",

    "player_pass_attempts",
    "player_pass_completions",
    "player_pass_interceptions",
    "player_pass_longest_completion",
    "player_pass_yds",
    "player_pass_tds",

    "player_pats",
    "player_kicking_points",
    "player_field_goals",
    "player_defensive_interceptions",

    "player_anytime_td",
    "player_1st_td",
    "player_last_td",
}

# ---- player props (alternate)
PLAYER_ALT_MARKETS = {
    "player_receptions_alternate",
    "player_reception_yds_alternate",
    "player_reception_longest_alternate",
    "player_reception_tds_alternate",

    "player_rush_attempts_alternate",
    "player_rush_yds_alternate",
    "player_rush_longest_alternate",
    "player_rush_tds_alternate",

    "player_sacks_alternate",
    "player_solo_tackles_alternate",
    "player_tackles_assists_alternate",

    "player_pass_attempts_alternate",
    "player_pass_completions_alternate",
    "player_pass_interceptions_alternate",
    "player_pass_longest_completion_alternate",
    "player_pass_yds_alternate",
    "player_pass_tds_alternate",

    "player_pats_alternate",
    "player_kicking_points_alternate",
    "player_field_goals_alternate",
    "player_defensive_interceptions_alternate",
}

# ---- full whitelist
VALID_MARKETS = (
    GAME_MARKETS
    | QH_MARKETS
    | PLAYER_MARKETS
    | PLAYER_ALT_MARKETS
)

# ---- alias / “fixups” (left -> right)
# You can add more legacy names here as you find them.
ALIASES = {
    "player_passing_yards": "player_pass_yds",
    "player_receiving_yards": "player_reception_yds",
    "player_passing_tds": "player_pass_tds",
    "player_passing_attempts": "player_pass_attempts",
    "player_passing_completions": "player_pass_completions",
    "player_passing_longest_completion": "player_pass_longest_completion",
    "player_rushing_yards": "player_rush_yds",
    "player_rushing_attempts": "player_rush_attempts",
    "player_rushing_tds": "player_rush_tds",
    "player_longest_reception": "player_reception_longest",
    "player_receiving_tds": "player_reception_tds",
}

NON_NFL_GLOBAL = {"btts", "draw_no_bet", "outrights", "h2h_lay"}
