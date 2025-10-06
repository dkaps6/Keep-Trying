# scripts/market_keys.py

# --- Canonical player-prop markets we’ll fetch by default ---
NFL_DEFAULT_MARKETS = [
    "player_pass_yds",
    "player_rush_yds",
    "player_reception_yds",
    "player_receptions",
    "player_anytime_td",
]

# --- Synonyms to try (in order) if a canonical key returns no outcomes ---
MARKET_SYNONYMS = {
    "player_pass_yds": [
        "player_pass_yds",
        "player_pass_yards",
        "player_passing_yards",
    ],
    "player_rush_yds": [
        "player_rush_yds",
        "player_rushing_yards",
        "player_rush_yards",
    ],
    "player_reception_yds": [
        "player_reception_yds",
        "player_receiving_yards",
        "player_reception_yards",
    ],
    "player_receptions": [
        "player_receptions",
    ],
    "player_anytime_td": [
        "player_anytime_td",
    ],
}

# --- Validation set (canonical + all synonyms) ---
VALID_MARKETS = sorted(
    set(m for v in MARKET_SYNONYMS.values() for m in v) | set(NFL_DEFAULT_MARKETS)
)
