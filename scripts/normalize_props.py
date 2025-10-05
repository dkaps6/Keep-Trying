# scripts/normalize_props.py
from __future__ import annotations
import pandas as pd

# map feed keys → friendly names your model expects
MARKET_KEYS = {
    "player_pass_yds": "pass_yards",
    "player_pass_tds": "pass_tds",
    "player_receptions": "receptions",
    "player_reception_yds": "rec_yards",
    "player_rush_yds": "rush_yards",
    "player_rush_attempts": "rush_att",
    "player_rush_reception_yds": "rush_rec_yards",
    "player_anytime_td": "anytime_td",
}

def normalize(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Input: outcome-level rows (both sides present), columns:
      event_id, home_team, away_team, commence_time, book, market, player, side, line, price
    Output: one row per (player,market,line,book,event) with prices for both sides.
    """
    df = df_raw.copy()
    df["market_key"] = df["market"].map(MARKET_KEYS).fillna(df["market"])
    df["side"] = df["side"].str.lower()

    # pivot prices into two columns
    over_yes = df[df["side"].isin(["over","yes"])].rename(columns={"price":"price_over"})
    under_no = df[df["side"].isin(["under","no"])].rename(columns={"price":"price_under"})
    key_cols = ["event_id","book","market_key","player","line"]
    left = over_yes[key_cols + ["price_over","home_team","away_team","commence_time"]]
    right = under_no[key_cols + ["price_under"]]
    merged = pd.merge(left, right, on=key_cols, how="outer")

    # drop dup columns if any
    merged["price_over"] = merged["price_over"].astype(float)
    merged["price_under"] = merged["price_under"].astype(float)
    merged["line"] = merged["line"].astype(float)

    # Keep even if one side missing (we can still price)
    cols = ["player","market_key","line","book","event_id","home_team","away_team","commence_time","price_over","price_under"]
    merged = merged[cols].drop_duplicates()

    return merged
