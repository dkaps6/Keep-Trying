# scripts/props_hybrid.py
from __future__ import annotations
import os
import pandas as pd

USE_ODDSAPI = os.getenv("ODDS_HYBRID_USE_ODDSAPI", "true").lower() not in ("0","false","no")
USE_BOOKS   = os.getenv("ODDS_HYBRID_USE_BOOKS",   "true").lower() not in ("0","false","no")

def _safe_concat(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    if a is None or a.empty:
        return b.copy() if b is not None else pd.DataFrame()
    if b is None or b.empty:
        return a.copy()
    out = pd.concat([a, b], ignore_index=True)
    out = out.drop_duplicates(subset=["event_id","player","market","book","vegas_line","price_name"], keep="first")
    return out

def get_props(*, books="draftkings", **kwargs) -> pd.DataFrame:
    """
    Hybrid union:
      1) Try OddsAPI v4 (if enabled)
      2) Try direct books (if enabled)
      3) Return union (or whichever is non-empty)
    """
    df_odds = pd.DataFrame()
    df_books = pd.DataFrame()

    if USE_ODDSAPI:
        try:
            from scripts.odds_api_v4 import get_props as odds_get
            df_odds = odds_get(books=books, **kwargs)
            print(f"[hybrid] oddsapi rows = {0 if df_odds is None else len(df_odds)}")
        except Exception as e:
            print(f"[hybrid] oddsapi failed: {e}")
            df_odds = pd.DataFrame()

    if USE_BOOKS:
        try:
            from scripts.props_books import get_props as books_get
            df_books = books_get(books=books, **kwargs)
            print(f"[hybrid] books rows = {0 if df_books is None else len(df_books)}")
        except Exception as e:
            print(f"[hybrid] books failed: {e}")
            df_books = pd.DataFrame()

    # If one side is empty, the other wins; if both empty, return empty (engine will log it).
    df = _safe_concat(df_odds, df_books)
    print(f"[hybrid] merged rows = {len(df)}")
    return df
