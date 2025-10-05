# scripts/validate_outputs.py
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd

REQ_PROPS_BASE = ["player", "market", "line"]
REQ_GAMES_BASE = ["game_id", "team"]

def _read(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p)
    except Exception as e:
        print(f"[validate] failed reading {p}: {e}", file=sys.stderr)
        sys.exit(2)

def _need(df, cols, name):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        print(f"[validate] {name} missing columns: {miss}", file=sys.stderr)
        sys.exit(2)
    if df.empty:
        print(f"[validate] {name} is empty", file=sys.stderr)
        sys.exit(2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--props", required=True)
    ap.add_argument("--games", required=True)
    a = ap.parse_args()

    p_props, p_games = Path(a.props), Path(a.games)
    if not p_props.exists() or not p_games.exists():
        print(f"[validate] missing outputs (props: {p_props.exists()}, games: {p_games.exists()})", file=sys.stderr)
        sys.exit(2)

    props = _read(p_props); games = _read(p_games)
    _need(props, REQ_PROPS_BASE, "props_priced.csv")
    _need(games, REQ_GAMES_BASE, "game_lines.csv")

    for c in REQ_PROPS_BASE:
        if props[c].isna().any():
            print(f"[validate] NaN in props_priced.csv.{c}", file=sys.stderr)
            sys.exit(2)

    print("[validate] outputs look good:",
          f"props={len(props)} rows, games={len(games)} rows")

if __name__ == "__main__":
    main()
