# scripts/providers/nflverse_entry.py — pure-python nflverse fallback
from __future__ import annotations
from pathlib import Path
import pandas as pd

def fetch(season: int, date: str | None = None) -> None:
    try:
        import nfl_data_py as nfl
    except Exception as e:
        raise RuntimeError(f"nfl_data_py not available: {e}")

    years = [int(season)]
    d = Path("data"); d.mkdir(parents=True, exist_ok=True)

    # Weekly player stats
    try:
        weekly = nfl.import_weekly_data(years=years)
        if isinstance(weekly, pd.DataFrame) and not weekly.empty:
            weekly.to_csv(d / "player_stats_week.csv", index=False)
            print(f"[nflverse] wrote data/player_stats_week.csv rows={len(weekly)}")
    except Exception as e:
        print(f"[nflverse] weekly failed: {e}")

    # Rosters
    try:
        rost = nfl.import_rosters(years=years)
        if isinstance(rost, pd.DataFrame) and not rost.empty:
            rost.to_csv(d / "rosters.csv", index=False)
            print(f"[nflverse] wrote data/rosters.csv rows={len(rost)}")
    except Exception as e:
        print(f"[nflverse] rosters failed: {e}")

    # PBP
    try:
        pbp = nfl.import_pbp_data(years=years)
        if isinstance(pbp, pd.DataFrame) and not pbp.empty:
            outp = d / f"pbp_{season}.parquet"
            try:
                pbp.to_parquet(outp, index=False)
            except Exception:
                outp = d / f"pbp_{season}.csv"
                pbp.to_csv(outp, index=False)
            print(f"[nflverse] wrote {outp} rows={len(pbp)}")
    except Exception as e:
        print(f"[nflverse] pbp failed: {e}")
