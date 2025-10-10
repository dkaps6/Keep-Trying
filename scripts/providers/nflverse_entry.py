from __future__ import annotations
from pathlib import Path
import pandas as pd
def fetch(season: int, date: str | None = None) -> None:
    try:
        import nfl_data_py as nfl
    except Exception as e:
        print(f"[nflverse] nfl_data_py not available: {e}"); return
    Path("data").mkdir(parents=True, exist_ok=True)
    try:
        weekly = nfl.import_weekly_data(years=[season])
        if isinstance(weekly, pd.DataFrame) and not weekly.empty:
            weekly.to_csv("data/player_stats_week.csv", index=False)
            print(f"[nflverse] wrote data/player_stats_week.csv rows={len(weekly)}")
    except Exception as e:
        print(f"[nflverse] weekly failed: {e}")
    try:
        rost = nfl.import_rosters(years=[season])
        if isinstance(rost, pd.DataFrame) and not rost.empty:
            rost.to_csv("data/rosters.csv", index=False)
            print(f"[nflverse] wrote data/rosters.csv rows={len(rost)}")
    except Exception as e:
        print(f"[nflverse] rosters failed: {e}")
    try:
        pbp = nfl.import_pbp_data(years=[season])
        if isinstance(pbp, pd.DataFrame) and not pbp.empty:
            outp = Path("data")/f"pbp_{season}.parquet"
            try: pbp.to_parquet(outp, index=False)
            except Exception: outp = Path("data")/f"pbp_{season}.csv"; pbp.to_csv(outp, index=False)
            print(f"[nflverse] wrote {outp} rows={len(pbp)}")
    except Exception as e:
        print(f"[nflverse] pbp failed: {e}")
