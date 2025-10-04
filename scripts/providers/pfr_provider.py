from __future__ import annotations
import os
import pandas as pd
from .base import WeeklyProvider, coerce_weekly_schema, empty_weekly_df

class PFRProvider(WeeklyProvider):
    """
    Lightweight fallback that reads a prepared CSV if you provide it:
      inputs/pfr_player_weekly_{season}.csv

    Columns expected (best-effort): gsis_id/player_id, player_name, team, position, week,
    receptions, receiving_yards, rushing_att, rushing_yards, passing_att, passing_yards,
    redzone_targets, targets

    You can export these via tools or your own scraper and drop into inputs/.
    """
    name = "pfr_manual"

    def fetch_weekly(self, season: int) -> pd.DataFrame:
        path = f"inputs/pfr_player_weekly_{season}.csv"
        if not os.path.exists(path):
            print(f"ℹ️  No {path} found; PFR provider returns empty.")
            return empty_weekly_df()
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"⚠️  Failed to read {path}: {e}")
            return empty_weekly_df()
        if df is None or df.empty:
            return empty_weekly_df()

        # Try to normalize column names
        df = df.rename(columns={
            "player_id":"gsis_id",
            "team":"recent_team",
            "pos":"position"
        })
        for c in ["gsis_id","player_name","recent_team","position","week"]:
            if c not in df.columns:
                df[c] = None

        # Numeric conversions for rolling
        for src, dst in [
            ("receptions","_rec"),
            ("receiving_yards","_rec_yds"),
            ("rushing_att","_rush_att"),
            ("rushing_yards","_rush_yds"),
            ("passing_att","_pass_att"),
            ("passing_yards","_pass_yds"),
            ("redzone_targets","_rz_tgts"),
            ("targets","_targets")
        ]:
            df[dst] = pd.to_numeric(df.get(src), errors="coerce").fillna(0)

        df = df.sort_values(["gsis_id","week"])
        grp = df.groupby("gsis_id", as_index=False)
        def l4(s): return s.rolling(4, min_periods=1).sum()

        out = df[["gsis_id","player_name","recent_team","position","week"]].copy()
        out["rec_l4"]       = grp["_rec"].transform(l4)
        out["rec_yds_l4"]   = grp["_rec_yds"].transform(l4)
        out["ra_l4"]        = grp["_rush_att"].transform(l4)
        out["ry_l4"]        = grp["_rush_yds"].transform(l4)
        out["pass_att_l4"]  = grp["_pass_att"].transform(l4)
        out["pass_yds_l4"]  = grp["_pass_yds"].transform(l4)
        out["rz_tgt_share_l4"] = (grp["_rz_tgts"].transform(l4) /
                                  grp["_targets"].transform(l4).replace(0, 0.0)).fillna(0.0)

        return coerce_weekly_schema(out)
