from __future__ import annotations
import pandas as pd, numpy as np

# sportsdataverse-py exposes ESPN helpers; import guarded so the provider
# simply returns empty when library/endpoints are unavailable
try:
    from sportsdataverse.nfl import load_espn_nfl_player_game
except Exception as _:
    load_espn_nfl_player_game = None

from .base import WeeklyProvider, coerce_weekly_schema, empty_weekly_df

class ESPNProvider(WeeklyProvider):
    name = "espn"

    def fetch_weekly(self, season: int) -> pd.DataFrame:
        if load_espn_nfl_player_game is None:
            print("ℹ️  sportsdataverse not installed/available; ESPN provider returns empty.")
            return empty_weekly_df()
        try:
            # Returns long table of player game stats for season
            df = load_espn_nfl_player_game(seasons=[season])
        except Exception as e:
            print(f"⚠️  ESPN weekly failed for {season}: {e}")
            return empty_weekly_df()
        if df is None or df.empty:
            return empty_weekly_df()

        # Normalize to our schema (best-effort mapping)
        # Column names may evolve; use soft mapping and fill defaults.
        # We create rolling last-4 features per player.
        df = df.rename(columns={
            "athlete_id":"gsis_id",
            "athlete_display_name":"player_name",
            "team_abbreviation":"recent_team",
            "athlete_position_abbreviation":"position",
            "week":"week"
        })

        needed = ["gsis_id","player_name","recent_team","position","week"]
        for c in needed:
            if c not in df.columns:
                df[c] = None
        df = df.dropna(subset=["gsis_id"]).copy()
        df["week"] = pd.to_numeric(df["week"], errors="coerce")

        # Basic stat columns (if present)
        rec        = pd.to_numeric(df.get("receptions"), errors="coerce").fillna(0)
        rec_yds    = pd.to_numeric(df.get("receiving_yards"), errors="coerce").fillna(0)
        rush_att   = pd.to_numeric(df.get("rushing_attempts"), errors="coerce").fillna(0)
        rush_yds   = pd.to_numeric(df.get("rushing_yards"), errors="coerce").fillna(0)
        pass_att   = pd.to_numeric(df.get("passing_attempts"), errors="coerce").fillna(0)
        pass_yds   = pd.to_numeric(df.get("passing_yards"), errors="coerce").fillna(0)
        rz_tgts    = pd.to_numeric(df.get("red_zone_targets"), errors="coerce").fillna(0)
        targets    = pd.to_numeric(df.get("targets"), errors="coerce").fillna(0)

        df["_rec"] = rec
        df["_rec_yds"] = rec_yds
        df["_rush_att"] = rush_att
        df["_rush_yds"] = rush_yds
        df["_pass_att"] = pass_att
        df["_pass_yds"] = pass_yds
        df["_rz_tgts"]  = rz_tgts
        df["_targets"]  = targets

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
                                  grp["_targets"].transform(l4).replace(0, np.nan)).fillna(0.0)

        return coerce_weekly_schema(out)
