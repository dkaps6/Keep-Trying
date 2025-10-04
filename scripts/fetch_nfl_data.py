def build_id_map(season: int) -> pd.DataFrame:
    """
    Basic player id map from rosters; resilient to nfl_data_py versions.

    Tries:
      1) nfl.import_rosters([season])  (newer nfl_data_py)
      2) nfl.import_players()          (fallback; filter to active/nearby seasons)

    Returns columns: player_name, gsis_id, recent_team, position
    """
    # Try modern rosters endpoint first
    roster_fn = getattr(nfl, "import_rosters", None)
    ros = pd.DataFrame()

    if callable(roster_fn):
        try:
            ros = roster_fn([season])
        except Exception as e:
            print(f"⚠️  import_rosters failed for {season}: {e}")

    # Fallback: import_players (wide table over many seasons)
    if ros is None or ros.empty:
        try:
            players = nfl.import_players()
        except Exception as e:
            print(f"⚠️  import_players failed: {e}")
            players = pd.DataFrame()

        if not players.empty:
            # Best-effort normalization: keep recent/active players with IDs
            keep_cols = {
                "player_name": "player_name",
                "full_name": "player_name",
                "gsis_id": "gsis_id",
                "gsisid": "gsis_id",
                "team": "recent_team",
                "recent_team": "recent_team",
                "position": "position",
                "pos": "position",
                "last_season": "last_season",
                "first_season": "first_season",
            }
            m = {c: keep_cols[c] for c in players.columns if c in keep_cols}
            pl = players.rename(columns=m)

            # Light filter to plausible actives around target season
            if "last_season" in pl.columns:
                pl = pl[pl["last_season"].fillna(0) >= (season - 2)]
            if "first_season" in pl.columns:
                pl = pl[pl["first_season"].fillna(season) <= season]

            for c in ["player_name", "gsis_id", "recent_team", "position"]:
                if c not in pl.columns:
                    pl[c] = None

            ros = pl[["player_name", "gsis_id", "recent_team", "position"]].dropna(subset=["player_name"]).drop_duplicates()

    # If still empty, return schema-correct empty
    if ros is None or ros.empty:
        print("ℹ️  id_map fallback produced empty output.")
        return pd.DataFrame(columns=["player_name", "gsis_id", "recent_team", "position"]).reset_index(drop=True)

    return ros[["player_name", "gsis_id", "recent_team", "position"]].drop_duplicates().reset_index(drop=True)
