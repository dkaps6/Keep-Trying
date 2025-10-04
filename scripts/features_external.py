# scripts/features_external.py
import os
import pandas as pd

REQS = {
    "team_form": {"team","pressure_z","pass_epa_z","proe","light_box_share","heavy_box_share","man_rate_z","zone_rate_z"},
    "player_form": {"gsis_id","week","recent_team","position",
                    "rec_l4","rec_yds_l4","pass_att_l4","pass_yds_l4","ra_l4","ry_l4","rz_tgt_share_l4"},
    "ids": {"player_name","gsis_id","recent_team","position"},
    # weather is optional
}

def _safe_read(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"⚠️  Failed to read {path}: {e}")
    else:
        print(f"ℹ️  Missing file (ok if not used): {path}")
    return pd.DataFrame()

def _check(df: pd.DataFrame, need: set[str], name: str):
    if df.empty:
        print(f"⚠️  {name} is empty.")
        return
    missing = need - set(df.columns)
    if missing:
        print(f"⚠️  {name} missing columns: {sorted(missing)}")
    else:
        print(f"✅ {name} columns OK ({len(df)} rows).")

def build_external(season: int):
    ids         = _safe_read("metrics/id_map.csv")
    player_form = _safe_read("metrics/player_form.csv")
    team_form   = _safe_read("metrics/team_form.csv")
    weather     = _safe_read("inputs/weather.csv")  # optional

    _check(ids, REQS["ids"], "id_map.csv")
    _check(player_form, REQS["player_form"], "player_form.csv")
    _check(team_form, REQS["team_form"], "team_form.csv")

    if not team_form.empty and "team" in team_form.columns:
        team_form = team_form.set_index("team")

    # quick peek (first 3 rows) so you can see sample values in the Actions log
    for name, df in [("ids", ids), ("player_form", player_form), ("team_form", team_form.reset_index() if not team_form.empty else team_form)]:
        if not df.empty:
            print(f"── {name} head() ──")
            print(df.head(3).to_string(index=False))

    return {"ids": ids, "player_form": player_form, "team_form": team_form, "weather": weather}
