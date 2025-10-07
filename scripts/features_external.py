# scripts/features_external.py
from __future__ import annotations
from pathlib import Path
from typing import List
import pandas as pd

TEAM_FORM_PREFERRED = Path("outputs/metrics/team_form.csv")
PLAYER_FORM_PREFERRED = Path("outputs/metrics/player_form.csv")

DATA_DIR = Path("data")
METRICS_DIR = Path("outputs/metrics")

# Optional context files you already keep
COVERAGE = DATA_DIR / "coverage.csv"            # defense_team, tag (top_shadow/heavy_man/heavy_zone)
CB_ASSIGN = DATA_DIR / "cb_assignments.csv"     # defense_team, receiver, cb, penalty or quality
INJURIES = DATA_DIR / "injuries.csv"            # player, team, status
ROLES    = DATA_DIR / "roles.csv"               # player, team, role
GAME_LINES = Path("outputs/game_lines.csv")     # event_id, home_team, away_team, home_wp, away_wp
WEATHER  = DATA_DIR / "weather.csv"             # event_id, wind_mph, temp_f, precip, altitude_ft, dome

def _safe_read(p: Path) -> pd.DataFrame:
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            pass
    return pd.DataFrame()

def load_team_form() -> pd.DataFrame:
    df = _safe_read(TEAM_FORM_PREFERRED)
    if df.empty:
        df = _safe_read(DATA_DIR / "team_form.csv")
    if df.empty:
        return pd.DataFrame()
    # Ensure minimal columns exist
    req = ["team","def_pass_epa","def_rush_epa","def_sack_rate","pace","proe","light_box_rate","heavy_box_rate"]
    for c in req:
        if c not in df.columns:
            df[c] = 0.0
    df["team"] = df["team"].astype(str).str.upper()
    return df

def load_player_form() -> pd.DataFrame:
    df = _safe_read(PLAYER_FORM_PREFERRED)
    if df.empty:
        df = _safe_read(DATA_DIR / "player_form.csv")
    if df.empty:
        return pd.DataFrame()
    req = ["player","team","position","target_share","rush_share",
           "rz_tgt_share","rz_carry_share","yprr_proxy","ypc","ypt","qb_ypa"]
    for c in req:
        if c not in df.columns:
            df[c] = 0.0
    df["team"] = df["team"].astype(str).str.upper()
    return df[req]

def load_context_extras() -> dict[str, pd.DataFrame]:
    return {
        "coverage": _safe_read(COVERAGE),
        "cb":       _safe_read(CB_ASSIGN),
        "inj":      _safe_read(INJURIES),
        "roles":    _safe_read(ROLES),
        "lines":    _safe_read(GAME_LINES),
        "wx":       _safe_read(WEATHER),
    }

def build_feature_frame() -> pd.DataFrame:
    team = load_team_form()
    players = load_player_form()
    if team.empty or players.empty:
        return pd.DataFrame()

    ctx = load_context_extras()

    # Coverage tags (defense_team, tag)
    cov = ctx["coverage"]
    if not cov.empty:
        cov = cov.groupby("defense_team")["tag"].apply(lambda s: ",".join(sorted(set(str(x) for x in s)))).reset_index()
        team = team.merge(cov.rename(columns={"defense_team":"team","tag":"coverage_tags"}), on="team", how="left")
    else:
        team["coverage_tags"] = ""

    # CB shadow penalties (defense_team, receiver, penalty)
    cb = ctx["cb"]
    if not cb.empty:
        # keep raw; pricing merges per receiver later if needed
        pass

    # Injuries / Roles (merged to players for redistribution logic)
    inj = ctx["inj"]
    if not inj.empty:
        inj["status"] = inj["status"].astype(str).str.title()
    roles = ctx["roles"]

    out = players.copy()
    out = out.merge(roles, on=["player","team"], how="left", suffixes=("",""))
    out = out.merge(inj[["player","team","status"]] if not inj.empty else pd.DataFrame(columns=["player","team","status"]),
                    on=["player","team"], how="left")

    # Attach team-level context to each player (for opponent mods later you’ll join per-game)
    out = out.merge(team, on="team", how="left", suffixes=("",""))

    # Weather & lines are joined later per event_id in pricing/props merge; keep them separate
    return out

def main():
    out = build_feature_frame()
    if out.empty:
        print("[features_external] No features (missing team or player form).")
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(DATA_DIR / "features_external.csv", index=False)
    print(f"[features_external] ✅ wrote {len(out)} rows → data/features_external.csv")

if __name__ == "__main__":
    main()
