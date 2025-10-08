#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root (adjust if needed)

def write_headers(path: Path, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        pd.DataFrame(columns=columns).to_csv(path, index=False)
        print(f"[bootstrap] wrote headers -> {path}")
    else:
        try:
            # ensure readable; if unreadable, overwrite with headers
            _ = pd.read_csv(path, nrows=1)
        except Exception:
            pd.DataFrame(columns=columns).to_csv(path, index=False)
            print(f"[bootstrap] fixed unreadable -> {path}")

# --- data/ (static / slow-changing inputs you may curate or fetch later)
write_headers(ROOT / "data" / "weather.csv",
              ["event_id","wind_mph","temp_f","precip","altitude_ft"])

write_headers(ROOT / "data" / "roles.csv",
              ["player","team","position","target_share","rush_share",
               "rz_tgt_share","rz_carry_share","yprr_proxy","ypc","ypt","qb_ypa"])

write_headers(ROOT / "data" / "injuries.csv",
              ["player","team","status"])

write_headers(ROOT / "data" / "coverage.csv",
              ["defense_team","tag"])

write_headers(ROOT / "data" / "cb_assignments.csv",
              ["defense_team","receiver","cb","quality","penalty"])

# --- outputs/ produced by fetchers; stub if upstream missing right now
write_headers(ROOT / "outputs" / "game_lines.csv",
              ["event_id","home_team","away_team","home_wp","away_wp"])

write_headers(ROOT / "outputs" / "props_raw.csv",
              ["book","market","player","team","line","over_odds","under_odds",
               "event_id","timestamp"])
