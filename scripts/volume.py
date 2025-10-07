# scripts/volume.py
from __future__ import annotations
import pandas as pd
from pathlib import Path

TEAM_FORM_PREFERRED = Path("outputs/metrics/team_form.csv")

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
        df = _safe_read(Path("data/team_form.csv"))
    if df.empty:
        return pd.DataFrame()
    for c in ["team","pace","proe"]:
        if c not in df.columns:
            df[c] = 0.0 if c != "team" else ""
    df["team"] = df["team"].astype(str).str.upper()
    return df

def estimate_plays(team_form: pd.DataFrame) -> pd.DataFrame:
    """
    Conservative plays model:
      plays_est = base_plays * (1 + 0.5*(Z(pace_off) + Z(pace_def))) if you supply both.
    If only single 'pace' is present (bundle), we z-score within league and use that.
    """
    if team_form.empty:
        return pd.DataFrame()

    df = team_form.copy()
    # Use 'pace' column directly; make a league z-score
    if "pace" not in df.columns:
        df["pace"] = 0.0
    mean = float(df["pace"].mean()) if len(df) else 0.0
    std  = float(df["pace"].std()) if len(df) else 1.0
    if std == 0: std = 1.0
    df["pace_z"] = (df["pace"] - mean) / std

    base_plays = 120.0  # combined snaps per game baseline (both teams)
    df["plays_est"] = base_plays * (1.0 + 0.5 * df["pace_z"].clip(-2, 2))

    # PROE tilt guides pass/run split later; keep it attached
    if "proe" not in df.columns:
        df["proe"] = 0.0
    return df[["team","pace","pace_z","proe","plays_est"]]

def build_volume_features() -> pd.DataFrame:
    team = load_team_form()
    if team.empty:
        return pd.DataFrame()
    return estimate_plays(team)

def main():
    out = build_volume_features()
    if out.empty:
        print("[volume] No team form → no volume features.")
        return
    Path("data").mkdir(parents=True, exist_ok=True)
    out.to_csv("data/volume_features.csv", index=False)
    print(f"[volume] ✅ wrote {len(out)} rows → data/volume_features.csv")

if __name__ == "__main__":
    main()
