#!/usr/bin/env python3
# scripts/build_data_pack.py
import argparse, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXT  = ROOT / "external" / "nflverse_bundle" / "outputs"
DATA = ROOT / "data"
OUTS = ROOT / "outputs"

def _read_csv(p: Path) -> pd.DataFrame:
    try:
        if p.exists() and p.stat().st_size > 0:
            return pd.read_csv(p)
    except Exception:
        try:
            return pd.read_csv(p, engine="python")
        except Exception:
            pass
    return pd.DataFrame()

def _write_csv(df: pd.DataFrame, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(df, pd.DataFrame) and not df.empty:
        df.to_csv(p, index=False)
        print(f"[write] {p.relative_to(ROOT)} rows={len(df)}")
    else:
        # write header-only if we know columns, else empty file
        if isinstance(df, pd.DataFrame):
            df.head(0).to_csv(p, index=False)
        else:
            p.write_text("")
        print(f"[write] {p.relative_to(ROOT)} rows=0 (empty)")

def build(season: int):
    # ---- WEATHER (per-event) ----
    # Prefer addon/weather if present; else leave header only
    wx_paths = [
        EXT / "weather" / f"weather_{season}.csv",
        ROOT / "external" / "nflverse_bundle" / "outputs" / "proe" / f"weather_{season}.csv",  # just in case
    ]
    wx = pd.DataFrame(columns=["event_id","wind_mph","temp_f","precip"])
    for p in wx_paths:
        df = _read_csv(p)
        if not df.empty:
            # normalize columns
            colmap = {}
            for c in df.columns:
                lc = c.lower()
                if lc in ("game_id","event_id","gid"): colmap[c] = "event_id"
                elif "wind" in lc: colmap[c] = "wind_mph"
                elif "temp" in lc: colmap[c] = "temp_f"
                elif "precip" in lc or "weather" in lc: colmap[c] = "precip"
            if colmap:
                df = df.rename(columns=colmap)
            wx = df[["event_id","wind_mph","temp_f","precip"]].drop_duplicates()
            break
    _write_csv(wx, DATA / "weather.csv")

    # ---- ROLES ---- (from addon derive_roles)
    roles = _read_csv(ROOT / "external" / "nflverse_bundle" / "outputs" / "roles" / f"roles_{season}.csv")
    # expected: player, team, position[, WR1/WR2/SLOT/RB1/TE1 role tag]
    if not roles.empty:
        role_cols = [c for c in roles.columns if c.lower() in {"player","team","position","role"}]
        if "role" not in {c.lower() for c in role_cols}:
            # synthesize a best-effort role column if separate flags exist
            role_guess = []
            for _,r in roles.iterrows():
                for k in ("WR1","WR2","SLOT","RB1","TE1"):
                    if k in roles.columns and r.get(k):
                        role_guess.append(k)
                        break
                else:
                    role_guess.append("")
            roles["role"] = role_guess
        roles = roles.rename(columns={"Player":"player","Team":"team","Position":"position"})
        roles = roles[["player","team","position","role"]]
    _write_csv(roles, DATA / "roles.csv")

    # ---- INJURIES ---- (prefer ESPN addon if present; else nflverse injuries)
    inj_pref = [
        ROOT / "external" / "nflverse_bundle" / "outputs" / "injuries_espn" / f"injuries_espn_{season}.csv",
        ROOT / "external" / "nflverse_bundle" / "outputs" / "injuries"      / f"injuries_{season}.csv",
    ]
    inj = pd.DataFrame(columns=["player","team","status"])
    for p in inj_pref:
        df = _read_csv(p)
        if not df.empty:
            # flexible rename
            cmap = {}
            for c in df.columns:
                lc = c.lower()
                if lc.startswith("player"): cmap[c] = "player"
                elif lc.startswith("team"): cmap[c] = "team"
                elif "status" in lc or "designation" in lc: cmap[c] = "status"
            df = df.rename(columns=cmap)
            have = [c for c in ("player","team","status") if c in df.columns]
            inj = df[have]
            break
    _write_csv(inj, DATA / "injuries.csv")

    # ---- COVERAGE TAGS ---- (from your coverage addon if produced)
    cov = _read_csv(ROOT / "external" / "nflverse_bundle" / "outputs" / "coverage" / f"coverage_{season}.csv")
    # expected: defense_team, tag
    _write_csv(cov, DATA / "coverage.csv")

    # ---- CB ASSIGNMENTS ----
    cb = _read_csv(ROOT / "external" / "nflverse_bundle" / "outputs" / "cb_assignments" / f"cb_assignments_{season}.csv")
    # expected: defense_team, receiver, cb, quality/penalty
    _write_csv(cb, DATA / "cb_assignments.csv")

    # ---- GAME LINES (home/away win prob from H2H) ----
    # If you already write this somewhere, map it here. Otherwise emit a header-only file.
    gl_src = _read_csv(OUTS / "game_lines.csv")  # if already created by some step
    if gl_src.empty:
        gl_src = pd.DataFrame(columns=["event_id","home_team","away_team","home_wp","away_wp"])
    _write_csv(gl_src, OUTS / "game_lines.csv")

    # ---- RAW PROPS (books scrape) ----
    pr = _read_csv(OUTS / "props_raw.csv")
    if pr.empty:
        pr = pd.DataFrame(columns=["event_id","player","market","line","over_odds","under_odds","book","ts"])
    _write_csv(pr, OUTS / "props_raw.csv")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    args = ap.parse_args()
    build(args.season)

if __name__ == "__main__":
    sys.exit(main())
