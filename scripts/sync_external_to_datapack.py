#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def write_if_nonempty(df: pd.DataFrame, path: Path, cols):
    ensure_dir(path)
    if isinstance(df, pd.DataFrame) and not df.empty:
        # keep only expected columns if present
        keep = [c for c in cols if c in df.columns]
        if keep:
            df = df[keep]
        df.to_csv(path, index=False)
        print(f"[sync] wrote {len(df)} rows -> {path}")
        return True
    return False

def main():
    # 1) Injuries -> data/injuries.csv
    # Try ESPN injuries addon first
    espn_inj = ROOT / "external" / "nflverse_bundle" / "outputs" / "injuries" / "injuries_2025.csv"
    msf_inj  = ROOT / "external" / "nflverse_bundle" / "outputs" / "injuries" / "injuries_msf_2025.csv"
    out_inj  = ROOT / "data" / "injuries.csv"
    inj_cols = ["player","team","status"]

    for cand in [espn_inj, msf_inj]:
        if cand.exists() and cand.stat().st_size > 0:
            try:
                df = pd.read_csv(cand)
                # Try to normalize column names
                rename = {
                    "player_name":"player",
                    "club":"team",
                    "team_abbr":"team",
                    "injury_status":"status",
                    "game_status":"status",
                    "status_desc":"status",
                }
                df = df.rename(columns=rename)
                if write_if_nonempty(df, out_inj, inj_cols):
                    break
            except Exception as e:
                print(f"[sync] WARN: could not read {cand}: {e}")
    else:
        # leave as header-only if nothing available
        pd.DataFrame(columns=inj_cols).to_csv(out_inj, index=False)
        print(f"[sync] no injuries source found -> header only {out_inj}")

    # 2) Roles -> data/roles.csv
    roles_src = ROOT / "external" / "nflverse_bundle" / "outputs" / "roles" / "roles_2025.csv"
    out_roles = ROOT / "data" / "roles.csv"
    role_cols = ["player","team","role"]
    if roles_src.exists() and roles_src.stat().st_size > 0:
        try:
            df = pd.read_csv(roles_src)
            rename = {"pos_role":"role", "position_role":"role"}
            df = df.rename(columns=rename)
            write_if_nonempty(df, out_roles, role_cols)
        except Exception as e:
            print(f"[sync] WARN reading roles: {e}")
            pd.DataFrame(columns=role_cols).to_csv(out_roles, index=False)
    else:
        pd.DataFrame(columns=role_cols).to_csv(out_roles, index=False)

    # 3) Coverage tags -> data/coverage.csv  (often static/manual)
    cov_src = ROOT / "external" / "nflverse_bundle" / "outputs" / "coverage" / "coverage_2025.csv"
    out_cov = ROOT / "data" / "coverage.csv"
    cov_cols = ["defense_team","tag"]
    if cov_src.exists() and cov_src.stat().st_size > 0:
        try:
            df = pd.read_csv(cov_src)
            write_if_nonempty(df, out_cov, cov_cols)
        except Exception as e:
            print(f"[sync] WARN reading coverage: {e}")
            pd.DataFrame(columns=cov_cols).to_csv(out_cov, index=False)
    else:
        # Leave header-only if you curate this manually
        pd.DataFrame(columns=cov_cols).to_csv(out_cov, index=False)

    # 4) CB assignments -> data/cb_assignments.csv  (often curated)
    cba_src = ROOT / "external" / "nflverse_bundle" / "outputs" / "coverage" / "cb_assignments_2025.csv"
    out_cba = ROOT / "data" / "cb_assignments.csv"
    cba_cols = ["defense_team","receiver"]
    if cba_src.exists() and cba_src.stat().st_size > 0:
        try:
            df = pd.read_csv(cba_src)
            write_if_nonempty(df, out_cba, cba_cols)
        except Exception as e:
            print(f"[sync] WARN reading cb_assignments: {e}")
            pd.DataFrame(columns=cba_cols).to_csv(out_cba, index=False)
    else:
        pd.DataFrame(columns=cba_cols).to_csv(out_cba, index=False)

    # 5) Weather -> data/weather.csv  (from your weather script, if it ran)
    wx_src1 = ROOT / "metrics" / "weather.csv"
    wx_src2 = ROOT / "data" / "weather_live.csv"
    out_wx  = ROOT / "data" / "weather.csv"
    wx_cols = ["event_id","wind_mph","temp_f","precip"]
    for cand in [wx_src1, wx_src2]:
        if cand.exists() and cand.stat().st_size > 0:
            try:
                df = pd.read_csv(cand)
                rename = {"wind":"wind_mph","temperature_f":"temp_f","conditions":"precip"}
                df = df.rename(columns=rename)
                if write_if_nonempty(df, out_wx, wx_cols):
                    break
            except Exception as e:
                print(f"[sync] WARN reading weather: {e}")
    else:
        pd.DataFrame(columns=wx_cols).to_csv(out_wx, index=False)

    # 6) Game lines -> outputs/game_lines.csv
    gl_src = ROOT / "external" / "nflverse_bundle" / "outputs" / "odds" / "game_lines_2025.csv"
    out_gl = ROOT / "outputs" / "game_lines.csv"
    gl_cols = ["event_id","home_team","away_team","home_wp","away_wp","commence_time"]
    if gl_src.exists() and gl_src.stat().st_size > 0:
        try:
            df = pd.read_csv(gl_src)
            rename = {
                "home_win_prob":"home_wp",
                "away_win_prob":"away_wp",
                "start_time":"commence_time",
            }
            df = df.rename(columns=rename)
            write_if_nonempty(df, out_gl, gl_cols)
        except Exception as e:
            print(f"[sync] WARN reading game_lines: {e}")
            pd.DataFrame(columns=gl_cols).to_csv(out_gl, index=False)
    else:
        pd.DataFrame(columns=gl_cols).to_csv(out_gl, index=False)

    # 7) Raw props -> outputs/props_raw.csv (only if you run a props fetcher)
    pr_src = ROOT / "external" / "nflverse_bundle" / "outputs" / "odds" / "props_raw_2025.csv"
    out_pr = ROOT / "outputs" / "props_raw.csv"
    pr_cols = ["id","commence_time","home_team","away_team",
               "bookmaker_key","bookmaker_title",
               "market_api","market_internal","player","side","line","price"]
    if pr_src.exists() and pr_src.stat().st_size > 0:
        try:
            df = pd.read_csv(pr_src)
            write_if_nonempty(df, out_pr, pr_cols)
        except Exception as e:
            print(f"[sync] WARN reading props_raw: {e}")
            pd.DataFrame(columns=pr_cols).to_csv(out_pr, index=False)
    else:
        pd.DataFrame(columns=pr_cols).to_csv(out_pr, index=False)

    print("[sync] done.")

if __name__ == "__main__":
    main()
