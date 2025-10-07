# scripts/build_player_form.py
from __future__ import annotations
import argparse, warnings, time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import requests
import warnings

# Optional primaries
try:
    import nflreadpy as nrp
    HAS_NFLREADPY = True
except Exception:
    HAS_NFLREADPY = False
try:
    import nfl_data_py as nfl
    HAS_NFL_DATA_PY = True
except Exception:
    HAS_NFL_DATA_PY = False

# Fallbacks already present in your repo
from .sources.apisports import season_team_player_tables as apisports_tables
from .sources.mysportsfeeds import season_team_player_tables as msf_tables

# Bundle outputs we prefer
BUNDLE_PLAYER_FORM = Path("outputs/metrics/player_form.csv")

# ESPN (light fallback when nflverse isn’t present for current year)
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/v2/sports/football/nfl/scoreboard"
ESPN_BOX = "https://site.api.espn.com/apis/v2/sports/football/nfl/boxscore"


# -----------------------------
# Shared helpers
# -----------------------------
def _to_pandas(obj):
    try:
        import polars as pl  # type: ignore
        if isinstance(obj, pl.DataFrame):
            return obj.to_pandas()
    except Exception:
        pass
    return obj


def _http_json(url: str, params=None, tries=3, backoff=0.6):
    params = params or {}
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except Exception:
            pass
        time.sleep(backoff * (2 ** i))
    return None


# -----------------------------
# nflverse loaders
# -----------------------------
def _load_pbp(season: int) -> pd.DataFrame:
    if HAS_NFLREADPY:
        try:
            print(f"[build_player_form] USING PBP (nflreadpy) {season}")
            df = nrp.load_pbp(seasons=[season])
            return _to_pandas(df)
        except Exception as e:
            warnings.warn(f"nflreadpy.load_pbp failed: {type(e).__name__}: {e}")
    if HAS_NFL_DATA_PY:
        try:
            print(f"[build_player_form] FALLBACK PBP (nfl_data_py) {season}")
            df = nfl.import_pbp_data([season])
            return _to_pandas(df)
        except Exception as e:
            warnings.warn(f"nfl_data_py.import_pbp_data failed: {type(e).__name__}: {e}")
    return pd.DataFrame()


def _load_weekly(season: int) -> pd.DataFrame:
    if HAS_NFLREADPY:
        try:
            print(f"[build_player_form] USING WEEKLY (nflreadpy) {season}")
            df = nrp.load_player_stats(seasons=[season])
            return _to_pandas(df)
        except Exception as e:
            warnings.warn(f"nflreadpy.load_player_stats failed: {type(e).__name__}: {e}")
    if HAS_NFL_DATA_PY:
        try:
            print(f"[build_player_form] FALLBACK WEEKLY (nfl_data_py) {season}")
            df = nfl.import_weekly_data([season])
            return _to_pandas(df)
        except Exception as e:
            warnings.warn(f"nfl_data_py.import_weekly_data failed: {type(e).__name__}: {e}")
    return pd.DataFrame()


# -----------------------------
# ESPN lightweight fallback
# -----------------------------
def _espn_week_events(season: int, week: int) -> List[str]:
    data = _http_json(ESPN_SCOREBOARD, params={"week": week, "seasontype": 2, "dates": season})
    if not data:
        return []
    return [e.get("id") for e in data.get("events", []) if e.get("id")]

def _espn_box_players(event_id: str) -> Dict[str, Any] | None:
    return _http_json(ESPN_BOX, params={"event": event_id})

def _parse_box_to_rows(event_id: str, box: Dict[str, Any] | None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not box:
        return pd.DataFrame(), pd.DataFrame()
    teams: Dict[str, Dict[str, float]] = {}
    players_rows: List[Dict[str, Any]] = []
    for t in box.get("teams", []):
        abbr = (t.get("team", {}) or {}).get("abbreviation") or (t.get("team", {}) or {}).get("displayName")
        teams[abbr] = {"pass_att": 0.0, "rush_att": 0.0}
        for grp in t.get("players", []):
            for pl in grp.get("athletes", []):
                name = pl.get("athlete", {}).get("displayName") or pl.get("athlete", {}).get("shortName")
                row = {"team": abbr, "player": name,
                       "targets": 0.0, "receptions": 0.0, "carries": 0.0,
                       "pass_yards": 0.0, "rush_yards": 0.0, "rec_yards": 0.0, "attempts": 0.0}
                for s in pl.get("stats", []):
                    nm = (s.get("name") or "").lower()
                    st = s.get("statistics") or {}
                    if nm == "passing":
                        row["attempts"] += float(st.get("attempts") or 0)
                        row["pass_yards"] += float(st.get("yards") or 0)
                        teams[abbr]["pass_att"] += float(st.get("attempts") or 0)
                    elif nm == "rushing":
                        carr = float(st.get("attempts") or st.get("carries") or 0)
                        row["carries"] += carr
                        row["rush_yards"] += float(st.get("yards") or 0)
                        teams[abbr]["rush_att"] += carr
                    elif nm == "receiving":
                        row["targets"] += float(st.get("targets") or 0)
                        row["receptions"] += float(st.get("receptions") or 0)
                        row["rec_yards"] += float(st.get("yards") or 0)
                players_rows.append(row)
    team_rows = [{"team": abbr, "pass_att": v["pass_att"], "rush_att": v["rush_att"], "plays": v["pass_att"]+v["rush_att"], "event_id": event_id}
                 for abbr, v in teams.items()]
    return pd.DataFrame(team_rows), pd.DataFrame(players_rows)

def _espn_players_table(season: int) -> pd.DataFrame:
    rows = []
    # scan weeks conservatively
    for wk in range(1, 23):
        evts = _espn_week_events(season, wk)
        if not evts and wk > 3:
            break
        for eid in evts:
            _, pdf = _parse_box_to_rows(eid, _espn_box_players(eid))
            if not pdf.empty:
                pdf["week"] = wk
                rows.append(pdf)
            time.sleep(0.08)
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)

    team_tot = (df.groupby("team")
                .agg(team_targets=("targets", "sum"),
                     team_rushes=("carries", "sum"),
                     team_attempts=("attempts", "sum")).reset_index())
    out = df.groupby(["team","player"], as_index=False).sum(numeric_only=True)
    out = out.merge(team_tot, on="team", how="left")
    out["target_share"] = out["targets"] / out["team_targets"].clip(lower=1)
    out["rush_share"]   = out["carries"] / out["team_rushes"].clip(lower=1)
    out["ypt"]          = out["rec_yards"] / out["targets"].clip(lower=1)
    out["ypc"]          = out["rush_yards"] / out["carries"].clip(lower=1)
    out["qb_ypa"]       = out["pass_yards"] / out["attempts"].clip(lower=1)
    out["yprr_proxy"]   = out["rec_yards"] / out["team_attempts"].clip(lower=1)
    out["rz_tgt_share"] = 0.0
    out["rz_carry_share"] = 0.0
    out["position"] = out.apply(lambda r: "QB" if (r.get("attempts",0) or 0) > 0 else ("RB" if (r.get("carries",0) or 0) > 0 else ("WR/TE" if (r.get("targets",0) or 0) > 0 else "")), axis=1)
    keep = ["player","team","position","target_share","rush_share","rz_tgt_share","rz_carry_share","yprr_proxy","ypc","ypt","qb_ypa"]
    for c in keep:
        if c not in out.columns: out[c] = 0.0
    return out[keep].fillna(0.0).sort_values(["team","player"]).reset_index(drop=True)


# -----------------------------
# Transforms from nflverse tables
# -----------------------------
def _from_pbp(pbp: pd.DataFrame, season: int) -> pd.DataFrame:
    cur = pbp.copy()
    if "season" in cur.columns:
        cur = cur[cur["season"] == season].copy()
    if cur.empty:
        return pd.DataFrame()

    cur["is_pass"] = (cur.get("pass", 0) == 1) if "pass" in cur.columns else cur.get("play_type","").astype(str).str.contains("pass", case=False, na=False)
    cur["is_rush"] = (cur.get("rush", 0) == 1) if "rush" in cur.columns else cur.get("play_type","").astype(str).str.contains("rush", case=False, na=False)
    cur["is_rz"]   = (cur.get("yardline_100", 100) <= 20)

    rec = (cur[cur["is_pass"]]
           .groupby(["posteam","receiver_player_name"], dropna=True)
           .agg(targets=("receiver_player_name","count"),
                rec_yards=("yards_gained","sum"),
                rz_targets=("is_rz","sum"))
           .reset_index().rename(columns={"posteam":"team","receiver_player_name":"player"}))
    team_attempts = (cur[cur["is_pass"]].groupby("posteam").size().rename("team_attempts")).reset_index().rename(columns={"posteam":"team"})
    rec = rec.merge(team_attempts, on="team", how="left")
    rec["target_share"] = rec["targets"] / rec["team_attempts"].clip(lower=1)
    rec["ypt"]          = rec["rec_yards"] / rec["targets"].clip(lower=1)
    rec["yprr_proxy"]   = rec["rec_yards"] / rec["team_attempts"].clip(lower=1)
    rec["rz_tgt_share"] = rec["rz_targets"] / rec["targets"].clip(lower=1)

    rush = (cur[cur["is_rush"]]
            .groupby(["posteam","rusher_player_name"], dropna=True)
            .agg(carries=("rusher_player_name","count"),
                 rush_yards=("yards_gained","sum"),
                 rz_carries=("is_rz","sum"))
            .reset_index().rename(columns={"posteam":"team","rusher_player_name":"player"}))
    team_rushes = (cur[cur["is_rush"]].groupby("posteam").size().rename("team_rushes")).reset_index().rename(columns={"posteam":"team"})
    rush = rush.merge(team_rushes, on="team", how="left")
    rush["rush_share"]   = rush["carries"] / rush["team_rushes"].clip(lower=1)
    rush["ypc"]          = rush["rush_yards"] / rush["carries"].clip(lower=1)
    rush["rz_carry_share"] = rush["rz_carries"] / rush["carries"].clip(lower=1)

    qb = (cur[cur["is_pass"]]
          .groupby(["posteam","passer_player_name"], dropna=True)
          .agg(pass_yards=("yards_gained","sum"),
               attempts=("is_pass","sum"))
          .reset_index().rename(columns={"posteam":"team","passer_player_name":"player"}))
    qb["qb_ypa"] = qb["pass_yards"] / qb["attempts"].clip(lower=1)

    out = pd.merge(rec[["player","team","target_share","rz_tgt_share","yprr_proxy","ypt"]],
                   rush[["player","team","rush_share","rz_carry_share","ypc"]],
                   on=["player","team"], how="outer")
    out = pd.merge(out, qb[["player","team","qb_ypa"]], on=["player","team"], how="outer")

    out["position"] = out.apply(lambda r:
        "QB" if pd.notna(r.get("qb_ypa"))
        else ("RB" if pd.notna(r.get("rush_share"))
              else ("WR/TE" if pd.notna(r.get("target_share")) else "")), axis=1)
    for c in ["target_share","rush_share","rz_tgt_share","rz_carry_share","yprr_proxy","ypc","ypt","qb_ypa"]:
        out[c] = out[c].fillna(0.0)
    return out.sort_values(["team","player"]).reset_index(drop=True)


def _from_weekly(wk: pd.DataFrame) -> pd.DataFrame:
    if wk is None or wk.empty:
        return pd.DataFrame()
    # normalize common alt names
    ren = {
        "pass_attempts":"attempts",
        "rush_attempts":"carries",
        "receiving_yards":"rec_yards",
        "rushing_yards":"rush_yards",
        "passing_yards":"pass_yards",
    }
    for a,b in ren.items():
        if a in wk.columns and b not in wk.columns:
            wk = wk.rename(columns={a:b})

    team_tot = (wk.groupby("team")
                  .agg(team_targets=("targets","sum"),
                       team_rushes=("carries","sum"),
                       team_attempts=("attempts","sum")).reset_index())
    rec = (wk.groupby(["team","player"])
             .agg(targets=("targets","sum"),
                  receptions=("receptions","sum"),
                  rec_yards=("rec_yards","sum"),
                  carries=("carries","sum"),
                  rush_yards=("rush_yards","sum"),
                  pass_yards=("pass_yards","sum"),
                  attempts=("attempts","sum")).reset_index())
    out = rec.merge(team_tot, on="team", how="left")
    out["target_share"] = out["targets"] / out["team_targets"].clip(lower=1)
    out["rush_share"]   = out["carries"] / out["team_rushes"].clip(lower=1)
    out["ypt"]          = out["rec_yards"] / out["targets"].clip(lower=1)
    out["ypc"]          = out["rush_yards"] / out["carries"].clip(lower=1)
    out["qb_ypa"]       = out["pass_yards"] / out["attempts"].clip(lower=1)
    out["yprr_proxy"]   = out["rec_yards"] / out["team_attempts"].clip(lower=1)
    out["rz_tgt_share"] = 0.0
    out["rz_carry_share"] = 0.0
    out["position"] = out.apply(lambda r:
        "QB" if (r.get("attempts",0) or 0) > 0 else
        ("RB" if (r.get("carries",0) or 0) > 0 else
         ("WR/TE" if (r.get("targets",0) or 0) > 0 else "")), axis=1)
    keep = ["player","team","position","target_share","rush_share","rz_tgt_share","rz_carry_share","yprr_proxy","ypc","ypt","qb_ypa"]
    for c in keep:
        if c not in out.columns: out[c] = 0.0
    return out[keep].fillna(0.0).sort_values(["team","player"]).reset_index(drop=True)


# -----------------------------
# Builder (with bundle preference)
# -----------------------------
# --- REPLACE your current helper with this ---
def _maybe_player_form_from_bundle() -> pd.DataFrame:
    """
    Try to read player_form from the external bundle.
    Returns an empty DataFrame if the file is missing, empty, or malformed.
    """
    p = BUNDLE_PLAYER_FORM
    need = ["player","team","position","target_share","rush_share",
            "rz_tgt_share","rz_carry_share","yprr_proxy","ypc","ypt","qb_ypa"]

    # Guard: file present and non-empty?
    if not p.exists():
        print(f"[build_player_form] bundle not found at {p} — skipping.")
        return pd.DataFrame(columns=need)
    try:
        if p.stat().st_size < 5:
            print(f"[build_player_form] bundle is empty at {p} — skipping.")
            return pd.DataFrame(columns=need)
    except Exception:
        pass

    # Read with robust options
    try:
        df = pd.read_csv(p)
    except pd.errors.EmptyDataError:
        print(f"[build_player_form] bundle {p} is empty — skipping.")
        return pd.DataFrame(columns=need)
    except Exception as e:
        warnings.warn(f"Failed reading {p}: {type(e).__name__}: {e}")
        return pd.DataFrame(columns=need)

    if df.empty:
        print(f"[build_player_form] bundle {p} produced 0 rows — skipping.")
        return pd.DataFrame(columns=need)

    # Minimal schema check; fill anything missing so downstream code is safe
    for c in need:
        if c not in df.columns:
            df[c] = 0.0
    if "team" in df.columns:
        df["team"] = df["team"].astype(str).str.upper()

    keep = need + [c for c in df.columns if c not in need and c in ("week","season")]
    return df[keep].reset_index(drop=True)

# --- In your builder, ensure this exact early branch exists ---
def build_player_form(season: int, history: str) -> pd.DataFrame:
    # 0) External bundle (preferred if present)
    pf = _maybe_player_form_from_bundle()
    if not pf.empty:
        print(f"[build_player_form] ✅ using bundled player_form ({len(pf)} rows)")
        return pf

    # 1) PBP (best)
    pbp = _load_pbp(season)
    pf  = _from_pbp(pbp, season)
    if not pf.empty:
        return pf

    # 2) Weekly (nflverse)
    wk = _load_weekly(season)
    pf = _from_weekly(wk)
    if not pf.empty:
        return pf

    # 3) ESPN fallback
    print("[build_player_form] ⚠️ Using ESPN fallback for player shares/efficiency")
    pf = _espn_players_table(season)
    if not pf.empty:
        return pf

    # 4) API-SPORTS fallback
    print("[build_player_form] ⚠️ Using API-SPORTS fallback for player shares")
    _, p = apisports_tables(season)
    pf = _from_weekly(p)
    if not pf.empty:
        return pf

    # 5) MySportsFeeds fallback
    print("[build_player_form] ⚠️ Using MySportsFeeds fallback for player shares")
    _, p = msf_tables(season)
    pf = _from_weekly(p)
    return pf


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--history", default="2019-2024")
    ap.add_argument("--write", default="data/player_form.csv")
    args = ap.parse_args()

    Path("data").mkdir(parents=True, exist_ok=True)
    df = build_player_form(args.season, args.history)
    df.to_csv(args.write, index=False)
    print(f"[build_player_form] ✅ wrote {len(df)} rows → {args.write}")


if __name__ == "__main__":
    main()
