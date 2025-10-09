#!/usr/bin/env python3
from __future__ import annotations

import sys, os, argparse
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import pandas as pd, numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent if ROOT.name == "nflverse_bundle" else ROOT.parent
OUT_BUNDLE = ROOT / "outputs"
OUT_METRICS = REPO / "outputs" / "metrics"
DATA_MIRROR = REPO / "data"

def _import_or_none(modname: str):
    try:
        return __import__(modname, fromlist=["*"])
    except Exception:
        return None

nflverse = _import_or_none("scripts.providers.nflverse")
espn  = _import_or_none("scripts.providers.espn_pbp")
msf   = _import_or_none("scripts.providers.msf")
apis  = _import_or_none("scripts.providers.apisports")
gsis  = _import_or_none("scripts.providers.nflgsis")

def _mkdirs():
    for p in (OUT_BUNDLE, OUT_METRICS, DATA_MIRROR):
        p.mkdir(parents=True, exist_ok=True)

def _ok(df: Optional[pd.DataFrame]) -> bool:
    try: return isinstance(df, pd.DataFrame) and not df.empty
    except Exception: return False

def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0: return pd.DataFrame()
    for eng in (None, "python"):
        try:
            return pd.read_csv(path) if not eng else pd.read_csv(path, engine=eng)
        except Exception:
            continue
    return pd.DataFrame()

def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if _ok(df):
        df.to_csv(path, index=False); print(f"[write] {path.relative_to(REPO)} rows={len(df)}")
    else:
        pd.DataFrame().to_csv(path, index=False); print(f"[write-empty] {path.relative_to(REPO)} rows=0")

def _z(s: pd.Series) -> pd.Series:
    arr = pd.to_numeric(s, errors="coerce").values
    mu, sd = np.nanmean(arr), np.nanstd(arr)
    if not np.isfinite(sd) or sd == 0: return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd

PRIMARY_PATHS: Dict[str, Callable[[int], List[Path]]] = {
    "pbp":              lambda s: [OUT_BUNDLE / "pbp" / f"pbp_{s}_{s}.csv"],
    "injuries":         lambda s: [OUT_BUNDLE / "injuries" / f"injuries_{s}.csv"],
    "schedules":        lambda s: [OUT_BUNDLE / "schedules" / f"schedules_{s}.csv"],
    "rosters":          lambda s: [OUT_BUNDLE / "rosters" / f"rosters_{s}.csv"],
    "depth_charts":     lambda s: [OUT_BUNDLE / "depth_charts" / f"depth_charts_{s}.csv"],
    "snap_counts":      lambda s: [OUT_BUNDLE / "snap_counts" / f"snap_counts_{s}.csv"],
    "team_stats_week":  lambda s: [OUT_BUNDLE / "team_stats" / f"team_stats_week_{s}.csv"],
    "player_stats_week":lambda s: [OUT_BUNDLE / "player_stats" / f"player_stats_week_{s}.csv"],
    "proe_week":        lambda s: [ROOT / "outputs" / "proe" / f"proe_week_{s}.csv"],
    "box_week":         lambda s: [OUT_BUNDLE / "box_counts" / f"defense_box_rates_week_{s}.csv"],
}

FALLBACKS: Dict[str, List] = {
    "pbp": [
        (lambda s: nflverse.pbp(s)) if nflverse and hasattr(nflverse,"pbp") else None,
        (lambda s: espn.pbp(s)) if espn and hasattr(espn,"pbp") else None,
        (lambda s: msf.pbp(s)) if msf and hasattr(msf,"pbp") else None,
        (lambda s: apis.pbp(s)) if apis and hasattr(apis,"pbp") else None,
        (lambda s: gsis.pbp(s)) if gsis and hasattr(gsis,"pbp") else None,
    ],
    "team_stats_week": [
        (lambda s: nflverse.team_stats_week(s)) if nflverse and hasattr(nflverse,"team_stats_week") else None,
        (lambda s: espn.team_stats_week(s)) if espn and hasattr(espn,"team_stats_week") else None,
        (lambda s: msf.team_stats_week(s)) if msf and hasattr(msf,"team_stats_week") else None,
        (lambda s: apis.team_stats_week(s)) if apis and hasattr(apis,"team_stats_week") else None,
        (lambda s: gsis.team_stats_week(s)) if gsis and hasattr(gsis,"team_stats_week") else None,
    ],
    "player_stats_week": [
        (lambda s: nflverse.player_stats_week(s)) if nflverse and hasattr(nflverse,"player_stats_week") else None,
        (lambda s: espn.player_stats_week(s)) if espn and hasattr(espn,"player_stats_week") else None,
        (lambda s: msf.player_stats_week(s)) if msf and hasattr(msf,"player_stats_week") else None,
        (lambda s: apis.player_stats_week(s)) if apis and hasattr(apis,"player_stats_week") else None,
        (lambda s: gsis.player_stats_week(s)) if gsis and hasattr(gsis,"player_stats_week") else None,
    ],
    "injuries": [
        (lambda s: nflverse.injuries(s)) if nflverse and hasattr(nflverse,"injuries") else None,
        (lambda s: espn.injuries(s)) if espn and hasattr(espn,"injuries") else None,
        (lambda s: msf.injuries(s)) if msf and hasattr(msf,"injuries") else None,
        (lambda s: apis.injuries(s)) if apis and hasattr(apis,"injuries") else None,
        (lambda s: gsis.injuries(s)) if gsis and hasattr(gsis,"injuries") else None,
    ],
    "schedules": [
        (lambda s: nflverse.schedules(s)) if nflverse and hasattr(nflverse,"schedules") else None,
        (lambda s: espn.schedules(s)) if espn and hasattr(espn,"schedules") else None,
        (lambda s: msf.schedules(s)) if msf and hasattr(msf,"schedules") else None,
        (lambda s: apis.schedules(s)) if apis and hasattr(apis,"schedules") else None,
        (lambda s: gsis.schedules(s)) if gsis and hasattr(gsis,"schedules") else None,
    ],
    "rosters": [
        (lambda s: nflverse.rosters(s)) if nflverse and hasattr(nflverse,"rosters") else None,
        (lambda s: espn.rosters(s)) if espn and hasattr(espn,"rosters") else None,
        (lambda s: msf.rosters(s)) if msf and hasattr(msf,"rosters") else None,
        (lambda s: apis.rosters(s)) if apis and hasattr(apis,"rosters") else None,
        (lambda s: gsis.rosters(s)) if gsis and hasattr(gsis,"rosters") else None,
    ],
    "depth_charts": [
        (lambda s: nflverse.depth_charts(s)) if nflverse and hasattr(nflverse,"depth_charts") else None,
        (lambda s: espn.depth_charts(s)) if espn and hasattr(espn,"depth_charts") else None,
        (lambda s: msf.depth_charts(s)) if msf and hasattr(msf,"depth_charts") else None,
        (lambda s: apis.depth_charts(s)) if apis and hasattr(apis,"depth_charts") else None,
        (lambda s: gsis.depth_charts(s)) if gsis and hasattr(gsis,"depth_charts") else None,
    ],
    "snap_counts": [
        (lambda s: nflverse.snap_counts(s)) if nflverse and hasattr(nflverse,"snap_counts") else None,
        (lambda s: espn.snap_counts(s)) if espn and hasattr(espn,"snap_counts") else None,
        (lambda s: msf.snap_counts(s)) if msf and hasattr(msf,"snap_counts") else None,
        (lambda s: apis.snap_counts(s)) if apis and hasattr(apis,"snap_counts") else None,
        (lambda s: gsis.snap_counts(s)) if gsis and hasattr(gsis,"snap_counts") else None,
    ],
}

def _get_or_resolve(name: str, season: int, cache: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = cache.get(name)
    if _ok(df): return df
    for p in PRIMARY_PATHS.get(name, lambda s: [])(season):
        df = _read_csv(p)
        if _ok(df): cache[name]=df; return df
    for fn in FALLBACKS.get(name, []):
        if fn is None: continue
        try:
            df = fn(season)
            if _ok(df):
                cache[name]=df; return df
        except Exception as e:
            print(f"[resolve] {name} provider failed: {e}")
    cache[name] = pd.DataFrame(); return cache[name]

def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _bool_col(df: pd.DataFrame, names: List[str]) -> pd.Series:
    for c in names:
        if c in df.columns:
            v = df[c]
            if v.dtype == bool: return v.fillna(False)
            return _safe_num(v).fillna(0) > 0
    return pd.Series(False, index=getattr(df, "index", None))

def _pick(df: pd.DataFrame, *cands: str) -> Optional[str]:
    for c in cands:
        if c in df.columns: return c
    return None

def derive_team_from_pbp(pbp: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not _ok(pbp):
        return pd.DataFrame(columns=[
            "team","def_pressure_rate_z","def_pass_epa_z","def_rush_epa_z","def_sack_rate_z",
            "pace_z","light_box_rate_z","heavy_box_rate_z","ay_per_att_z","plays_est","proe","rz_rate"
        ]), pd.DataFrame(columns=["team","week","proe"])
    posteam = _pick(pbp, "posteam","offense_team","PossessionTeam") or "posteam"
    defteam = _pick(pbp, "defteam","defense_team","DefenseTeam") or "defteam"
    weekcol = _pick(pbp, "week","game_week","Week") or "week"
    game_id = _pick(pbp, "game_id","old_game_id","gameid","gameId") or "game_id"
    epa = _pick(pbp, "epa","EPA")
    air = _pick(pbp, "air_yards","AirYards")
    yline = _pick(pbp, "yardline_100","yardline","yardline_200","YardLine")
    pass_flag = _bool_col(pbp, ["pass","is_pass","qb_dropback","PassAttempt"])
    rush_flag = _bool_col(pbp, ["rush","is_rush","RushAttempt"]) & ~pass_flag

    teams = pd.unique(pd.concat([pbp[posteam], pbp[defteam]], ignore_index=True)).astype(str)
    env = pd.DataFrame({"team": sorted([t for t in teams if t and t != "nan"])})

    if epa:
        try:
            off_pass = pbp.loc[pass_flag & pbp[posteam].notna(), [posteam, epa]].groupby(posteam, as_index=False)[epa].mean().rename(columns={posteam:"team", epa:"off_pass_epa"})
            off_rush = pbp.loc[rush_flag & pbp[posteam].notna(), [posteam, epa]].groupby(posteam, as_index=False)[epa].mean().rename(columns={posteam:"team", epa:"off_rush_epa"})
            env = env.merge(off_pass, on="team", how="left").merge(off_rush, on="team", how="left")
            env["def_pass_epa"] = -env["off_pass_epa"].fillna(0.0)
            env["def_rush_epa"] = -env["off_rush_epa"].fillna(0.0)
        except Exception:
            env["def_pass_epa"] = 0.0; env["def_rush_epa"] = 0.0
    else:
        env["def_pass_epa"] = 0.0; env["def_rush_epa"] = 0.0

    if air:
        attempts = pbp.loc[pbp[posteam].notna(), [posteam]].assign(att=pass_flag.astype(int)).groupby(posteam, as_index=False)["att"].sum().rename(columns={posteam:"team"})
        air_sum = pbp.loc[pass_flag & pbp[posteam].notna(), [posteam, air]].groupby(posteam, as_index=False)[air].sum().rename(columns={posteam:"team", air:"air_sum"})
        env = env.merge(attempts, on="team", how="left").merge(air_sum, on="team", how="left")
        env["ay_per_att"] = env["air_sum"].fillna(0.0) / env["att"].replace(0, np.nan)
    else:
        env["ay_per_att"] = np.nan

    sec_rem = _pick(pbp, "game_seconds_remaining","game_seconds","GameSeconds")
    score_diff = _pick(pbp, "score_differential","score_diff","ScoreDiff")
    env["pace"] = 28.0
    if sec_rem and score_diff:
        p = pbp.loc[pbp[posteam].notna() & pbp[score_diff].between(-7,7, inclusive="both")].copy()
        p = p.sort_values([c for c in ["game_id","old_game_id","drive","play_id"] if c in p.columns])
        p["sec"] = pd.to_numeric(p[sec_rem], errors="coerce"); p["delta"] = -p.groupby([posteam])["sec"].diff().fillna(np.nan)
        pace = p.groupby(posteam, as_index=False)["delta"].median().rename(columns={posteam:"team","delta":"sec_per_play_neutral"})
        env = env.merge(pace, on="team", how="left")
        env["pace"] = env["sec_per_play_neutral"].fillna(env.get("sec_per_play_neutral", pd.Series([28.0]*len(env))).median())
    env["plays_est"] = (3600.0 / env.get("sec_per_play_neutral", pd.Series([28.0]*len(env))).replace(0,np.nan)).fillna(0.0)

    proe_week = pd.DataFrame(columns=["team","week","proe"])
    down = _pick(pbp, "down"); ytg = _pick(pbp, "ydstogo","yards_to_go","yds_to_go"); yline_use = yline
    if down and ytg and yline_use and weekcol:
        df = pbp.loc[pbp[posteam].notna() & pbp[down].notna(), [posteam, weekcol, down, ytg, yline_use]].copy()
        df["is_pass"] = pass_flag.loc[df.index].astype(int) if hasattr(pass_flag, "loc") else 0
        grp = df.groupby([posteam, weekcol], as_index=False)["is_pass"].mean().rename(columns={posteam:"team", weekcol:"week", "is_pass":"pass_rate"})
        league = grp["pass_rate"].mean() if not grp.empty else 0.5
        grp["proe"] = grp["pass_rate"] - league
        proe_week = grp[["team","week","proe"]]
        env = env.merge(grp.groupby("team", as_index=False)["proe"].mean(), on="team", how="left")
    else:
        env["proe"] = 0.0

    env["rz_rate"] = 0.0
    if yline:
        df = pbp.loc[pbp[posteam].notna() & pbp[yline].notna(), [game_id, posteam, yline]].copy().rename(columns={posteam:"team"})
        df = df.sort_values([c for c in ["game_id","old_game_id","drive","play_id"] if c in pbp.columns])
        df["is_rz"] = pd.to_numeric(df[yline], errors="coerce") <= 20
        df["prev_is_rz"] = df.groupby([game_id,"team"])["is_rz"].shift(1).fillna(False)
        df["rz_entry"] = df["is_rz"] & (~df["prev_is_rz"])
        trips = df.groupby([game_id,"team"], as_index=False)["rz_entry"].sum().rename(columns={"rz_entry":"rz_trips"})
        rz = trips.groupby("team", as_index=False)["rz_trips"].mean().rename(columns={"rz_trips":"rz_trips_per_game"})
        env = env.merge(rz, on="team", how="left")
        env["rz_rate"] = env["rz_trips_per_game"].fillna(0.0)

    out = pd.DataFrame({
        "team": env["team"],
        "def_pressure_rate_z": 0.0,
        "def_pass_epa_z": _z(env.get("def_pass_epa", pd.Series([0.0]*len(env))).fillna(0.0)),
        "def_rush_epa_z": _z(env.get("def_rush_epa", pd.Series([0.0]*len(env))).fillna(0.0)),
        "def_sack_rate_z": 0.0,
        "pace_z": _z(env["pace"].fillna(28.0)),
        "light_box_rate_z": 0.0,
        "heavy_box_rate_z": 0.0,
        "ay_per_att_z": _z(env["ay_per_att"].fillna(0.0)),
        "plays_est": env["plays_est"].fillna(0.0),
        "proe": env["proe"].fillna(0.0),
        "rz_rate": env["rz_rate"].fillna(0.0),
    })
    return out, proe_week

def derive_player_from_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    if not _ok(pbp):
        return pd.DataFrame(columns=[
            "player","team","position","target_share","rush_share","rz_tgt_share","rz_carry_share","yprr_proxy","ypc","ypt","qb_ypa"
        ])
    posteam = _pick(pbp, "posteam","offense_team","PossessionTeam") or "posteam"
    rec_name  = _pick(pbp, "receiver_player_name","receiver","receiver_name","target_player_name","targeted_receiver","target_name")
    rush_name = _pick(pbp, "rusher_player_name","rusher","rusher_name")
    pass_name = _pick(pbp, "passer_player_name","passer","passer_name","qb_player_name")

    yline = _pick(pbp, "yardline_100","yardline","yardline_200","YardLine")
    pass_flag = _bool_col(pbp, ["pass","is_pass","qb_dropback","PassAttempt"])
    rush_flag = _bool_col(pbp, ["rush","is_rush","RushAttempt"]) & ~pass_flag

    targ_df = pd.DataFrame()
    if rec_name:
        targ_df = (pbp.loc[pass_flag & pbp[rec_name].notna() & pbp[posteam].notna(), [posteam, rec_name]]
                    .assign(tgt=1).groupby([posteam, rec_name], as_index=False)["tgt"].sum()
                    .rename(columns={posteam:"team", rec_name:"player"}))
    team_tgts = targ_df.groupby("team", as_index=False)["tgt"].sum() if not targ_df.empty else pd.DataFrame(columns=["team","tgt"])

    rush_df = pd.DataFrame()
    if rush_name:
        rush_df = (pbp.loc[rush_flag & pbp[rush_name].notna() & pbp[posteam].notna(), [posteam, rush_name]]
                    .assign(rush=1).groupby([posteam, rush_name], as_index=False)["rush"].sum()
                    .rename(columns={posteam:"team", rush_name:"player"}))
    team_rush = rush_df.groupby("team", as_index=False)["rush"].sum() if not rush_df.empty else pd.DataFrame(columns=["team","rush"])

    rz_mask = pd.Series(False, index=pbp.index)
    if yline: rz_mask = pd.to_numeric(pbp[yline], errors="coerce") <= 20
    rz_tgt_df = pd.DataFrame()
    if rec_name:
        rz_tgt_df = (pbp.loc[rz_mask & pass_flag & pbp[rec_name].notna() & pbp[posteam].notna(), [posteam, rec_name]]
                        .assign(rz_tgt=1).groupby([posteam, rec_name], as_index=False)["rz_tgt"].sum()
                        .rename(columns={posteam:"team", rec_name:"player"}))
    team_rz_tgts = rz_tgt_df.groupby("team", as_index=False)["rz_tgt"].sum() if not rz_tgt_df.empty else pd.DataFrame(columns=["team","rz_tgt"])

    rz_car_df = pd.DataFrame()
    if rush_name:
        rz_car_df = (pbp.loc[rz_mask & rush_flag & pbp[rush_name].notna() & pbp[posteam].notna(), [posteam, rush_name]]
                        .assign(rz_car=1).groupby([posteam, rush_name], as_index=False)["rz_car"].sum()
                        .rename(columns={posteam:"team", rush_name:"player"}))
    team_rz_car = rz_car_df.groupby("team", as_index=False)["rz_car"].sum() if not rz_car_df.empty else pd.DataFrame(columns=["team","rz_car"])

    pf = pd.DataFrame(columns=["player","team"])
    for base in [targ_df, rush_df, rz_tgt_df, rz_car_df]:
        if not base.empty:
            pf = pf.merge(base, on=["team","player"], how="outer") if not pf.empty else base.copy()
    if pf.empty:
        pf = pd.DataFrame(columns=["player","team","tgt","rush","rz_tgt","rz_car"])

    for c in ["tgt","rush","rz_tgt","rz_car"]:
        if c not in pf.columns: pf[c] = 0.0
        pf[c] = pd.to_numeric(pf[c], errors="coerce").fillna(0.0)

    pf = pf.merge(team_tgts, on="team", how="left").merge(team_rush, on="team", how="left", suffixes=("","_team"))
    pf = pf.merge(team_rz_tgts, on="team", how="left").merge(team_rz_car, on="team", how="left")

    pf["target_share"] = pf["tgt"] / pf["tgt_team"].replace(0,np.nan)
    pf["rush_share"] = pf["rush"] / pf["rush_team"].replace(0,np.nan)
    pf["rz_tgt_share"] = pf["rz_tgt"] / pf.groupby("team")["rz_tgt"].transform(lambda s: s.sum() if s.sum() else np.nan)
    pf["rz_carry_share"] = pf["rz_car"] / pf.groupby("team")["rz_car"].transform(lambda s: s.sum() if s.sum() else np.nan)

    out = pf[["player","team"]].copy()
    out["position"] = ""
    for c in ["target_share","rush_share","rz_tgt_share","rz_carry_share","yprr_proxy","ypc","ypt","qb_ypa"]:
        if c in pf.columns: out[c] = pf[c]
        else: out[c] = 0.0
        out[c] = out[c].fillna(0.0)
    return out

def compose_team_form(season: int, cache: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pbp = _get_or_resolve("pbp", season, cache)
    tf, proe_week = derive_team_from_pbp(pbp)
    box = _get_or_resolve("box_week", season, cache)
    if _ok(box) and {"team","def_light_box_rate","def_heavy_box_rate"} <= set(box.columns):
        b = box.groupby("team", as_index=False)[["def_light_box_rate","def_heavy_box_rate"]].mean()
        tf = tf.merge(b, on="team", how="left")
        tf["light_box_rate_z"] = _z(tf["def_light_box_rate"].fillna(0.0))
        tf["heavy_box_rate_z"] = _z(tf["def_heavy_box_rate"].fillna(0.0))
    cols = ["team","def_pressure_rate_z","def_pass_epa_z","def_rush_epa_z","def_sack_rate_z",
            "pace_z","light_box_rate_z","heavy_box_rate_z","ay_per_att_z","plays_est","proe","rz_rate"]
    for c in cols:
        if c not in tf.columns: tf[c] = 0.0 if c != "team" else ""
    return tf[cols], proe_week

def compose_player_form(season: int, cache: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    pbp = _get_or_resolve("pbp", season, cache)
    pf = derive_player_from_pbp(pbp)
    return pf[["player","team","position","target_share","rush_share","rz_tgt_share","rz_carry_share","yprr_proxy","ypc","ypt","qb_ypa"]]

def _get_or_resolve(name: str, season: int, cache: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = cache.get(name)
    if _ok(df): return df
    for p in PRIMARY_PATHS.get(name, lambda s: [])(season):
        df = _read_csv(p)
        if _ok(df): cache[name]=df; return df
    for fn in FALLBACKS.get(name, []):
        if fn is None: continue
        try:
            df = fn(season)
            if _ok(df): cache[name]=df; return df
        except Exception as e:
            print(f"[resolve] {name} provider failed: {e}")
    cache[name]=pd.DataFrame(); return cache[name]

def seasons_from_args(args) -> List[int]:
    if args.seasons:
        return [int(x.strip()) for x in args.seasons.split(",") if x.strip()]
    if args.start and args.end:
        return list(range(int(args.start), int(args.end) + 1))
    return [int(args.season)]

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--seasons", type=str, default=None)
    ap.add_argument("--strict", type=int, default=int(os.getenv("STRICT","0")))
    args = ap.parse_args()
    seasons = seasons_from_args(args)
    _mkdirs()
    print(f"[make_all] seasons={seasons} strict={args.strict}")
    for s in seasons:
        cache: Dict[str, pd.DataFrame] = {}
        team_form, proe_week = compose_team_form(s, cache)
        player_form = compose_player_form(s, cache)
        _write_csv(team_form, OUT_METRICS / "team_form.csv")
        _write_csv(player_form, OUT_METRICS / "player_form.csv")
        _write_csv(team_form, DATA_MIRROR / "team_form.csv")
        _write_csv(player_form, DATA_MIRROR / "player_form.csv")
        if _ok(proe_week):
            _write_csv(proe_week, ROOT / "outputs" / "proe" / f"proe_week_{s}.csv")
    print("✅ make_all completed")
    return 0

if __name__ == "__main__":
    sys.exit(main())
