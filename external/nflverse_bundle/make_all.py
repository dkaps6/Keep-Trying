#!/usr/bin/env python3
from __future__ import annotations

import argparse, sys, subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
OUT_BUNDLE = ROOT / "outputs"
OUT_METRICS = REPO / "outputs" / "metrics"
DATA_MIRROR = REPO / "data"

# ---------- dirs ----------
def _mkdirs():
    for p in (OUT_BUNDLE, OUT_METRICS, DATA_MIRROR):
        p.mkdir(parents=True, exist_ok=True)

# ---------- dynamic imports for providers ----------
def _import_or_none(modname: str):
    try:
        return __import__(modname, fromlist=["*"])
    except Exception:
        return None

nflverse = _import_or_none("scripts.providers.nflverse")
msf  = _import_or_none("scripts.providers.msf")
apis = _import_or_none("scripts.providers.apisports")
gsis = _import_or_none("scripts.providers.nflgsis")

# ---------- io helpers ----------
def _ok(df: Optional[pd.DataFrame]) -> bool:
    try:
        return isinstance(df, pd.DataFrame) and not df.empty
    except Exception:
        return False

def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, engine="python")
        except Exception:
            return pd.DataFrame()

def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not _ok(df):
        try:
            (df.head(0) if isinstance(df, pd.DataFrame) else pd.DataFrame()).to_csv(path, index=False)
        except Exception:
            path.write_text("")
        print(f"[write] {path.relative_to(REPO)}  rows=0")
    else:
        df.to_csv(path, index=False)
        print(f"[write] {path.relative_to(REPO)}  rows={len(df)}")

def _run(cmd: List[str]) -> int:
    print(">>", " ".join(cmd))
    return subprocess.call(cmd)

def _z(col: pd.Series) -> pd.Series:
    try:
        arr = pd.to_numeric(col, errors="coerce").values
        mu, sd = np.nanmean(arr), np.nanstd(arr)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.zeros(len(col)), index=col.index)
        return (col - mu) / sd
    except Exception:
        return pd.Series(np.zeros(len(col)), index=col.index)

# ---------- primary bundle paths ----------
PRIMARY_PATHS: Dict[str, Callable[[int], List[Path]]] = {
    "schedules":        lambda s: [OUT_BUNDLE / "schedules" / f"schedules_{s}.csv"],
    "injuries":         lambda s: [OUT_BUNDLE / "injuries" / f"injuries_{s}.csv"],
    "team_stats_week":  lambda s: [OUT_BUNDLE / "team_stats" / f"team_stats_week_{s}.csv"],
    "team_stats_reg":   lambda s: [OUT_BUNDLE / "team_stats" / f"team_stats_reg_{s}.csv"],
    "player_stats_week":lambda s: [OUT_BUNDLE / "player_stats" / f"player_stats_week_{s}.csv"],
    "player_stats_reg": lambda s: [OUT_BUNDLE / "player_stats" / f"player_stats_reg_{s}.csv"],
    "depth_charts":     lambda s: [OUT_BUNDLE / "depth_charts" / f"depth_charts_{s}.csv"],
    "snap_counts":      lambda s: [OUT_BUNDLE / "snap_counts" / f"snap_counts_{s}.csv"],
    "rosters":          lambda s: [OUT_BUNDLE / "rosters" / f"rosters_{s}.csv"],
    "rosters_weekly":   lambda s: [OUT_BUNDLE / "rosters_weekly" / f"rosters_weekly_{s}.csv"],
    "participation":    lambda s: [OUT_BUNDLE / "participation" / f"participation_{s}.csv"],
    "pbp":              lambda s: [OUT_BUNDLE / "pbp" / f"pbp_{s}_{s}.csv"],
    "proe_week":        lambda s: [ROOT / "outputs" / "proe" / f"proe_week_{s}.csv"],
    "box_week":         lambda s: [OUT_BUNDLE / "box_counts" / f"defense_box_rates_week_{s}.csv"],
}

# ---------- fallbacks (priority: nflverse -> msf -> apisports -> gsis) ----------
FALLBACKS: Dict[str, List] = {
    "schedules": [
        (lambda s: nflverse.schedules(s)) if nflverse else None,
        (lambda s: msf.schedules(s)) if msf and hasattr(msf, "schedules") else None,
        (lambda s: apis.schedules(s)) if apis else None,
        (lambda s: gsis.schedules(s)) if gsis else None,
    ],
    "injuries": [
        (lambda s: nflverse.injuries(s)) if nflverse and hasattr(nflverse, "injuries") else None,
        (lambda s: msf.injuries(s)) if msf else None,
        (lambda s: apis.injuries(s)) if apis else None,
        (lambda s: gsis.injuries(s)) if gsis else None,
    ],
    "pbp": [
        (lambda s: nflverse.pbp(s)) if nflverse else None,
    ],
    "team_stats_week": [
        (lambda s: nflverse.team_stats_week(s)) if nflverse else None,
        (lambda s: msf.team_stats_week(s)) if msf else None,
    ],
    "team_stats_reg": [
        (lambda s: nflverse.team_stats_reg(s)) if nflverse else None,
        (lambda s: msf.team_stats_reg(s)) if msf else None,
    ],
    "player_stats_week": [
        (lambda s: nflverse.player_stats_week(s)) if nflverse else None,
        (lambda s: msf.player_stats_week(s)) if msf else None,
    ],
    "player_stats_reg": [
        (lambda s: nflverse.player_stats_reg(s)) if nflverse else None,
        (lambda s: msf.player_stats_reg(s)) if msf else None,
    ],
    "rosters": [
        (lambda s: nflverse.rosters(s)) if nflverse else None,
        (lambda s: msf.rosters(s)) if msf and hasattr(msf, "rosters") else None,
    ],
    "rosters_weekly": [
        (lambda s: nflverse.rosters_weekly(s)) if nflverse else None,
    ],
    "depth_charts": [
        (lambda s: nflverse.depth_charts(s)) if nflverse else None,
        (lambda s: gsis.depth_charts(s)) if gsis else None,
    ],
    "snap_counts": [
        (lambda s: nflverse.snap_counts(s)) if nflverse else None,
    ],
    "participation": [
        (lambda s: nflverse.participation(s)) if nflverse else None,
    ],
    "proe_week": [],
    "box_week":  [],
}

# ---------- computed / last-resort ----------
def _compute_proxy(key: str, season: int, cache: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    if key == "team_stats_week" and _ok(cache.get("pbp")):
        pbp = cache["pbp"]
        tcol = "posteam" if "posteam" in pbp.columns else ("offense_team" if "offense_team" in pbp.columns else None)
        if tcol and "week" in pbp.columns:
            out = (
                pbp.loc[pbp[tcol].notna()]
                   .groupby([tcol, "week"], as_index=False)
                   .size()
                   .rename(columns={tcol: "team", "size": "plays"})
            )
            return out
    if key == "team_stats_reg":
        wk = cache.get("team_stats_week")
        if _ok(wk):
            gcols = [c for c in ["team"] if c in wk.columns]
            return wk.groupby(gcols, as_index=False).sum(numeric_only=True)
    if key == "player_stats_reg":
        wk = cache.get("player_stats_week")
        if _ok(wk):
            keys = [c for c in ["team","player","position"] if c in wk.columns]
            return wk.groupby(keys, as_index=False).sum(numeric_only=True)
    return None

def resolve_table(key: str, season: int, cache: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    # 1) primary
    for p in PRIMARY_PATHS.get(key, lambda s: [])(season):
        df = _read_csv(p)
        if _ok(df): return df
    # 2) providers
    for fn in FALLBACKS.get(key, []):
        if fn is None: continue
        try:
            df = fn(season)
        except Exception:
            df = None
        if _ok(df): return df
    # 3) computed
    df = _compute_proxy(key, season, cache)
    if _ok(df): return df
    return pd.DataFrame()

# ============================
# === DERIVED METRICS LAYER ==
# ============================

def _pick(colnames: List[str], df: pd.DataFrame) -> Optional[str]:
    for c in colnames:
        if c in df.columns:
            return c
    return None

def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _bool_col(df: pd.DataFrame, names: List[str]) -> pd.Series:
    c = _pick(names, df)
    if c is None: return pd.Series(False, index=df.index)
    v = df[c]
    if v.dtype == bool: return v.fillna(False)
    return _safe_num(v).fillna(0) > 0

def _name_col(df: pd.DataFrame, names: List[str], default: str="") -> pd.Series:
    c = _pick(names, df)
    return df[c].astype(str).fillna(default) if c else pd.Series(default, index=df.index)

def derive_team_from_pbp(pbp: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      team_env  : team-level env metrics (EPA splits, pace, plays_est, ay_per_att, proe, rz_rate)
      proe_week : optional weekly proe table (team, week, proe)
    """
    if not _ok(pbp):
        return pd.DataFrame(), pd.DataFrame()

    # Column normalization (nflfastR style preferred)
    posteam = _pick(["posteam","offense_team"], pbp) or "posteam"
    defteam = _pick(["defteam","defense_team"], pbp) or "defteam"
    weekcol = _pick(["week","game_week"], pbp) or "week"
    game_id = _pick(["game_id","old_game_id","gameid","gameId"], pbp) or "game_id"
    epa = _pick(["epa"], pbp)
    pass_flag = _bool_col(pbp, ["pass","is_pass","qb_dropback"])
    rush_flag = _bool_col(pbp, ["rush","is_rush"]) & ~pass_flag

    # --- EPA splits (defense is negative of offense EPA allowed) ---
    teams = pd.unique(pd.concat([pbp[posteam], pbp[defteam]], ignore_index=True)).astype(str)
    env = pd.DataFrame({"team": sorted([t for t in teams if t and t != "nan"])})

    env["def_pass_epa"] = 0.0
    env["def_rush_epa"] = 0.0
    if epa:
        off_pass = (
            pbp.loc[pass_flag & pbp[posteam].notna(), [posteam, epa]]
               .groupby(posteam, as_index=False)[epa].mean()
               .rename(columns={posteam: "team", epa: "off_pass_epa"})
        )
        off_rush = (
            pbp.loc[rush_flag & pbp[posteam].notna(), [posteam, epa]]
               .groupby(posteam, as_index=False)[epa].mean()
               .rename(columns={posteam: "team", epa: "off_rush_epa"})
        )
        env = env.merge(off_pass, on="team", how="left").merge(off_rush, on="team", how="left")
        env["def_pass_epa"] = -env["off_pass_epa"].fillna(0.0)
        env["def_rush_epa"] = -env["off_rush_epa"].fillna(0.0)

    # --- aDOT / air_yards per attempt ---
    air = _pick(["air_yards"], pbp)
    attempts = (
        pbp.loc[pbp[posteam].notna(), [posteam]]
           .assign(att=pass_flag.astype(int))
           .groupby(posteam, as_index=False)["att"].sum()
           .rename(columns={posteam:"team"})
    )
    if air:
        air_sum = (
            pbp.loc[pass_flag & pbp[posteam].notna(), [posteam, air]]
               .groupby(posteam, as_index=False)[air].sum()
               .rename(columns={posteam:"team", air:"air_sum"})
        )
        env = env.merge(attempts, on="team", how="left").merge(air_sum, on="team", how="left")
        env["ay_per_att"] = env["air_sum"].fillna(0.0) / env["att"].replace(0, np.nan)
    else:
        env["ay_per_att"] = np.nan

    # --- Neutral pace (sec/play in neutral score) ---
    sec_rem = _pick(["game_seconds_remaining","game_seconds"], pbp)
    score_diff = _pick(["score_differential","score_diff"], pbp)
    qtr = _pick(["qtr","quarter"], pbp)
    env["pace"] = 0.0
    if sec_rem and score_diff:
        # filter neutral situations (within one score), offense-only plays
        p = pbp.loc[pbp[posteam].notna() & (pbp[score_diff].between(-7, 7, inclusive="both"))].copy()
        # approximate seconds per play using negative diffs of game_seconds_remaining
        # order plays per game/possession chronology
        order_cols = [c for c in ["game_id","old_game_id","drive","play_id","index"] if c in p.columns]
        if not order_cols: order_cols = [p.index.name] if p.index.name else [p.index]
        p = p.sort_values(order_cols)
        p["sec"] = _safe_num(p[sec_rem])
        p["delta"] = -p.groupby([posteam])[ "sec" ].diff().fillna(np.nan)
        pace = p.groupby(posteam, as_index=False)["delta"].median()
        pace = pace.rename(columns={posteam:"team","delta":"sec_per_play_neutral"})
        env = env.merge(pace, on="team", how="left")
        env["pace"] = env["sec_per_play_neutral"].fillna(env["sec_per_play_neutral"].median())
        env["pace"] = -env["pace"].fillna(28.0)  # default ~28 sec/play

    # --- PROE (league expectation by down/dist/field/score/quarter/time) ---
    down = _pick(["down"], pbp)
    ytg  = _pick(["ydstogo","yards_to_go","yds_to_go"], pbp)
    yline= _pick(["yardline_100","yardline"], pbp)
    time = sec_rem
    proe_week = pd.DataFrame(columns=["team","week","proe"])
    env["proe"] = 0.0
    if down and ytg and yline and weekcol:
        df = pbp.loc[pbp[posteam].notna() & pbp[down].notna(), [posteam, weekcol, down, ytg, yline]].copy()
        df["is_pass"] = pass_flag.loc[df.index].astype(int)

        # bins
        df["down_b"]   = df[down].clip(1,4)
        df["ytg_b"]    = pd.cut(_safe_num(df[ytg]), bins=[-1,2,6,10,25,99], labels=["short","med","long","xlong","hail"])
        df["yl_b"]     = pd.cut(_safe_num(df[yline]), bins=[-1,20,50,80,110], labels=["rz","mid","deep","backed"])

        # add score bins, quarter bins, time bins (more expressive expectation)
        if score_diff in pbp.columns:
            df2 = pbp.loc[df.index, [score_diff]].copy()
            df["score_b"] = pd.cut(_safe_num(df2[score_diff]), bins=[-99,-14,-7,-3,3,7,14,99],
                                   labels=["<-14","-14..-7","-7..-3","-3..3","3..7","7..14",">14"])
        else:
            df["score_b"] = "0"

        if qtr:
            qmap = pd.Series(pbp[qtr], index=pbp.index).reindex(df.index)
            df["qtr_b"] = qmap.clip(1,4).fillna(1).astype(int)
        else:
            df["qtr_b"] = 1

        if time:
            tmap = pd.Series(_safe_num(pbp[time]), index=pbp.index).reindex(df.index)
            # bins emphasize late/early situations
            df["time_b"] = pd.cut(tmap, bins=[-1,120,600,1200,1800,3600], labels=["last2m","late","Q3","Q2","Q1"])
        else:
            df["time_b"] = "Q1"

        # league baseline expectation across multi-dim buckets
        exp_keys = ["down_b","ytg_b","yl_b","score_b","qtr_b","time_b"]
        exp_tbl = (
            df.groupby(exp_keys, as_index=False)["is_pass"]
              .mean()
              .rename(columns={"is_pass":"exp_pass"})
        )
        df = df.merge(exp_tbl, on=exp_keys, how="left")
        # if a rare bucket is NaN (no league history), back off to simpler (down_b, ytg_b, yl_b)
        if df["exp_pass"].isna().any():
            exp_tbl_simple = (
                df.groupby(["down_b","ytg_b","yl_b"], as_index=False)["is_pass"]
                  .mean()
                  .rename(columns={"is_pass":"exp_pass_simple"})
            )
            df = df.merge(exp_tbl_simple, on=["down_b","ytg_b","yl_b"], how="left")
            df["exp_pass"] = df["exp_pass"].fillna(df["exp_pass_simple"]).fillna(df["is_pass"].mean())
            df = df.drop(columns=["exp_pass_simple"], errors="ignore")

        grp = df.groupby([posteam, weekcol], as_index=False)[["is_pass","exp_pass"]].mean()
        grp["proe"] = grp["is_pass"] - grp["exp_pass"]
        grp = grp.rename(columns={posteam: "team", weekcol: "week"})
        proe_week = grp[["team","week","proe"]]
        proe_season = grp.groupby("team", as_index=False)["proe"].mean()
        env = env.merge(proe_season, on="team", how="left")

    # --- Red-zone trip rate (first entries per game) ---
    # define entry when yardline_100 <= 20 and previous play for same team+game was >20 or different game/team
    yline_use = yline
    env["rz_rate"] = 0.0
    if yline_use:
        df = pbp.loc[pbp[posteam].notna() & pbp[yline_use].notna(), [game_id, posteam, yline_use]].copy()
        df = df.rename(columns={posteam:"team"})
        # sort by game sequence if available
        order_cols = [c for c in ["game_id","old_game_id","drive","play_id","index"] if c in pbp.columns]
        if order_cols:
            df = pbp.loc[df.index, [game_id, posteam, yline_use] + order_cols].copy().rename(columns={posteam:"team"})
            df = df.sort_values(order_cols)
        df["is_rz"] = _safe_num(df[yline_use]) <= 20
        # first entry boolean using shift within team+game
        df["prev_is_rz"] = df.groupby([game_id, "team"])["is_rz"].shift(1).fillna(False)
        df["rz_entry"] = df["is_rz"] & (~df["prev_is_rz"])
        trips = df.groupby([game_id, "team"], as_index=False)["rz_entry"].sum().rename(columns={"rz_entry":"rz_trips"})
        # games played per team
        games = df.groupby(["team", game_id], as_index=False).size().groupby("team", as_index=False).size().rename(columns={"size":"games"})
        rz = trips.groupby("team", as_index=False)["rz_trips"].mean().rename(columns={"rz_trips":"rz_trips_per_game"})
        env = env.merge(rz, on="team", how="left")
        env["rz_rate"] = env["rz_trips_per_game"].fillna(0.0)

    # --- plays_est from pace ---
    env["plays_est"] = (3600.0 / env["sec_per_play_neutral"].replace(0, np.nan)).fillna(0.0) if "sec_per_play_neutral" in env.columns else 0.0

    # final assemble
    out = pd.DataFrame({
        "team": env["team"],
        "def_pressure_rate_z": 0.0,  # no free NGS; keep z of 0
        "def_pass_epa_z": _z(env["def_pass_epa"].fillna(0.0)),
        "def_rush_epa_z": _z(env["def_rush_epa"].fillna(0.0)),
        "def_sack_rate_z": 0.0,
        "pace_z": _z(env["pace"].fillna(0.0)),
        "light_box_rate_z": 0.0,
        "heavy_box_rate_z": 0.0,
        "ay_per_att_z": _z(env["ay_per_att"].fillna(0.0)),
        "plays_est": env["plays_est"].fillna(0.0),
        "proe": env["proe"].fillna(0.0),
        "rz_rate": env["rz_rate"].fillna(0.0),
    })

    return out, proe_week

def derive_player_from_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Returns per-player form with shares/efficiency proxies.

    Columns:
      player, team, position (blank), target_share, rush_share, rz_tgt_share, rz_carry_share,
      yprr_proxy, ypc, ypt, qb_ypa
    """
    if not _ok(pbp):
        return pd.DataFrame(columns=[
            "player","team","position","target_share","rush_share","rz_tgt_share","rz_carry_share",
            "yprr_proxy","ypc","ypt","qb_ypa"
        ])

    posteam = _pick(["posteam","offense_team"], pbp) or "posteam"
    weekcol = _pick(["week","game_week"], pbp) or "week"

    # Names (nflfastR cols)
    rec_name  = _pick(["receiver_player_name","receiver","receiver_name"], pbp)
    rush_name = _pick(["rusher_player_name","rusher","rusher_name"], pbp)
    pass_name = _pick(["passer_player_name","passer","passer_name","qb_player_name"], pbp)

    yards_gained = _pick(["yards_gained","yards"], pbp)
    rush_yards_col = yards_gained  # on rush plays, yards_gained is rush yards
    yac = _pick(["yards_after_catch","yac"], pbp)
    yline= _pick(["yardline_100","yardline"], pbp)

    pass_flag = _bool_col(pbp, ["pass","is_pass","qb_dropback"])
    rush_flag = _bool_col(pbp, ["rush","is_rush"]) & ~pass_flag

    # --- team totals for shares ---
    targ_df = pd.DataFrame()
    if rec_name:
        targ_df = (pbp.loc[pass_flag & pbp[rec_name].notna() & pbp[posteam].notna(), [posteam, rec_name]]
                      .assign(tgt=1)
                      .groupby([posteam, rec_name], as_index=False)["tgt"].sum()
                      .rename(columns={posteam:"team", rec_name:"player"}))

    team_tgts = targ_df.groupby("team", as_index=False)["tgt"].sum() if _ok(targ_df) else pd.DataFrame(columns=["team","tgt"])

    rush_df = pd.DataFrame()
    if rush_name:
        rush_df = (pbp.loc[rush_flag & pbp[rush_name].notna() & pbp[posteam].notna(), [posteam, rush_name]]
                      .assign(rush=1)
                      .groupby([posteam, rush_name], as_index=False)["rush"].sum()
                      .rename(columns={posteam:"team", rush_name:"player"}))
    team_rush = rush_df.groupby("team", as_index=False)["rush"].sum() if _ok(rush_df) else pd.DataFrame(columns=["team","rush"])

    # --- red-zone shares (yardline_100 <= 20) ---
    rz_mask = pd.Series(False, index=pbp.index)
    if yline:
        rz_mask = _safe_num(pbp[yline]) <= 20

    rz_tgt_df = pd.DataFrame()
    if rec_name:
        rz_tgt_df = (pbp.loc[rz_mask & pass_flag & pbp[rec_name].notna() & pbp[posteam].notna(), [posteam, rec_name]]
                        .assign(rz_tgt=1)
                        .groupby([posteam, rec_name], as_index=False)["rz_tgt"].sum()
                        .rename(columns={posteam:"team", rec_name:"player"}))
    team_rz_tgts = rz_tgt_df.groupby("team", as_index=False)["rz_tgt"].sum() if _ok(rz_tgt_df) else pd.DataFrame(columns=["team","rz_tgt"])

    rz_car_df = pd.DataFrame()
    if rush_name:
        rz_car_df = (pbp.loc[rz_mask & rush_flag & pbp[rush_name].notna() & pbp[posteam].notna(), [posteam, rush_name]]
                        .assign(rz_car=1)
                        .groupby([posteam, rush_name], as_index=False)["rz_car"].sum()
                        .rename(columns={posteam:"team", rush_name:"player"}))
    team_rz_car = rz_car_df.groupby("team", as_index=False)["rz_car"].sum() if _ok(rz_car_df) else pd.DataFrame(columns=["team","rz_car"])

    # --- efficiency: receiving yards, YAC; rushing yards for YPC ---
    rec_yds_df = pd.DataFrame()
    if rec_name and yards_gained:
        rec_yds_df = (pbp.loc[pass_flag & pbp[rec_name].notna() & pbp[posteam].notna(), [posteam, rec_name, yards_gained]]
                         .groupby([posteam, rec_name], as_index=False)[yards_gained].sum()
                         .rename(columns={posteam:"team", rec_name:"player", yards_gained:"rec_yards"}))
    yac_df = pd.DataFrame()
    if rec_name and yac:
        yac_df = (pbp.loc[pass_flag & pbp[rec_name].notna() & pbp[posteam].notna(), [posteam, rec_name, yac]]
                     .groupby([posteam, rec_name], as_index=False)[yac].sum()
                     .rename(columns={posteam:"team", rec_name:"player", yac:"yac_sum"}))
    rush_yds_df = pd.DataFrame()
    if rush_name and rush_yards_col:
        rush_yds_df = (pbp.loc[rush_flag & pbp[rush_name].notna() & pbp[posteam].notna(), [posteam, rush_name, rush_yards_col]]
                          .groupby([posteam, rush_name], as_index=False)[rush_yards_col].sum()
                          .rename(columns={posteam:"team", rush_name:"player", rush_yards_col:"rush_yards"}))

    # merge per-player frame
    pf = pd.DataFrame(columns=["player","team"])
    for base in [targ_df, rush_df, rz_tgt_df, rz_car_df, rec_yds_df, yac_df, rush_yds_df]:
        if _ok(base):
            pf = pf.merge(base, on=["team","player"], how="outer") if _ok(pf) else base.copy()
    if not _ok(pf):
        pf = pd.DataFrame(columns=["player","team","tgt","rush","rz_tgt","rz_car","rec_yards","yac_sum","rush_yards"])

    # fill NaNs
    for c in ["tgt","rush","rz_tgt","rz_car","rec_yards","yac_sum","rush_yards"]:
        if c not in pf.columns: pf[c] = 0.0
        pf[c] = _safe_num(pf[c]).fillna(0.0)

    # shares
    pf = pf.merge(team_tgts, on="team", how="left", suffixes=("","_team")).merge(team_rush, on="team", how="left", suffixes=("","_team"))
    pf = pf.merge(team_rz_tgts, on="team", how="left").merge(team_rz_car, on="team", how="left", suffixes=("","_team"))
    pf["target_share"] = pf["tgt"] / pf["tgt_team"].replace(0, np.nan)
    pf["rush_share"]   = pf["rush"] / pf["rush_team"].replace(0, np.nan)
    pf["rz_tgt_share"] = pf["rz_tgt"] / pf["rz_tgt_team"].replace(0, np.nan)
    pf["rz_carry_share"] = pf["rz_car"] / pf["rz_car_team"].replace(0, np.nan)

    # ypt / ypc / yprr_proxy
    pf["ypt"] = pf["rec_yards"] / pf["tgt"].replace(0, np.nan)
    pf["ypc"] = pf["rush_yards"] / pf["rush"].replace(0, np.nan)
    pf["yprr_proxy"] = pf["ypt"]  # stable proxy without routes

    # QB YPA (by passer)
    qb = pd.DataFrame(columns=["player","team","qb_ypa"])
    if pass_name and yards_gained:
        pa = (pbp.loc[pass_flag & pbp[pass_name].notna() & pbp[posteam].notna(), [posteam, pass_name]]
                .assign(att=1)
                .groupby([posteam, pass_name], as_index=False)["att"].sum()
                .rename(columns={posteam:"team", pass_name:"player"}))
        py = (pbp.loc[pass_flag & pbp[pass_name].notna() & pbp[posteam].notna(), [posteam, pass_name, yards_gained]]
                .groupby([posteam, pass_name], as_index=False)[yards_gained].sum()
                .rename(columns={posteam:"team", pass_name:"player", yards_gained:"pass_yards"}))
        qb = pa.merge(py, on=["team","player"], how="left")
        qb["qb_ypa"] = qb["pass_yards"] / qb["att"].replace(0, np.nan)
        qb = qb[["team","player","qb_ypa"]]

    pf = pf.merge(qb, on=["team","player"], how="left")

    pf2 = pf[["player","team"]].copy()
    pf2["position"] = ""
    for c in ["target_share","rush_share","rz_tgt_share","rz_carry_share","yprr_proxy","ypc","ypt","qb_ypa"]:
        if c not in pf.columns: pf[c] = np.nan
        pf2[c] = pf[c].astype(float)

    # clean NaN -> 0 where appropriate
    for c in ["target_share","rush_share","rz_tgt_share","rz_carry_share","yprr_proxy","ypc","ypt","qb_ypa"]:
        pf2[c] = pf2[c].fillna(0.0)

    return pf2

# ============================
# ===== COMPOSERS ============
# ============================

def compose_team_form(season: int, cache: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns team_form and proe_week (for optional export)
    """
    # Resolve pbp early for derivations
    pbp = cache.get("pbp") or resolve_table("pbp", season, cache); cache["pbp"] = pbp
    tf_der, proe_week = derive_team_from_pbp(pbp)

    # Optional sources for box rates, etc.
    box   = cache.get("box_week")        or resolve_table("box_week",        season, cache); cache["box_week"]        = box
    proe  = cache.get("proe_week")       or resolve_table("proe_week",       season, cache); cache["proe_week"]       = proe

    # integrate box rates if present
    if _ok(box) and {"team","def_light_box_rate","def_heavy_box_rate"} <= set(box.columns):
        b = box.groupby("team", as_index=False)[["def_light_box_rate","def_heavy_box_rate"]].mean()
        tf_der = tf_der.merge(b, on="team", how="left")
        tf_der["light_box_rate_z"] = _z(tf_der["def_light_box_rate"].fillna(0.0))
        tf_der["heavy_box_rate_z"] = _z(tf_der["def_heavy_box_rate"].fillna(0.0))

    # prefer addon PROE season if present (otherwise derived)
    if _ok(proe) and {"team","proe"} <= set(proe.columns):
        p = proe.groupby("team", as_index=False)["proe"].mean()
        tf_der = tf_der.drop(columns=["proe"], errors="ignore").merge(p, on="team", how="left")

    # ensure required cols exist
    cols = ["team",
            "def_pressure_rate_z","def_pass_epa_z","def_rush_epa_z","def_sack_rate_z",
            "pace_z","light_box_rate_z","heavy_box_rate_z","ay_per_att_z",
            "plays_est","proe","rz_rate"]
    for c in cols:
        if c not in tf_der.columns:
            tf_der[c] = 0.0 if c != "team" else ""

    return tf_der[cols], proe_week

def compose_player_form(season: int, cache: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    # Start from PBP-derived player form
    pbp = cache.get("pbp") or resolve_table("pbp", season, cache); cache["pbp"] = pbp
    pf = derive_player_from_pbp(pbp)

    # Merge optional addons
    roles = _read_csv(ROOT / "outputs" / "roles" / f"roles_{season}.csv")
    rbm   = _read_csv(ROOT / "outputs" / "rb_metrics" / f"rb_metrics_{season}.csv")
    inj   = cache.get("injuries") or resolve_table("injuries", season, cache); cache["injuries"] = inj

    if _ok(roles):
        # where roles provide position & overrides for shares, prefer them
        for col, default in [
            ("player",""), ("team",""), ("position",""),
            ("target_share",np.nan), ("rush_share",np.nan),
            ("rz_tgt_share",np.nan), ("rz_carry_share",np.nan),
            ("yprr_proxy",np.nan), ("ypc",np.nan), ("ypt",np.nan), ("qb_ypa",np.nan),
        ]:
            if col not in roles.columns: roles[col] = default
        pf = pf.merge(roles[pf.columns], on=["player","team"], how="outer", suffixes=("", "_role"))
        # fill from roles if present
        for c in ["position","target_share","rush_share","rz_tgt_share","rz_carry_share","yprr_proxy","ypc","ypt","qb_ypa"]:
            rc = f"{c}_role"
            if rc in pf.columns:
                pf[c] = pf[c].where(pf[c].notna() & (pf[c] != 0), pf[rc])
        drop_cols = [c for c in pf.columns if c.endswith("_role")]
        pf = pf.drop(columns=drop_cols)

    if _ok(rbm) and {"player","team"} <= set(rbm.columns):
        for c in ["success_rate","yac","explosive_rate"]:
            if c not in rbm.columns: rbm[c] = np.nan
        pf = pf.merge(rbm[["player","team","success_rate","yac","explosive_rate"]],
                      on=["player","team"], how="left")

    if _ok(inj):
        inj2 = inj.rename(columns={"status":"inj_status"})
        for c in ["player","team","inj_status"]:
            if c not in inj2.columns: inj2[c] = "" if c != "inj_status" else ""
        pf = pf.merge(inj2[["player","team","inj_status"]], on=["player","team"], how="left")

    # final NA cleanup
    pf["position"] = pf["position"].fillna("")
    for c in ["target_share","rush_share","rz_tgt_share","rz_carry_share","yprr_proxy","ypc","ypt","qb_ypa"]:
        if c not in pf.columns: pf[c] = 0.0
        pf[c] = pf[c].fillna(0.0)

    return pf[[
        "player","team","position",
        "target_share","rush_share","rz_tgt_share","rz_carry_share",
        "yprr_proxy","ypc","ypt","qb_ypa"
    ]]

# ============================
# ===== BUNDLE DRIVER ========
# ============================

def seasons_from_args(args) -> List[int]:
    if args.seasons:
        return [int(x.strip()) for x in args.seasons.split(",") if x.strip()]
    if args.start and args.end:
        return list(range(int(args.start), int(args.end) + 1))
    return [int(args.season)]

def fetch_bundle(seasons: List[int]) -> None:
    fetch = ROOT / "fetch_all.py"
    if fetch.exists():
        for s in seasons:
            rc = _run([sys.executable, str(fetch), "--season", str(s)])
            if rc != 0:
                print(f"[warn] fetch_all.py returned {rc} for {s}; continuing")
    addons = [
        ("addons/derive_proe.py", []),
        ("addons/aggregate_box_counts.py", []),
        ("addons/derive_roles.py", []),
        ("addons/derive_rb_metrics.py", []),
        ("addons/fetch_injuries_espn.py", []),
    ]
    for rel, extra in addons:
        scr = ROOT / rel
        if not scr.exists(): continue
        for s in seasons:
            rc = _run([sys.executable, str(scr), "--season", str(s), *extra])
            if rc != 0:
                print(f"[warn] {rel} returned {rc} for {s}; continuing")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--seasons", type=str, default=None)
    args = ap.parse_args()

    _mkdirs()
    seasons = seasons_from_args(args)
    print(f"[make_all] seasons={seasons}")

    fetch_bundle(seasons)

    for s in seasons:
        print(f"[compose] season {s}")
        cache: Dict[str, pd.DataFrame] = {}
        cache["pbp"] = resolve_table("pbp", s, cache)  # early for proxies

        team_form, proe_week = compose_team_form(s, cache)
        player_form = compose_player_form(s, cache)

        # write outputs
        _write_csv(team_form, OUT_METRICS / "team_form.csv")
        _write_csv(player_form, OUT_METRICS / "player_form.csv")
        _write_csv(team_form, DATA_MIRROR / "team_form.csv")
        _write_csv(player_form, DATA_MIRROR / "player_form.csv")

        # optional export of weekly proe (debug/inspection)
        if _ok(proe_week):
            out_proe = ROOT / "outputs" / "proe" / f"proe_week_{s}.csv"
            _write_csv(proe_week, out_proe)

    print("✅ make_all completed.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
