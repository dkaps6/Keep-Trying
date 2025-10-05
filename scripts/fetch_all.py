# scripts/fetch_all.py
# -*- coding: utf-8 -*-
"""
Fetch external features from free sources and write CSVs into metrics/ and inputs/.
Also writes a JSON report summarizing which sources/metrics populated.

Outputs:
  metrics/team_form.csv
  metrics/team_week_form.csv
  metrics/player_form.csv
  inputs/id_map.csv
  inputs/weather.csv      (optional; contains schedule-weather shells)
  metrics/fetch_report.json (coverage + source status)

This script is resilient to nflverse schema changes (e.g., drive_result) and logs what was/wasn't populated.
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import typing as t
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np

# Free HTTP fetch (for PFR; will gracefully handle 403 and mark unavailable)
import requests

# nflverse python client
try:
    import nfl_data_py as nfl
except Exception as e:
    nfl = None

# ------------------------
# Config / constants
# ------------------------

SEASON_DEFAULT = 2025

OUT_METRICS_DIR = "metrics"
OUT_INPUTS_DIR = "inputs"
REPORT_PATH = os.path.join(OUT_METRICS_DIR, "fetch_report.json")

# "Explosive" thresholds used commonly in analytics
EXPLOSIVE_PASS_YDS = 20
EXPLOSIVE_RUSH_YDS = 15

# Neutral situation definition
NEUTRAL_SCORE_DIFF = 7
NEUTRAL_QUARTERS = {1, 2, 3}
EARLY_DOWNS = {1, 2}

# PFR endpoints (often 403 from CI; we still try and record status)
PFR_OPPONENT_URL = "https://www.pro-football-reference.com/years/{season}/opp.htm"
PFR_DRIVES_URL = "https://www.pro-football-reference.com/years/{season}/drives.htm"

# ------------------------
# Utilities / helpers
# ------------------------

def mkdirs():
    os.makedirs(OUT_METRICS_DIR, exist_ok=True)
    os.makedirs(OUT_INPUTS_DIR, exist_ok=True)

@dataclass
class SourceStatus:
    name: str
    ok: bool
    rows: int
    detail: str = ""

@dataclass
class CoverageReport:
    season: int
    generated_at: str
    sources: list[SourceStatus]
    metrics_rows: dict
    notes: list[str]

    def to_json(self) -> str:
        payload = dict(
            season=self.season,
            generated_at=self.generated_at,
            sources=[asdict(s) for s in self.sources],
            metrics_rows=self.metrics_rows,
            notes=self.notes,
        )
        return json.dumps(payload, indent=2)

def log(msg: str):
    print(msg, flush=True)

def safe_len(df: t.Optional[pd.DataFrame]) -> int:
    try:
        return 0 if df is None else int(df.shape[0])
    except Exception:
        return 0

def _ensure_drive_result(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee 'drive_result' on pbp. Map from 'drive_end_event' if available,
    else create placeholder.
    """
    if 'drive_result' in pbp.columns:
        return pbp
    if 'drive_end_event' in pbp.columns:
        map_ev = {
            'Touchdown': 'Touchdown',
            'Opp touchdown': 'Opp touchdown',
            'Field goal': 'Field goal',
            'Missed FG': 'Missed FG',
            'Punt': 'Punt',
            'Fumble': 'Fumble',
            'Interception': 'Interception',
            'Safety': 'Safety',
            'End of half': 'End of half',
            'End of game': 'End of game',
            'Downs': 'Turnover on downs',
            'Timeout': 'Timeout',
        }
        dr = pbp['drive_end_event'].fillna('Unknown').astype(str)
        pbp['drive_result'] = dr.map(map_ev).fillna(dr)
        return pbp
    pbp['drive_result'] = 'Unknown'
    return pbp

def _neutral_mask(df: pd.DataFrame) -> pd.Series:
    """Neutral game state mask."""
    # score differential from offense perspective
    # nflverse columns can be posteam_score/defteam_score or score_differential; support both
    if 'score_differential' in df.columns:
        diff = df['score_differential']
    else:
        if {'posteam_score', 'defteam_score'}.issubset(df.columns):
            diff = df['posteam_score'] - df['defteam_score']
        else:
            # no score info; mark all as False
            return pd.Series([False]*len(df), index=df.index)

    q_ok = df.get('qtr', 0).isin(NEUTRAL_QUARTERS)
    down_ok = df.get('down', 0).isin(EARLY_DOWNS)
    return (diff.abs() <= NEUTRAL_SCORE_DIFF) & q_ok & down_ok

def _is_pass_play(df: pd.DataFrame) -> pd.Series:
    # nflverse has 'pass' boolean for dropbacks; fallback to play_type
    if 'pass' in df.columns:
        return df['pass'] == 1
    if 'play_type' in df.columns:
        return df['play_type'].astype(str).str.lower().eq('pass')
    # fallback: none
    return pd.Series([False]*len(df), index=df.index)

def _is_rush_play(df: pd.DataFrame) -> pd.Series:
    if 'rush' in df.columns:
        return df['rush'] == 1
    if 'play_type' in df.columns:
        return df['play_type'].astype(str).str.lower().eq('run')
    return pd.Series([False]*len(df), index=df.index)

def _yards_gained(df: pd.DataFrame) -> pd.Series:
    if 'yards_gained' in df.columns:
        return pd.to_numeric(df['yards_gained'], errors='coerce').fillna(0)
    return pd.Series([0]*len(df), index=df.index)

def _in_red_zone(df: pd.DataFrame) -> pd.Series:
    """
    Approximate: offense snaps with yards_to_goal <= 20. nflverse has 'yardline_100' or 'yardline' variants.
    """
    # yardline_100 is yards to opponent goal from offense perspective (0 at opp goal)
    y100 = None
    for cand in ['yardline_100', 'yardline_50_to_ydline', 'ydstogo100']:
        if cand in df.columns:
            y100 = pd.to_numeric(df[cand], errors='coerce')
            break
    # if not available, try yardline as numeric from own 0..50..opp 50..opp 0 is tricky; skip
    if y100 is None:
        return pd.Series([False]*len(df), index=df.index)
    # inside opp 20 means y100 <= 20
    return y100 <= 20

def _rate(n: pd.Series, d: pd.Series) -> pd.Series:
    d = d.replace(0, np.nan)
    return (n / d).fillna(0.0)

# ------------------------
# Fetchers
# ------------------------

def fetch_pbp(season: int) -> tuple[pd.DataFrame|None, SourceStatus]:
    """
    Download season PBP from nflverse. (nfl_data_py)
    """
    if nfl is None:
        return None, SourceStatus("nflverse_pbp", False, 0, "nfl_data_py not installed/imported")

    try:
        log(f"► Downloading play-by-play for season={season} …")
        pbp = nfl.import_pbp_data([season], downcast=True)
        log(f"✓ PBP rows: {pbp.shape[0]:,}")
        return pbp, SourceStatus("nflverse_pbp", True, pbp.shape[0], "")
    except Exception as e:
        return None, SourceStatus("nflverse_pbp", False, 0, f"{type(e).__name__}: {e}")

def fetch_id_map(season: int) -> tuple[pd.DataFrame|None, SourceStatus]:
    """
    Player id crosswalk (will be sparse out of season).
    """
    if nfl is None:
        return None, SourceStatus("nflverse_ids", False, 0, "nfl_data_py not installed/imported")
    try:
        ids = nfl.import_ids()
        # keep a small subset
        keep = [c for c in ['gsis_id', 'pfr_id', 'pfr_player_id', 'player_name',
                            'recent_team', 'position'] if c in ids.columns]
        out = ids[keep].drop_duplicates()
        return out, SourceStatus("nflverse_ids", True, out.shape[0], "")
    except Exception as e:
        return None, SourceStatus("nflverse_ids", False, 0, f"{type(e).__name__}: {e}")

def try_fetch_pfr_csv(url: str, timeout: int = 20) -> tuple[pd.DataFrame|None, str]:
    """
    Attempt to fetch a PFR table via simple HTTP. Many CI environments hit 403.
    Return (df, detail_message).
    """
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}: Forbidden/blocked likely (url={url})"
        # crude table read; PFR tables can be commented; let pandas try anyway
        dfs = pd.read_html(r.text)
        if not dfs:
            return None, "No tables found"
        # choose the largest table heuristic
        df = max(dfs, key=lambda d: d.shape[0]*d.shape[1])
        return df, "OK"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

def fetch_pfr_opp(season: int) -> tuple[pd.DataFrame|None, SourceStatus]:
    url = PFR_OPPONENT_URL.format(season=season)
    log(f"GET {url}")
    df, detail = try_fetch_pfr_csv(url)
    ok = df is not None
    rows = 0 if df is None else df.shape[0]
    return df, SourceStatus("pfr_opponent", ok, rows, detail)

def fetch_pfr_drives(season: int) -> tuple[pd.DataFrame|None, SourceStatus]:
    url = PFR_DRIVES_URL.format(season=season)
    log(f"GET {url}")
    df, detail = try_fetch_pfr_csv(url)
    ok = df is not None
    rows = 0 if df is None else df.shape[0]
    return df, SourceStatus("pfr_drives", ok, rows, detail)

# ------------------------
# Feature builders
# ------------------------

def build_team_week_form(pbp: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Weekly team features (explosive pass/rush rates, neutral pass rate, red-zone proxies).
    One row per (season, week, team).
    """
    df = pbp.copy()

    # Basic filters: regular plays only
    if 'play_type_nfl' in df.columns:
        # exclude non-plays like 'no_play', etc.
        mask_play = df['play_type_nfl'].notna()
        df = df[mask_play]

    # group key
    wk = df.get('week', np.nan)
    team = df.get('posteam', df.get('pos_team', df.get('offense', np.nan)))
    df = df.assign(_week=wk, _team=team)

    # Explosive flags
    yds = _yards_gained(df)
    pass_flag = _is_pass_play(df)
    rush_flag = _is_rush_play(df)
    exp_pass = (pass_flag) & (yds >= EXPLOSIVE_PASS_YDS)
    exp_rush = (rush_flag) & (yds >= EXPLOSIVE_RUSH_YDS)

    # Neutral situation + pass
    neutral = _neutral_mask(df)
    neutral_pass = neutral & pass_flag

    # Red-zone flags
    rz = _in_red_zone(df)
    rz_pass = rz & pass_flag
    rz_rush = rz & rush_flag

    g = df.groupby(['_team', '_week'], dropna=False)
    agg = pd.DataFrame({
        'plays': g.size(),
        'pass_plays': g[pass_flag].size(),
        'rush_plays': g[rush_flag].size(),
        'exp_pass_cnt': g[exp_pass].size(),
        'exp_rush_cnt': g[exp_rush].size(),
        'neutral_plays': g[neutral].size(),
        'neutral_pass_plays': g[neutral_pass].size(),
        'rz_plays': g[rz].size(),
        'rz_pass_plays': g[rz_pass].size(),
        'rz_rush_plays': g[rz_rush].size(),
    }).reset_index().rename(columns={'_team': 'team', '_week':'week'})

    # Rates
    agg['exp_pass_rate'] = _rate(agg['exp_pass_cnt'], agg['pass_plays'])
    agg['exp_rush_rate'] = _rate(agg['exp_rush_cnt'], agg['rush_plays'])
    agg['neutral_pass_rate'] = _rate(agg['neutral_pass_plays'], agg['neutral_plays'])
    agg['rz_pass_rate'] = _rate(agg['rz_pass_plays'], agg['rz_plays'])
    agg['rz_rush_rate'] = _rate(agg['rz_rush_plays'], agg['rz_plays'])

    agg.insert(0, 'season', season)
    return agg

def build_team_form(pbp: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Season-to-date team features (aggregate of week form).
    """
    week = build_team_week_form(pbp, season)
    g = week.groupby(['season', 'team'], dropna=False)
    out = g[['plays', 'pass_plays', 'rush_plays',
             'exp_pass_cnt', 'exp_rush_cnt',
             'neutral_plays', 'neutral_pass_plays',
             'rz_plays', 'rz_pass_plays', 'rz_rush_plays']].sum().reset_index()

    out['exp_pass_rate'] = _rate(out['exp_pass_cnt'], out['pass_plays'])
    out['exp_rush_rate'] = _rate(out['exp_rush_cnt'], out['rush_plays'])
    out['neutral_pass_rate'] = _rate(out['neutral_pass_plays'], out['neutral_plays'])
    out['rz_pass_rate'] = _rate(out['rz_pass_plays'], out['rz_plays'])
    out['rz_rush_rate'] = _rate(out['rz_rush_plays'], out['rz_plays'])
    return out

def build_player_form(pbp: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Lightweight per-player usage snapshot from pbp only (counts & proxies).
    If actual snap routes/targets are needed, you'd merge in route/snap tables later.

    Output one row per (season, player_id/gsis, team, position) with:
      pass_att, rush_att, targets (from pass attempts to player), rec (completions), air_yards if available, etc.
    """
    df = pbp.copy()

    posteam = df.get('posteam')
    defenses = df.get('defteam')

    # Identify skill involvement
    is_pass = _is_pass_play(df)
    is_rush = _is_rush_play(df)

    # Player columns in nflverse:
    pass_thrower = df.get('passer_id')
    rusher = df.get('rusher_id')
    receiver = df.get('receiver_id')
    complete = df.get('complete_pass', pd.Series([0]*len(df), index=df.index)).fillna(0).astype(int)

    # Build skinny event-level tables
    rows = []
    # Passing attempts by passer
    if pass_thrower is not None:
        gb = df[is_pass & pass_thrower.notna()].groupby([pass_thrower, posteam], dropna=False).size().reset_index(name="pass_att")
        gb.rename(columns={pass_thrower.name: 'player_id', posteam.name: 'team'}, inplace=True)
        gb['stat'] = 'pass_att'
        rows.append(gb)

    # Rush attempts by rusher
    if rusher is not None:
        gb = df[is_rush & rusher.notna()].groupby([rusher, posteam], dropna=False).size().reset_index(name="rush_att")
        gb.rename(columns={rusher.name: 'player_id', posteam.name: 'team'}, inplace=True)
        gb['stat'] = 'rush_att'
        rows.append(gb)

    # Targets & receptions by receiver
    if receiver is not None:
        tg = df[is_pass & receiver.notna()].groupby([receiver, posteam], dropna=False).size().reset_index(name="targets")
        tg.rename(columns={receiver.name: 'player_id', posteam.name: 'team'}, inplace=True)
        tg['stat'] = 'targets'
        rows.append(tg)

        rc = df[is_pass & receiver.notna()].groupby([receiver, posteam], dropna=False)['complete_pass'].sum().reset_index(name="receptions")
        rc.rename(columns={receiver.name: 'player_id', posteam.name: 'team'}, inplace=True)
        rc['stat'] = 'receptions'
        rows.append(rc)

    if not rows:
        return pd.DataFrame(columns=['season','player_id','team','pass_att','rush_att','targets','receptions'])

    tall = pd.concat(rows, ignore_index=True).fillna(0)
    # pivot to wide
    wide = tall.pivot_table(index=['player_id','team'], columns='stat', values=['pass_att','rush_att','targets','receptions'], aggfunc='sum', fill_value=0)
    # flatten columns
    wide.columns = [c[0] for c in wide.columns]
    wide = wide.reset_index()
    wide.insert(0, 'season', season)
    return wide

# ------------------------
# main
# ------------------------

def main(argv=None):
    import datetime as dt

    season = SEASON_DEFAULT
    if argv is None:
        argv = sys.argv[1:]

    # --season 2025
    for i, a in enumerate(argv):
        if a == "--season" and i+1 < len(argv):
            try:
                season = int(argv[i+1])
            except:
                pass

    mkdirs()

    sources: list[SourceStatus] = []
    notes: list[str] = []
    metrics_rows: dict[str, int] = {}

    # 1) PBP (nflverse)
    pbp, st = fetch_pbp(season)
    sources.append(st)
    if pbp is None or pbp.empty:
        notes.append("pbp unavailable; all derived metrics will be empty.")
        # still write empty outputs for consistency
        team_week = pd.DataFrame()
        team_form = pd.DataFrame()
        player_form = pd.DataFrame()
    else:
        # schema guard
        pbp = _ensure_drive_result(pbp)

        # 2) team-week and team season
        log("► Building team_week_form …")
        team_week = build_team_week_form(pbp, season)
        log(f"✓ Wrote metrics/team_week_form.csv ({team_week.shape[0]} rows)")
        team_week.to_csv(os.path.join(OUT_METRICS_DIR, "team_week_form.csv"), index=False)
        metrics_rows['team_week_form'] = int(team_week.shape[0])

        log("► Building team_form …")
        team_form = build_team_form(pbp, season)
        log(f"✓ Wrote metrics/team_form.csv ({team_form.shape[0]} rows)")
        team_form.to_csv(os.path.join(OUT_METRICS_DIR, "team_form.csv"), index=False)
        metrics_rows['team_form'] = int(team_form.shape[0])

        # 3) player_form (usage proxies from pbp)
        log("► Building player_form …")
        player_form = build_player_form(pbp, season)
        log(f"✓ Wrote metrics/player_form.csv ({player_form.shape[0]} rows)")
        player_form.to_csv(os.path.join(OUT_METRICS_DIR, "player_form.csv"), index=False)
        metrics_rows['player_form'] = int(player_form.shape[0])

    # 4) id map (nflverse)
    ids, st_ids = fetch_id_map(season)
    sources.append(st_ids)
    if ids is None:
        pd.DataFrame().to_csv(os.path.join(OUT_INPUTS_DIR, "id_map.csv"), index=False)
        metrics_rows['id_map'] = 0
    else:
        ids.to_csv(os.path.join(OUT_INPUTS_DIR, "id_map.csv"), index=False)
        metrics_rows['id_map'] = int(ids.shape[0])

    # 5) Weather (optional stub; still write header)
    weather_cols = ['game_id','stadium','roof','surface','temp','wind','precip']
    pd.DataFrame(columns=weather_cols).to_csv(os.path.join(OUT_INPUTS_DIR, "weather.csv"), index=False)
    metrics_rows['weather'] = 0

    # 6) PFR (try; likely 403 in CI)
    pfr_opp, st_opp = fetch_pfr_opp(season)
    sources.append(st_opp)
    if not st_opp.ok:
        notes.append(f"PFR opponent table unavailable: {st_opp.detail}")

    pfr_drives, st_drv = fetch_pfr_drives(season)
    sources.append(st_drv)
    if not st_drv.ok:
        notes.append(f"PFR drives table unavailable: {st_drv.detail}")

    # Summarize + write report
    rep = CoverageReport(
        season=season,
        generated_at=dt.datetime.utcnow().isoformat() + "Z",
        sources=sources,
        metrics_rows=metrics_rows,
        notes=notes,
    )
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(rep.to_json())

    # Human-readable summary
    print("\n===== External Metrics Fetch Summary =====")
    for s in sources:
        flag = "OK " if s.ok else "ERR"
        print(f"[{flag}] {s.name:16} rows={s.rows}  {s.detail}")
    print("---- Outputs ----")
    for k, v in metrics_rows.items():
        print(f"{k:18}: {v}")
    if notes:
        print("---- Notes ----")
        for n in notes:
            print("-", n)
    print(f"Report written: {REPORT_PATH}")
    print("=========================================\n")

if __name__ == "__main__":
    main()
