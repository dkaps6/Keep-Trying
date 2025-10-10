from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

def _safe(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

def _upp(x: pd.Series) -> pd.Series:
    return x.astype(str).str.upper()

def _try_import_nfl():
    try:
        import nfl_data_py as nfl
        return nfl
    except Exception:
        return None

def _read_csv(p: str) -> pd.DataFrame:
    path = Path(p)
    if path.exists() and path.stat().st_size>0:
        try: return pd.read_csv(path)
        except Exception: pass
    return pd.DataFrame()


def load_nflverse_pbp(season: int) -> pd.DataFrame:
    nfl = _try_import_nfl()
    if not nfl: return pd.DataFrame()
    try: return nfl.import_pbp_data(years=[season])
    except Exception: return pd.DataFrame()

def _red_zone_shares_from_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    if pbp is None or pbp.empty: 
        return pd.DataFrame(columns=["player","team","rz_tgt_share","rz_carry_share"])
    df = pbp.copy()
    # Standardize
    for c in ["posteam","defteam","rusher_player_name","receiver_player_name","passer_player_name"]:
        if c in df.columns: df[c] = df[c].astype(str)
    df["team"] = df.get("posteam", "").astype(str).str.upper()
    # RZ filter: yardline_100 <= 20 (if missing, approximate by red_zone boolean if present)
    if "yardline_100" in df.columns:
        df = df[df["yardline_100"] <= 20]
    elif "red_zone" in df.columns:
        df = df[df["red_zone"] == 1]
    else:
        return pd.DataFrame(columns=["player","team","rz_tgt_share","rz_carry_share"])

    # rushes
    df["is_rush"] = (df.get("rush", 0).fillna(0) > 0)
    df["is_pass"] = (df.get("pass", 0).fillna(0) > 0)

    rush = df[df["is_rush"]].copy()
    rush["player"] = rush.get("rusher_player_name","").astype(str)
    rush = rush[rush["player"].ne("")]
    rz_carry_by_player = rush.groupby(["player","team"]).size().rename("rz_carries")

    # targets in RZ (use receiver name)
    rec = df[df["is_pass"]].copy()
    rec["player"] = rec.get("receiver_player_name","").astype(str)
    rec = rec[rec["player"].ne("")]
    rz_tgts_by_player = rec.groupby(["player","team"]).size().rename("rz_tgts")

    tot_rz_carries = rush.groupby("team").size().rename("team_rz_carries")
    tot_rz_tgts = rec.groupby("team").size().rename("team_rz_tgts")

    out = pd.DataFrame(index=set(rz_carry_by_player.index).union(set(rz_tgts_by_player.index)))
    out = out.join(rz_carry_by_player, how="outer").join(rz_tgts_by_player, how="outer")
    out = out.reset_index().rename(columns={"level_0":"player","level_1":"team"})
    out["team"] = out["team"].astype(str).str.upper()
    out = out.merge(tot_rz_carries.reset_index(), on="team", how="left")
    out = out.merge(tot_rz_tgts.reset_index(), on="team", how="left")
    out["rz_carry_share"] = (out["rz_carries"] / out["team_rz_carries"].replace(0, pd.NA)).fillna(0.0).clip(0,1)
    out["rz_tgt_share"]   = (out["rz_tgts"] / out["team_rz_tgts"].replace(0, pd.NA)).fillna(0.0).clip(0,1)
    return out[["player","team","rz_tgt_share","rz_carry_share"]]
def load_nflverse_weekly(season: int) -> pd.DataFrame:
    nfl = _try_import_nfl()
    if not nfl: return pd.DataFrame()
    try: return nfl.import_weekly_data(years=[season])
    except Exception: return pd.DataFrame()

def load_espn_players() -> pd.DataFrame: return _read_csv("data/player_form_espn.csv")
def load_gsis_players() -> pd.DataFrame: return _read_csv("data/player_form_nflgsis.csv")
def load_msf_players()  -> pd.DataFrame: return _read_csv("data/player_form_msf.csv")
def load_api_players()  -> pd.DataFrame: return _read_csv("data/player_form_apisports.csv")

def _aggregate_shares(weekly: pd.DataFrame) -> pd.DataFrame:
    if weekly.empty:
        return pd.DataFrame(columns=["player","team","position","target_share","rush_share","rz_tgt_share","rz_carry_share","yprr_proxy","ypc","ypt","qb_ypa","mu","sd","sd_widen"])
    w = weekly.copy()
    for c in ["recent_team","team","player_name","position"]:
        if c in w.columns: w[c]=w[c].astype(str)
    w["team_u"] = _upp(w.get("recent_team") if "recent_team" in w.columns else w.get("team"))
    for c in ["targets","receptions","rushing_attempts","rushing_yards","receiving_yards","attempts","passing_yards"]:
        if c not in w.columns: w[c]=0
    team_tgts = w.groupby("team_u")["targets"].sum().rename("team_targets")
    team_carr = w.groupby("team_u")["rushing_attempts"].sum().rename("team_carries")
    agg = w.groupby(["player_name","team_u","position"], as_index=False).agg({
        "targets":"sum","rushing_attempts":"sum","rushing_yards":"sum","receiving_yards":"sum","attempts":"sum","passing_yards":"sum"
    }).merge(team_tgts, left_on="team_u", right_index=True, how="left").merge(team_carr, left_on="team_u", right_index=True, how="left")
    agg["target_share"]=(agg["targets"]/agg["team_targets"].replace(0,np.nan)).fillna(0).clip(0,1)
    agg["rush_share"]=(agg["rushing_attempts"]/agg["team_carries"].replace(0,np.nan)).fillna(0).clip(0,1)
    agg["ypt"]=(agg["receiving_yards"]/agg["targets"].replace(0,np.nan)).fillna(0)
    agg["ypc"]=(agg["rushing_yards"]/agg["rushing_attempts"].replace(0,np.nan)).fillna(0)
    agg["qb_ypa"]=(agg["passing_yards"]/agg["attempts"].replace(0,np.nan)).fillna(0)
    agg["yprr_proxy"]=(agg["receiving_yards"]/(agg["targets"]*1.5).replace(0,np.nan)).fillna(0).clip(0.4,3.5)
    agg["rz_tgt_share"]=0.0; agg["rz_carry_share"]=0.0
    agg["mu"]=None; agg["sd"]=None; agg["sd_widen"]=1.0
    out = agg.rename(columns={"player_name":"player","team_u":"team"})[["player","team","position","target_share","rush_share","rz_tgt_share","rz_carry_share","yprr_proxy","ypc","ypt","qb_ypa","mu","sd","sd_widen"]]
    out["team"]=_upp(out["team"])
    return out

def _merge_fallback_cols(base: pd.DataFrame, fb: pd.DataFrame, cols: list) -> pd.DataFrame:
    if fb is None or fb.empty or "player" not in fb.columns or "team" not in fb.columns: return base
    out=base.copy(); fb=fb.copy(); fb["team"]=_upp(fb["team"]); fb["player"]=fb["player"].astype(str)
    out=out.merge(fb[["player","team"]+cols], on=["player","team"], how="left", suffixes=("", "_fb"))
    for c in cols:
        colf=f"{c}_fb"
        if colf in out.columns:
            out[c]=out[c].where(~out[c].isna(), out[colf])
            out.drop(columns=[colf], inplace=True)
    return out

def build_player_form(season: int) -> pd.DataFrame:
    Path("data").mkdir(parents=True, exist_ok=True); Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    weekly = load_nflverse_weekly(season)
    pbp = load_nflverse_pbp(season)
    base = _aggregate_shares(weekly)
    rz = _red_zone_shares_from_pbp(pbp)
    if not rz.empty:
        base = base.merge(rz, on=['player','team'], how='left', suffixes=('','_rz'))
        for c in ['rz_tgt_share','rz_carry_share']:
            rzcol = f"{c}_rz"
            if rzcol in base.columns:
                base[c] = base[c].where(~base[c].isna(), base[rzcol])
                base.drop(columns=[rzcol], inplace=True)
    need=["target_share","rush_share","rz_tgt_share","rz_carry_share","yprr_proxy","ypc","ypt","qb_ypa","mu","sd","sd_widen"]
    base=_merge_fallback_cols(base, load_espn_players(), need)
    base=_merge_fallback_cols(base, load_gsis_players(), need)
    base=_merge_fallback_cols(base, load_msf_players(), need)
    base=_merge_fallback_cols(base, load_api_players(), need)
    base.to_csv("data/player_form.csv", index=False); base.to_csv("outputs/metrics/player_form.csv", index=False)
    print(f"[build_player_form] ✅ rows={len(base)} → data/player_form.csv & outputs/metrics/player_form.csv")
    return base
