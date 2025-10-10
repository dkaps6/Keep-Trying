from __future__ import annotations
import argparse, pandas as pd, numpy as np
from pathlib import Path
from scripts.utils.names import normalize_team

def _z(x): return (x - x.mean())/x.std(ddof=0) if x.std(ddof=0)>0 else x*0

def _nflverse_team(season:int) -> pd.DataFrame:
    import nfl_data_py as nfl
    pbp = nfl.import_pbp_data([season])
    # team offense vs opponent defense
    grp = pbp.groupby("defteam").agg(
        def_pass_epa = ("epa", lambda s: s[pbp.loc[s.index,"pass"]==1].mean()),
        def_rush_epa = ("epa", lambda s: s[pbp.loc[s.index,"rush"]==1].mean()),
        def_sack_rate = ("qb_hit", lambda s: (pbp.loc[s.index,"sack"].sum())/max(1,len(s))),
        light_box_rate = ("box_players", lambda s: (pbp.loc[s.index,"box_players"]<=6).mean() if "box_players" in pbp else np.nan),
        heavy_box_rate = ("box_players", lambda s: (pbp.loc[s.index,"box_players"]>=8).mean() if "box_players" in pbp else np.nan),
    ).reset_index().rename(columns={"defteam":"team"})
    # pace & PROE (rough from situation-neutral: 1st/2nd, close score)
    nu = pbp[(pbp.down.isin([1,2])) & (pbp.score_differential.abs()<=7)]
    pace = nu.groupby("posteam").apply(lambda d: d["game_seconds_remaining"].diff().abs().dropna().mean()).reset_index()
    pace.columns=["team","pace"]
    proe = nu.groupby("posteam")["pass"].mean().reset_index().rename(columns={"posteam":"team","pass":"proe"})
    df = grp.merge(pace, on="team", how="left").merge(proe, on="team", how="left")
    return df

def _fallback_csv(name:str) -> pd.DataFrame:
    p = Path(f"data/{name}.csv")
    return pd.read_csv(p) if p.exists() and p.stat().st_size>0 else pd.DataFrame()

def _merge_colwise(base:pd.DataFrame, fb:pd.DataFrame, on="team"):
    if fb.empty: return base
    cols = [c for c in fb.columns if c!=on]
    base = base.merge(fb[[on]+cols], on=on, how="left", suffixes=("","_fb"))
    for c in cols:
        if c not in base.columns: continue
        fbcol = f"{c}_fb"
        base[c] = np.where(base[c].isna(), base[fbcol], base[c])
        if fbcol in base.columns: base.drop(columns=[fbcol], inplace=True)
    return base

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--season", type=int, required=True)
    a=ap.parse_args()
    Path("data").mkdir(exist_ok=True); Path("outputs/metrics").mkdir(parents=True, exist_ok=True)

    df = _nflverse_team(a.season)
    df["team"]=df["team"].map(normalize_team)
    # column-by-column fallbacks from provider-specific CSVs if present
    for src in ["team_form_espn","team_form_nflgsis","team_form_msf","team_form_apisports"]:
        df = _merge_colwise(df, _fallback_csv(src))

    # z-scores for pricing
    for c in ["def_pass_epa","def_rush_epa","def_sack_rate","light_box_rate","heavy_box_rate","pace","proe"]:
        if c in df.columns: df[c+"_z"] = _z(df[c].fillna(df[c].median()))

    df.to_csv("data/team_form.csv", index=False)
    df.to_csv("outputs/metrics/team_form.csv", index=False)
    print(f"[team_form] wrote rows={len(df)} → data/team_form.csv")

if __name__=="__main__":
    main()
