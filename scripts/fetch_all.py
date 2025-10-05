# scripts/fetch_all.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests


PFR_BASE = "https://www.pro-football-reference.com"

# PFR -> common NFL abbrev harmonization
PFR_TO_STD = {
    "GNB": "GB", "KAN": "KC", "NWE": "NE", "SFO": "SF", "TAM": "TB",
    "NOR": "NO", "SDG": "LAC", "LAC": "LAC", "LAR": "LAR", "STL": "LAR",
    "RAI": "LV", "OAK": "LV", "LVR": "LV", "CRD": "ARI", "PHO": "ARI", "ARI": "ARI",
    "RAV": "BAL", "CLT": "IND", "HTX": "HOU", "JAX": "JAX",
    "NYJ": "NYJ", "NYG": "NYG", "BUF": "BUF", "MIA": "MIA", "CIN": "CIN",
    "CLE": "CLE", "PIT": "PIT", "CHI": "CHI", "DET": "DET", "MIN": "MIN",
    "ATL": "ATL", "CAR": "CAR", "NOS": "NO", "DAL": "DAL", "PHI": "PHI",
    "WAS": "WAS", "SEA": "SEA", "TEN": "TEN", "JAC": "JAX", "HOU": "HOU",
    "BAL": "BAL", "IND": "IND", "DEN": "DEN", "ARI": "ARI", "TB": "TB",
    "GB": "GB", "KC": "KC", "SF": "SF", "NO": "NO", "LA": "LAR", "LAR": "LAR",
}


def _ensure_dirs() -> Dict[str, Path]:
    paths = {"metrics": Path("metrics"), "inputs": Path("inputs")}
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    print(f"Wrote {path} ({len(df)} rows)")


def _get(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/123.0 Safari/537.36"
        )
    }
    print(f"GET {url}")
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.text


def _read_tables(url: str) -> List[pd.DataFrame]:
    html = _get(url)
    dfs = pd.read_html(html)
    return dfs


def _std_team(s: str) -> str:
    s = str(s).strip().replace("*", "")
    return PFR_TO_STD.get(s, s).upper()


def _z(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    mu, sd = np.nanmean(x), np.nanstd(x)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - mu) / sd


# ------------------------------
# Season aggregates from PFR /years/{season}/opp.htm
# ------------------------------
def _agg_team_defense_from_pfr(season: int) -> pd.DataFrame:
    """Season-to-date team defense proxies with pressure, EPA-proxy, funnels."""
    url = f"{PFR_BASE}/years/{season}/opp.htm"
    try:
        df = max(_read_tables(url), key=lambda d: d.shape[1])
    except Exception as e:
        print(f"opp.htm fetch failed: {e}")
        return pd.DataFrame(columns=["team","pressure_z","pass_epa_z","run_funnel","pass_funnel"])

    # flexible column pick
    def pick(cols: List[str]) -> Optional[str]:
        for c in df.columns:
            c0 = str(c).lower()
            if any(w in c0 for w in cols):
                return c
        return None

    team_c = pick(["team"])
    pass_att_c = pick(["pass att","att"])
    sacks_c = pick(["sk"])
    qb_hits_c = pick(["qb hits"])
    sk_pct_c = pick(["sk%"])
    pass_ypa_c = pick(["yds/att","pass y/a","yards per pass"])
    rush_ypc_c = pick(["rush y/a","yards per rush"])

    if team_c is None:
        return pd.DataFrame(columns=["team","pressure_z","pass_epa_z","run_funnel","pass_funnel"])

    out = df.copy()
    out["team"] = out[team_c].map(_std_team)
    out = out[~out["team"].isna() & (out["team"] != "LEAGUE TOTAL")]

    def num(c): 
        return pd.to_numeric(out[c], errors="coerce") if c in out.columns else np.nan

    att = num(pass_att_c)
    sacks = num(sacks_c)
    qbh = num(qb_hits_c)
    skpct = num(sk_pct_c)
    ypa = num(pass_ypa_c)
    ypc = num(rush_ypc_c)

    # pressure proxy
    if skpct.notna().any():
        pressure = skpct
    else:
        pressure = (sacks.add(qbh.fillna(0), fill_value=0) / att.replace(0, np.nan)) * 100.0

    inv_ypa = -ypa  # lower ypa allowed => better defense => higher z

    out = pd.DataFrame({
        "team": out["team"],
        "pressure_z": _z(pressure).fillna(0),
        "pass_epa_z": _z(inv_ypa).fillna(0),
        "pass_ypa_allowed": ypa,
        "rush_ypc_allowed": ypc,
    })

    p_med = np.nanmedian(out["pass_ypa_allowed"])
    r_med = np.nanmedian(out["rush_ypc_allowed"])
    out["pass_funnel"] = ((out["rush_ypc_allowed"] <= r_med) & (out["pass_ypa_allowed"] > p_med)).astype(int)
    out["run_funnel"]  = ((out["pass_ypa_allowed"] <= p_med) & (out["rush_ypc_allowed"] > r_med)).astype(int)

    return out[["team","pressure_z","pass_epa_z","run_funnel","pass_funnel"]]


# ------------------------------
# Drive stats /years/{season}/drives.htm
# ------------------------------
def _drive_stats(season: int) -> pd.DataFrame:
    url = f"{PFR_BASE}/years/{season}/drives.htm"
    try:
        dfs = _read_tables(url)
    except Exception as e:
        print(f"drives.htm fetch failed: {e}")
        return pd.DataFrame(columns=["team","drives_z","plays_per_drive","yds_per_drive","score_pct"])

    # Heuristic: pick the "Team Drive Stats" table (has 'Team' & 'Plays/Drive')
    def looks_like(d: pd.DataFrame) -> bool:
        cols = " ".join(map(str, d.columns)).lower()
        return ("team" in cols) and ("plays/drive" in cols or "plays/ drv" in cols or "pl/drive" in cols)

    drive = next((d for d in dfs if looks_like(d)), pd.DataFrame())
    if drive.empty:
        # try the widest
        drive = max(dfs, key=lambda d: d.shape[1]) if dfs else pd.DataFrame()
    if drive.empty:
        return pd.DataFrame(columns=["team","drives_z","plays_per_drive","yds_per_drive","score_pct"])

    # best-effort column picks
    def pick(cols: List[str]) -> Optional[str]:
        for c in drive.columns:
            c0 = str(c).lower()
            if any(w in c0 for w in cols):
                return c
        return None

    team_c = pick(["team"])
    ppd_c  = pick(["plays/drive","pl/drive"])
    ypd_c  = pick(["yards/drive","yds/drive","yds/ drv"])
    scp_c  = pick(["score%", "scores/drive","score pct","pts/drive"])  # available variants

    d = drive.copy()
    d["team"] = d[team_c].map(_std_team) if team_c in d.columns else d.index.astype(str)
    d = d[~d["team"].isna() & (d["team"] != "LEAGUE TOTAL")]

    def num(c):
        return pd.to_numeric(d[c], errors="coerce") if c in d.columns else np.nan

    ppd = num(ppd_c)
    ypd = num(ypd_c)
    scp = num(scp_c)

    # composite "defense-hardness via drives" proxy: fewer opp yards/drive + fewer scores/drive
    # Here we use offense team table (drives for the team). To convert to defense context, we
    # take the inverse signals: strong defense => lower opp yds/drive, lower opp score%.
    # Using negatives to make "higher is better defense" for z-score.
    comp = _z(-ypd) + _z(-scp)

    out = pd.DataFrame({
        "team": d["team"],
        "plays_per_drive": ppd,
        "yds_per_drive": ypd,
        "score_pct": scp,
        "drives_z": _z(comp).fillna(0),
    })
    out = out.groupby("team", as_index=False).agg("first")
    return out[["team","drives_z","plays_per_drive","yds_per_drive","score_pct"]]


# ------------------------------
# Explosive plays allowed – proxied from opp.htm (20+ pass & rush)
# ------------------------------
def _explosive_allowed(season: int) -> pd.DataFrame:
    url = f"{PFR_BASE}/years/{season}/opp.htm"
    try:
        df = max(_read_tables(url), key=lambda d: d.shape[1])
    except Exception as e:
        print(f"opp.htm for explosive failed: {e}")
        return pd.DataFrame(columns=["team","explosive_z","explosive_rate"])

    def pick(cols: List[str]) -> Optional[str]:
        for c in df.columns:
            c0 = str(c).lower()
            if any(w in c0 for w in cols):
                return c
        return None

    team_c = pick(["team"])
    plays_c = pick(["plays"])  # total plays defended
    # these vary a lot on PFR; cast a wide net
    p20_c = pick(["pass 20", "20+ pass", "pass plays 20", "pass plays of 20"])
    r20_c = pick(["rush 20", "20+ rush", "rush plays 20", "rush plays of 20"])

    if team_c is None:
        return pd.DataFrame(columns=["team","explosive_z","explosive_rate"])

    d = df.copy()
    d["team"] = d[team_c].map(_std_team)
    d = d[~d["team"].isna() & (d["team"] != "LEAGUE TOTAL")]

    def n(c):
        return pd.to_numeric(d[c], errors="coerce") if c in d.columns else np.nan

    plays = n(plays_c)
    p20 = n(p20_c)
    r20 = n(r20_c)

    # If 20+ columns are missing, fall back to a weak proxy from longest plays columns (very noisy).
    if p20.isna().all() and r20.isna().all():
        long_pass_c = pick(["long pass","longest pass","long pass play"])
        long_rush_c = pick(["long rush","longest rush","long rush play"])
        lp = n(long_pass_c)
        lr = n(long_rush_c)
        proxy = _z(lp) + _z(lr)
        return pd.DataFrame({
            "team": d["team"],
            "explosive_rate": np.nan,
            "explosive_z": _z(proxy).fillna(0)
        })

    explosive = p20.fillna(0).add(r20.fillna(0), fill_value=0)
    rate = explosive / plays.replace(0, np.nan)
    return pd.DataFrame({
        "team": d["team"],
        "explosive_rate": rate,
        "explosive_z": _z(-rate).fillna(0),  # lower rate allowed => better defense => higher z
    })


# ------------------------------
# Weekly defensive splits – /teams/{abbr}/{season}/gamelog/
# ------------------------------
def _team_abbrs_from_opp(season: int) -> List[str]:
    url = f"{PFR_BASE}/years/{season}/opp.htm"
    try:
        df = max(_read_tables(url), key=lambda d: d.shape[1])
    except Exception:
        return []
    # team col with links holds abbrs in href: /teams/kan/2025.htm
    # pandas read_html loses the href; fallback to team names and map via dict guess
    teams = df[df.columns[0]].astype(str).str.replace("*", "", regex=False).str.strip()
    teams = [PFR_TO_STD.get(t, t).upper() for t in teams if t and t.lower() != "league total"]
    # get unique and filter obvious junk
    teams = sorted({t for t in teams if t.isalpha() and len(t) <= 3})
    return teams


def _gamelog_weekly_defense(season: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    teams = _team_abbrs_from_opp(season)
    # PFR gamelog path uses lowercase 3-letter
    pfr_guess = {v: k.lower() for k, v in PFR_TO_STD.items()}  # std->pfr-ish
    for t in teams:
        pfr = pfr_guess.get(t, t.lower())
        url = f"{PFR_BASE}/teams/{pfr}/{season}/gamelog/"
        try:
            dfs = _read_tables(url)
        except Exception as e:
            print(f"gamelog fetch failed for {t}: {e}")
            continue

        # pick widest table that has 'Week' and 'Opp'
        def looks_like(d: pd.DataFrame) -> bool:
            cols = " ".join(map(str, d.columns)).lower()
            return ("week" in cols) and ("opp" in cols)

        g = next((d for d in dfs if looks_like(d)), (pd.DataFrame()))
        if g.empty:
            g = max(dfs, key=lambda d: d.shape[1]) if dfs else pd.DataFrame()
        if g.empty:
            continue

        # flexible picks
        def pick(cols: List[str]) -> Optional[str]:
            for c in g.columns:
                c0 = str(c).lower()
                if any(w in c0 for w in cols):
                    return c
            return None

        week_c = pick(["week"])
        opp_c  = pick(["opp"])
        opp_pass_yds_c = pick(["opp pass yds","opp passing yards","opp pass yards"])
        opp_pass_att_c = pick(["opp pass att","opp passes"])
        opp_rush_yds_c = pick(["opp rush yds","opp rushing yards"])
        opp_rush_att_c = pick(["opp rush att","opp rushes"])
        opp_sacks_c    = pick(["sk","sacks"])  # team sacks vs opp

        if week_c is None or opp_c is None:
            continue

        gl = g.copy()
        # remove non-regular rows (bye, playoffs headers, etc.)
        def to_int(x):
            try:
                return int(x)
            except:
                return np.nan

        gl["week"] = pd.to_numeric(gl[week_c].apply(to_int), errors="coerce")
        gl = gl.dropna(subset=["week"])
        gl["week"] = gl["week"].astype(int)

        def num(c):
            return pd.to_numeric(gl[c], errors="coerce") if c in gl.columns else np.nan

        pass_y = num(opp_pass_yds_c)
        pass_a = num(opp_pass_att_c)
        rush_y = num(opp_rush_yds_c)
        rush_a = num(opp_rush_att_c)
        sacks  = num(opp_sacks_c)

        ypa = pass_y / pass_a.replace(0, np.nan)
        ypc = rush_y / rush_a.replace(0, np.nan)

        for i in range(len(gl)):
            rows.append({
                "team": t,
                "week": int(gl.iloc[i]["week"]),
                "opp": str(gl.iloc[i][opp_c]).upper() if opp_c in gl.columns else "",
                "pass_att_allowed": float(pass_a.iloc[i]) if len(pass_a) > i else np.nan,
                "pass_yds_allowed": float(pass_y.iloc[i]) if len(pass_y) > i else np.nan,
                "rush_att_allowed": float(rush_a.iloc[i]) if len(rush_a) > i else np.nan,
                "rush_yds_allowed": float(rush_y.iloc[i]) if len(rush_y) > i else np.nan,
                "sacks": float(sacks.iloc[i]) if len(sacks) > i else np.nan,
                "ypa_allowed": float(ypa.iloc[i]) if len(ypa) > i else np.nan,
                "ypc_allowed": float(ypc.iloc[i]) if len(ypc) > i else np.nan,
            })

    wk = pd.DataFrame(rows)
    if wk.empty:
        return pd.DataFrame(columns=[
            "team","week","opp","pass_att_allowed","pass_yds_allowed",
            "rush_att_allowed","rush_yds_allowed","sacks","ypa_allowed","ypc_allowed"
        ])
    # clean team & opp codes
    wk["team"] = wk["team"].map(_std_team)
    wk["opp"]  = wk["opp"].map(_std_team)
    wk = wk.sort_values(["team","week"]).reset_index(drop=True)
    return wk


# ------------------------------
# Stubs to keep engine/rules happy
# ------------------------------
def _empty_player_form() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "gsis_id","week","usage_routes","usage_targets","usage_carries","mu_pred","sigma_pred"
    ])


def _empty_id_map() -> pd.DataFrame:
    return pd.DataFrame(columns=["player_name","gsis_id","recent_team","position"])


def _weather_shell() -> pd.DataFrame:
    return pd.DataFrame(columns=["game_id","wind_mph","temp_f"])


# ------------------------------
# Orchestrator
# ------------------------------
def build_all(season: int) -> None:
    paths = _ensure_dirs()

    # season aggregates
    agg = _agg_team_defense_from_pfr(season)
    drv = _drive_stats(season)
    xpl = _explosive_allowed(season)

    # merge into team_form
    team_form = agg.merge(drv, on="team", how="left").merge(xpl, on="team", how="left")

    # keep the older columns engine expects; extras are harmless
    if "week" not in team_form.columns:
        team_form["week"] = 0

    _save_csv(team_form, paths["metrics"] / "team_form.csv")

    # weekly defensive splits
    wk = _gamelog_weekly_defense(season)
    _save_csv(wk, paths["metrics"] / "team_week_form.csv")

    # stubs
    _save_csv(_empty_player_form(), paths["metrics"] / "player_form.csv")
    _save_csv(_empty_id_map(), paths["inputs"] / "id_map.csv")
    _save_csv(_weather_shell(), paths["inputs"] / "weather.csv")


def main():
    ap = argparse.ArgumentParser(description="Fetch external metrics from PFR (season aggregates + weekly).")
    ap.add_argument("--season", type=int, required=True)
    args = ap.parse_args()
    build_all(args.season)


if __name__ == "__main__":
    main()
