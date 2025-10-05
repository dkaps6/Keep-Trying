# scripts/fetch_all.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests


PFR_BASE = "https://www.pro-football-reference.com"

# Map PFR quirks -> common abbreviations
PFR_TO_STD = {
    "GNB": "GB", "KAN": "KC", "NWE": "NE", "SFO": "SF", "TAM": "TB",
    "NOR": "NO", "SDG": "LAC", "LAC": "LAC", "LAR": "LAR", "STL": "LAR",
    "RAI": "LV", "OAK": "LV", "LVR": "LV", "CRD": "ARI", "PHO": "ARI", "ARI": "ARI",
    "RAV": "BAL", "CLT": "IND", "HTX": "HOU", "JAX": "JAX",
    "NYJ": "NYJ", "NYG": "NYG", "BUF": "BUF", "MIA": "MIA", "CIN": "CIN",
    "CLE": "CLE", "PIT": "PIT", "CHI": "CHI", "DET": "DET", "MIN": "MIN",
    "ATL": "ATL", "CAR": "CAR", "DAL": "DAL", "PHI": "PHI",
    "WAS": "WAS", "SEA": "SEA", "TEN": "TEN", "HOU": "HOU",
    "BAL": "BAL", "IND": "IND", "DEN": "DEN", "TB": "TB", "GB": "GB", "KC": "KC",
    "SF": "SF", "NO": "NO", "LA": "LAR", "LAR": "LAR",
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
        return pd.DataFrame(columns=["team","pressure_z","pass_epa_z","run_funnel","pass_funnel",
                                     "pass_ypa_allowed","rush_ypc_allowed"])

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
        return pd.DataFrame(columns=["team","pressure_z","pass_epa_z","run_funnel","pass_funnel",
                                     "pass_ypa_allowed","rush_ypc_allowed"])

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

    return out


# ------------------------------
# Drive stats /years/{season}/drives.htm (Team + Opponent)
# ------------------------------
def _drive_stats_off_def(season: int) -> pd.DataFrame:
    url = f"{PFR_BASE}/years/{season}/drives.htm"
    try:
        dfs = _read_tables(url)
    except Exception as e:
        print(f"drives.htm fetch failed: {e}")
        return pd.DataFrame(columns=[
            "team",
            "plays_per_drive", "yds_per_drive", "score_pct",
            "opp_plays_per_drive", "opp_yds_per_drive", "opp_score_pct",
            "drives_z"
        ])

    def looks_like(d: pd.DataFrame) -> bool:
        cols = " ".join(map(str, d.columns)).lower()
        return ("team" in cols) and ("plays/drive" in cols or "pl/drive" in cols)

    tables = [d for d in dfs if looks_like(d)]
    if not tables:
        tables = [max(dfs, key=lambda d: d.shape[1])] if dfs else []

    # Heuristic: first = Team Drive Stats, second = Opponent Drive Stats
    team_df = tables[0] if tables else pd.DataFrame()
    opp_df  = tables[1] if len(tables) > 1 else pd.DataFrame()

    def pick(df: pd.DataFrame, keys: List[str]) -> Optional[str]:
        for c in df.columns:
            c0 = str(c).lower()
            if any(w in c0 for w in keys):
                return c
        return None

    def clean_one(df: pd.DataFrame, is_opp: bool = False) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        team_c = pick(df, ["team"])
        ppd_c  = pick(df, ["plays/drive","pl/drive"])
        ypd_c  = pick(df, ["yards/drive","yds/drive","yds/ drv"])
        scp_c  = pick(df, ["score%", "scores/drive","score pct","pts/drive"])

        d = df.copy()
        d["team"] = d[team_c].map(_std_team) if team_c in d.columns else d.index.astype(str)
        d = d[~d["team"].isna() & (d["team"] != "LEAGUE TOTAL")]

        def num(c):
            return pd.to_numeric(d[c], errors="coerce") if c in d.columns else np.nan

        out = pd.DataFrame({"team": d["team"]})
        out["ppd"] = num(ppd_c)
        out["ypd"] = num(ypd_c)
        out["scp"] = num(scp_c)
        suffix = "" if not is_opp else "opp_"
        out = out.rename(columns={
            "ppd": f"{suffix}plays_per_drive",
            "ypd": f"{suffix}yds_per_drive",
            "scp": f"{suffix}score_pct",
        })
        return out.groupby("team", as_index=False).agg("first")

    team_clean = clean_one(team_df, is_opp=False)
    opp_clean  = clean_one(opp_df,  is_opp=True)

    merged = team_clean.merge(opp_clean, on="team", how="left")

    # Composite defensive difficulty via opponent yards/drive + opponent score%
    comp_def = _z(-merged.get("opp_yds_per_drive")) + _z(-merged.get("opp_score_pct"))
    merged["drives_z"] = _z(comp_def).fillna(0)

    return merged[[
        "team",
        "plays_per_drive", "yds_per_drive", "score_pct",
        "opp_plays_per_drive", "opp_yds_per_drive", "opp_score_pct",
        "drives_z"
    ]]


# ------------------------------
# Explosive plays allowed – split pass/rush from opp.htm
# ------------------------------
def _explosive_allowed_split(season: int) -> pd.DataFrame:
    url = f"{PFR_BASE}/years/{season}/opp.htm"
    try:
        df = max(_read_tables(url), key=lambda d: d.shape[1])
    except Exception as e:
        print(f"opp.htm for explosive failed: {e}")
        return pd.DataFrame(columns=[
            "team","explosive_rate","explosive_z",
            "explosive_pass_rate","explosive_rush_rate",
            "explosive_pass_z","explosive_rush_z"
        ])

    def pick(cols: List[str]) -> Optional[str]:
        for c in df.columns:
            c0 = str(c).lower()
            if any(w in c0 for w in cols):
                return c
        return None

    team_c = pick(["team"])
    plays_c = pick(["plays"])
    p20_c = pick(["pass 20", "20+ pass", "pass plays 20", "pass plays of 20"])
    r20_c = pick(["rush 20", "20+ rush", "rush plays 20", "rush plays of 20"])

    if team_c is None or plays_c is None:
        return pd.DataFrame(columns=[
            "team","explosive_rate","explosive_z",
            "explosive_pass_rate","explosive_rush_rate",
            "explosive_pass_z","explosive_rush_z"
        ])

    d = df.copy()
    d["team"] = d[team_c].map(_std_team)
    d = d[~d["team"].isna() & (d["team"] != "LEAGUE TOTAL")]

    def n(c):
        return pd.to_numeric(d[c], errors="coerce") if c in d.columns else np.nan

    plays = n(plays_c)
    p20 = n(p20_c)
    r20 = n(r20_c)

    # Fallback: if split columns missing, return combined only
    if p20.isna().all() and r20.isna().all():
        # try "longest" as super-weak proxy for combined explosive
        long_pass_c = pick(["long pass","longest pass","long pass play"])
        long_rush_c = pick(["long rush","longest rush","long rush play"])
        lp = n(long_pass_c)
        lr = n(long_rush_c)
        combined_proxy = _z(lp) + _z(lr)
        return pd.DataFrame({
            "team": d["team"],
            "explosive_rate": np.nan,
            "explosive_z": _z(combined_proxy).fillna(0),
            "explosive_pass_rate": np.nan,
            "explosive_rush_rate": np.nan,
            "explosive_pass_z": np.nan,
            "explosive_rush_z": np.nan,
        })

    combined = p20.fillna(0).add(r20.fillna(0), fill_value=0)
    combined_rate = combined / plays.replace(0, np.nan)

    pass_rate = p20 / plays.replace(0, np.nan)
    rush_rate = r20 / plays.replace(0, np.nan)

    return pd.DataFrame({
        "team": d["team"],
        "explosive_rate": combined_rate,
        "explosive_z": _z(-combined_rate).fillna(0),  # lower allowed => better defense
        "explosive_pass_rate": pass_rate,
        "explosive_rush_rate": rush_rate,
        "explosive_pass_z": _z(-pass_rate).fillna(0),
        "explosive_rush_z": _z(-rush_rate).fillna(0),
    })


# ------------------------------
# Weekly defensive splits – /teams/{abbr}/{season}/gamelog/
# + Neutral pass rate (defense) proxied from 1-score games
# ------------------------------
def _team_abbrs_from_opp(season: int) -> List[str]:
    url = f"{PFR_BASE}/years/{season}/opp.htm"
    try:
        df = max(_read_tables(url), key=lambda d: d.shape[1])
    except Exception:
        return []
    teams = df[df.columns[0]].astype(str).str.replace("*", "", regex=False).str.strip()
    teams = [PFR_TO_STD.get(t, t).upper() for t in teams if t and t.lower() != "league total"]
    teams = sorted({t for t in teams if t.isalpha() and len(t) <= 3})
    return teams


def _gamelog_weekly_defense_and_neutral(season: int) -> (pd.DataFrame, pd.DataFrame):
    rows: List[Dict[str, object]] = []
    neutral_rows: List[Dict[str, object]] = []
    teams = _team_abbrs_from_opp(season)
    # std->pfr guess
    pfr_guess = {v: k.lower() for k, v in PFR_TO_STD.items()}
    for t in teams:
        pfr = pfr_guess.get(t, t.lower())
        url = f"{PFR_BASE}/teams/{pfr}/{season}/gamelog/"
        try:
            dfs = _read_tables(url)
        except Exception as e:
            print(f"gamelog fetch failed for {t}: {e}")
            continue

        def looks_like(d: pd.DataFrame) -> bool:
            cols = " ".join(map(str, d.columns)).lower()
            return ("week" in cols) and ("opp" in cols)

        g = next((d for d in dfs if looks_like(d)), (pd.DataFrame()))
        if g.empty:
            g = max(dfs, key=lambda d: d.shape[1]) if dfs else pd.DataFrame()
        if g.empty:
            continue

        def pick(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
            for c in df.columns:
                c0 = str(c).lower()
                if any(w in c0 for w in candidates):
                    return c
            return None

        week_c = pick(g, ["week"])
        opp_c  = pick(g, ["opp"])
        opp_pass_yds_c = pick(g, ["opp pass yds","opp passing yards","opp pass yards"])
        opp_pass_att_c = pick(g, ["opp pass att","opp passes"])
        opp_rush_yds_c = pick(g, ["opp rush yds","opp rushing yards"])
        opp_rush_att_c = pick(g, ["opp rush att","opp rushes"])
        opp_sacks_c    = pick(g, ["sk","sacks"])  # team sacks vs opp
        tm_pts_c = pick(g, ["tm","pf","points for"])  # team points
        op_pts_c = pick(g, ["opp pts","pa","points against","points opp"])

        gl = g.copy()

        def to_int(x):
            try:
                return int(x)
            except:
                return np.nan

        if week_c is None or opp_c is None:
            continue

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

        # Weekly rows
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

        # Neutral-pass-rate (defense) – restrict to one-score games (final margin ≤7)
        if tm_pts_c and op_pts_c and opp_pass_att_c and opp_rush_att_c:
            tm_pts = num(tm_pts_c)
            op_pts = num(op_pts_c)
            margin = (tm_pts - op_pts).abs()
            neutral_mask = margin <= 7
            na = pass_a[neutral_mask]
            nr = rush_a[neutral_mask]
            rate = na / (na.add(nr, fill_value=0).replace(0, np.nan))
            neutral_rows.append({
                "team": t,
                "neutral_pass_rate_def": np.nanmean(rate) if len(rate) else np.nan
            })

    wk = pd.DataFrame(rows)
    if not wk.empty:
        wk["team"] = wk["team"].map(_std_team)
        wk["opp"]  = wk["opp"].map(_std_team)
        wk = wk.sort_values(["team","week"]).reset_index(drop=True)
    else:
        wk = pd.DataFrame(columns=[
            "team","week","opp","pass_att_allowed","pass_yds_allowed",
            "rush_att_allowed","rush_yds_allowed","sacks","ypa_allowed","ypc_allowed"
        ])

    neutral = pd.DataFrame(neutral_rows)
    if not neutral.empty:
        neutral["team"] = neutral["team"].map(_std_team)
        neutral = neutral.groupby("team", as_index=False).agg("first")
        neutral["neutral_pass_rate_def_z"] = _z(neutral["neutral_pass_rate_def"]).fillna(0)
    else:
        neutral = pd.DataFrame(columns=["team","neutral_pass_rate_def","neutral_pass_rate_def_z"])

    return wk, neutral


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

    # Season aggregates
    agg = _agg_team_defense_from_pfr(season)
    drv = _drive_stats_off_def(season)
    xpl = _explosive_allowed_split(season)

    # Weekly defensive splits + neutral pass rate (defense)
    wk, neutral = _gamelog_weekly_defense_and_neutral(season)

    # Merge into team_form
    team_form = agg.merge(drv, on="team", how="left").merge(xpl, on="team", how="left").merge(neutral, on="team", how="left")

    # Red-zone proxy (offense & defense) from drives:
    #   rz_trips_off_proxy  ≈ scoring% * plays/drive (per-drive chance scaled by drive length)
    #   rz_trips_def_proxy  ≈ opp scoring% * opp plays/drive
    # You can swap to: drives_per_game * score% if you later add games/drives counts.
    team_form["rz_trips_off_proxy"] = pd.to_numeric(team_form.get("score_pct"), errors="coerce") * pd.to_numeric(team_form.get("plays_per_drive"), errors="coerce")
    team_form["rz_trips_def_proxy"] = pd.to_numeric(team_form.get("opp_score_pct"), errors="coerce") * pd.to_numeric(team_form.get("opp_plays_per_drive"), errors="coerce")
    team_form["rz_trips_off_z"] = _z(team_form["rz_trips_off_proxy"]).fillna(0)
    team_form["rz_trips_def_z"] = _z(team_form["rz_trips_def_proxy"]).fillna(0)

    if "week" not in team_form.columns:
        team_form["week"] = 0

    _save_csv(team_form, paths["metrics"] / "team_form.csv")
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
