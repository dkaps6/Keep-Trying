# scripts/features_external.py
# Loads and merges external features onto props:
# - team_form.csv (plays_est, PROE, RZ rate, and *_z splits)
# - player_form.csv (shares + efficiency proxies)
# - weather.csv (wind/temp/precip/dome/altitude)
# - injuries.csv, roles.csv (role-aware caps/redistribution used by pricing)
# - coverage.csv, cb_assignments.csv (coverage/CB shadow)
# - game_lines.csv (home_wp/away_wp → script escalators)
#
# Exposes:
#   enrich_props(props: pd.DataFrame) -> pd.DataFrame
#   main() for optional CLI (reads outputs/props_raw.csv → outputs/props_enriched.csv)

from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

# -------- helpers --------

def _read_csv(path: str, required: List[str] | None = None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=required or [])
    try:
        df = pd.read_csv(p)
    except Exception:
        return pd.DataFrame(columns=required or [])
    if required:
        for c in required:
            if c not in df.columns:
                df[c] = pd.NA
    return df

def _lower_join(a: pd.DataFrame, b: pd.DataFrame, on: List[str]) -> pd.DataFrame:
    # case-insensitive join fallback for player names
    a2, b2 = a.copy(), b.copy()
    for k in on:
        a2[f"__{k}__"] = a2[k].astype(str).str.lower()
        b2[f"__{k}__"] = b2[k].astype(str).str.lower()
    out = a2.merge(
        b2.drop(columns=[c for c in on if c in b2.columns]),
        left_on=[f"__{k}__" for k in on],
        right_on=[f"__{k}__" for k in on],
        how="left"
    )
    out.drop(columns=[f"__{k}__" for k in on], inplace=True, errors="ignore")
    return out

# -------- loaders --------

def load_team_form(path: str = "data/team_form.csv") -> pd.DataFrame:
    req = [
        "team",
        "def_pressure_rate_z","def_pass_epa_z","def_rush_epa_z","def_sack_rate_z",
        "pace_z","light_box_rate_z","heavy_box_rate_z","ay_per_att_z",
        "plays_est","proe","rz_rate",
    ]
    return _read_csv(path, req)

def load_player_form(path: str = "data/player_form.csv") -> pd.DataFrame:
    req = [
        "player","team","position",
        "target_share","rush_share","rz_tgt_share","rz_carry_share",
        "yprr_proxy","ypc","ypt","qb_ypa"
    ]
    return _read_csv(path, req)

def load_weather(path: str = "data/weather.csv") -> pd.DataFrame:
    req = ["event_id","wind_mph","temp_f","precip","altitude_ft","dome","home_team","away_team","commence_time"]
    return _read_csv(path, req)

def load_injuries(path: str = "data/injuries.csv") -> pd.DataFrame:
    req = ["player","team","status"]
    return _read_csv(path, req)

def load_roles(path: str = "data/roles.csv") -> pd.DataFrame:
    req = ["player","team","role"]
    return _read_csv(path, req)

def load_coverage(path: str = "data/coverage.csv") -> pd.DataFrame:
    req = ["defense_team","tag"]
    return _read_csv(path, req)

def load_cb_assignments(path: str = "data/cb_assignments.csv") -> pd.DataFrame:
    # either 'quality' or 'penalty' may exist
    df = _read_csv(path, ["defense_team","receiver"])
    for c in ("cb","quality","penalty"):
        if c not in df.columns:
            df[c] = pd.NA
    return df

def load_game_lines(path: str = "outputs/game_lines.csv") -> pd.DataFrame:
    req = ["event_id","home_team","away_team","home_wp","away_wp","commence_time"]
    return _read_csv(path, req)

# -------- merge entrypoint --------

def enrich_props(props: pd.DataFrame) -> pd.DataFrame:
    if props is None or props.empty:
        return props

    # Normalize keys
    if "id" in props.columns and "event_id" not in props.columns:
        props = props.rename(columns={"id":"event_id"})
    # market_internal expected from your ingestion
    for col in ("player","team","market_internal"):
        if col not in props.columns:
            props[col] = pd.NA

    # ---- team_form on team ----
    tf = load_team_form()
    if not tf.empty:
        props = props.merge(tf, on="team", how="left")

    # ---- player_form on (player, team) with fallback case-insensitive join ----
    pf = load_player_form()
    if not pf.empty:
        props = props.merge(pf, on=["player","team"], how="left")
        if props["target_share"].isna().mean() > 0.6:
            props = _lower_join(props, pf, on=["player","team"])

    # ---- weather on event_id ----
    wx = load_weather()
    if not wx.empty:
        keep = ["event_id","wind_mph","temp_f","precip","altitude_ft","dome"]
        props = props.merge(wx[keep], on="event_id", how="left")
    else:
        for c in ["wind_mph","temp_f","precip","altitude_ft","dome"]:
            if c not in props.columns:
                props[c] = pd.NA

    # ---- injuries/roles on (player, team) ----
    inj = load_injuries()
    if not inj.empty:
        inj["status"] = inj["status"].fillna("Unknown")
        props = props.merge(inj, on=["player","team"], how="left", suffixes=("","_inj"))

    rl = load_roles()
    if not rl.empty:
        props = props.merge(rl, on=["player","team"], how="left", suffixes=("","_role"))
        if "role_role" in props.columns and "role" in props.columns:
            props["role"] = props["role"].fillna(props["role_role"])
            props.drop(columns=["role_role"], inplace=True, errors="ignore")

    # ---- coverage tags on defense team (infer from opponent if present) ----
    # If your props carry 'opp_team', this is trivial; otherwise, skip or infer later in pricing.
    cov = load_coverage()
    if not cov.empty and "opp_team" in props.columns:
        cov_tag = (cov.groupby("defense_team")["tag"]
                      .apply(lambda s: "|".join(sorted(set(str(x) for x in s if pd.notna(x)))))
                      .rename("coverage_tags")).reset_index()
        props = props.merge(cov_tag, left_on="opp_team", right_on="defense_team", how="left")
        props.drop(columns=["defense_team"], inplace=True, errors="ignore")

    # ---- CB assignments (receiver matchups) ----
    cba = load_cb_assignments()
    if not cba.empty and "opp_team" in props.columns:
        props = props.merge(
            cba.rename(columns={"defense_team":"opp_team","receiver":"player"}),
            on=["opp_team","player"], how="left", suffixes=("","_cb")
        )

    # ---- game lines (for script escalators in pricing) ----
    gl = load_game_lines()
    if not gl.empty:
        props = props.merge(gl[["event_id","home_wp","away_wp"]], on="event_id", how="left")

    return props

# optional CLI: props_raw -> props_enriched
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Enrich props with external features")
    ap.add_argument("--in-props", default="outputs/props_raw.csv")
    ap.add_argument("--out", default="outputs/props_enriched.csv")
    args = ap.parse_args()

    df = _read_csv(args.in_props)
    if df.empty:
        print(f"[features_external] input empty: {args.in_props}")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        return
    out = enrich_props(df)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[features_external] wrote {len(out)} rows → {args.out}")

if __name__ == "__main__":
    main()
