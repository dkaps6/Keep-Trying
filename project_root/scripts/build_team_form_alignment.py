#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mapping-free alignment for team_form.
- Derives slate from props_priced.csv (fallback: props_priced.xlsx, then props_raw.csv)
- Normalizes team names (handles JAX/LV/LA Rams/etc.)
- Aligns team_form to the full slate (no needs_mapping.csv required)
- Writes:
  1) data/out/team_form_normalized_join.csv
  2) data/out/team_form_rows_not_in_slate.csv
  3) data/out/mapping_free_team_form_summary.json
"""

import os, re, json, argparse
import pandas as pd

ALIASES = {
    "ny giants":"new york giants","ny jets":"new york jets",
    "jax":"jacksonville jaguars","kc":"kansas city chiefs",
    "gb":"green bay packers","no":"new orleans saints",
    "tb":"tampa bay buccaneers","lv":"las vegas raiders",
    "la rams":"los angeles rams","la chargers":"los angeles chargers",
    "sf":"san francisco 49ers","ne":"new england patriots",
    "wsh":"washington commanders","was":"washington commanders","wft":"washington commanders",
    "ari":"arizona cardinals","bal":"baltimore ravens","buf":"buffalo bills",
    "chi":"chicago bears","cin":"cincinnati bengals","cle":"cleveland browns",
    "dal":"dallas cowboys","den":"denver broncos","det":"detroit lions",
    "hou":"houston texans","ind":"indianapolis colts","mia":"miami dolphins",
    "min":"minnesota vikings","phi":"philadelphia eagles","pit":"pittsburgh steelers",
    "sea":"seattle seahawks","ten":"tennessee titans","atl":"atlanta falcons",
    "car":"carolina panthers","sd":"los angeles chargers",
    "st louis rams":"los angeles rams",
}

def normalize_team(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return x
    s = re.sub(r"[\.\-_/]", " ", str(x).strip().lower())
    s = re.sub(r"\s+", " ", s).strip()
    return ALIASES.get(s, s)

def read_flexible_csv(path):
    """Try multiple delimiters; fall back to python engine with on_bad_lines=skip."""
    if not os.path.exists(path):
        return None
    for sep in [None, ",", "\t", "|", ";"]:
        try:
            return pd.read_csv(path, sep=sep, engine="python", on_bad_lines="skip")
        except Exception:
            continue
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception:
        return None

def derive_slate(in_dir, props_priced_filename, props_raw_filename, props_priced_xlsx):
    """
    Prefer priced CSV, then priced XLSX, then raw CSV.
    Expect columns: home_team, away_team
    """
    # 1) priced CSV
    pp_csv = os.path.join(in_dir, props_priced_filename) if props_priced_filename else None
    if pp_csv and os.path.exists(pp_csv):
        df = read_flexible_csv(pp_csv)
        if df is not None and {"home_team","away_team"}.issubset(df.columns):
            return pd.unique(pd.concat([df["home_team"], df["away_team"]], ignore_index=True).dropna().astype(str))

    # 2) priced XLSX
    pp_xlsx = os.path.join(in_dir, props_priced_xlsx) if props_priced_xlsx else None
    if pp_xlsx and os.path.exists(pp_xlsx):
        try:
            df = pd.read_excel(pp_xlsx)
            if {"home_team","away_team"}.issubset(df.columns):
                return pd.unique(pd.concat([df["home_team"], df["away_team"]], ignore_index=True).dropna().astype(str))
        except Exception:
            pass

    # 3) raw CSV
    pr_csv = os.path.join(in_dir, props_raw_filename) if props_raw_filename else None
    if pr_csv and os.path.exists(pr_csv):
        df = read_flexible_csv(pr_csv)
        if df is not None and {"home_team","away_team"}.issubset(df.columns):
            return pd.unique(pd.concat([df["home_team"], df["away_team"]], ignore_index=True).dropna().astype(str))

    raise RuntimeError("Could not derive slate: props_priced.csv/.xlsx or props_raw.csv with home_team/away_team not found.")

def locate_team_col(tf: pd.DataFrame) -> str:
    """Find a team column inside team_form by name heuristics."""
    candidates = [c for c in tf.columns if c.lower() in ("team","team_name","name","franchise","club")]
    if candidates:
        return candidates[0]
    # looser heuristic
    for c in tf.columns:
        if "team" in c.lower():
            return c
    raise RuntimeError("Could not find a team column in team_form.csv (looked for team/team_name/name/franchise/club).")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs-dir", default="data/input", help="Directory containing team_form.csv and props files.")
    ap.add_argument("--outputs-dir", default="data/out", help="Directory to write outputs.")
    ap.add_argument("--team-form", default="team_form.csv", help="Filename of team_form source CSV (in inputs-dir).")
    ap.add_argument("--props-priced", default="props_priced.csv", help="Filename of priced props CSV (in inputs-dir).")
    ap.add_argument("--props-priced-xlsx", default="props_priced.xlsx", help="Filename of priced props XLSX (in inputs-dir).")
    ap.add_argument("--props-raw", default="props_raw.csv", help="Filename of raw props CSV (in inputs-dir).")
    ap.add_argument("--overwrite-team-form", action="store_true", help="Optional: also overwrite the input team_form.csv with normalized version.")
    args = ap.parse_args()

    in_dir = args.inputs_dir
    out_dir = args.outputs_dir
    os.makedirs(out_dir, exist_ok=True)

    # --- Load inputs ---
    team_form_path = os.path.join(in_dir, args.team_form)
    if not os.path.exists(team_form_path):
        raise FileNotFoundError(f"Missing input: {team_form_path}")

    tf = read_flexible_csv(team_form_path)
    if tf is None or tf.empty:
        raise RuntimeError(f"team_form could not be read or is empty: {team_form_path}")

    team_col = locate_team_col(tf)

    # --- Build slate from props ---
    slate = derive_slate(
        in_dir=in_dir,
        props_priced_filename=args.props_priced,
        props_raw_filename=args.props_raw,
        props_priced_xlsx=args.props_priced_xlsx
    )
    slate_df = pd.DataFrame({"team": slate})
    slate_df["team_norm"] = slate_df["team"].map(normalize_team)

    # --- Normalize team_form and align ---
    tf["team_norm"] = tf[team_col].map(normalize_team)
    tf["in_slate"] = tf["team_norm"].isin(slate_df["team_norm"])

    # --- Write artifacts ---
    out_join = os.path.join(out_dir, "team_form_normalized_join.csv")
    out_missing = os.path.join(out_dir, "team_form_rows_not_in_slate.csv")
    out_summary = os.path.join(out_dir, "mapping_free_team_form_summary.json")

    tf.to_csv(out_join, index=False)
    tf[~tf["in_slate"]].to_csv(out_missing, index=False)

    summary = {
        "inputs_dir": in_dir,
        "outputs_dir": out_dir,
        "team_form_source": team_form_path,
        "team_col_used": team_col,
        "unique_slate_teams": int(pd.Series(slate_df["team_norm"].unique()).shape[0]),
        "team_form_unique_before": int(pd.Series(tf[team_col].unique()).dropna().shape[0]),
        "team_form_unique_after_filter": int(pd.Series(tf.loc[tf["in_slate"], "team_norm"].unique()).dropna().shape[0]),
        "missing_rows_count": int((~tf["in_slate"]).sum()),
        "notes": "No needs_mapping.csv required. Slate derived from props. Normalize this same way before any merges."
    }
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)

    # Optional overwrite to keep downstream code unchanged
    if args.overwrite_team_form:
        tf_overwrite = tf.drop(columns=["in_slate"], errors="ignore")
        tf_overwrite.to_csv(team_form_path, index=False)

    print(f"Wrote:\n- {out_join}\n- {out_missing}\n- {out_summary}")
    if args.overwrite_team_form:
        print(f"Also overwrote: {team_form_path}")

if __name__ == "__main__":
    main()
