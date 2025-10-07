#!/usr/bin/env python3
"""
Unified driver:
1) Calls ./nflverse_csv_fetcher/make_all.py (free pulls + addons) if present.
2) Uses secrets for paid feeds (MSF/NFLGSIS/Odds/API-Sports) — stubs that won't crash if missing.
3) Composes:
     outputs/metrics/team_form.csv
     outputs/metrics/player_form.csv
"""
from __future__ import annotations
import argparse, os, sys, subprocess
from pathlib import Path
import pandas as pd

def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(rc)

def exists_nonempty(p) -> bool:
    p = Path(p)
    return p.exists() and p.is_file() and p.stat().st_size > 0

def latest_csv(folder: Path, stem_contains: str) -> Path | None:
    if not folder.exists(): return None
    c = sorted([p for p in folder.glob("*.csv") if stem_contains in p.name],
               key=lambda q: q.stat().st_mtime, reverse=True)
    return c[0] if c else None

def safe_left_join(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if left is None or left.empty:  return right.copy()
    if right is None or right.empty: return left.copy()
    k = [c for c in keys if c in left.columns and c in right.columns]
    return left if not k else left.merge(right, on=k, how="left")

def compose_team_form(out_root: Path, season: int) -> None:
    team_stats_dir = out_root / "team_stats"
    proe_dir       = out_root / "proe"
    box_dir        = out_root / "box_counts"

    tf = pd.DataFrame()
    reg = latest_csv(team_stats_dir, f"team_stats_reg_{season}")
    if reg and exists_nonempty(reg):
        t = pd.read_csv(reg)
        keep = [c for c in t.columns if c in (
            "team","season","passing_epa","rushing_epa","def_pass_epa",
            "def_rush_epa","epa_per_play","def_epa_per_play","sack_rate"
        )]
        if keep: tf = t[keep].copy()

    proe = latest_csv(proe_dir, f"team_proe_season_{season}")
    if proe and exists_nonempty(proe):
        p = pd.read_csv(proe).rename(columns={"posteam":"team"})
        tf = safe_left_join(tf, p[["season","team","season_proe"]], ["season","team"])

    dbox = latest_csv(box_dir, f"defense_box_rates_week_{season}")
    if dbox and exists_nonempty(dbox):
        d = pd.read_csv(dbox)
        g = d.groupby(["season","team"], as_index=False).agg({
            "def_light_box_rate":"mean",
            "def_heavy_box_rate":"mean"
        })
        tf = safe_left_join(tf, g, ["season","team"])

    tf = tf.rename(columns={
        "passing_epa":"off_pass_epa",
        "rushing_epa":"off_rush_epa",
        "epa_per_play":"off_epa_play",
        "def_epa_per_play":"def_epa_play",
        "season_proe":"proe",
        "sack_rate":"def_sack_rate",
    })
    out = out_root / "metrics" / "team_form.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    tf.to_csv(out, index=False)
    print(f"[compose] wrote {out} ({len(tf)} rows)")

def compose_player_form(out_root: Path, season: int) -> None:
    pstats_dir = out_root / "player_stats"
    roles_dir  = out_root / "roles"
    rbm_dir    = out_root / "rb_metrics"

    EXPECT = ["team","player","position","target_share","rush_share",
              "rz_tgt_share","rz_carry_share","yprr_proxy","ypc","ypt","qb_ypa"]

    base = pd.DataFrame(columns=[
        "season","week","team","player_display_name","position",
        "targets","receptions","receiving_yards","rush_attempts","rushing_yards","air_yards"
    ])

    pw = latest_csv(pstats_dir, f"player_stats_week_{season}")
    if pw and exists_nonempty(pw):
        dfp = pd.read_csv(pw)
        have = [c for c in base.columns if c in dfp.columns]
        base = dfp[have].copy()

    if not base.empty and {"targets","team"}.issubset(base.columns):
        team_tot = base.groupby(["season","week","team"], as_index=False)["targets"].sum()\
                       .rename(columns={"targets":"team_targets"})
        base = base.merge(team_tot, on=["season","week","team"], how="left")
        base["target_share"] = (base["targets"] / base["team_targets"]).fillna(0.0)
    else:
        base["target_share"] = 0.0

    roles_csv = latest_csv(roles_dir, f"roles_weekly_{season}")
    if roles_csv and exists_nonempty(roles_csv):
        rr = pd.read_csv(roles_csv).rename(columns={"role_label":"role"})
        base = base.merge(rr[["season","week","team","player_display_name","role"]],
                          on=["season","week","team","player_display_name"], how="left")

    rbw = latest_csv(rbm_dir, f"rb_metrics_week_{season}")
    if rbw and exists_nonempty(rbw):
        r = pd.read_csv(rbw).rename(columns={
            "posteam":"team",
            "rusher_player_name":"player_display_name",
            "yards_per_carry":"ypc"
        })
        base = base.merge(r[["season","week","team","player_display_name","ypc"]],
                          on=["season","week","team","player_display_name"], how="left")
    else:
        base["ypc"] = 0.0

    base = base.rename(columns={"player_display_name":"player"})
    out_df = pd.DataFrame()
    out_df["team"]           = base.get("team", "")
    out_df["player"]         = base.get("player", "")
    out_df["position"]       = base.get("position", "")
    out_df["target_share"]   = base.get("target_share", 0.0).fillna(0.0)
    out_df["rush_share"]     = 0.0
    out_df["rz_tgt_share"]   = 0.0
    out_df["rz_carry_share"] = 0.0
    out_df["yprr_proxy"]     = 0.0
    out_df["ypc"]            = base.get("ypc", 0.0).fillna(0.0)
    out_df["ypt"]            = 0.0
    out_df["qb_ypa"]         = 0.0

    out_df = out_df[EXPECT].fillna(0.0)
    out = out_root / "metrics" / "player_form.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"[compose] wrote {out} ({len(out_df)} rows)")

def fetch_msf(_season: int) -> None:
    if not (os.getenv("MSF_KEY") and os.getenv("MSF_PASSWORD")):
        print("[msf] secrets not set; skipping")
        return
    print("[msf] creds detected — add MSF endpoints here when ready.")

def fetch_gsis(_season: int) -> None:
    if not (os.getenv("NFLGSIS_USERNAME") and os.getenv("NFLGSIS_PASSWORD")):
        print("[gsis] secrets not set; skipping")
        return
    print("[gsis] creds detected — add GSIS client calls here if you have partner endpoints.")

def fetch_odds(_season: int) -> None:
    if not os.getenv("THE_ODDS_API_KEY"):
        print("[odds] THE_ODDS_API_KEY not set; skipping")
        return
    print("[odds] key detected — add Odds API pulls here if you want them staged.")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--skip-pbp", action="store_true")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent
    out_root = repo / args.out
    out_root.mkdir(parents=True, exist_ok=True)

    fetcher = repo / "nflverse_csv_fetcher" / "make_all.py"
    if fetcher.exists():
        cmd = [sys.executable, str(fetcher), "--season", str(args.season)]
        if args.skip_pbp: cmd.append("--skip-pbp")
        run(cmd)
    else:
        print("::warning ::nflverse_csv_fetcher/make_all.py missing — skipping free fetch.")

    fetch_msf(args.season)
    fetch_gsis(args.season)
    fetch_odds(args.season)

    compose_team_form(out_root, args.season)
    compose_player_form(out_root, args.season)

    must = [out_root / "metrics" / "team_form.csv", out_root / "metrics" / "player_form.csv"]
    for m in must:
        if not exists_nonempty(m):
            print(f"::error ::missing or empty {m}")
            return 1

    print("✅ make_all completed.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
