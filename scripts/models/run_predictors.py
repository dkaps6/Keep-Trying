# scripts/models/run_predictors.py
from __future__ import annotations
import os, math, json
from pathlib import Path
import pandas as pd

# Internal models
from scripts.models import ensemble
from scripts.models import monte_carlo, bayes_hier, markov, ml_ensemble
from scripts.config import LOG_DIR, RUN_ID, MONTE_CARLO_TRIALS

# ---------- helpers ----------
def american_to_prob(odds):
    try:
        o = float(odds)
    except Exception:
        return None
    if o > 0:
        return 100.0 / (o + 100.0)
    if o < 0:
        return (-o) / ((-o) + 100.0)
    return None

def devig_two_way(p_over, p_under):
    # simple fair prob normalization
    if p_over is None and p_under is None: return None
    if p_over is None: return 1.0 - p_under
    if p_under is None: return p_over
    z = p_over + p_under
    if z <= 0: return None
    return p_over / z

def fair_odds_from_p(p):
    if p is None or p <= 0 or p >= 1: return None
    # return American odds
    if p >= 0.5:
        return -int(round(100 * p / (1 - p)))
    return int(round(100 * (1 - p) / p))

def tier_from_edge(edge):
    if edge is None: return "RED"
    if edge >= 0.06: return "ELITE"
    if edge >= 0.04: return "GREEN"
    if edge >= 0.01: return "AMBER"
    return "RED"

# ---------- load inputs ----------
def _read_csv(p, empty_cols=None):
    p = Path(p)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=empty_cols or [])
    try:
        df = pd.read_csv(p)
        return df
    except Exception:
        return pd.DataFrame(columns=empty_cols or [])

def make_features_row(player_row, team_row):
    # base μ/σ priors (can be refined by your pricing.py if available)
    feats = {
        "mu": player_row.get("mu", None),
        "sd": player_row.get("sd", None),
        "sd_widen": player_row.get("sd_widen", 1.0),
        "eff_mu": player_row.get("eff_mu", None),
        "eff_sd": player_row.get("eff_sd", None),
        "p_market_fair": player_row.get("p_market_fair", 0.5),
        "target_share": player_row.get("target_share", 0.0),
        "rush_share": player_row.get("rush_share", 0.0),
        "qb_ypa": player_row.get("qb_ypa", 0.0),
        "light_box_rate": team_row.get("light_box_rate", 0.0),
        "heavy_box_rate": team_row.get("heavy_box_rate", 0.0),
        "def_pressure_z": team_row.get("def_pressure_rate_z", 0.0),
        "def_pass_epa_z": team_row.get("def_pass_epa_z", 0.0),
        "pace_z": team_row.get("pace_z", 0.0),
        "proe": team_row.get("proe", 0.0),
    }
    return feats

def run(season: int):
    Path("outputs").mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # core inputs (tolerant if some are missing)
    player_form = _read_csv("data/player_form.csv", empty_cols=[
        "player","team","position","target_share","rush_share","yprr_proxy","ypc","ypt","qb_ypa"
    ])
    team_form   = _read_csv("data/team_form.csv", empty_cols=[
        "team","pace_z","proe","def_pass_epa_z","def_rush_epa_z","def_pressure_rate_z",
        "light_box_rate","heavy_box_rate"
    ])
    props_raw   = _read_csv("outputs/props_raw.csv", empty_cols=[
        "player","team","market","line","over_odds","under_odds","book","commence_time"
    ])

    # normalize join keys
    for df, col in [(player_form,"team"), (team_form,"team"), (props_raw,"team")]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper()

    if props_raw.empty:
        # ensure a file exists so pipeline never crashes
        props_raw = pd.DataFrame(columns=["player","team","market","line","over_odds","under_odds","book","commence_time"])

    # compute de-vigged market prob per row (when both sides present)
    p_market_fair = []
    for _, r in props_raw.iterrows():
        p_o = american_to_prob(r.get("over_odds"))
        p_u = american_to_prob(r.get("under_odds"))
        p_market_fair.append(devig_two_way(p_o, p_u))
    props_raw["p_market_fair"] = p_market_fair

    # merge features
    pf = player_form.copy()
    tf = team_form.copy()
    merged = props_raw.merge(pf, on=["player","team"], how="left", suffixes=("","_pf"))
    merged = merged.merge(tf, on=["team"], how="left", suffixes=("","_tf"))

    # prepare legs
    legs = []
    rows_out = []
    providers_seen = {os.getenv("PROVIDER_USED", "unknown")}

    for _, r in merged.iterrows():
        player = str(r.get("player","")).strip()
        team   = str(r.get("team","")).strip()
        market = str(r.get("market","")).strip()
        try:
            line = float(r.get("line"))
        except Exception:
            # skip props without numeric line
            continue

        # collect per-team features
        team_row = r if "pace_z" in r.index else {}
        feats = make_features_row(r, team_row)
        from scripts.models.__init__ import Leg
        leg = Leg(player_id=f"{player}|{team}|{market}|{line}",
                  player=player, team=team, market=market, line=line,
                  features=feats)

        # run ensemble (internally calls Monte Carlo/Bayes/Markov/ML)
        blended = ensemble.blend(leg, context={
            "w_mc": 0.25, "w_bayes": 0.25, "w_markov": 0.25, "w_ml": 0.25
        })
        p_market = blended.get("p_market", 0.5)
        p_final  = blended.get("p_final", 0.5)
        edge     = None if p_market is None else (p_final - p_market)
        fair     = fair_odds_from_p(p_final)

        rows_out.append({
            "player": player, "team": team, "market": market, "line": line,
            "vegas_prob": p_market, "model_prob": p_final, "edge": edge,
            "fair_odds": fair, "tier": tier_from_edge(edge),
            "notes": blended.get("notes","")
        })

    df_out = pd.DataFrame(rows_out).sort_values(["tier","edge"], ascending=[True, False])
    out_path = Path("outputs/master_model_predictions.csv")
    df_out.to_csv(out_path, index=False)

    # logging summary
    summary = {
        "run_id": RUN_ID,
        "season": int(season),
        "rows": int(len(df_out)),
        "mc_trials": int(MONTE_CARLO_TRIALS),
        "providers_used": sorted(list(providers_seen)),
    }
    (LOG_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    df_out.to_csv(LOG_DIR / "master_model_predictions.csv", index=False)

    print(f"[predictors] ✅ wrote {len(df_out)} rows → {out_path}")
    print(f"[predictors] log → {LOG_DIR}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    args = ap.parse_args()
    run(args.season)

