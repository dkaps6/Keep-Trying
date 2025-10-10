from __future__ import annotations
import os, json, math
from pathlib import Path
import pandas as pd
from scripts.models import ensemble
from scripts.config import LOG_DIR, RUN_ID, MONTE_CARLO_TRIALS

def american_to_prob(odds):
    try: o=float(odds)
    except Exception: return None
    return 100.0/(o+100.0) if o>0 else (-o)/((-o)+100.0) if o<0 else None

def tier_from_edge(e):
    if e is None: return 'RED'
    if e>=0.06: return 'ELITE'
    if e>=0.04: return 'GREEN'
    if e>=0.01: return 'AMBER'
    return 'RED'

def _read_csv(p, empty_cols=None):
    p=Path(p)
    if not p.exists() or p.stat().st_size==0: return pd.DataFrame(columns=empty_cols or [])
    try: return pd.read_csv(p)
    except Exception: return pd.DataFrame(columns=empty_cols or [])

# ---------- Anytime TD: team TD expectation from totals/spreads ----------
def _team_expected_tds(team: str, odds_game: pd.DataFrame, merged_rows: pd.DataFrame) -> float:
    """
    λ_team ≈ team points / 6.8. If totals+spreads are available, use them.
    Fallback: pace-based rough proxy.
    """
    try:
        g = odds_game[(odds_game['home_team']==team) | (odds_game['away_team']==team)]
        total = float(g[g['market']=='totals']['point'].dropna().head(1).values[0]) if not g.empty else None
        spread_series = g[g['market']=='spreads']['point'].dropna().head(1)
        spread = float(spread_series.values[0]) if not spread_series.empty else None
        if total is not None:
            if spread is None:
                team_pts = total/2.0
            else:
                is_home = (not g.empty) and (g.iloc[0]['home_team']==team)
                # Negative spread for home team => favored; tilt points its way
                sgn = -1.0 if (is_home and spread<0) or ((not is_home) and spread>0) else 1.0
                team_pts = total/2.0 + sgn*abs(spread)/2.0
            return max(0.1, team_pts/6.8)
    except Exception:
        pass

    # Fallback: soft pace proxy
    try:
        trows = merged_rows[merged_rows['team']==team]
        plays = float(trows.get('pace').dropna().mean() or 60.0)
        return max(0.1, (plays*0.20)/6.8)  # 0.20 pts/play → rough TD conversion
    except Exception:
        return 1.8

# ---------- Defense-aware rush/pass TD mix ----------
def _sigmoid(x):  # keeps values bounded 0..1
    return 1.0/(1.0+math.exp(-x))

def _rush_mix_from_def(opp_row: pd.Series | None) -> float:
    """
    Return fraction of team TDs expected to be RUSHING (0..1),
    using opponent defense: def_rush_epa vs def_pass_epa, box rates, pressure.
    Positive EPA allowed => worse defense (more efficient offense).
    """
    if opp_row is None or opp_row.empty:
        return 0.50
    dr = float(opp_row.get('def_rush_epa', 0.0))
    dp = float(opp_row.get('def_pass_epa', 0.0))
    sack = float(opp_row.get('def_sack_rate', 0.0))
    light = float(opp_row.get('light_box_rate', 0.0))
    heavy = float(opp_row.get('heavy_box_rate', 0.0))

    # Base tilt by relative weakness rush vs pass (scale to ~[-1,1])
    base = (dr - dp) * 2.5  # tune: 2.5 gives sensible slope across z-ish values
    # Box counts: more light boxes -> easier to run; heavy boxes -> harder
    box_adj = (light - heavy) * 1.5
    # Pressure/sacks depress passing TDs => tilt to run
    press_adj = sack * 1.2

    mix = _sigmoid(base + box_adj + press_adj)               # 0..1
    return min(0.85, max(0.15, mix))  # clamp to avoid extremes

# ---------- Player-level Anytime TD from Poisson ----------
def p_anytime_td(position: str,
                 rz_tgt_share: float,
                 rz_carry_share: float,
                 team_lambda_td: float,
                 rush_mix: float) -> tuple[float, float]:
    """
    Returns: (p_score_any, player_lambda)
    Compose player share using position weights, then split by rush vs pass mix.
    """
    pos = (position or '').upper()
    if pos.startswith('RB'):
        tgt_w, car_w = 0.25, 0.75
    elif pos.startswith('TE'):
        tgt_w, car_w = 0.80, 0.20
    elif pos.startswith('QB'):
        tgt_w, car_w = 0.05, 0.20
    else:  # WR / default
        tgt_w, car_w = 0.85, 0.15

    rz_tgt_share = max(0.0, min(1.0, float(rz_tgt_share or 0.0)))
    rz_carry_share = max(0.0, min(1.0, float(rz_carry_share or 0.0)))

    # Effective TD share = pass share on (1 - rush_mix) + carry share on rush_mix
    eff_share = tgt_w*rz_tgt_share*(1.0 - rush_mix) + car_w*rz_carry_share*rush_mix
    eff_share = max(0.0, min(1.0, eff_share))

    lam_player = max(0.01, float(team_lambda_td))*eff_share
    p_any = 1.0 - math.exp(-lam_player)
    return p_any, lam_player

def p_two_plus_td(lam_player: float) -> float:
    """P(X>=2) for Poisson(λ) = 1 - e^{-λ}(1+λ)."""
    return 1.0 - math.exp(-lam_player)*(1.0 + lam_player)

# ---------- Feature assembly ----------
def make_features_row(r):
    return {
        'mu': r.get('mu'),'sd': r.get('sd'),'sd_widen': r.get('sd_widen',1.0),
        'eff_mu': r.get('eff_mu'),'eff_sd': r.get('eff_sd'),
        'p_market_fair': r.get('p_market_fair',0.5),
        'target_share': r.get('target_share',0.0),'rush_share': r.get('rush_share',0.0),
        'qb_ypa': r.get('qb_ypa',0.0),'light_box_rate': r.get('light_box_rate',0.0),
        'heavy_box_rate': r.get('heavy_box_rate',0.0),'def_sack_rate': r.get('def_sack_rate',0.0),
        'def_pass_epa': r.get('def_pass_epa',0.0),'pace': r.get('pace',0.0),'proe': r.get('proe',0.0),
    }

def run(season: int):
    Path('outputs').mkdir(parents=True, exist_ok=True); LOG_DIR.mkdir(parents=True, exist_ok=True)
    pf=_read_csv('data/player_form.csv', ['player','team','position'])
    tf=_read_csv('data/team_form.csv', ['team'])
    props=_read_csv('outputs/props_raw.csv', ['player','team','opp_team','market','line','over_odds','under_odds','book','commence_time'])
    odds_game=_read_csv('outputs/odds_game.csv', ['event_id','commence_time','sport_key','home_team','away_team','market','point','book'])

    for df,col in [(pf,'team'),(tf,'team'),(props,'team')]:
        if col in df.columns: df[col]=df[col].astype(str).str.upper()
    if 'opp_team' in props.columns:
        props['opp_team'] = props['opp_team'].astype(str).str.upper()

    # market de-vig for continuous props
    p_market_fair=[]
    for _,r in props.iterrows():
        p_o=american_to_prob(r.get('over_odds')); p_u=american_to_prob(r.get('under_odds'))
        if p_o is None and p_u is None: p_market_fair.append(0.5)
        elif p_o is None: p_market_fair.append(1-p_u)
        elif p_u is None: p_market_fair.append(p_o)
        else: s=p_o+p_u; p_market_fair.append(p_o/s if s>0 else 0.5)
    props['p_market_fair']=p_market_fair

    merged=props.merge(pf, on=['player','team'], how='left', suffixes=('','_pf')).merge(tf, on='team', how='left', suffixes=('','_tf'))

    rows=[]; provider=os.getenv('PROVIDER_USED','unknown')

    for _,r in merged.iterrows():
        market=str(r.get('market',''))

        # ---------- Anytime TD (Yes) ----------
        if market=='player_anytime_td':
            team=str(r.get('team','')); opp=str(r.get('opp_team','')); position=str(r.get('position',''))
            lam_team = _team_expected_tds(team, odds_game, merged)
            opp_row = tf[tf['team']==opp].head(1).squeeze() if not tf.empty else pd.Series(dtype='float64')
            rush_mix = _rush_mix_from_def(opp_row)
            p_final, lam_player = p_anytime_td(position, r.get('rz_tgt_share',0.0), r.get('rz_carry_share',0.0), lam_team, rush_mix)
            p_mkt = american_to_prob(r.get('over_odds')) or 0.5
            edge = p_final - p_mkt
            fair=None
            if 0<p_final<1:
                fair = -int(round(100*p_final/(1-p_final))) if p_final>=0.5 else int(round(100*(1-p_final)/p_final))
            rows.append({
                'player':str(r.get('player','')),'team':team,'market':market,'line':1.0,
                'vegas_prob':p_mkt,'model_prob':p_final,'edge':edge,'fair_odds':fair,
                'tier':tier_from_edge(edge),'notes':f'Anytime TD (λ_team={lam_team:.2f}, rush_mix={rush_mix:.2f})'
            })
            continue

        # ---------- 2+ TDs ----------
        if market in ('player_2_or_more_tds','player_two_plus_tds'):
            team=str(r.get('team','')); opp=str(r.get('opp_team','')); position=str(r.get('position',''))
            lam_team = _team_expected_tds(team, odds_game, merged)
            opp_row = tf[tf['team']==opp].head(1).squeeze() if not tf.empty else pd.Series(dtype='float64')
            rush_mix = _rush_mix_from_def(opp_row)
            _, lam_player = p_anytime_td(position, r.get('rz_tgt_share',0.0), r.get('rz_carry_share',0.0), lam_team, rush_mix)
            p_final = p_two_plus_td(lam_player)
            # Some books quote "Yes" price as under_odds; we keep symmetry using de-vig if present.
            p_mkt = american_to_prob(r.get('over_odds')) or american_to_prob(r.get('under_odds')) or 0.5
            edge = p_final - p_mkt
            fair=None
            if 0<p_final<1:
                fair = -int(round(100*p_final/(1-p_final))) if p_final>=0.5 else int(round(100*(1-p_final)/p_final))
            rows.append({
                'player':str(r.get('player','')),'team':team,'market':market,'line':2.0,
                'vegas_prob':p_mkt,'model_prob':p_final,'edge':edge,'fair_odds':fair,
                'tier':tier_from_edge(edge),'notes':f'2+ TDs (λ_player={lam_player:.2f})'
            })
            continue

        # ---------- Continuous markets (existing path) ----------
        try:
            line=float(r.get('line'))
        except Exception:
            continue
        feats={
            'mu': r.get('mu'),'sd': r.get('sd'),'sd_widen': r.get('sd_widen',1.0),
            'eff_mu': r.get('eff_mu'),'eff_sd': r.get('eff_sd'),
            'p_market_fair': r.get('p_market_fair',0.5),
            'target_share': r.get('target_share',0.0),'rush_share': r.get('rush_share',0.0),
            'qb_ypa': r.get('qb_ypa',0.0),'light_box_rate': r.get('light_box_rate',0.0),
            'heavy_box_rate': r.get('heavy_box_rate',0.0),'def_sack_rate': r.get('def_sack_rate',0.0),
            'def_pass_epa': r.get('def_pass_epa',0.0),'pace': r.get('pace',0.0),'proe': r.get('proe',0.0),
        }

        from scripts.models import Leg
        leg=Leg(player_id=f"{r.get('player','')}|{r.get('team','')}|{market}|{line}",
                player=str(r.get('player','')), team=str(r.get('team','')),
                market=market, line=line, features=feats)

        blended=ensemble.blend(leg, context={'w_mc':0.25,'w_bayes':0.25,'w_markov':0.25,'w_ml':0.25})
        p_mkt=blended.get('p_market',0.5); p_final=blended.get('p_final',0.5)
        edge=(p_final-p_mkt) if p_mkt is not None else None
        fair=None
        if p_final is not None and 0<p_final<1:
            fair = -int(round(100*p_final/(1-p_final))) if p_final>=0.5 else int(round(100*(1-p_final)/p_final))

        rows.append({
            'player':str(r.get('player','')), 'team':str(r.get('team','')),
            'market':market, 'line':line,
            'vegas_prob':p_mkt,'model_prob':p_final,'edge':edge,'fair_odds':fair,
            'tier':tier_from_edge(edge),'notes':blended.get('notes','')
        })

    out=pd.DataFrame(rows).sort_values(['tier','edge'], ascending=[True, False])
    out_path=Path('outputs/master_model_predictions.csv'); out.to_csv(out_path, index=False)
    (LOG_DIR/'summary.json').write_text(json.dumps({'run_id':RUN_ID,'season':int(season),'rows':int(len(out)),'mc_trials':int(MONTE_CARLO_TRIALS),'provider':os.getenv('PROVIDER_USED','unknown')}, indent=2))
    out.to_csv(LOG_DIR/'master_model_predictions.csv', index=False)
    print(f"[predictors] ✅ wrote {len(out)} rows → {out_path}")
    print(f"[predictors] log → {LOG_DIR}")

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--season', type=int, required=True)
    a=ap.parse_args(); run(a.season)
