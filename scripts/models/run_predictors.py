from __future__ import annotations
import os, json
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
    props=_read_csv('outputs/props_raw.csv', ['player','team','market','line','over_odds','under_odds','book','commence_time'])

    for df,col in [(pf,'team'),(tf,'team'),(props,'team')]:
        if col in df.columns: df[col]=df[col].astype(str).str.upper()

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
        try: line=float(r.get('line'))
        except Exception: continue
        feats=make_features_row(r)
        player=str(r.get('player','')); team=str(r.get('team','')); market=str(r.get('market',''))
        from scripts.models import Leg
        leg=Leg(player_id=f"{player}|{team}|{market}|{line}", player=player, team=team, market=market, line=line, features=feats)
        blended=ensemble.blend(leg, context={'w_mc':0.25,'w_bayes':0.25,'w_markov':0.25,'w_ml':0.25})
        p_mkt=blended.get('p_market',0.5); p_final=blended.get('p_final',0.5); edge=(p_final-p_mkt) if p_mkt is not None else None
        fair = None
        if p_final is not None and 0<p_final<1:
            fair = -int(round(100*p_final/(1-p_final))) if p_final>=0.5 else int(round(100*(1-p_final)/p_final))
        rows.append({'player':player,'team':team,'market':market,'line':line,'vegas_prob':p_mkt,'model_prob':p_final,'edge':edge,'fair_odds':fair,'tier':tier_from_edge(edge),'notes':blended.get('notes','')})
    out=pd.DataFrame(rows).sort_values(['tier','edge'], ascending=[True, False])
    out_path=Path('outputs/master_model_predictions.csv'); out.to_csv(out_path, index=False)
    (LOG_DIR/'summary.json').write_text(json.dumps({'run_id':RUN_ID,'season':int(season),'rows':int(len(out)),'mc_trials':int(MONTE_CARLO_TRIALS),'provider':provider}, indent=2))
    out.to_csv(LOG_DIR/'master_model_predictions.csv', index=False)
    print(f"[predictors] ✅ wrote {len(out)} rows → {out_path}"); print(f"[predictors] log → {LOG_DIR}")

if __name__=='__main__':
    import argparse; ap=argparse.ArgumentParser(); ap.add_argument('--season', type=int, required=True); a=ap.parse_args(); run(a.season)
