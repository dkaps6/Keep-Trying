import os, argparse, pathlib, pandas as pd, requests
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--books', default='draftkings,fanduel,betmgm,caesars')
    ap.add_argument('--markets', default='')
    ap.add_argument('--date', default='')
    ap.add_argument('--out', default='outputs/props_raw.csv')
    a=ap.parse_args()
    out=pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    cols=['player','team','opp_team','event_id','market','line','over_odds','under_odds','book','commence_time','sport_key']
    pd.DataFrame(columns=cols).to_csv(out, index=False)
    key=os.getenv('ODDS_API_KEY','').strip()
    if not key:
        print('[oddsapi] key missing; wrote header-only CSV'); return 0
    base='https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds'
    params={'regions':'us','oddsFormat':'american','markets':'h2h,spreads,totals,player_pass_yds,player_rec_yds,player_rush_yds,player_receptions,player_anytime_td','apiKey':key}
    if a.date: params['dateFormat']='iso'; params['commenceTimeFrom']=a.date
    try:
        r=requests.get(base, params=params, timeout=30); r.raise_for_status(); data=r.json()
    except Exception as e:
        print(f'[oddsapi] request failed: {e}'); return 0
    rows=[]
    for g in data:
        home=(g.get('home_team') or '').upper(); away=(g.get('away_team') or '').upper()
        for bk in g.get('bookmakers', []):
            bk_key=bk.get('key')
            for mk in bk.get('markets', []):
                mkey=mk.get('key')
                for ou in mk.get('outcomes', []):
                    name=ou.get('description') or ou.get('name'); line=ou.get('point'); price=ou.get('price'); team=(ou.get('team') or '').upper()
                    opp=away if team==home else home if team==away else ''
                    if name and (line is not None) and (price is not None):
                        rows.append([name,team,opp,g.get('id'),mkey,line,price,None,bk_key,g.get('commence_time'),g.get('sport_key','nfl')])
    pd.DataFrame(rows, columns=cols).to_csv(out, index=False); print(f"[oddsapi] wrote {out} rows={len(rows)}"); return 0
if __name__=='__main__': main()


# also write game-level totals/spreads if present
    games_rows=[]; games_cols=['event_id','commence_time','sport_key','home_team','away_team','market','point','book']
    for g in data:
        for bk in g.get('bookmakers', []):
            for mk in bk.get('markets', []):
                if mk.get('key') in ('h2h','spreads','totals'):
                    point = None
                    if mk.get('key')=='totals' and mk.get('outcomes'):
                        # use first as total line
                        point = mk['outcomes'][0].get('point')
                    elif mk.get('key')=='spreads' and mk.get('outcomes'):
                        point = mk['outcomes'][0].get('point')
                    games_rows.append([g.get('id'), g.get('commence_time'), g.get('sport_key'),
                                       g.get('home_team'), g.get('away_team'), mk.get('key'), point, bk.get('key')])
    if games_rows:
        pd.DataFrame(games_rows, columns=games_cols).to_csv('outputs/odds_game.csv', index=False)
        print(f"[oddsapi] wrote outputs/odds_game.csv rows={len(games_rows)}")
