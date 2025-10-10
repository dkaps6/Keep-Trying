import os, argparse, pathlib, pandas as pd, requests

DEFAULT_MARKETS = ",".join([
    # core player markets
    "player_pass_yds","player_rec_yds","player_rush_yds","player_receptions",
    # combos & extras
    "player_rush_rec_yds",
    # TD markets
    "player_anytime_td","player_2_or_more_tds",
    # keep game lines for λ_team
    "h2h","spreads","totals",
])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--books', default='draftkings,fanduel,betmgm,caesars')
    ap.add_argument('--markets', default=DEFAULT_MARKETS)
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
    params={'regions':'us','oddsFormat':'american','markets':a.markets,'apiKey':key}
    if a.date: params['dateFormat']='iso'; params['commenceTimeFrom']=a.date

    try:
        r=requests.get(base, params=params, timeout=45); r.raise_for_status(); data=r.json()
    except Exception as e:
        print(f'[oddsapi] request failed: {e}'); return 0

    # ----- player props -----
    rows=[]
    for g in data:
        home=(g.get('home_team') or '').upper(); away=(g.get('away_team') or '').upper()
        for bk in g.get('bookmakers', []):
            bk_key=bk.get('key')
            for mk in bk.get('markets', []):
                mkey=mk.get('key')
                # Only ingest player markets here; game markets handled below
                if not str(mkey).startswith("player_"): 
                    continue
                for ou in mk.get('outcomes', []):
                    name=ou.get('description') or ou.get('name')
                    team=(ou.get('team') or '').upper()
                    # Opponent heuristic
                    opp=away if team==home else home if team==away else ''
                    line=ou.get('point') if 'point' in ou else (1.0 if mkey in ('player_anytime_td','player_2_or_more_tds') else None)
                    # Odds: we try to map both “Over/Under” and “Yes/No” style outputs
                    price=ou.get('price'); side=ou.get('name','').lower()
                    over_odds=price if (side in ('over','yes')) else None
                    under_odds=price if (side in ('under','no')) else None
                    if name and (line is not None) and (price is not None):
                        rows.append([name,team,opp,g.get('id'),mkey,line,over_odds,under_odds,bk_key,g.get('commence_time'),g.get('sport_key','nfl')])

    pd.DataFrame(rows, columns=cols).to_csv(out, index=False)
    print(f"[oddsapi] wrote {out} rows={len(rows)}")

    # ----- game lines to support team totals / spreads -----
    games_rows=[]; games_cols=['event_id','commence_time','sport_key','home_team','away_team','market','point','book']
    for g in data:
        for bk in g.get('bookmakers', []):
            for mk in bk.get('markets', []):
                if mk.get('key') in ('h2h','spreads','totals'):
                    point = None
                    if mk.get('key')=='totals' and mk.get('outcomes'):
                        point = mk['outcomes'][0].get('point')
                    elif mk.get('key')=='spreads' and mk.get('outcomes'):
                        point = mk['outcomes'][0].get('point')
                    games_rows.append([g.get('id'), g.get('commence_time'), g.get('sport_key'),
                                       g.get('home_team'), g.get('away_team'), mk.get('key'), point, bk.get('key')])
    if games_rows:
        pd.DataFrame(games_rows, columns=games_cols).to_csv('outputs/odds_game.csv', index=False)
        print(f"[oddsapi] wrote outputs/odds_game.csv rows={len(games_rows)}")

    return 0

if __name__=='__main__':
    main()
