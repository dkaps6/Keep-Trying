import os, argparse, pathlib, pandas as pd, requests, time

DEFAULT_MARKETS = ",".join([
    "player_pass_yds","player_rec_yds","player_rush_yds","player_receptions",
    "player_rush_rec_yds","player_anytime_td","player_2_or_more_tds",
    "h2h","spreads","totals",
])

def _write(df, path):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def _call(base, params):
    r = requests.get(base, params=params, timeout=45)
    for k in ("x-requests-remaining","x-requests-used"):
        if k in r.headers:
            print(f"[oddsapi] {k}: {r.headers[k]}")
    r.raise_for_status()
    return r.json()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--books', default='draftkings,fanduel,betmgm,caesars')
    ap.add_argument('--markets', default=DEFAULT_MARKETS)
    ap.add_argument('--date', default='')
    ap.add_argument('--out', default='outputs/props_raw.csv')
    a=ap.parse_args()

    key=os.getenv('ODDS_API_KEY','').strip()
    if not key:
        print('[oddsapi] key missing; writing empty CSVs'); 
        _write(pd.DataFrame(), a.out); _write(pd.DataFrame(), "outputs/odds_game.csv"); return 0

    base='https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds'
    params={'regions':'us','oddsFormat':'american','markets':a.markets,'apiKey':key,'bookmakers':a.books}

    tries=[]
    tries.append(('full', dict(params)))
    if a.date:
        pdate=dict(params); pdate['dateFormat']='iso'; pdate['commenceTimeFrom']=a.date
        tries.insert(0, ('with_date', pdate))
    p_game={'regions':'us','oddsFormat':'american','markets':'h2h,spreads,totals','apiKey':key,'bookmakers':a.books}
    tries.append(('game_only', p_game))

    data=None; tag=""
    for tag,pa in tries:
        try:
            print(f"[oddsapi] request tag={tag} markets={pa.get('markets')}")
            data=_call(base, pa)
        except Exception as e:
            print(f"[oddsapi] error tag={tag}: {e}")
            data=None
        if data: break
        time.sleep(1.0)

    cols=['player','team','opp_team','event_id','market','line','over_odds','under_odds','book','commence_time','sport_key','position']
    rows=[]
    games_rows=[]; games_cols=['event_id','commence_time','sport_key','home_team','away_team','market','point','book']

    if not data:
        print("[oddsapi] no data after retries; writing empty CSVs")
        _write(pd.DataFrame(columns=cols), a.out)
        _write(pd.DataFrame(columns=games_cols), "outputs/odds_game.csv")
        return 0

    for g in data:
        home=(g.get('home_team') or '').upper(); away=(g.get('away_team') or '').upper()
        for bk in g.get('bookmakers', []):
            bk_key=bk.get('key')
            for mk in bk.get('markets', []):
                mkey=mk.get('key')
                if mkey in ('h2h','spreads','totals'):
                    point=None
                    if mk.get('outcomes'):
                        point = mk['outcomes'][0].get('point')
                    games_rows.append([g.get('id'), g.get('commence_time'), g.get('sport_key'),
                                       home, away, mkey, point, bk_key])
                    continue
                if not str(mkey).startswith("player_"):
                    continue
                for ou in mk.get('outcomes', []):
                    name=ou.get('description') or ou.get('name')
                    team=(ou.get('team') or '').upper()
                    opp=away if team==home else home if team==away else ''
                    line=ou.get('point') if 'point' in ou else (1.0 if mkey in ('player_anytime_td','player_2_or_more_tds') else None)
                    price=ou.get('price'); side=(ou.get('name','') or '').lower()
                    over_odds=price if (side in ('over','yes')) else None
                    under_odds=price if (side in ('under','no')) else None
                    rows.append([name,team,opp,g.get('id'),mkey,line,over_odds,under_odds,bk_key,g.get('commence_time'),g.get('sport_key','nfl'),ou.get('position') or ''])

    props=pd.DataFrame(rows, columns=cols)
    games=pd.DataFrame(games_rows, columns=games_cols)
    _write(props, a.out); _write(games, "outputs/odds_game.csv")
    print(f"[oddsapi] wrote props={len(props)} games={len(games)} tag={tag}")
    return 0

if __name__=='__main__':
    main()
