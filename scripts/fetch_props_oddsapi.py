import os, argparse, pathlib, pandas as pd, requests, time

# Candidate market names (we’ll probe and keep only what the API accepts)
CANDIDATE_PLAYER_MARKETS = [
    # Common v4 names (exactly as docs/use in prod)
    "player_pass_yards",
    "player_rush_yards",
    "player_rec_yards",
    "player_receptions",
    "player_rush_rec_yards",
    "player_anytime_td",
    "player_pass_tds",
    # Some books use these aliases; probing will tell us if they’re valid
    "player_passing_yards", "player_rushing_yards", "player_receiving_yards",
    "player_rush_and_receive_yards", "player_two_or_more_tds", "player_2_or_more_tds",
]

GAME_MARKETS = ["h2h", "spreads", "totals"]

def _write(df, path):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def _call(url, params):
    r = requests.get(url, params=params, timeout=45)
    for k in ("x-requests-remaining","x-requests-used"):
        if k in r.headers:
            print(f"[oddsapi] {k}: {r.headers[k]}")
    if r.status_code == 422:
        raise ValueError("422")
    r.raise_for_status()
    return r.json()

def _fetch_market(url, key, books):
    """Return (ok, data_or_none). If market unsupported, ok=False, data=None."""
    params={'regions':'us','oddsFormat':'american','bookmakers':books,'markets':key}
    try:
        print(f"[oddsapi] probe market={key}")
        data=_call(url, params)
        return True, data
    except ValueError as e:
        print(f"[oddsapi] 422 unsupported market={key}")
        return False, None
    except Exception as e:
        print(f"[oddsapi] error market={key}: {e}")
        return False, None

def _flatten_props(payload):
    cols=['player','team','opp_team','event_id','market','line','over_odds','under_odds','book','commence_time','sport_key','position']
    rows=[]
    for g in payload:
        home=(g.get('home_team') or '').upper(); away=(g.get('away_team') or '').upper()
        for bk in g.get('bookmakers', []):
            bk_key=bk.get('key')
            for mk in bk.get('markets', []):
                mkey=mk.get('key')
                if not str(mkey).startswith("player_"):
                    continue
                for ou in mk.get('outcomes', []):
                    name=ou.get('description') or ou.get('name')
                    team=(ou.get('team') or '').upper()
                    opp=away if team==home else home if team==away else ''
                    line=ou.get('point') if 'point' in ou else (1.0 if 'anytime_td' in mkey else None)
                    price=ou.get('price'); side=(ou.get('name','') or '').lower()
                    over_odds=price if (side in ('over','yes')) else None
                    under_odds=price if (side in ('under','no')) else None
                    rows.append([name,team,opp,g.get('id'),mkey,line,over_odds,under_odds,bk_key,g.get('commence_time'),g.get('sport_key','nfl'),ou.get('position') or ''])
    return pd.DataFrame(rows, columns=cols)

def _flatten_games(payload):
    cols=['event_id','commence_time','sport_key','home_team','away_team','market','point','book']
    rows=[]
    for g in payload:
        home=(g.get('home_team') or '').upper(); away=(g.get('away_team') or '').upper()
        for bk in g.get('bookmakers', []):
            bk_key=bk.get('key')
            for mk in bk.get('markets', []):
                mkey=mk.get('key')
                if mkey not in GAME_MARKETS: continue
                point=None
                if mk.get('outcomes'): point = mk['outcomes'][0].get('point')
                rows.append([g.get('id'), g.get('commence_time'), g.get('sport_key'), home, away, mkey, point, bk_key])
    return pd.DataFrame(rows, columns=cols)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--books', default='draftkings,fanduel,betmgm,caesars')
    ap.add_argument('--date', default='')  # kept for compatibility; we ignore date to avoid 422 combos
    ap.add_argument('--out', default='outputs/props_raw.csv')
    a=ap.parse_args()

    key=os.getenv('ODDS_API_KEY','').strip()
    if not key:
        print('[oddsapi] key missing; writing empty CSVs'); 
        _write(pd.DataFrame(), a.out); _write(pd.DataFrame(), "outputs/odds_game.csv"); return 0

    base='https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds'

    # 1) fetch game markets once
    try:
        print(f"[oddsapi] request game markets={','.join(GAME_MARKETS)}")
        game_json=_call(base, {'regions':'us','oddsFormat':'american','bookmakers':a.books,'markets':','.join(GAME_MARKETS)})
    except Exception as e:
        print(f"[oddsapi] game error: {e}")
        game_json=[]

    games=_flatten_games(game_json)
    _write(games, "outputs/odds_game.csv")
    print(f"[oddsapi] wrote games={len(games)}")

    # 2) probe player markets individually so unsupported keys don't 422 the whole request
    all_payload=[]
    supported=[]
    for mk in CANDIDATE_PLAYER_MARKETS:
        ok, data = _fetch_market(base, mk, a.books)
        if not ok or not data:
            continue
        supported.append(mk)
        all_payload.extend(data)

        # polite pacing to stay clear of per-second throttle
        time.sleep(0.4)

    if supported:
        print(f"[oddsapi] supported player markets: {supported}")
    else:
        print("[oddsapi] no supported player markets detected")

    props = _flatten_props(all_payload)
    _write(props, a.out)
    print(f"[oddsapi] wrote props={len(props)}")

    return 0

if __name__=='__main__':
    main()
