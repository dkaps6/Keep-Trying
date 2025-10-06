# scripts/props_hybrid.py
from __future__ import annotations
import os, time, typing as T
import pandas as pd
import requests

# ---------------- Canonical NFL market keys + aliases (inlined) ----------------

GAME_MARKETS = {
    "h2h", "spreads", "totals",
    "alternate_spreads", "alternate_totals",
}
QH_MARKETS = {
    "h2h_q1","h2h_q2","h2h_q3","h2h_q4","h2h_h1","h2h_h2",
    "spreads_q1","spreads_q2","spreads_q3","spreads_q4","spreads_h1","spreads_h2",
    "totals_q1","totals_q2","totals_q3","totals_q4","totals_h1","totals_h2",
    "alternate_spreads_q1","alternate_spreads_q2","alternate_spreads_q3","alternate_spreads_q4",
    "alternate_spreads_h1","alternate_spreads_h2",
    "alternate_totals_q1","alternate_totals_q2","alternate_totals_q3","alternate_totals_q4",
    "alternate_totals_h1","alternate_totals_h2",
}
PLAYER_MARKETS = {
    "player_receptions","player_reception_yds","player_reception_longest","player_reception_tds",
    "player_rush_attempts","player_rush_yds","player_rush_longest","player_rush_tds",
    "player_sacks","player_solo_tackles","player_tackles_assists",
    "player_pass_attempts","player_pass_completions","player_pass_interceptions",
    "player_pass_longest_completion","player_pass_yds","player_pass_tds",
    "player_pats","player_kicking_points","player_field_goals","player_defensive_interceptions",
    "player_anytime_td","player_1st_td","player_last_td",
}
PLAYER_ALT_MARKETS = {
    "player_receptions_alternate","player_reception_yds_alternate",
    "player_reception_longest_alternate","player_reception_tds_alternate",
    "player_rush_attempts_alternate","player_rush_yds_alternate",
    "player_rush_longest_alternate","player_rush_tds_alternate",
    "player_sacks_alternate","player_solo_tackles_alternate","player_tackles_assists_alternate",
    "player_pass_attempts_alternate","player_pass_completions_alternate",
    "player_pass_interceptions_alternate","player_pass_longest_completion_alternate",
    "player_pass_yds_alternate","player_pass_tds_alternate",
    "player_pats_alternate","player_kicking_points_alternate",
    "player_field_goals_alternate","player_defensive_interceptions_alternate",
}
VALID_MARKETS = GAME_MARKETS | QH_MARKETS | PLAYER_MARKETS | PLAYER_ALT_MARKETS

ALIASES = {
    # legacy -> canonical
    "player_passing_yards": "player_pass_yds",
    "player_receiving_yards": "player_reception_yds",
    "player_passing_tds": "player_pass_tds",
    "player_passing_attempts": "player_pass_attempts",
    "player_passing_completions": "player_pass_completions",
    "player_passing_longest_completion": "player_pass_longest_completion",
    "player_rushing_yards": "player_rush_yds",
    "player_rushing_attempts": "player_rush_attempts",
    "player_rushing_tds": "player_rush_tds",
    "player_longest_reception": "player_reception_longest",
    "player_receiving_tds": "player_reception_tds",
}

NON_NFL_GLOBAL = {"btts", "draw_no_bet", "outrights", "h2h_lay"}

NFL_SPORT = "americanfootball_nfl"
BASE = "https://api.the-odds-api.com/v4"
DEFAULT_BOOKS = ["draftkings", "fanduel"]
DEFAULT_REGIONS = "us"
DEFAULT_ODDS_FORMAT = "american"
DEFAULT_DATEFORMAT = "iso"

def _log(msg:str)->None: print(f"[props_hybrid] {msg}")

def _env_api_key()->str:
    return (os.getenv("THE_ODDS_API_KEY") or os.getenv("ODDS_API_KEY") or "").strip()

def _coerce_books(books:T.Optional[T.Union[str,T.List[str]]])->str:
    if not books:
        return ",".join(DEFAULT_BOOKS)
    if isinstance(books, str):
        raw = [b.strip().lower() for b in books.split(",") if b.strip()]
    else:
        raw = [str(b).strip().lower() for b in books if str(b).strip()]
    dedup=[]
    for b in raw:
        if b not in dedup: dedup.append(b)
    return ",".join(dedup)

def _alias_and_filter_markets(markets:T.Optional[T.Union[str,T.List[str]]])->T.List[str]:
    if not markets: return []
    if isinstance(markets, str):
        mlist=[m.strip().lower() for m in markets.split(",") if m.strip()]
    else:
        mlist=[str(m).strip().lower() for m in markets if str(m).strip()]
    fixed=[]
    for m in mlist:
        m = ALIASES.get(m, m)
        if m in NON_NFL_GLOBAL: continue
        if m in VALID_MARKETS: fixed.append(m)
    out=[]
    for m in fixed:
        if m not in out: out.append(m)
    return out

def _http_get(url:str, params:dict)->tuple[dict,dict]:
    last=None
    for i in range(3):
        try:
            r=requests.get(url, params=params, timeout=30)
            hdrs={k.lower():v for k,v in r.headers.items()}
            if r.status_code==200:
                return r.json(), hdrs
            try: body=r.json()
            except Exception: body=r.text
            last=f"HTTP {r.status_code}: {body}"
        except Exception as e:
            last=repr(e)
        time.sleep(0.8*(i+1))
    raise RuntimeError(f"GET failed after retries: {url}\nDetail: {last}")

def _print_credits(hdrs:dict)->None:
    left=hdrs.get("x-requests-remaining")
    used=hdrs.get("x-requests-used")
    reset=hdrs.get("x-requests-reset")
    if any([left,used,reset]):
        _log(f"credits | used={used} remaining={left} reset={reset}")

def _list_event_ids(api_key:str, date:str, books_csv:str)->list[str]:
    url=f"{BASE}/sports/{NFL_SPORT}/odds"
    params=dict(apiKey=api_key, regions=DEFAULT_REGIONS, markets="h2h",
                oddsFormat=DEFAULT_ODDS_FORMAT, dateFormat=DEFAULT_DATEFORMAT,
                bookmakers=books_csv)
    data,h=_http_get(url, params); _print_credits(h)
    ev=[]
    if isinstance(data, list):
        for evobj in data:
            eid=str(evobj.get("id") or evobj.get("eventId") or evobj.get("key") or "")
            if eid: ev.append(eid)
    out=[]
    for e in ev:
        if e not in out: out.append(e)
    _log(f"event shells fetched: {len(out)}")
    return out

# --- helper: check if payload contains any outcomes
def _has_outcomes(payload: dict) -> bool:
    """Return True if any bookmaker in payload has any market with any outcomes."""
    for bk in (payload.get("bookmakers") or []):
        for mk in (bk.get("markets") or []):
            oc = mk.get("outcomes") or []
            if oc:
                return True
    return False


# --- robust event-prop fetcher with per-market fallback
def _fetch_event_props(api_key: str, event_id: str, books_csv: str, markets: list[str]) -> list[dict]:
    """
    Fetch props for one event. First try all markets together.
    If the payload has no outcomes, retry one-market-at-a-time and merge.
    """
    if not markets:
        return []

    base_url = f"{BASE}/sports/{NFL_SPORT}/events/{event_id}/odds"
    common = dict(
        apiKey=api_key,
        regions=DEFAULT_REGIONS,
        oddsFormat=DEFAULT_ODDS_FORMAT,
        dateFormat=DEFAULT_DATEFORMAT,
        bookmakers=books_csv,
    )

    # 1) bulk request for all markets
    params = dict(common, markets=",".join(markets))
    data, hdrs = _http_get(base_url, params)
    _print_credits(hdrs)

    agg: list[dict] = []
    if isinstance(data, dict):
        data.setdefault("eventId", event_id)
        if _has_outcomes(data):
            agg.append(data)
            _log(f"event {event_id}: outcomes (bulk) ✓")
            return agg

    # 2) fallback: one-market-at-a-time
    _log(f"event {event_id}: no outcomes in bulk call → retrying per-market")
    for m in markets:
        p = dict(common, markets=m)
        d, h = _http_get(base_url, p)
        _print_credits(h)
        if isinstance(d, dict):
            d.setdefault("eventId", event_id)
            if _has_outcomes(d):
                agg.append(d)

    if agg:
        _log(f"event {event_id}: outcomes (per-market) ✓ {len(agg)} payload(s)")
    else:
        _log(f"event {event_id}: still no outcomes for requested markets")

    return agg


# ---------- NEW: flatten to one-row-per-outcome with explicit "market" ----------

def _flatten_event_payloads(payloads:list[dict])->list[dict]:
    flat=[]
    for ev in payloads:
        if not isinstance(ev, dict): continue
        event_id = ev.get("id") or ev.get("eventId") or ev.get("key")
        commence = ev.get("commence_time") or ev.get("commenceTime")
        home = ev.get("home_team") or ev.get("homeTeam")
        away = ev.get("away_team") or ev.get("awayTeam")
        bookmakers = ev.get("bookmakers") or []

        for bk in bookmakers or []:
            bk_key = (bk.get("key") or bk.get("title") or "").lower()
            for mk in bk.get("markets") or []:
                mkey = mk.get("key")  # <-- this is the market key we need
                # odds-api v4 has "outcomes" per market (list of dicts)
                for oc in mk.get("outcomes") or []:
                    row = {
                        "eventId": event_id,
                        "commence_time": commence,
                        "home_team": home,
                        "away_team": away,
                        "bookmaker": bk_key,
                        "market": mkey,                 # <- REQUIRED by normalize
                        "outcome_name": oc.get("name"),
                        "price": oc.get("price"),
                        "point": oc.get("point"),
                    }
                    # helpful echoes for player props
                    # (many books put player name in outcome_name already)
                    player = oc.get("description") or oc.get("player") or None
                    if player:
                        row["player"] = player
                    flat.append(row)
    return flat

# ---------------- public entry point ----------------

def get_props(
    *,
    api_key: str | None = None,
    date: str = "today",
    season: int | None = None,
    cap: int = 0,
    markets: str | list[str] | None = None,
    books: str | list[str] | None = None,
    order: str = "odds",
    team_filter: list[str] | None = None,
    selection: str | None = None,
    event_ids: list[str] | None = None,
    window: int | str | None = None,
    hours: int | None = None,
    **_,
) -> pd.DataFrame:
    api_key = (api_key or _env_api_key()).strip()
    if not api_key:
        raise RuntimeError("Missing THE_ODDS_API_KEY in environment.")

    books_csv = _coerce_books(books)
    clean_markets = _alias_and_filter_markets(markets)
    if not clean_markets:
        clean_markets = ["player_pass_yds","player_rush_yds","player_reception_yds",
                         "player_receptions","player_anytime_td"]
        _log(f"no markets passed; using defaults: {','.join(clean_markets)}")
    else:
        _log(f"markets => {','.join(clean_markets)}")

    if not event_ids:
        event_ids = _list_event_ids(api_key, date=date, books_csv=books_csv)
        if cap and cap>0: event_ids = event_ids[:cap]
    else:
        _log(f"using supplied event_ids={len(event_ids)} (cap={cap})")
        if cap and cap>0: event_ids = event_ids[:cap]

    if not event_ids:
        _log("no events available with current filters; returning empty.")
        return pd.DataFrame()

    all_payloads=[]
    for eid in event_ids:
        payloads=_fetch_event_props(api_key, eid, books_csv, clean_markets)
        # tag the top-level with the event id if the payload omitted it
        for p in payloads:
            p.setdefault("eventId", eid)
        all_payloads.extend(payloads)

    if not all_payloads:
        _log("no props returned from Odds API.")
        return pd.DataFrame()

    flat_rows = _flatten_event_payloads(all_payloads)
    if not flat_rows:
        _log("flatten produced 0 rows.")
        return pd.DataFrame()

    df = pd.DataFrame(flat_rows)
    _log(f"raw rows (post-flatten): {len(df)}")
    return df
