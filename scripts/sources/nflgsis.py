from __future__ import annotations
import os, time
from typing import Dict, Any, List, Tuple
import requests
import pandas as pd
from bs4 import BeautifulSoup

LOGIN_URL = "https://www.nflgsis.com/GameStatsLive/Auth/Login"
HOME_URL  = "https://www.nflgsis.com/GameStatsLive/"
PBP_URL   = "https://www.nflgsis.com/GameStatsLive/GameDetail/PlayByPlay?gameId={game_id}"
BOX_URL   = "https://www.nflgsis.com/GameStatsLive/GameDetail?gameId={game_id}"

def login_session() -> requests.Session:
    user = os.getenv("NFLGSIS_USERNAME", "")
    pw   = os.getenv("NFLGSIS_PASSWORD", "")
    if not user or not pw:
        raise EnvironmentError("NFLGSIS_USERNAME / NFLGSIS_PASSWORD not set.")
    s = requests.Session()
    r = s.post(LOGIN_URL, data={"UserName": user, "Password": pw}, timeout=25, allow_redirects=True)
    if r.status_code != 200:
        raise RuntimeError(f"GSIS login HTTP {r.status_code}")
    # Try landing page; don’t hard-fail on branding differences.
    _ = s.get(HOME_URL, timeout=20)
    return s

def list_games(session: requests.Session) -> List[Dict[str, Any]]:
    try:
        r = session.get(HOME_URL, timeout=25)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        out: List[Dict[str, Any]] = []
        for a in soup.select("a[href*='gameId=']"):
            href = a.get("href", "")
            if "gameId=" not in href: 
                continue
            gid = href.split("gameId=")[-1].split("&")[0]
            title = (a.text or "").strip()
            if gid and gid.isdigit():
                out.append({"id": gid, "title": title})
        # de-dup
        uniq, seen = [], set()
        for g in out:
            if g["id"] in seen: 
                continue
            seen.add(g["id"]); uniq.append(g)
        return uniq
    except Exception:
        return []

def get_pbp(session: requests.Session, game_id: str) -> pd.DataFrame:
    try:
        r = session.get(PBP_URL.format(game_id=game_id), timeout=25)
        try:
            js = r.json()
            if isinstance(js, list): return pd.DataFrame(js)
            if isinstance(js, dict) and "data" in js: return pd.DataFrame(js["data"])
        except Exception:
            pass
        soup = BeautifulSoup(r.text, "html.parser")
        tbl = soup.find("table")
        if not tbl: return pd.DataFrame()
        rows = [[td.get_text(strip=True) for td in tr.select("td")] for tr in tbl.select("tr")]
        rows = [r for r in rows if r]
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

def _html_table_to_df(table) -> pd.DataFrame:
    try:
        headers = [th.get_text(strip=True) for th in table.select("thead th")]
        if not headers:
            first = table.find("tr")
            headers = [th.get_text(strip=True) for th in first.find_all(["th","td"])]
            body_rows = first.find_all_next("tr")
        else:
            body_rows = table.select("tbody tr")
        rows = []
        for tr in body_rows:
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if cells: rows.append(dict(zip(headers, cells)))
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

def _to_f(x) -> float:
    try:
        if x is None or x == "": return 0.0
        return float(str(x).replace(",",""))
    except Exception:
        return 0.0

def get_box_players(session: requests.Session, game_id: str) -> pd.DataFrame:
    try:
        r = session.get(BOX_URL.format(game_id=game_id), timeout=25)
        soup = BeautifulSoup(r.text, "html.parser")
        tables = soup.find_all("table")
        rows = []
        for t in tables:
            df = _html_table_to_df(t)
            if df.empty: continue
            cols = {c:c.lower() for c in df.columns}
            df = df.rename(columns=cols)
            if "player" not in df.columns: continue
            for _, rr in df.iterrows():
                rows.append({
                    "team": str(rr.get("team","")).strip(),
                    "player": str(rr.get("player") or rr.get("Player") or "").strip(),
                    "attempts": _to_f(rr.get("attempts") or rr.get("att")),
                    "pass_yards": _to_f(rr.get("pass_yards")),
                    "carries": _to_f(rr.get("carries") or rr.get("car")),
                    "rush_yards": _to_f(rr.get("rush_yards")),
                    "targets": _to_f(rr.get("targets") or rr.get("tgt")),
                    "receptions": _to_f(rr.get("receptions") or rr.get("rec")),
                    "rec_yards": _to_f(rr.get("rec_yards") or rr.get("yds")),
                })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

def team_player_tables(session: requests.Session, game_ids: List[str], limit: int | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    teams, players = [], []
    count = 0
    for gid in game_ids:
        if limit and count >= limit: break
        count += 1
        pbp = get_pbp(session, gid)
        pass_att, rush_att, team_abbrs = {}, {}, set()
        if not pbp.empty:
            lc = [c.lower() for c in pbp.columns]
            off_col = next((c for c in ["posteam","off","offense","team","home","away"] if c in lc), None)
            type_col = next((c for c in ["play_type","type","result","description","desc"] if c in lc), None)
            if off_col and type_col:
                oi, ti = lc.index(off_col), lc.index(type_col)
                for _, row in pbp.iterrows():
                    team = str(row.iloc[oi]).strip()
                    ptyp = str(row.iloc[ti]).lower()
                    if not team: continue
                    team_abbrs.add(team)
                    if "pass" in ptyp: pass_att[team] = pass_att.get(team,0.0)+1.0
                    elif ("rush" in ptyp) or ("run" in ptyp): rush_att[team] = rush_att.get(team,0.0)+1.0
        box = get_box_players(session, gid)
        if not box.empty:
            players.append(box)
        for tm in (team_abbrs or set([""])):
            pa, ra = pass_att.get(tm,0.0), rush_att.get(tm,0.0)
            teams.append({"team": tm, "pass_att": pa, "rush_att": ra, "plays": pa+ra, "event_id": gid})
        time.sleep(0.08)
    team_df = pd.DataFrame(teams)
    player_df = pd.concat(players, ignore_index=True) if players else pd.DataFrame()
    return team_df, player_df
