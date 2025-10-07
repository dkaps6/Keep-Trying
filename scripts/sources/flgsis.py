# scripts/sources/nflgsis.py
from __future__ import annotations
import os, time
from typing import Dict, Any, List, Tuple
import requests
import pandas as pd
from bs4 import BeautifulSoup

"""
NFLGSIS (GameStatsLive) authenticated helpers.

This module logs in with your credentials and provides two light utilities:
- list_games(): get minimal set of available game ids/titles from the portal
- get_pbp():   fetch play-by-play for a game (JSON if available, otherwise parse HTML)
- team_player_tables(): derive TEAM and PLAYER rollups from multiple games (best-effort)

Notes:
- NFLGSIS is a private portal without a public API; this code uses form login and
  page requests behind the session cookies.
- If the portal changes HTML/flows, the parser may need small tweaks.
- We stay conservative: if anything fails, we return empty DataFrames rather than crash.
"""

LOGIN_URL = "https://www.nflgsis.com/GameStatsLive/Auth/Login"
HOME_URL  = "https://www.nflgsis.com/GameStatsLive/"
# This endpoint often returns JSON for PBP (depends on the view/config)
PBP_URL   = "https://www.nflgsis.com/GameStatsLive/GameDetail/PlayByPlay?gameId={game_id}"
# Some sites expose a PlayerStats grid; keep as optional (might be HTML only)
BOX_URL   = "https://www.nflgsis.com/GameStatsLive/GameDetail?gameId={game_id}"

def login_session() -> requests.Session:
    user = os.getenv("NFLGSIS_USERNAME", "")
    pw   = os.getenv("NFLGSIS_PASSWORD", "")
    if not user or not pw:
        raise EnvironmentError("NFLGSIS_USERNAME / NFLGSIS_PASSWORD not set.")
    s = requests.Session()
    # Simple POST; if site uses antiforgery tokens, this may require a pre-get + token capture.
    r = s.post(LOGIN_URL, data={"UserName": user, "Password": pw}, timeout=25, allow_redirects=True)
    # Heuristic pass check: 200 and landing page accessible
    if r.status_code != 200:
        raise RuntimeError(f"GSIS login HTTP {r.status_code}")
    test = s.get(HOME_URL, timeout=20)
    if test.status_code != 200 or "Game Stats Live" not in test.text:
        # Don't hard-fail; some tenants brand text differently. Keep going.
        pass
    return s

def list_games(session: requests.Session) -> List[Dict[str, Any]]:
    try:
        r = session.get(HOME_URL, timeout=25)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        out: List[Dict[str, Any]] = []
        # Grab any link with a gameId query param
        for a in soup.select("a[href*='gameId=']"):
            href = a.get("href", "")
            if "gameId=" not in href: 
                continue
            gid = href.split("gameId=")[-1].split("&")[0]
            title = (a.text or "").strip()
            if gid and gid.isdigit():
                out.append({"id": gid, "title": title})
        # De-dup
        seen = set(); uniq = []
        for g in out:
            if g["id"] in seen: 
                continue
            seen.add(g["id"]); uniq.append(g)
        return uniq
    except Exception:
        return []

def get_pbp(session: requests.Session, game_id: str) -> pd.DataFrame:
    """Try JSON first, then fallback to table HTML parse."""
    try:
        r = session.get(PBP_URL.format(game_id=game_id), timeout=25)
        # If JSON: build dataframe directly
        try:
            js = r.json()
            if isinstance(js, list):
                return pd.DataFrame(js)
            if isinstance(js, dict) and "data" in js:
                return pd.DataFrame(js["data"])
        except Exception:
            pass  # not JSON, try HTML
        soup = BeautifulSoup(r.text, "html.parser")
        tbl = soup.find("table")
        if not tbl:
            return pd.DataFrame()
        rows = []
        for tr in tbl.select("tr"):
            cells = [td.get_text(strip=True) for td in tr.select("td")]
            if cells: rows.append(cells)
        df = pd.DataFrame(rows)
        return df
    except Exception:
        return pd.DataFrame()

def get_box_players(session: requests.Session, game_id: str) -> pd.DataFrame:
    """
    Best-effort: scrape a box/players view if present, normalize to columns:
    team, player, attempts, pass_yards, carries, rush_yards, targets, receptions, rec_yards
    Returns empty on parse issues (we only use as a late fallback).
    """
    try:
        r = session.get(BOX_URL.format(game_id=game_id), timeout=25)
        soup = BeautifulSoup(r.text, "html.parser")
        # heuristic: look for any tables with 'Passing', 'Rushing', 'Receiving' headings
        sections = soup.find_all(["section","div"], string=lambda s: s and any(w in s for w in ["Passing","Rushing","Receiving"]))
        if not sections:
            # fallback: parse every table and try to infer
            tables = soup.find_all("table")
        else:
            tables = []
            for sec in sections:
                tables += sec.find_all("table")
        rows = []
        team_hint = None
        for t in tables:
            head = (t.find_previous("h3") or t.find_previous("h2") or t.find_previous("h4"))
            header_text = (head.get_text(strip=True) if head else "").lower()
            df = _html_table_to_df(t)
            if df.empty:
                continue
            if "player" not in [c.lower() for c in df.columns]:
                continue
            # Standardize columns
            cols = {c:c.lower() for c in df.columns}
            df = df.rename(columns=cols)
            team = team_hint or ""
            # Extract metrics
            attempts = df.get("att") or df.get("attempts")
            pass_yd = df.get("yds") if "passing" in header_text else None
            carries = df.get("att") if "rushing" in header_text else df.get("car")
            rush_yd = df.get("yds") if "rushing" in header_text else None
            tgts    = df.get("tgt") or df.get("targets")
            recs    = df.get("rec") or df.get("receptions")
            rec_yd  = df.get("yds") if "receiv" in header_text else None

            for _, rrow in df.iterrows():
                rows.append({
                    "team": team,
                    "player": str(rrow.get("player","")).strip(),
                    "attempts": _to_f(rrow.get("att") or rrow.get("attempts")),
                    "pass_yards": _to_f(rrow.get("yds")) if "passing" in header_text else 0.0,
                    "carries": _to_f(rrow.get("car") or rrow.get("att")) if "rushing" in header_text else 0.0,
                    "rush_yards": _to_f(rrow.get("yds")) if "rushing" in header_text else 0.0,
                    "targets": _to_f(rrow.get("tgt") or rrow.get("targets")),
                    "receptions": _to_f(rrow.get("rec") or rrow.get("receptions")),
                    "rec_yards": _to_f(rrow.get("yds")) if "receiv" in header_text else 0.0,
                })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

def _html_table_to_df(table) -> pd.DataFrame:
    try:
        headers = [th.get_text(strip=True) for th in table.select("thead th")]
        if not headers:
            # try first row
            first = table.find("tr")
            headers = [th.get_text(strip=True) for th in first.find_all(["th","td"])]
            body_rows = first.find_all_next("tr")
        else:
            body_rows = table.select("tbody tr")
        rows = []
        for tr in body_rows:
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if cells:
                rows.append(dict(zip(headers, cells)))
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

def _to_f(x) -> float:
    try:
        if x is None or x == "":
            return 0.0
        return float(str(x).replace(",",""))
    except Exception:
        return 0.0

def team_player_tables(session: requests.Session, game_ids: List[str], limit: int | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate multiple games into:
      team_df:   team, pass_att, rush_att, plays, event_id
      player_df: team, player, attempts, pass_yards, carries, rush_yards, targets, receptions, rec_yards
    """
    teams, players = [], []
    count = 0
    for gid in game_ids:
        if limit and count >= limit:
            break
        count += 1
        pbp = get_pbp(session, gid)
        # team totals from PBP if we can infer; otherwise later from box
        pass_att = {}; rush_att = {}
        team_abbrs = set()
        if not pbp.empty:
            # Heuristics: try to find columns that look like team/play_type
            lc = [c.lower() for c in pbp.columns]
            # common shapes: ["Quarter", "Time", "Off", "Def", "Down", "ToGo", "YdLine", "Type", "Description"]
            off_col = None
            for cand in ["posteam","off","offense","offense_team","team","home","away"]:
                off_col = cand if cand in lc else off_col
            type_col = None
            for cand in ["play_type","type","result","desc","description"]:
                type_col = cand if cand in lc else type_col
            if off_col is not None and type_col is not None:
                off_ix  = lc.index(off_col)
                type_ix = lc.index(type_col)
                for _, row in pbp.iterrows():
                    team = str(row.iloc[off_ix]).strip()
                    ptyp = str(row.iloc[type_ix]).lower()
                    if not team:
                        continue
                    team_abbrs.add(team)
                    if "pass" in ptyp:
                        pass_att[team] = pass_att.get(team, 0.0) + 1.0
                    elif "rush" in ptyp or "run" in ptyp:
                        rush_att[team] = rush_att.get(team, 0.0) + 1.0
        # box parse for players
        box = get_box_players(session, gid)
        if not box.empty and "player" in [c.lower() for c in box.columns]:
            # Try to infer team column if present; else blank
            team_col = None
            for c in box.columns:
                if c.lower() in {"team","tm"}:
                    team_col = c; break
            for _, r in box.iterrows():
                players.append({
                    "team": str(r.get(team_col,"")).strip() if team_col else "",
                    "player": str(r.get("player") or r.get("Player") or "").strip(),
                    "attempts": _to_f(r.get("attempts")),
                    "pass_yards": _to_f(r.get("pass_yards")),
                    "carries": _to_f(r.get("carries")),
                    "rush_yards": _to_f(r.get("rush_yards")),
                    "targets": _to_f(r.get("targets")),
                    "receptions": _to_f(r.get("receptions")),
                    "rec_yards": _to_f(r.get("rec_yards")),
                })
        # finalize team rows (even if some fields are zero)
        for tm in (team_abbrs or set([""])):  # ensure at least one team row per game if we saw anything
            pa = pass_att.get(tm, 0.0); ra = rush_att.get(tm, 0.0)
            teams.append({"team": tm, "pass_att": pa, "rush_att": ra, "plays": pa+ra, "event_id": gid})
        time.sleep(0.1)
    return pd.DataFrame(teams), pd.DataFrame(players)
