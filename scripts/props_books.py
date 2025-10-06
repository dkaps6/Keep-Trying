# scripts/props_books.py
from __future__ import annotations

import json
import re
import time
import urllib.request
import urllib.error
from typing import Any, Dict, Iterable, List, Optional
from dataclasses import dataclass
import pandas as pd

# ---------- Public config ----------
DEFAULT_MARKETS = [
    "player_receptions",
    "player_receiving_yds",
    "player_rush_yds",
    "player_rush_attempts",
    "player_pass_yds",
    "player_pass_tds",
    "player_anytime_td",
]

BOOK_ALIASES = {
    "dk": "draftkings", "draftkings": "draftkings",
    "fd": "fanduel", "fanduel": "fanduel",
    "mgm": "betmgm", "betmgm": "betmgm",
    "cz": "caesars", "czrs": "caesars", "caesars": "caesars",
}

# ---------- Helpers ----------

def _http_json(url: str, headers: Optional[Dict[str, str]] = None, retries: int = 3, sleep: float = 0.8) -> Any:
    last = None
    for _ in range(max(1, retries)):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            last = e
            time.sleep(sleep)
    raise RuntimeError(f"[props_books] GET failed: {url}\n{last}")

def _norm_books_arg(books: Optional[str | Iterable[str]]) -> List[str]:
    if not books:
        return ["draftkings"]  # default to DK for coverage
    if isinstance(books, str):
        items = [b.strip() for b in books.split(",") if b.strip()]
    else:
        items = list(books)
    return [BOOK_ALIASES.get(b.lower(), b.lower()) for b in items]

def _mk_rows_template() -> List[Dict[str, Any]]:
    return []

def _to_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    # enforce column order even if empty
    want = [
        "event_id","event_name","home_team","away_team","commence_time",
        "player","market","book","price_name","vegas_line","vegas_price"
    ]
    for col in want:
        if col not in df.columns:
            df[col] = pd.Series(dtype="object")
    return df[want]

# ---------- Market mapping & matchers ----------

@dataclass
class MarketSig:
    key: str
    # case-insensitive substrings that identify the DK/FanDuel label
    hints: List[str]

MARKETS = [
    MarketSig("player_receptions",      ["receptions"]),
    MarketSig("player_receiving_yds",   ["receiving yards"]),
    MarketSig("player_rush_yds",        ["rushing yards"]),
    MarketSig("player_rush_attempts",   ["rushing attempts"]),
    MarketSig("player_pass_yds",        ["passing yards"]),
    MarketSig("player_pass_tds",        ["passing tds", "passing touchdowns"]),
    MarketSig("player_anytime_td",      ["anytime touchdown", "to score a touchdown"]),
]

def _guess_market_key(text: str) -> Optional[str]:
    t = text.lower()
    for m in MARKETS:
        if any(h in t for h in m.hints):
            return m.key
    return None

# ---------- DraftKings (DK) ----------

def _dk_fetch_props(markets: List[str]) -> pd.DataFrame:
    """
    DraftKings NFL event group 88808.
    We walk the offerCategories tree and collect player props, mapping by label hints.
    """
    # Main event group (NFL)
    url = "https://sportsbook.draftkings.com/sites/US-SB/api/v5/eventgroups/88808?format=json"
    data = _http_json(url, headers={"User-Agent": "Mozilla/5.0"})
    eg = (data or {}).get("eventGroup") or {}

    # Build event lookup (id -> metadata)
    events_index: Dict[str, Dict[str, Any]] = {}
    for ev in eg.get("events", []) or []:
        events_index[str(ev.get("eventId"))] = {
            "event_id": str(ev.get("eventId")),
            "home_team": ev.get("homeTeam"),
            "away_team": ev.get("awayTeam"),
            "name": f"{ev.get('awayTeam')} @ {ev.get('homeTeam')}",
            "commence_time": ev.get("startDate"),
        }

    rows = _mk_rows_template()

    # Traverse all offers (props are usually under 'offerCategories' → 'offerSubcategoryDescriptors')
    for cat in eg.get("offerCategories", []) or []:
        for sub in cat.get("offerSubcategoryDescriptors", []) or []:
            subname = str(sub.get("name") or "")
            for offers in sub.get("offers", []) or []:
                # each 'offers' is a list of markets for same bet
                for market in offers or []:
                    label = str(market.get("label") or subname)
                    # market may be a player prop; try to identify
                    mk = _guess_market_key(label)
                    if mk is None:
                        # try by outcome labels
                        out_labels = " ".join(str(o.get("label") or "") for o in market.get("outcomes", []) or [])
                        mk = _guess_market_key(out_labels) or _guess_market_key(subname)
                    if mk is None or mk not in markets:
                        continue

                    # collect outcomes
                    for oc in market.get("outcomes", []) or []:
                        # player name appears in 'participant' or 'label'
                        player = oc.get("participant") or oc.get("label")
                        if not player:
                            continue
                        # find event metadata
                        ev_id = str(market.get("eventId") or oc.get("eventId") or "")
                        ev_meta = events_index.get(ev_id, {})
                        rows.append({
                            "event_id": ev_meta.get("event_id") or ev_id,
                            "event_name": ev_meta.get("name") or "",
                            "home_team": ev_meta.get("home_team") or "",
                            "away_team": ev_meta.get("away_team") or "",
                            "commence_time": ev_meta.get("commence_time") or "",
                            "player": player,
                            "market": mk,
                            "book": "draftkings",
                            "price_name": str(oc.get("label") or ""),
                            "vegas_line": oc.get("line"),
                            "vegas_price": oc.get("oddsAmerican"),
                        })

    return _to_frame(rows)

# ---------- FanDuel / BetMGM / Caesars stubs ----------
# The structure is here so we can easily turn them on; for now they return empty
# unless you flip the flag and we wire the exact endpoints for your region.

def _fd_fetch_props(markets: List[str]) -> pd.DataFrame:
    # Placeholder: return empty DataFrame until we finalize endpoint wiring (region-dependent).
    return _to_frame([])

def _mgm_fetch_props(markets: List[str]) -> pd.DataFrame:
    return _to_frame([])

def _cz_fetch_props(markets: List[str]) -> pd.DataFrame:
    return _to_frame([])

FETCHERS = {
    "draftkings": _dk_fetch_props,
    "fanduel": _fd_fetch_props,
    "betmgm": _mgm_fetch_props,
    "caesars": _cz_fetch_props,
}

# ---------- Public entrypoint ----------

def get_props(
    date: Optional[str] = None,
    season: Optional[str] = None,
    window: Optional[int] = None,
    hours: Optional[int] = None,
    lookahead: Optional[int] = None,
    cap: int = 0,
    markets: Optional[str | Iterable[str]] = None,
    order: str = "odds",
    books: Optional[str | Iterable[str]] = None,
    team_filter: Optional[Iterable[str]] = None,
    selection: Optional[str] = None,
    event_ids: Optional[Iterable[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    wanted_markets: List[str]
    if markets is None:
        wanted_markets = DEFAULT_MARKETS
    elif isinstance(markets, str):
        wanted_markets = [m.strip() for m in markets.split(",") if m.strip()]
    else:
        wanted_markets = list(markets)

    wanted_books = _norm_books_arg(books)
    frames: List[pd.DataFrame] = []

    for b in wanted_books:
        fn = FETCHERS.get(b)
        if not fn:
            continue
        try:
            df = fn(wanted_markets)
        except Exception as e:
            print(f"[props_books] {b} fetch failed: {e}")
            df = _to_frame([])
        if cap and not df.empty:
            df = df.head(cap)
        frames.append(df)

    if not frames:
        return _to_frame([])

    df_all = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    # Basic de-dupe
    if not df_all.empty:
        df_all = df_all.drop_duplicates(subset=["event_id","player","market","book","vegas_line","price_name"], keep="first")
    print(f"[props_books] merged rows = {len(df_all)} (books={','.join(wanted_books)})")
    return df_all
