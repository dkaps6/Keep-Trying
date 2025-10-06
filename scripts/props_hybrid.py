import requests
import time
from typing import List, Optional, Dict, Any
from urllib.parse import urlencode

# =====================================================
# CONFIG
# =====================================================
BASE_ODDS_URL = "https://api.the-odds-api.com/v4/sports"
SPORT_KEY = "americanfootball_nfl"

ALLOWED_BOOKS = {"draftkings", "fanduel"}
DEFAULT_BOOKS = "draftkings,fanduel"

DEFAULT_MARKETS = [
    "player_receptions",
    "player_receiving_yards",
    "player_rushing_yards",
    "player_rush_attempts",
    "player_passing_yards",
    "player_passing_tds",
    "player_touchdown_anytime",
]

# =====================================================
# UTILITIES
# =====================================================

def _csv(val) -> str:
    """Normalize list or string to comma-separated string."""
    if not val:
        return ""
    if isinstance(val, str):
        return ",".join([x.strip() for x in val.split(",") if x.strip()])
    return ",".join([str(x).strip() for x in val if str(x).strip()])


def _clean_books(user_books) -> str:
    """Normalize and cap to DK + FD."""
    if not user_books:
        return DEFAULT_BOOKS
    parts = set(_csv(user_books).split(","))
    allowed = parts & ALLOWED_BOOKS
    return _csv(allowed) if allowed else DEFAULT_BOOKS


def _log_credits(headers: Dict[str, Any]):
    """Display remaining Odds API credits."""
    used = headers.get("x-requests-used") or headers.get("X-Requests-Used")
    rem = headers.get("x-requests-remaining") or headers.get("X-Requests-Remaining")
    if used or rem:
        print(f"[credits] used={used} remaining={rem}")


def _http_get(url: str, retries: int = 2, sleep: float = 0.8) -> Dict[str, Any]:
    """Simple GET with retry + credit counter logging."""
    last = None
    for _ in range(max(1, retries)):
        r = requests.get(url, timeout=25)
        last = r
        if r.ok:
            _log_credits(r.headers)
            return r.json()
        time.sleep(sleep)
    raise RuntimeError(
        f"GET failed after retries: {url}\nDetail: {getattr(last, 'text', '')[:400]}"
    )


# =====================================================
# CORE FETCHING
# =====================================================

def fetch_event_shells(api_key: str, books=None, regions="us") -> List[Dict[str, Any]]:
    """Fetch list of upcoming NFL events (to get event IDs)."""
    books_csv = _clean_books(books)
    qs = {
        "regions": regions,
        "markets": "h2h",
        "oddsFormat": "american",
        "dateFormat": "iso",
        "bookmakers": books_csv,
        "apiKey": api_key,
    }
    url = f"{BASE_ODDS_URL}/{SPORT_KEY}/odds?{urlencode(qs)}"
    print(f"[props_hybrid] Fetching event shells: {url}")
    data = _http_get(url)
    if not data:
        print("[props_hybrid] No events returned.")
        return []
    print(f"[props_hybrid] Retrieved {len(data)} events.")
    return data


def fetch_player_props(api_key: str, event_id: str, markets=None, books=None) -> Dict[str, Any]:
    """Fetch player props for a single event."""
    if not markets:
        markets = DEFAULT_MARKETS
    qs = {
        "regions": "us",
        "markets": _csv(markets),
        "oddsFormat": "american",
        "dateFormat": "iso",
        "bookmakers": _clean_books(books),
        "apiKey": api_key,
    }
    url = f"{BASE_ODDS_URL}/{SPORT_KEY}/events/{event_id}/odds?{urlencode(qs)}"
    print(f"[props_hybrid] Fetching props for event {event_id}")
    return _http_get(url)


def get_props(api_key: str, books=None, markets=None, limit: int = 0):
    """Master orchestrator — fetch events, then all props."""
    if not markets:
        markets = DEFAULT_MARKETS

    event_shells = fetch_event_shells(api_key, books=books)
    if not event_shells:
        print("[props_hybrid] No events found for given parameters.")
        return []

    if limit > 0:
        event_shells = event_shells[:limit]

    all_props = []
    for event in event_shells:
        event_id = event.get("id")
        if not event_id:
            continue
        props_data = fetch_player_props(api_key, event_id, markets, books)
        if props_data:
            all_props.append({"event": event, "props": props_data})
        time.sleep(0.5)
    print(f"[props_hybrid] Finished fetching props for {len(all_props)} events.")
    return all_props


# =====================================================
# MAIN EXECUTION (if standalone)
# =====================================================

if __name__ == "__main__":
    import os
    API_KEY = os.getenv("THE_ODDS_API_KEY") or input("Enter THE_ODDS_API_KEY: ").strip()
    props = get_props(API_KEY, books="draftkings,fanduel", limit=2)
    print(f"✅ Completed run. Events fetched: {len(props)}")
