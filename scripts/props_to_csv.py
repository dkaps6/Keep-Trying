# scripts/props_to_csv.py
from __future__ import annotations
import argparse, os, sys, math, time, json, datetime as dt
from typing import List, Dict, Any, Tuple
import pandas as pd
import requests

# ---------------------------
# Config / helpers
# ---------------------------

SPORT_KEY = "americanfootball_nfl"
BASE = "https://api.the-odds-api.com/v4/sports"

def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def _parse_iso(s: str) -> dt.datetime:
    # Odds API returns ISO8601 strings (with Z)
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))

def _american_to_prob(odds: float) -> float:
    try:
        o = float(odds)
    except Exception:
        return float("nan")
    if o == 0:
        return float("nan")
    if o > 0:
        return 100.0 / (o + 100.0)
    return (-o) / ((-o) + 100.0)

def _devig_two_way(p_over: float, p_under: float) -> Tuple[float, float]:
    # Return fair probabilities (over, under) from vigged probabilities
    d = p_over + p_under
    if d == 0 or not math.isfinite(d):
        return float("nan"), float("nan")
    return p_over / d, p_under / d

def _headers() -> Dict[str, str]:
    return {"User-Agent": "props-to-csv/1.0 (+https://github.com/dkaps6/Keep-Trying)"}

def _http(url: str, params: Dict[str, Any]) -> Any:
    for i in range(3):
        r = requests.get(url, params=params, headers=_headers(), timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (402, 403, 404):
            # 402: out of credits; 403/404: no access / not found
            try:
                j = r.json()
            except Exception:
                j = {"error": r.text}
            raise RuntimeError(f"Odds API error {r.status_code}: {j}")
        time.sleep(0.5 * (2 ** i))
    r.raise_for_status()
    return None

# ---------------------------
# Core fetch
# ---------------------------

def get_props(
    *,
    api_key: str | None = None,
    date: str = "today",
    season: str = "",
    markets: List[str] | None = None,
    books: List[str] | None = None,
    order: str = "odds",
    teams_filter: List[str] | None = None,
    selection_filter: str | None = None,
    window_hours: int = 0,        # <—— the new parameter you needed
    cap: int = 0                  # optional: hard-cap events processed
) -> pd.DataFrame:
    """
    Fetch player props from The Odds API v4 and return a normalized DataFrame.
    This version accepts `window_hours` to restrict to games starting within N hours (0 = no filter).
    """
    key = api_key or os.getenv("THE_ODDS_API_KEY") or os.getenv("ODDS_API_KEY")
    if not key:
        raise RuntimeError("Missing THE_ODDS_API_KEY / ODDS_API_KEY")

    # Reasonable defaults if user passes comma string
    markets = markets or [
        "player_pass_yards",
        "player_pass_tds",
        "player_rush_yards",
        "player_rush_attempts",
        "player_receptions",
        "player_receiving_yards",
        "player_rush_receiving_yards",
        "player_anytime_td"
    ]
    books = books or ["draftkings", "fanduel", "betmgm", "caesars"]
    teams_filter = [t.strip().lower() for t in (teams_filter or []) if t.strip()]
    selection_filter = (selection_filter or "").strip()

    # Prep time-window filter
    cutoff_lo = _now_utc()
    cutoff_hi = None
    if window_hours and int(window_hours) > 0:
        cutoff_hi = cutoff_lo + dt.timedelta(hours=int(window_hours))

    params_common = {
        "apiKey": key,
        "regions": "us",
        "oddsFormat": "american",
        "dateFormat": "iso",
        "markets": ",".join(markets),
        "bookmakers": ",".join(books),
        "sort": order,
    }

    url = f"{BASE}/{SPORT_KEY}/odds"
    raw = _http(url, params_common)

    rows: List[Dict[str, Any]] = []
    processed_events = 0

    for ev in raw or []:
        try:
            event_id = ev.get("id")
            home = (ev.get("home_team") or "").strip()
            away = (ev.get("away_team") or "").strip()
            commence_time_iso = ev.get("commence_time")
            if not commence_time_iso:
                continue
            commence = _parse_iso(commence_time_iso)

            # window filter
            if cutoff_hi is not None:
                if not (cutoff_lo <= commence <= cutoff_hi):
                    continue

            # team filter
            if teams_filter:
                if not any(q in home.lower() or q in away.lower() for q in teams_filter):
                    continue

            for bk in ev.get("bookmakers", []):
                book_key = bk.get("key")
                # if a book isn't in our requested list (API bug/variance), skip
                if books and book_key not in books:
                    continue
                for mk in bk.get("markets", []):
                    mkey = mk.get("key")
                    for out in mk.get("outcomes", []):
                        # Player props usually have "description"/"name" fields
                        sel = out.get("description") or out.get("name") or ""
                        if selection_filter and selection_filter.lower() not in sel.lower():
                            continue
                        side = out.get("name") or out.get("label") or ""  # Over/Under/Yes/No
                        price = out.get("price")
                        point = out.get("point")
                        # Normalize probabilities
                        p = _american_to_prob(price)

                        rows.append({
                            "event_id": event_id,
                            "commence_time": commence_time_iso,
                            "home_team": home,
                            "away_team": away,
                            "book": book_key,
                            "market": mkey,
                            "selection": sel,
                            "side": side,
                            "line": point,
                            "price": price,
                            "prob_vig": p,   # vigged single-side prob
                        })

            processed_events += 1
            if cap and processed_events >= cap:
                break
        except Exception:
            # be forgiving—skip bad event entries
            continue

    if not rows:
        return pd.DataFrame(columns=[
            "event_id","commence_time","home_team","away_team","book",
            "market","selection","side","line","price",
            "prob_vig_over","prob_vig_under","prob_fair_over","prob_fair_under"
        ])

    df = pd.DataFrame(rows)

    # Build two-way pairs (Over/Under) or (Yes/No) for devig
    # We'll compute per market/line/selection/book the fair probs
    def _pair_key(r):
        # key that pairs the two sides of the same prop
        return (r["event_id"], r["book"], r["market"], r["selection"], r["line"])

    df["_pair"] = df.apply(_pair_key, axis=1)

    # split
    is_over = df["side"].str.lower().eq("over")
    is_under = df["side"].str.lower().eq("under")
    is_yes = df["side"].str.lower().eq("yes")
    is_no = df["side"].str.lower().eq("no")

    # Compute vigged side probs
    df["prob_vig"] = df["prob_vig"].astype(float)

    # Prepare devig containers
    df["prob_vig_over"] = pd.NA
    df["prob_vig_under"] = pd.NA
    df["prob_fair_over"] = pd.NA
    df["prob_fair_under"] = pd.NA

    # For each pair, apply devig if both sides exist
    for key, sub in df.groupby("_pair"):
        po = float(sub.loc[is_over & (sub["_pair"] == key), "prob_vig"].head(1).fillna(pd.NA) or pd.Series([float("nan")]))
        pu = float(sub.loc[is_under & (sub["_pair"] == key), "prob_vig"].head(1).fillna(pd.NA) or pd.Series([float("nan")]))
        py = float(sub.loc[is_yes  & (sub["_pair"] == key), "prob_vig"].head(1).fillna(pd.NA) or pd.Series([float("nan")]))
        pn = float(sub.loc[is_no   & (sub["_pair"] == key), "prob_vig"].head(1).fillna(pd.NA) or pd.Series([float("nan")]))

        # Over/Under first
        over, under = float("nan"), float("nan")
        if math.isfinite(po) and math.isfinite(pu) and po > 0 and pu > 0:
            over, under = _devig_two_way(po, pu)

        # TD yes/no etc.
        yes, no = float("nan"), float("nan")
        if math.isfinite(py) and math.isfinite(pn) and py > 0 and pn > 0:
            yes, no = _devig_two_way(py, pn)

        mask = df["_pair"] == key
        if math.isfinite(over):
            df.loc[mask, "prob_fair_over"] = over
            df.loc[mask, "prob_vig_over"]  = po if math.isfinite(po) else pd.NA
        if math.isfinite(under):
            df.loc[mask, "prob_fair_under"] = under
            df.loc[mask, "prob_vig_under"]  = pu if math.isfinite(pu) else pd.NA
        if math.isfinite(yes):
            # Use "over" slots for YES/NO to stay schema-compatible with the pipeline;
            # downstream code already understands that for markets like 'player_anytime_td'
            df.loc[mask, "prob_fair_over"] = yes
            df.loc[mask, "prob_vig_over"]  = py if math.isfinite(py) else pd.NA
        if math.isfinite(no):
            df.loc[mask, "prob_fair_under"] = no
            df.loc[mask, "prob_vig_under"]  = pn if math.isfinite(pn) else pd.NA

    # Clean / order
    keep = [
        "event_id","commence_time","home_team","away_team","book",
        "market","selection","side","line","price",
        "prob_vig_over","prob_vig_under","prob_fair_over","prob_fair_under"
    ]
    for c in keep:
        if c not in df.columns:
            df[c] = pd.NA
    out = df[keep].drop_duplicates().reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="today")
    ap.add_argument("--season", default="")
    ap.add_argument("--books", default="draftkings,fanduel,betmgm,caesars")
    ap.add_argument("--markets", default="")
    ap.add_argument("--order", default="odds")
    ap.add_argument("--teams", default="")
    ap.add_argument("--selection", default="")
    ap.add_argument("--cap", type=int, default=0)
    ap.add_argument("--window", type=int, default=0, help="Only include games starting within N hours (0 = off)")
    ap.add_argument("--write", default="outputs/props_raw.csv")
    args = ap.parse_args()

    books = [b.strip() for b in (args.books or "").split(",") if b.strip()]
    markets = [m.strip() for m in (args.markets or "").split(",") if m.strip()]
    teams = [t.strip() for t in (args.teams or "").split(",") if t.strip()]
    selection = args.selection or ""

    df = get_props(
        date=args.date,
        season=args.season,
        markets=markets or None,
        books=books or None,
        order=args.order,
        teams_filter=teams or None,
        selection_filter=selection or None,
        window_hours=args.window,   # <—— pass through
        cap=args.cap
    )

    os.makedirs(os.path.dirname(args.write), exist_ok=True)
    df.to_csv(args.write, index=False)
    print(f"[props_to_csv] ✅ wrote {len(df)} rows → {args.write}")


if __name__ == "__main__":
    main()
