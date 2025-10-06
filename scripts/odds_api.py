# --- replace your existing helper with this ---

import re

def _apply_selection_filters(events_list, teams=None, events=None, selection=None):
    """
    Apply optional filters to a list of events.
    - teams: list[str] of substrings to match in home/away/team names. None/[]/"all" -> no team filter
    - events: list[str] of Odds API event IDs. None/[]/"all" -> no event-id filter
    - selection: string/regex to match event names. None/"" -> no selection filter
    Returns the filtered list.
    """

    # 1) selection
    sel_regex = None
    if selection is not None:
        s = str(selection).strip()
        if s != "":
            try:
                sel_regex = re.compile(s, re.IGNORECASE)
            except Exception:
                # treat as plain substring if regex fails
                sel_regex = re.compile(re.escape(s), re.IGNORECASE)

    def _sel_ok(ev):
        if sel_regex is None:
            return True
        name = str(getattr(ev, "name", "") or getattr(ev, "title", "") or "")
        return bool(sel_regex.search(name))

    # 2) team filter
    team_terms = None
    if teams is not None:
        if isinstance(teams, str):
            teams = [t.strip() for t in teams.split(",") if t.strip()]
        if isinstance(teams, list) and len(teams) > 0 and str(teams).lower() not in ("all",):
            team_terms = [t.lower() for t in teams]

    def _team_ok(ev):
        if team_terms is None:
            return True
        # Try to get home/away or team names safely
        home = str(getattr(ev, "home_team", "")).lower()
        away = str(getattr(ev, "away_team", "")).lower()
        name = str(getattr(ev, "name", "") or getattr(ev, "title", "") or "").lower()
        bucket = f"{home} {away} {name}"
        return any(term in bucket for term in team_terms)

    # 3) event-id filter
    event_ids = None
    if events is not None:
        if isinstance(events, str):
            events = [e.strip() for e in events.split(",") if e.strip()]
        if isinstance(events, list) and len(events) > 0 and str(events).lower() not in ("all",):
            event_ids = set(events)

    def _id_ok(ev):
        if event_ids is None:
            return True
        eid = getattr(ev, "id", None) or getattr(ev, "event_id", None) or getattr(ev, "key", None)
        return (eid in event_ids)

    filtered = [ev for ev in events_list if _sel_ok(ev) and _team_ok(ev) and _id_ok(ev)]
    return filtered
