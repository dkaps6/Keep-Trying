TEAM_ALIASES = {
    "JAC": "JAX", "WSH": "WAS",
    "OAK": "LV", "SD": "LAC", "STL": "LAR",
    "LA":  "LAR",
}

def norm_team(s: str) -> str:
    if s is None: return ""
    s = str(s).strip().upper()
    return TEAM_ALIASES.get(s, s)

def norm_player(s: str) -> str:
    if s is None: return ""
    return " ".join(str(s).split())
