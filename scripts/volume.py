# scripts/volume.py
def consensus_spread_total(game_df):
    """
    Expect columns: event_id, home_spread, total.
    If not present, return empty or best-effort.
    """
    cols = {"event_id","home_spread","total"}
    if not cols.issubset(set(game_df.columns)):
        return game_df[["event_id"]].assign(home_spread=0.0, total=43.5)
    return game_df[["event_id","home_spread","total"]].copy()

def team_volume_estimates(ev_row, is_home: bool, tf_row=None, weather=None):
    """
    Returns (plays, pass_rate, rush_rate, win_prob).
    plays is left coarse for now; pass_rate breathes with PROE + weather.
    """
    plays = 63.0
    pass_rate = 0.56
    rush_rate = 1.0 - pass_rate

    # PROE (pass rate over expected)
    proe = 0.0
    try:
        proe = float(tf_row.get("proe", 0.0)) if tf_row is not None else 0.0
    except Exception:
        pass
    pass_rate += 0.6 * proe

    # Weather nudges
    if weather:
        try:
            wind = float(weather.get("wind_mph", 0) or 0)
            precip = str(weather.get("precip","")).lower()
        except Exception:
            wind, precip = 0.0, ""
        if wind >= 15:  pass_rate -= 0.02
        if "rain" in precip or "snow" in precip: pass_rate -= 0.02

    # Clamp and compute rush_rate
    pass_rate = min(max(pass_rate, 0.40), 0.66)
    rush_rate = 1.0 - pass_rate

    # Win prob from spread (home_spread = home − away; negative means home favored)
    win_prob = 0.5
    try:
        spread = float(ev_row.get("home_spread", 0.0) or 0.0)
        if is_home:
            win_prob = 0.5 + min(max(-spread / 20.0, -0.25), 0.25)
        else:
            win_prob = 0.5 + min(max(spread / 20.0, -0.25), 0.25)
    except Exception:
        pass

    return plays, pass_rate, rush_rate, win_prob
