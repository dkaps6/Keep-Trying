import pandas as pd

def make_team_last4_from_player_form(player_form):
    """
    Returns a dict keyed by team with team-level L4 totals of targets and rush attempts.
    """
    # Sum per team-week then rolling isn't trivial here; we already have player-form L4.
    tmp = player_form.groupby(["team","week"], as_index=False).agg(
        tgt_team=("tgt_l4","sum"),
        rush_att_team=("ra_l4","sum")
    )
    # Take latest week per team as "current"
    latest = tmp.sort_values("week").groupby("team").tail(1)
    latest = latest.rename(columns={"tgt_team":"tgt_team_l4","rush_att_team":"rush_att_team_l4"})
    return latest.set_index("team").to_dict(orient="index")
