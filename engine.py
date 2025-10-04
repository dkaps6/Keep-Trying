# engine.py
import os
import re
import pandas as pd

from scripts.odds_api import fetch_game_lines, fetch_props_all_events
from scripts.features_external import build_external
from scripts.id_map import map_players
from scripts.pricing import (
    american_to_prob, devig_two_way, blend, prob_to_american,
    edge_pct, kelly_fraction, tier
)
from scripts.model_core import (
    base_sigma, over_prob_normal,
    player_shares,
    mu_receptions, mu_rec_yards, mu_rush_atts, mu_rush_yards, mu_pass_yards
)
from scripts.elite_rules import (
    funnel_multiplier, volatility_widen, pace_smoothing, sack_to_attempts
)
from scripts.engine_helpers import make_team_last4_from_player_form
from scripts.rules_engine import apply_rules
from scripts.volume import consensus_spread_total, team_volume_estimates
from scripts.td_model import (
    implied_team_totals, total_to_td_lambda, player_td_probability,
    qb_pass_tds_lambda, yes_prob_from_lambda
)
# from scripts.calibration import apply_shrinkage  # optional; enable after week 1
# from scripts.sgp import build_sgp  # optional

def _norm(val) -> str:
    if val is None: return ""
    s = str(val).strip()
    if s.lower() in {"nan","none","null"}: return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+"," ",s).strip()

def run_pipeline(target_date: str, season: int, out_dir: str = "outputs"):
    # 1) Odds
    game_df  = fetch_game_lines()
    props_df = fetch_props_all_events()

    cons = consensus_spread_total(game_df)  # event_id, home_spread, total

    # 2) External features (ids, player_form, team_form, weather)
    ext = build_external(season)
    ids = ext["ids"]; team_form = ext["team_form"]; pform = ext["player_form"]; weather_df = ext["weather"]

    team_l4_map = make_team_last4_from_player_form(pform)

    # 3) Map player names → ids/position/team
    props_df = map_players(props_df, ids)

    # simple weather lookup
    weather_by_event = {}
    if not weather_df.empty and "event_id" in weather_df.columns:
        weather_by_event = {str(r["event_id"]): r for _, r in weather_df.iterrows()}

    def team_ctx(team):
        return team_l4_map.get(team, {"tgt_team_l4": 30.0, "rush_att_team_l4": 25.0})

    rows = []
    for _, r in props_df.iterrows():
        market = r["market"]
        if market not in {
            "player_reception_yds","player_receptions",
            "player_rush_yds","player_rush_attempts",
            "player_pass_yds","player_pass_tds",
            "player_anytime_td",
        }:
            continue

        price = r["price"]; line = r["point"]; side = r["outcome"]
        # market fair (de-vig)
        p_raw = american_to_prob(price)
        p_over_raw  = p_raw if side in ("Over","Yes") else None
        p_under_raw = p_raw if side in ("Under","No") else None
        p_over_fair, p_under_fair = devig_two_way(p_over_raw, p_under_raw)
        market_p_fair = p_over_fair if side in ("Over","Yes") else p_under_fair

        # volume
        ev = cons[cons["event_id"].eq(r["event_id"])].head(1)
        ev_row = ev.iloc[0] if not ev.empty else {}
        team_code = r.get("recent_team") or ""
        is_home = _norm(team_code) and (_norm(team_code) in _norm(r.get("home") or ""))
        w = weather_by_event.get(str(r.get("event_id")), None)

        # team form row for PROE, etc.
        tf_row = None
        opp = r.get("opponent") or r.get("opp") or ""
        if not team_form.empty and opp in team_form.index:
            tf_row = team_form.loc[opp].to_dict()

        plays, pass_rate, rush_rate, win_prob = team_volume_estimates(ev_row, bool(is_home), tf_row=tf_row, weather=w)

        # player shares/efficiency
        pid = r.get("gsis_id")
        pr = pform[pform["gsis_id"] == pid]
        pr = pr.sort_values("week").tail(1).iloc[0].to_dict() if not pr.empty else {}
        shares = player_shares(pr, team_ctx(r.get("recent_team")))
        tgt_share, rush_share, catch_rate, ypr, ypc = shares

        pass_mult = funnel_multiplier(True,  def_rush_epa_z=tf_row.get("pass_epa_z",0.0) if tf_row else 0.0,
                                           def_pass_epa_z=tf_row.get("pass_epa_z",0.0) if tf_row else 0.0)
        rush_mult = funnel_multiplier(False, def_rush_epa_z=tf_row.get("pass_epa_z",0.0) if tf_row else 0.0,
                                            def_pass_epa_z=tf_row.get("pass_epa_z",0.0) if tf_row else 0.0)
        plays = pace_smoothing(plays, 0.0, 0.0)

        sigma = base_sigma(market)
        model_prob_override = None

        if market == "player_receptions":
            mu_rec = mu_receptions(plays, pass_rate, tgt_share, catch_rate) * pass_mult
            mu = mu_rec

        elif market == "player_reception_yds":
            mu_rec = mu_receptions(plays, pass_rate, tgt_share, catch_rate) * pass_mult
            mu = mu_rec_yards(mu_rec, ypr)

        elif market == "player_rush_attempts":
            atts = mu_rush_atts(plays, rush_rate, rush_share) * rush_mult
            atts = sack_to_attempts(atts, sack_rate_above_avg=0.0)
            mu = atts

        elif market == "player_rush_yds":
            atts = mu_rush_atts(plays, rush_rate, rush_share) * rush_mult
            mu = mu_rush_yards(atts, ypc)

        elif market == "player_pass_yds":
            qb_ypa = (pr.get("pass_yds_l4",0)/max(pr.get("pass_att_l4",1),1)) if pr else 6.8
            dropbacks = plays * pass_rate
            mu = mu_pass_yards(dropbacks, qb_ypa, z_opp_pressure=tf_row.get("pressure_z",0.0) if tf_row else 0.0,
                                              z_opp_pass_epa=tf_row.get("pass_epa_z",0.0) if tf_row else 0.0)
            sigma = volatility_widen(sigma, pressure_mismatch=False, qb_inconsistent=False)

        elif market == "player_pass_tds":
            total_pts = float(ev_row.get("total", 43.5) or 43.5)
            spread    = float(ev_row.get("home_spread", 0.0) or 0.0)  # home - away
            home_total, away_total = implied_team_totals(total_pts, spread, home_is_favorite=(spread<0))
            my_team_total = home_total if is_home else away_total
            lam_td = total_to_td_lambda(my_team_total)
            lam_pass = qb_pass_tds_lambda(lam_td, pass_rate)
            model_prob_override = yes_prob_from_lambda(lam_pass)
            mu = lam_pass
            sigma = max(0.6, sigma * 0.04)

        elif market == "player_anytime_td":
            total_pts = float(ev_row.get("total", 43.5) or 43.5)
            spread    = float(ev_row.get("home_spread", 0.0) or 0.0)
            home_total, away_total = implied_team_totals(total_pts, spread, home_is_favorite=(spread<0))
            my_team_total = home_total if is_home else away_total
            lam_td = total_to_td_lambda(my_team_total)
            rz_share = float(pr.get("rz_tgt_share_l4", 0.15) if pr else 0.15)
            goal_bias = 1.15 if str(r.get("position","")).upper() in {"RB","FB"} else 1.0
            p_yes = player_td_probability(lam_td, rz_share, goal_line_bias=goal_bias)
            model_prob_override = p_yes
            mu = p_yes
            sigma = 1.0

        else:
            mu = 0.0

        # Elite rules (now fed real z-scores where available)
        opp_pressure_z = float(tf_row.get("pressure_z",0.0)) if tf_row else 0.0
        opp_pass_epa_z = float(tf_row.get("pass_epa_z",0.0)) if tf_row else 0.0
        heavy_man  = (float(tf_row.get("man_rate_z",0.0))  if tf_row else 0.0) > 0.8
        heavy_zone = (float(tf_row.get("zone_rate_z",0.0)) if tf_row else 0.0) > 0.8
        light_box_share = tf_row.get("light_box_share", None) if tf_row else None
        heavy_box_share = tf_row.get("heavy_box_share", None) if tf_row else None

        mu, sigma, notes = apply_rules(
            market=market, side=side, mu=mu, sigma=sigma,
            player=pr, team_ctx=team_ctx(r.get("recent_team")),
            opp_pressure_z=opp_pressure_z, opp_pass_epa_z=opp_pass_epa_z,
            run_funnel=False, pass_funnel=False,
            alpha_limited=False, tough_shadow=False,
            heavy_man=heavy_man, heavy_zone=heavy_zone,
            team_ay_att_z=0.0, light_box_share=light_box_share, heavy_box_share=heavy_box_share,
            win_prob=win_prob, qb_inconsistent=False, pressure_mismatch=False,
        )

        # Optional: μ shrinkage from calibration
        # mu = apply_shrinkage(market, mu)

        # Price at quoted line
        if model_prob_override is not None:
            model_p = model_prob_override if side in ("Over","Yes") else (1.0 - model_prob_override)
        else:
            model_p = over_prob_normal(line, mu, sigma) if side in ("Over","Yes") else 1 - over_prob_normal(line, mu, sigma)

        p_blend = blend(model_p, market_p_fair, w_model=0.65)
        fair_american = prob_to_american(p_blend)
        edge = edge_pct(p_blend, market_p_fair)
        kelly = kelly_fraction(p_blend, price, cap=0.05)
        tier_label = tier(edge)

        rows.append({
            **r.to_dict(),
            "model_mu": mu, "model_sigma": sigma,
            "model_prob": model_p, "market_prob_fair": market_p_fair,
            "blend_prob": p_blend, "blend_fair_odds": fair_american,
            "edge_pct": edge, "edge_pp": 100.0 * edge,
            "kelly_cap": kelly, "tier": tier_label,
            "notes": notes,
            "plays_est": plays, "pass_rate_est": pass_rate, "rush_rate_est": rush_rate,
            "win_prob_est": win_prob,
        })

    out = pd.DataFrame(rows)
    os.makedirs(out_dir or "outputs", exist_ok=True)
    out.to_csv(f"{out_dir}/props_priced.csv", index=False)
    game_df.to_csv(f"{out_dir}/game_lines.csv", index=False)

    # Optional: build SGPs
    # sgp_df = build_sgp(out, max_legs=3, min_edge=0.02)
    # sgp_df.to_csv(f"{out_dir}/sgp.csv", index=False)

    return {"game_lines": game_df, "props_priced": out}

