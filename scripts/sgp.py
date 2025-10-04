# scripts/sgp.py
import itertools
import pandas as pd

# placeholder correlations; swap to copula later if you like
CORR = {
    ("QB_PASS_YDS","WR_YDS"): 0.6,
    ("QB_PASS_YDS","RB_RUSH_YDS"): -0.4,
    ("WR1_YDS","WR2_YDS"): 0.2,
}

def joint_prob_independent(ps):
    p = 1.0
    for x in ps: p *= max(min(float(x),1.0),0.0)
    return p

def build_sgp(priced_df: pd.DataFrame, max_legs=3, min_edge=0.02):
    pool = priced_df[priced_df["edge_pct"] > min_edge].copy()
    pool = pool.sort_values("edge_pct", ascending=False).head(150)
    sgps = []
    for L in range(2, max_legs+1):
        for idxs in itertools.combinations(pool.index, L):
            block = pool.loc[list(idxs)]
            jp = joint_prob_independent(block["blend_prob"].tolist())  # TODO: corr-adjust
            # convert probability to fair american
            fair = -100/jp if jp>=0.5 else 100*(1-jp)/jp
            sgps.append({
                "n_legs": L,
                "legs": " | ".join(block.get("description", block["market"]).astype(str).tolist()),
                "joint_prob": jp,
                "fair_odds": fair
            })
    return pd.DataFrame(sgps)
