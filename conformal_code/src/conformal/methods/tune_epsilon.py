import numpy as np
import pandas as pd
from conformal.features import Y_TRUE_COL, Y_PRED_COL, FORECAST_DATE_COL
from .method_helpers import add_season, weighted_quantile, circ_week_dist, Pi_shrinkage, kernel_fn, parse_year_week_from_yyyww, n_eff


def summarize_eval(df_eval, alpha):
    coverage = float(df_eval["covered"].mean())
    mean_width = float(df_eval["width"].mean())
    median_width = float(df_eval["width"].median())
    return {
        "coverage": coverage,
        "mean_width": mean_width,
        "median_width": median_width,
    }

def objective_width_subject_to_group_coverage(df_eval, alpha, width_stat="mean", penalty_scale=1000.0):
    target = 1 - alpha

    if width_stat == "mean":
        width = float(df_eval["width"].mean())
    elif width_stat == "median":
        width = float(df_eval["width"].median())
    else:
        raise ValueError("width_stat must be 'mean' or 'median'")

    group_cov = df_eval.groupby(["Region", "Season"], observed=False)["covered"].mean()
    shortfalls = np.maximum(target - group_cov, 0.0)

    if np.all(shortfalls == 0):
        return width
    else:
        return width + penalty_scale * float(shortfalls.mean())
    

def run_kernel_regionmix_conformal(cal_df, eval_df, eps, alpha, tau_weeks, kernel_kind):

    cal = parse_year_week_from_yyyww(cal_df)
    ev  = parse_year_week_from_yyyww(eval_df)

    cal = add_season(cal)
    ev  = add_season(ev)

    cal["score"] = (cal[Y_TRUE_COL] - cal[Y_PRED_COL]).abs()

    regions = sorted(cal["Region"].unique())
    r_to_idx = {r:i for i,r in enumerate(regions)}
    Pi = Pi_shrinkage(regions, eps=eps)

    # Precompute kernel lookup K[w_test-1, w_cal-1]
    weeks = np.arange(1, 54)
    D = circ_week_dist(weeks[:, None], weeks[None, :], period=52)
    K = kernel_fn(D, tau=tau_weeks, kind=kernel_kind)

    cal_scores = cal["score"].to_numpy()
    cal_week   = cal["week_num"].to_numpy()
    cal_ridx   = cal["Region"].map(r_to_idx).to_numpy()

    qhat = np.empty(len(ev), dtype=float)
    neff = np.empty(len(ev), dtype=float)

    for j, row in enumerate(ev.itertuples(index=False)):
        wx = int(getattr(row, "week_num"))
        rx = r_to_idx[getattr(row, "Region")]

        w_week = K[wx-1, cal_week-1]
        w_reg  = Pi[cal_ridx, rx]
        w = w_week * w_reg

        s = w.sum()
        if s <= 0:
            qhat[j] = float(np.quantile(cal_scores, 1 - alpha))
            neff[j] = np.nan
        else:
            qhat[j] = weighted_quantile(cal_scores, w / s, 1 - alpha)
            neff[j] = n_eff(w)

    out = ev.copy()
    out["q_hat"] = qhat
    out["n_eff"] = neff
    out["lower"] = out[Y_PRED_COL] - out["q_hat"]
    out["upper"] = out[Y_PRED_COL] + out["q_hat"]
    out["covered"] = (out[Y_TRUE_COL] >= out["lower"]) & (out[Y_TRUE_COL] <= out["upper"])
    out["width"] = out["upper"] - out["lower"]
    return out