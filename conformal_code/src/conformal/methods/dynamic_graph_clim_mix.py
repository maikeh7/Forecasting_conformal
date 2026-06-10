import numpy as np
import pandas as pd
from .method_helpers import add_season, weighted_quantile, circ_week_dist, Pi_shrinkage, kernel_fn, parse_year_week_from_yyyww, n_eff
from conformal.features import Y_PRED_COL, Y_TRUE_COL, ALPHA, FORECAST_DATE_COL, CURRENT_WEEK_COL
from conformal.config import HORIZONS
from conformal.utils import apply_symmetric_intervals
from conformal.metrics import add_interval_score
from conformal.base import ConformalResult, validate_prediction_df
from conformal.methods.method_helpers import calculate_detrended_intervals
from conformal.hetGP.HetGP_Fitting import run_hetGP_fitting
from scipy.stats import norm

Z90 = 1.6448536269514722

rho_by_horizon = {
    1: 0.15,
    2: 0.35,
    3: 0.50,
    4: 0.60,
}

    
def effective_n(weights):
    w = np.asarray(weights, dtype=float)
    s = w.sum()
    if s <= 0:
        return np.nan
    w = w / s
    return float(1.0 / np.sum(w ** 2))

    

def add_interval_score_from_cols(
    df,
    *,
    alpha,
    y_true_col,
    lower_col,
    upper_col,
    score_col,
):
    """
    Interval score for a central (1-alpha) interval.
    Lower is better.
    """
    out = df.copy()

    y = out[y_true_col]
    l = out[lower_col]
    u = out[upper_col]

    out[score_col] = (
        (u - l)
        + (2.0 / alpha) * (l - y).clip(lower=0)
        + (2.0 / alpha) * (y - u).clip(lower=0)
    )

    return out


def add_clim_variance_mix_to_df(
    df,
    *,
    hetgp_pred_df,
    rho,
    alpha,
    y_true_col,
    y_pred_col,
    sd_clim_col="hetgp_predSD",
    q_col="q_hat",
    merge_cols=("Region", "week_num"),
    prefix="dynamic_varmix",
    mixing_type="convex",
):
    """
    Keep the dynamic graph CP mean, but mix/inflate variance using climatology.

    Convex variance mixing:
        var_final = (1-rho) * var_cp + rho * var_clim

    Inflation-only variant:
        var_final = var_cp + rho * max(var_clim - var_cp, 0)

    This function assumes df already has q_hat from dynamic graph conformal.
    """
    out = df.copy()

    # Merge climatology predictive SD by Region/week_num.
    keep_cols = list(merge_cols) + [sd_clim_col]
    hetgp_small = hetgp_pred_df[keep_cols].drop_duplicates(subset=list(merge_cols))

    out = pd.merge(
        out,
        hetgp_small,
        on=list(merge_cols),
        how="left",
    )

    if out[sd_clim_col].isna().any():
        n_missing = out[sd_clim_col].isna().sum()
        raise ValueError(
            f"Missing {sd_clim_col} for {n_missing} rows after merging hetGP predictions."
        )

    # Dynamic graph CP sigma from rowwise conformal radius.
    out[f"{prefix}_sigma_cp"] = out[q_col] / Z90

    var_cp = out[f"{prefix}_sigma_cp"].to_numpy() ** 2
    var_clim = out[sd_clim_col].to_numpy() ** 2

    if mixing_type == "convex":
        var_final = (1.0 - rho) * var_cp + rho * var_clim
    elif mixing_type == "inflation_only":
        var_final = var_cp + rho * np.maximum(var_clim - var_cp, 0.0)
    else:
        raise ValueError("mixing_type must be 'convex' or 'inflation_only'.")

    out[f"{prefix}_rho"] = rho
    out[f"{prefix}_sd"] = np.sqrt(np.maximum(var_final, 1e-12))

    z = norm.ppf(1 - alpha / 2)

    out[f"{prefix}_lower"] = out[y_pred_col] - z * out[f"{prefix}_sd"]
    out[f"{prefix}_upper"] = out[y_pred_col] + z * out[f"{prefix}_sd"]
    out[f"{prefix}_width"] = out[f"{prefix}_upper"] - out[f"{prefix}_lower"]

    out[f"{prefix}_covered"] = (
        (out[y_true_col] >= out[f"{prefix}_lower"])
        & (out[y_true_col] <= out[f"{prefix}_upper"])
    )

    out = add_interval_score_from_cols(
        out,
        alpha=alpha,
        y_true_col=y_true_col,
        lower_col=f"{prefix}_lower",
        upper_col=f"{prefix}_upper",
        score_col=f"{prefix}_interval_score",
    )

    final_lower_col = f"{prefix}_lower"
    final_upper_col = f"{prefix}_upper"
    final_covered_col = f"{prefix}_covered"
    final_width_col = f"{prefix}_width"
    final_score_col = f"{prefix}_interval_score"
    
    df_final = out.copy()
    df_final["lower"] = df_final[final_lower_col]
    df_final["upper"] = df_final[final_upper_col]
    df_final["covered"] = df_final[final_covered_col]
    df_final["width"] = df_final[final_width_col] 
    df_final["interval_score"] = df_final[f"{prefix}_interval_score"]

    return df_final

def add_clim_variance_mix_to_dynamic_result(
    dynamic_result,
    *,
    rho_by_horizon,
    alpha,
    y_true_col,
    y_pred_col,
    sd_clim_col="hetgp_predSD",
    prefix="dynamic_varmix",
    mixing_type="convex",
):
    """
    Adds variance-mixed climatology intervals to each horizon in a dynamic graph
    ConformalResult.

    rho_by_horizon can be:
      - dict like {1: 0.2, 2: 0.4, 3: 0.5, 4: 0.6}
      - scalar float, used for all horizons
    """
    hetgp_pred_df = run_hetGP_fitting(
        train_years=list((x for x in range(1980, 2017))),
        test_years=list((2017, 2018, 2019, 2020)), # test years doesn't matter here--we only train on 1980-2016
    )
    
    new_predictions_by_horizon = {}
    new_metadata_by_horizon = {}

    for h, dfh in dynamic_result.predictions_by_horizon.items():

        if isinstance(rho_by_horizon, dict):
            rho = rho_by_horizon[h]
        else:
            rho = float(rho_by_horizon)

        dfh_varmix = add_clim_variance_mix_to_df(
            dfh,
            hetgp_pred_df=hetgp_pred_df,
            rho=rho,
            alpha=alpha,
            y_true_col=y_true_col,
            y_pred_col=y_pred_col,
            sd_clim_col=sd_clim_col,
            q_col="q_hat",
            prefix=prefix,
            mixing_type=mixing_type,
        )

        new_predictions_by_horizon[h] = dfh_varmix

        meta = dict(dynamic_result.metadata_by_horizon.get(h, {}))
        meta.update({
            "variance_mixing": True,
            "variance_mixing_type": mixing_type,
            "rho": rho,
            "sd_clim_col": sd_clim_col,
        })
        new_metadata_by_horizon[h] = meta

        print(f"\n--- Dynamic Graph + Climatology Variance Mix | Horizon {h} ---")
        print(f"rho = {rho:.3f}")
        print(f"mixing_type = {mixing_type}")
        print(f"Coverage: {dfh_varmix[f'{prefix}_covered'].mean():.3f}")
        print(f"Median width: {dfh_varmix[f'{prefix}_width'].median():.3f}")
        print(f"Mean width: {dfh_varmix[f'{prefix}_width'].mean():.3f}")
        print(f"Mean interval score: {dfh_varmix[f'{prefix}_interval_score'].mean():.3f}")

    return ConformalResult(
        method=f"{dynamic_result.method}_{prefix}_{mixing_type}",
        predictions_by_horizon=new_predictions_by_horizon,
        metadata_by_horizon=new_metadata_by_horizon,
        config={
            **dynamic_result.config,
            "variance_mixing": True,
            "variance_mixing_type": mixing_type,
            "prefix": prefix,
        },
    )

# ============================================================
# MAIN METHOD
# ============================================================
def dynamic_graph_conformal(cal_df,
                            test_df, 
                            y_true_col, 
                            y_pred_col, 
                            alpha, 
                            forecast_date_col,
                            current_week_col,
                            T_window=20, 
                            tau_weeks=4.0,
                            graph_method="corr", 
                            week_kernel="exp"):
    """
    Dynamic graph weighted conformal:
      w_j(x*) ∝ g_t(r_j, r*) * k(w_j, w*)
    """

    cal = parse_year_week_from_yyyww(cal_df)
    test = parse_year_week_from_yyyww(test_df)

    cal = add_season(cal)
    test = add_season(test)

    cal["score"] = (cal[y_true_col] - cal[y_pred_col]).abs()

    # build region panel from ALL available observations up through test period
    # for a real deployment, this should only contain information available up to each time.
    all_df = pd.concat(
        [
            cal[[forecast_date_col, "Region", current_week_col]],
            test[[forecast_date_col, "Region", current_week_col]],
        ],
        ignore_index=True
    ).drop_duplicates(subset=[forecast_date_col, "Region"])

    region_panel = build_region_panel(all_df, value_col=current_week_col)

    # calibration arrays
    cal_scores = cal["score"].to_numpy()
    cal_weeks = cal["week_num"].to_numpy()
    cal_regions = cal["Region"].to_numpy()

    qhat = np.empty(len(test), dtype=float)
    neff = np.empty(len(test), dtype=float)

    # precompute static fallback global quantile
    global_q = float(np.quantile(cal_scores, 1 - alpha))

    # cache graph by test date since many rows share same date
    graph_cache = {}

    for i, row in enumerate(test.itertuples(index=False)):
        test_date = int(getattr(row, forecast_date_col))
        r_star = getattr(row, "Region")
        w_star = int(getattr(row, "week_num"))

        # dynamic graph for this test date
        if test_date not in graph_cache:
            G_t = compute_dynamic_similarity_matrix(
                region_panel,
                current_date=test_date,
                T_window=T_window,
                method=graph_method
            )
            graph_cache[test_date] = G_t
        else:
            G_t = graph_cache[test_date]

        # fallback if insufficient history
        if G_t is None:
            qhat[i] = global_q
            neff[i] = np.nan
            continue

        # region similarity for each calibration point
        # similarity from test region r_star to cal region r_j
        # because G_t is row-normalized, use row r_star
        if r_star not in G_t.index:
            qhat[i] = global_q
            neff[i] = np.nan
            continue
            
        # Long vec of corrs of all rj w/ r_star
        g_vec = np.array([G_t.loc[r_star, rj] if rj in G_t.columns else 0.0 for rj in cal_regions])

        # week similarity
        # d_week is long vec of corrs between weeks in cal and test point week
        d_week = circ_week_dist(cal_weeks, w_star, period=52)
        k_vec = kernel_fn(d_week, tau=tau_weeks, kind=week_kernel)

        # combined weights
        w = g_vec * k_vec
        s = w.sum()

        if s <= 0:
            qhat[i] = global_q
            neff[i] = np.nan
        else:
            qhat[i] = weighted_quantile(cal_scores, w / s, 1 - alpha)
            neff[i] = effective_n(w)

    out = test.copy()
    out["q_hat"] = qhat
    out["n_eff"] = neff

    df_pi = apply_symmetric_intervals(
            out,
            q=out["q_hat"].values,
            y_true_col=y_true_col,
            y_pred_col=y_pred_col,
            q_col_name="q_hat",
    )

    return df_pi

    
def run_dynamic_graph_cp(cal_files,
                         test_files,
                         week_kernel,
                         tau_weeks,
                         y_true_col,
                         y_pred_col, 
                         alpha, 
                         forecast_date_col,
                         current_week_col,
                         T_window, 
                         graph_method,
                         y_trend_col="trend_pred_K",
                        ):

    pred_df_val = run_hetGP_fitting(
        train_years=list(x for x in range(1980, 2017)),
        test_years=list(val_test_years),
    )
    
    predictions_by_horizon = {}
    metadata_by_horizon = {}
    
    for h in HORIZONS:
        cal_df = pd.read_csv(cal_files[h])
        test_df = pd.read_csv(test_files[h])
    
        out = dynamic_graph_conformal(
            cal_df=cal_df,
            test_df=test_df,
            y_true_col=y_true_col,
            y_pred_col=y_pred_col,
            alpha=alpha,
            forecast_date_col=forecast_date_col,
            current_week_col=current_week_col,
            T_window=T_window,
            tau_weeks=tau_weeks,
            graph_method="corr",   # try "cosine" too
            week_kernel=week_kernel
        )

        overall_cov = float(out["covered"].mean())
        median_width = float(out["width"].median())
        mean_width = float(out["width"].mean())
        median_neff = float(np.nanmedian(out["n_eff"]))
    
        print(f"\n--- Dynamic Graph Conformal | Horizon {h} ---")
        print(f"Overall coverage: {overall_cov:.3f}")
        print(f"Median width: {median_width:.3f}")
        print(f"Mean width: {mean_width:.3f}")
        print(f"Median n_eff: {median_neff:.1f}")
    
        print("\nCoverage by Region:")
        print(out.groupby("Region")["covered"].mean().round(3))
    
        print("\nCoverage by Region x Season:")
        print(
            out.groupby(["Region", "Season"])["covered"]
            .mean()
            .unstack("Season")
            .round(3)
        )

        df_pi = add_interval_score(df=out, alpha=alpha, y_true_col=y_true_col)
        df_pi = calculate_detrended_intervals(df=df_pi, y_true_col=y_true_col, y_pred_col=y_pred_col, y_trend_col=y_trend_col)
        
        validate_prediction_df(df_pi)

        predictions_by_horizon[h] = df_pi
        metadata_by_horizon[h] = {"kernel_kind": week_kernel, 
                                  "tau_weeks": tau_weeks,
                                  "T_window": T_window
                                 }
    
    return ConformalResult(
        method="dynamic_graph",
        predictions_by_horizon=predictions_by_horizon,
        metadata_by_horizon=metadata_by_horizon,
        config={
            "alpha": alpha,
            "y_true_col": y_true_col,
            "y_pred_col": y_pred_col,
            "date_col": forecast_date_col,
        },
    )
