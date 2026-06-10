import numpy as np
import pandas as pd
from .method_helpers import add_season, weighted_quantile, circ_week_dist, Pi_shrinkage, kernel_fn, parse_year_week_from_yyyww, n_eff
from conformal.features import Y_PRED_COL, Y_TRUE_COL, ALPHA, FORECAST_DATE_COL, CURRENT_WEEK_COL
from conformal.config import HORIZONS
from conformal.utils import apply_symmetric_intervals
from conformal.metrics import add_interval_score
from conformal.base import ConformalResult, validate_prediction_df
from conformal.methods.method_helpers import calculate_detrended_intervals

def build_region_panel(df, value_col="current_week"):
    """
    Build wide matrix:
      rows   = forecast_date (YYYYWW)
      cols   = Region
      values = current_week anomaly
    Assumes one row per region per date.
    """
    wide = (
        df.pivot_table(
            index=FORECAST_DATE_COL,
            columns="Region",
            values=value_col,
            aggfunc="first"
        )
        .sort_index()
    )
    return wide

    
def effective_n(weights):
    w = np.asarray(weights, dtype=float)
    s = w.sum()
    if s <= 0:
        return np.nan
    w = w / s
    return float(1.0 / np.sum(w ** 2))

    
def compute_dynamic_similarity_matrix(panel_wide, 
                                      current_date,
                                      T_window,
                                      method="corr"):
    """
    panel_wide: wide matrix indexed by forecast_date, columns=Region
    current_date: integer YYYYWW for the test point
    T_window: number of past rows to use, excluding current_date
    method: 'corr' or 'cosine'

    Returns:
      G: DataFrame of region-region similarities, row-normalized
    """
    if current_date not in panel_wide.index:
        raise ValueError(f"current_date {current_date} not found in region panel")

    idx = panel_wide.index.get_loc(current_date)

    # use only past rows, never current or future
    start = max(0, idx - T_window)
    window = panel_wide.iloc[start:idx].copy()

    # if too few rows, fallback later
    if len(window) < max(4, T_window // 3):
        return None

    # fill missing values with column means inside window
    window = window.apply(lambda col: col.fillna(col.mean()), axis=0)

    regions = list(window.columns)

    if method == "corr":
        G = window.corr()
    elif method == "cosine":
        X = window.to_numpy().T  # shape: regions x time
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Xn = X / norms
        G = pd.DataFrame(Xn @ Xn.T, index=regions, columns=regions)
    else:
        raise ValueError("method must be 'corr' or 'cosine'")

    # clip negatives to zero
    G = G.clip(lower=0.0)

    # add tiny diagonal stabilization
    for r in G.index:
        G.loc[r, r] = max(G.loc[r, r], 1e-8)

    # row normalize so each row sums to 1
    row_sums = G.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    G = G.div(row_sums, axis=0)

    return G

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
