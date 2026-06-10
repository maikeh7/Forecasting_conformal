# conformal/weighted_cp.py
import numpy as np
import pandas as pd

from conformal.base import ConformalResult, validate_prediction_df
from conformal.methods.method_helpers import calculate_detrended_intervals

from conformal.utils import (
    add_week_num,
    compute_scores,
    apply_symmetric_intervals,
    weighted_quantile,
    circ_week_dist,
    kernel_fn,
    make_region_mixing_matrix,
)
from conformal.metrics import add_interval_score


def run_weighted_region_conformal(
    cal_files: dict[int, str],
    test_files: dict[int, str],
    *,
    alpha: float,
    y_true_col: str,
    y_pred_col: str,
    date_col: str = "forecast_date",
    region_col: str = "Region",
    tau_weeks: float = 4.0,
    kernel: str = "exp",
    region_offdiag: float = 0.10,
    y_trend_col="trend_pred_K",
) -> ConformalResult:
    predictions_by_horizon = {}
    metadata_by_horizon = {}

    for h in sorted(cal_files):
        cal = pd.read_csv(cal_files[h])
        test = pd.read_csv(test_files[h])

        cal = add_week_num(cal, date_col=date_col)
        test = add_week_num(test, date_col=date_col)
        cal = compute_scores(cal, y_true_col, y_pred_col)

        regions = sorted(cal[region_col].unique())
        r_to_idx = {r: i for i, r in enumerate(regions)}

        Pi = make_region_mixing_matrix(regions, offdiag=region_offdiag)

        weeks = np.arange(1, 53)
        D = circ_week_dist(weeks[:, None], weeks[None, :], period=52)
        K = kernel_fn(D, tau=tau_weeks, kernel=kernel)

        cal_region_idx = cal[region_col].map(r_to_idx).values
        cal_week_idx = cal["week_num"].values.astype(int) - 1
        cal_scores = cal["score"].values

        q_hats = []
        n_effs = []

        for _, row in test.iterrows():
            test_region = row[region_col]
            test_week = int(row["week_num"]) - 1
            test_region_idx = r_to_idx[test_region]

            region_weights = Pi[test_region_idx, cal_region_idx]
            week_weights = K[test_week, cal_week_idx]
            weights = region_weights * week_weights

            q_hat = weighted_quantile(cal_scores, 1 - alpha, sample_weight=weights)
            q_hats.append(q_hat)

            sw = weights.sum()
            n_eff = (sw ** 2) / np.sum(weights ** 2) if np.sum(weights ** 2) > 0 else np.nan
            n_effs.append(n_eff)

        test2 = test.copy()
        test2["q_hat"] = q_hats
        test2["n_eff"] = n_effs

        df_pi = apply_symmetric_intervals(
            test2,
            q=test2["q_hat"].values,
            y_true_col=y_true_col,
            y_pred_col=y_pred_col,
            q_col_name="q_hat",
        )

        df_pi = add_interval_score(df=df_pi, alpha=alpha, y_true_col=y_true_col)
        df_pi = calculate_detrended_intervals(df=df_pi, y_true_col=y_true_col, y_pred_col=y_pred_col, y_trend_col=y_trend_col)
        
        validate_prediction_df(df_pi)
        predictions_by_horizon[h] = df_pi
        metadata_by_horizon[h] = {
            "regions": regions,
            "region_mixing_matrix": Pi,
            "tau_weeks": tau_weeks,
            "kernel": kernel,
            "region_offdiag": region_offdiag,
        }

    return ConformalResult(
        method="weighted_region_week",
        predictions_by_horizon=predictions_by_horizon,
        metadata_by_horizon=metadata_by_horizon,
        config={
            "alpha": alpha,
            "y_true_col": y_true_col,
            "y_pred_col": y_pred_col,
            "date_col": date_col,
            "region_col": region_col,
            "tau_weeks": tau_weeks,
            "kernel": kernel,
            "region_offdiag": region_offdiag,
        },
    )
