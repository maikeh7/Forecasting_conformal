# conformal/global_cp.py
import numpy as np
import pandas as pd
from pathlib import Path

#PROJECT_ROOT = Path(__file__).resolve().parents[2]
#sys.path.append(str(PROJECT_ROOT / "src"))

from conformal.base import ConformalResult, validate_prediction_df
from conformal.methods.method_helpers import calculate_detrended_intervals
from conformal.utils import (
    add_week_num,
    add_season,
    apply_symmetric_intervals,
)
from conformal.metrics import add_interval_score


def conformal_q(df_cal, alpha, y_true_col, y_pred_col):
    scores = (df_cal[y_true_col] - df_cal[y_pred_col]).abs().dropna().values
    return float(np.quantile(scores, 1 - alpha))


def run_global_conformal(
    cal_files: dict[int, str],
    test_files: dict[int, str],
    *,
    alpha: float,
    y_true_col: str,
    y_pred_col: str,
    date_col: str = "forecast_date",
    y_trend_col="trend_pred_K"
) -> ConformalResult:
    predictions_by_horizon = {}
    metadata_by_horizon = {}

    for h in sorted(cal_files):
        df_cal = pd.read_csv(cal_files[h])
        df_test = pd.read_csv(test_files[h])

        qh = conformal_q(df_cal, alpha, y_true_col, y_pred_col)

        # add metadata columns needed for downstream summaries/plots
        df_test = add_week_num(df_test, date_col=date_col)
        df_test = add_season(df_test)

        df_pi = apply_symmetric_intervals(df_test, qh, y_true_col, y_pred_col)
        df_pi = add_interval_score(df=df_pi, alpha=alpha, y_true_col=y_true_col)
        df_pi = calculate_detrended_intervals(df=df_pi, y_true_col=y_true_col, y_pred_col=y_pred_col, y_trend_col=y_trend_col)

        validate_prediction_df(df_pi)
        predictions_by_horizon[h] = df_pi
        metadata_by_horizon[h] = {"q_scalar": qh}

    return ConformalResult(
        method="global",
        predictions_by_horizon=predictions_by_horizon,
        metadata_by_horizon=metadata_by_horizon,
        config={
            "alpha": alpha,
            "y_true_col": y_true_col,
            "y_pred_col": y_pred_col,
            "date_col": date_col,
        },
    )