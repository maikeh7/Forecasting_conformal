# conformal/utils.py
import numpy as np
import pandas as pd
from pathlib import Path
from .features import FORECAST_DATE_COL
#from .config import BASE_DIR


def parse_year_week_from_yyyww(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    fd = out[FORECAST_DATE_COL].astype(int)
    out["year"] = (fd // 100).astype(int)
    out["week_num"] = (fd % 100).astype(int).clip(1, 53)
    return out

def add_week_num(df: pd.DataFrame, date_col: str = "forecast_date") -> pd.DataFrame:
    out = df.copy()
    out["week_num"] = out[date_col] % 100
    return out


def add_season(df: pd.DataFrame, week_col: str = "week_num") -> pd.DataFrame:
    out = df.copy()
    bins = [-1, 10, 22, 38, 48, 53]
    labels = ["Winter", "Spring", "Summer", "Fall", "Winter"]
    out["Season"] = pd.cut(
        out[week_col],
        bins=bins,
        labels=labels,
        include_lowest=True,
        ordered=False,
    )
    return out


def compute_scores(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    score_col: str = "score",
) -> pd.DataFrame:
    out = df.copy()
    out[score_col] = (out[y_true_col] - out[y_pred_col]).abs()
    return out


def apply_symmetric_intervals(
    df: pd.DataFrame,
    q,
    y_true_col: str,
    y_pred_col: str,
    q_col_name: str = "q_hat",
) -> pd.DataFrame:
    out = df.copy()

    if np.isscalar(q):
        out[q_col_name] = float(q)
    else:
        out[q_col_name] = q

    out["lower"] = out[y_pred_col] - out[q_col_name]
    out["upper"] = out[y_pred_col] + out[q_col_name]
    out["covered"] = (out[y_true_col] >= out["lower"]) & (out[y_true_col] <= out["upper"])
    out["width"] = out["upper"] - out["lower"]

    # normalize names used by downstream code
    out["y_true"] = out[y_true_col]
    out["y_pred"] = out[y_pred_col]


    return out

def thin(df: pd.DataFrame, n: None) -> pd.DataFrame:
    if n is None or len(df) <= n:
        return df
    idx = np.linspace(0, len(df) - 1, n).astype(int)
    return df.iloc[idx]



def weighted_quantile(values, quantile, sample_weight=None):
    values = np.asarray(values, dtype=float)
    if sample_weight is None:
        sample_weight = np.ones(len(values), dtype=float)
    else:
        sample_weight = np.asarray(sample_weight, dtype=float)

    mask = np.isfinite(values) & np.isfinite(sample_weight) & (sample_weight > 0)
    values = values[mask]
    sample_weight = sample_weight[mask]

    if len(values) == 0:
        return np.nan

    sorter = np.argsort(values)
    values = values[sorter]
    sample_weight = sample_weight[sorter]

    cum_w = np.cumsum(sample_weight)
    cutoff = quantile * cum_w[-1]
    return values[np.searchsorted(cum_w, cutoff, side="left")]


def circ_week_dist(a, b, period=52):
    d = np.abs(a - b)
    return np.minimum(d, period - d)


def kernel_fn(dist, tau=4.0, kernel="exp"):
    if kernel == "exp":
        return np.exp(-dist / tau)
    elif kernel == "gauss":
        return np.exp(-(dist ** 2) / (2 * tau ** 2))
    raise ValueError(f"Unknown kernel: {kernel}")


def make_region_mixing_matrix(regions, offdiag=0.10):
    n = len(regions)
    if n == 1:
        return np.ones((1, 1))
    mat = np.full((n, n), offdiag / (n - 1))
    np.fill_diagonal(mat, 1.0 - offdiag)
    return mat