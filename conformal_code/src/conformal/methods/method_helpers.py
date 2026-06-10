import numpy as np
import pandas as pd

from conformal.features import Y_TRUE_COL, Y_PRED_COL, FORECAST_DATE_COL

def calculate_detrended_intervals(df, y_true_col, y_pred_col, y_trend_col):
    df["y_true_detrended"] = df[y_true_col] - df[y_trend_col]
    df["y_pred_detrended"] = df[y_pred_col] - df[y_trend_col]
    df["lower_detrended"]  = df["lower"] - df[y_trend_col]
    df["upper_detrended"]  = df["upper"] - df[y_trend_col]
    return df

def add_season(df):
    """
    Meteorological-ish seasons using week-of-year.
    """
    # Weeks: Winter ~ (49-52, 1-9), Spring ~ 10-22, Summer ~ 23-35, Fall ~ 36-48
    #bins = [-1, 9, 22, 36, 48, 53]
    bins = [-1, 10, 22, 38, 48, 53]
    labels = ['Winter', 'Spring', 'Summer', 'Fall', 'Winter']
    df = df.copy()
    df['week_num'] = df[FORECAST_DATE_COL] % 100    
    
    # ordered=False allows us to have two 'winter' labels (start and end of year)
    df['Season'] = pd.cut(df['week_num'], bins=bins, labels=labels, ordered=False)
    return df

def weighted_quantile(values, weights, q):
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    m = np.isfinite(v) & np.isfinite(w) & (w >= 0)
    v = v[m]; w = w[m]
    if len(v) == 0 or w.sum() <= 0:
        return np.nan
    idx = np.argsort(v)
    v = v[idx]; w = w[idx]
    cw = np.cumsum(w)
    cw = cw / cw[-1]
    j = np.searchsorted(cw, q, side="left")
    return float(v[min(j, len(v) - 1)])

def circ_week_dist(w1, w2, period=52):
    d = np.abs(w1 - w2)
    return np.minimum(d, period - d)

def Pi_shrinkage(regions, eps: float) -> np.ndarray:
    R = len(regions)
    I = np.eye(R)
    U = np.ones((R, R)) / R
    return (1 - eps) * I + eps * U  # columns sum to 1

def kernel_fn(d, tau, kind="exp"):
    if kind == "exp":
        return np.exp(-d / tau)
    elif kind == "gauss":
        return np.exp(-(d / tau) ** 2)
    else:
        raise ValueError("kernel_kind must be 'exp' or 'gauss'")


def parse_year_week_from_yyyww(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    fd = out[FORECAST_DATE_COL].astype(int)
    out["year"] = (fd // 100).astype(int)
    out["week_num"] = (fd % 100).astype(int).clip(1, 53)
    return out

def n_eff(weights):
    w = np.asarray(weights, dtype=float)
    s = w.sum()
    if s <= 0:
        return np.nan
    w = w / s
    return float(1.0 / np.sum(w**2))
