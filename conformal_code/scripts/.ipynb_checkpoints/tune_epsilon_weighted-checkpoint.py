import numpy as np
import pandas as pd

alpha = 0.10
tau_weeks = 4.0
kernel_kind = "exp"  # "exp" or "gauss"
eps_grid = [0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50]

region_prefix = "Region_"
y_true_col = "y_true_kelvin"
y_pred_col = "y_pred_kelvin"
forecast_date_col = "forecast_date"  # int YYYYWW


# ---------- helpers ----------
def parse_year_week_from_yyyww(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    fd = out[forecast_date_col].astype(int)
    out["year"] = (fd // 100).astype(int)
    out["week_num"] = (fd % 100).astype(int).clip(1, 53)
    return out

def add_region_from_onehot(df: pd.DataFrame, prefix="Region_") -> pd.DataFrame:
    region_cols = [c for c in df.columns if c.startswith(prefix)]
    if not region_cols:
        raise ValueError(f"No region one-hot columns with prefix '{prefix}'.")
    rs = df[region_cols].sum(axis=1).astype(float)
    if not np.allclose(rs.values, 1.0):
        bad = (~np.isclose(rs.values, 1.0)).sum()
        raise ValueError(f"One-hot region rows do not sum to 1 for {bad} rows.")
    out = df.copy()
    out["Region"] = out[region_cols].idxmax(axis=1).str.replace(prefix, "", regex=False)
    return out
    
def add_season(df):
    """
    Meteorological-ish seasons using week-of-year.
    """
    # Weeks: Winter ~ (49-52, 1-9), Spring ~ 10-22, Summer ~ 23-35, Fall ~ 36-48
    #bins = [-1, 9, 22, 36, 48, 53]
    bins = [-1, 10, 22, 38, 48, 53]
    labels = ['Winter', 'Spring', 'Summer', 'Fall', 'Winter']
    df['week_num'] = df['forecast_date'] % 100    
    
    # ordered=False allows us to have two 'winter' labels (start and end of year)
    df['Season'] = pd.cut(df['week_num'], bins=bins, labels=labels, ordered=False)
    return df
    
def add_season_from_weeknum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses your cutoffs:
      Winter: week 0-10 and 49-53
      Spring: 11-22
      Summer: 23-38
      Fall:   39-48
    """
    out = df.copy()
    bins = [-1, 10, 22, 38, 48, 53]
    labels = ["Winter", "Spring", "Summer", "Fall", "Winter"]
    out["Season"] = pd.cut(out["week_num"], bins=bins, labels=labels, ordered=False)
    out["Season"] = out["Season"].astype(str)  # turn category into strings for grouping
    return out

def Pi_shrinkage(regions, eps: float) -> np.ndarray:
    R = len(regions)
    I = np.eye(R)
    U = np.ones((R, R)) / R
    return (1 - eps) * I + eps * U  # columns sum to 1

def circ_week_dist(w1, w2, period=52):
    d = np.abs(w1 - w2)
    return np.minimum(d, period - d)

def kernel_fn(d, tau, kind="exp"):
    if kind == "exp":
        return np.exp(-d / tau)
    elif kind == "gauss":
        return np.exp(-(d / tau) ** 2)
    else:
        raise ValueError("kernel_kind must be 'exp' or 'gauss'")

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

def n_eff(weights):
    w = np.asarray(weights, dtype=float)
    s = w.sum()
    if s <= 0:
        return np.nan
    w = w / s
    return float(1.0 / np.sum(w**2))

def run_kernel_regionmix_conformal(cal_df, eval_df, eps, alpha, tau_weeks, kernel_kind):

    cal = parse_year_week_from_yyyww(cal_df)
    ev  = parse_year_week_from_yyyww(eval_df)

    cal = add_season(cal)
    ev  = add_season(ev)

    cal["score"] = (cal[y_true_col] - cal[y_pred_col]).abs()

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
    out["lower"] = out[y_pred_col] - out["q_hat"]
    out["upper"] = out[y_pred_col] + out["q_hat"]
    out["covered"] = (out[y_true_col] >= out["lower"]) & (out[y_true_col] <= out["upper"])
    out["width"] = out["upper"] - out["lower"]
    return out

# objective (absolute residual of nominal and target coverage)
def coverage_objective_region_season(df_eval, alpha):
    target = 1 - alpha
    tab = df_eval.groupby(["Region", "Season"], observed=False)["covered"].mean()
    return float((tab - target).abs().mean())

def main():
    # Filepaths
    cal_files = {
        1: "/home/mfholth/subseasonal/weekly_data/conformal_UQ/Region_model_results/Calibration_long_1.csv",
        2: "/home/mfholth/subseasonal/weekly_data/conformal_UQ/Region_model_results/Calibration_long_2.csv",
        3: "/home/mfholth/subseasonal/weekly_data/conformal_UQ/Region_model_results/Calibration_long_3.csv",
        4: "/home/mfholth/subseasonal/weekly_data/conformal_UQ/Region_model_results/Calibration_long_4.csv",
    }
    test_files = {
        1: "/home/mfholth/subseasonal/weekly_data/conformal_UQ/Region_model_results/Test_long_1.csv",
        2: "/home/mfholth/subseasonal/weekly_data/conformal_UQ/Region_model_results/Test_long_2.csv",
        3: "/home/mfholth/subseasonal/weekly_data/conformal_UQ/Region_model_results/Test_long_3.csv",
        4: "/home/mfholth/subseasonal/weekly_data/conformal_UQ/Region_model_results/Test_long_4.csv",
    }
    eps_grid = [0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50]
    best_eps = {}
    
    for h in [1,2,3,4]:
        cal_all = pd.read_csv(cal_files[h])
        cal_all = parse_year_week_from_yyyww(cal_all)
    
        cal_train = cal_all[cal_all["year"].isin([2017, 2018, 2019])].copy()
        cal_val   = cal_all[cal_all["year"].isin([2020])].copy()
    
        rows = []
        for eps in eps_grid:
            val_out = run_kernel_regionmix_conformal(
                cal_df=cal_train,
                eval_df=cal_val,
                eps=eps,
                alpha=alpha,
                tau_weeks=tau_weeks,
                kernel_kind=kernel_kind,
            )
            obj = coverage_objective_region_season(val_out, alpha=alpha)
            cov = float(val_out["covered"].mean())
            medw = float(val_out["width"].median())
            med_neff = float(np.nanmedian(val_out["n_eff"]))
            rows.append((eps, obj, cov, medw, med_neff))
    
        rows.sort(key=lambda t: t[1])
        best = rows[0]
        best_eps[h] = best[0]
    
        print(f"\nH{h} tuning (tau={tau_weeks}, kernel={kernel_kind}):")
        for eps, obj, cov, medw, med_neff in rows:
            print(f"  eps={eps:>4} | obj={obj:.4f} | cov={cov:.3f} | medW={medw:.3f} | med n_eff={med_neff:.1f}")
        print(f"Best eps: {best[0]}")

    print("running full method on test data...")
    for h in [1,2,3,4]:
        cal_all = pd.read_csv(cal_files[h])
        test = pd.read_csv(test_files[h])
    
        out_test = run_kernel_regionmix_conformal(
            cal_df=cal_all,      # full calibration period
            eval_df=test,
            eps=best_eps[h],
            alpha=alpha,
            tau_weeks=tau_weeks,
            kernel_kind=kernel_kind,
        )
    
        overall = float(out_test["covered"].mean())
        medw = float(out_test["width"].median())
        med_neff = float(np.nanmedian(out_test["n_eff"]))
    
        print(f"\nH{h} TEST | tuned eps={best_eps[h]} | cov={overall:.3f} | medW={medw:.3f} | med n_eff={med_neff:.1f}")
    
        # Optional: inspect Region×Season coverage on test
        # tab = out_test.groupby(["Region","Season"])["covered"].mean().unstack("Season")
        tab = out_test.groupby("Region")["width"].mean()
        print(tab.round(3))

if __name__ == "__main__":
    main()