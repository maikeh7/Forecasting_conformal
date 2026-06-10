# conformal/metrics.py
import pandas as pd
import numpy as np
from .base import ConformalResult
from scipy.stats import norm
from scipy.optimize import brentq

Z90 = 1.6448536269514722  # central 90% normal interval


def weighted_rmse_step(y_true, y_pred, z_obs):
    z_abs = np.abs(z_obs)
    weights = np.where(z_abs < 1, 1.0,
               np.where(z_abs < 2, 2.0, 3.0))
    return np.sqrt(np.sum(weights * (y_true - y_pred) ** 2) / np.sum(weights))


def weighted_rmse_smooth(y_true, y_pred, z_obs, alpha=1.0):
    z_abs = np.abs(z_obs)
    weights = 1.0 + alpha * np.maximum(0.0, z_abs - 1.0)
    return np.sqrt(np.sum(weights * (y_true - y_pred) ** 2) / np.sum(weights))



def interval_score_from_cols(
    df,
    *,
    alpha,
    y_true_col,
    lower_col,
    upper_col,
):
    y = df[y_true_col]
    l = df[lower_col]
    u = df[upper_col]

    return (
        (u - l)
        + (2.0 / alpha) * (l - y).clip(lower=0)
        + (2.0 / alpha) * (y - u).clip(lower=0)
    )


def add_interval_score(
    df: pd.DataFrame,
    alpha: float,
    y_true_col: str = "y_true",
    lower_col: str = "lower",
    upper_col: str = "upper",
    score_col: str = "interval_score",
) -> pd.DataFrame:
    out = df.copy()

    y = out[y_true_col].to_numpy()
    l = out[lower_col].to_numpy()
    u = out[upper_col].to_numpy()

    width = u - l
    below = np.maximum(l - y, 0.0)
    above = np.maximum(y - u, 0.0)

    out[score_col] = width + (2.0 / alpha) * below + (2.0 / alpha) * above
    return out


def summarize_predictions(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    covered_col: str = "covered",
    width_col: str = "width",
    dropna: bool = False,
    observed: bool = False,
) -> pd.DataFrame:
    """
    Summarize a stacked prediction dataframe by arbitrary grouping columns.

    Parameters
    ----------
    df : pd.DataFrame
        Prediction dataframe, usually from `stacked_predictions()`.
    group_cols : list[str] | None
        Columns to group by, e.g. ["Region", "Season", "horizon"].
        If None or empty, returns one overall summary row.
    alpha : float
        Miscoverage level used for interval score.
    y_true_col, lower_col, upper_col, covered_col, width_col : str
        Column names used in metric computation.
    dropna : bool
        Passed to pandas groupby.
    observed : bool
        Passed to pandas groupby, useful for categorical columns.

    Returns
    -------
    pd.DataFrame
        Summary dataframe with one row per group.
    """
    #out = add_interval_score(
    #    df,
    #    alpha=alpha,
    #    y_true_col=y_true_col,
    #    lower_col=lower_col,
    #    upper_col=upper_col,
    #)
    out = df.copy()
    Method = out["method"][1]
    if group_cols is None:
        group_cols = []

    missing = [c for c in group_cols if c not in out.columns]
    if missing:
        raise ValueError(f"Grouping columns missing from dataframe: {missing}")

    metric_aggs = {
        "coverage": (covered_col, "mean"),
        "mean_width": (width_col, "mean"),
        "median_width": (width_col, "median"),
        "mean_interval_score": ("interval_score", "mean"),
        "median_interval_score": ("interval_score", "median"),
        "n": (covered_col, "size"),
    }

    if len(group_cols) == 0:
        summary = pd.DataFrame([{
            "coverage": out[covered_col].mean(),
            "mean_width": out[width_col].mean(),
            "median_width": out[width_col].median(),
            "mean_interval_score": out["interval_score"].mean(),
            "median_interval_score": out["interval_score"].median(),
            "n": len(out),
        }])
        return summary

    summary = (
        out.groupby(group_cols, dropna=dropna, observed=observed)
        .agg(**metric_aggs)
        .reset_index()
    )
    summary["method"] = Method

    return summary



def summarize_dynamic_varmix_result(
    dynamic_varmix_result,
    *,
    prefix="dynamic_varmix",
):
    rows = []

    for h, dfh in dynamic_varmix_result.predictions_by_horizon.items():
        grouped = (
            dfh.groupby("Region", observed=False)
            .agg(
                coverage=(f"{prefix}_covered", "mean"),
                mean_width=(f"{prefix}_width", "mean"),
                median_width=(f"{prefix}_width", "median"),
                mean_interval_score=(f"{prefix}_interval_score", "mean"),
                median_interval_score=(f"{prefix}_interval_score", "median"),
                n=(f"{prefix}_covered", "size"),
            )
            .reset_index()
        )

        grouped["horizon"] = h
        grouped["method"] = f"Dynamic graph + {prefix}"
        rows.append(grouped)

    return pd.concat(rows, ignore_index=True)



def compare_results(results: list[ConformalResult]) -> pd.DataFrame:
    frames = [summarize_result(r) for r in results]
    return pd.concat(frames, ignore_index=True)


# objective (absolute residual of nominal and target coverage)
def coverage_objective_region_season(df_eval):
    target = 1 - ALPHA
    tab = df_eval.groupby(["Region", "Season"], observed=False)["covered"].mean()
    return float((tab - target).abs().mean())


def interval_score(df_eval, alpha, y_true_col, lower_col="lower", upper_col="upper"):
    y = df_eval[y_true_col].to_numpy()
    l = df_eval[lower_col].to_numpy()
    u = df_eval[upper_col].to_numpy()

    width = u - l
    below = np.maximum(l - y, 0.0)
    above = np.maximum(y - u, 0.0)

    score = width + (2.0 / alpha) * below + (2.0 / alpha) * above
    return float(np.mean(score))


def add_interval_score(
    df: pd.DataFrame,
    alpha: float,
    y_true_col: str = "y_true",
    lower_col: str = "lower",
    upper_col: str = "upper",
    score_col: str = "interval_score",
) -> pd.DataFrame:
    out = df.copy()

    y = out[y_true_col].to_numpy()
    l = out[lower_col].to_numpy()
    u = out[upper_col].to_numpy()

    width = u - l
    below = np.maximum(l - y, 0.0)
    above = np.maximum(y - u, 0.0)

    out[score_col] = width + (2.0 / alpha) * below + (2.0 / alpha) * above
    return out


def A_gaussian_abs(a, var):
    """
    A(a, var) = E|Z| where Z ~ N(a, var)
    """
    s = np.sqrt(np.maximum(var, 1e-12))
    z = a / s
    return 2 * s * norm.pdf(z) + a * (2 * norm.cdf(z) - 1)


def crps_gaussian_mixture_2(y, mu_cp, sd_cp, mu_clim, sd_clim, lam):
    """
    CRPS for lambda*N(mu_cp, sd_cp^2) + (1-lambda)*N(mu_clim, sd_clim^2).
    Lower is better.
    """
    y = np.asarray(y)
    mu_cp = np.asarray(mu_cp)
    sd_cp = np.asarray(sd_cp)
    mu_clim = np.asarray(mu_clim)
    sd_clim = np.asarray(sd_clim)

    var_cp = sd_cp**2
    var_clim = sd_clim**2

    # First term: E|X - y|
    term1 = (
        lam * A_gaussian_abs(y - mu_cp, var_cp)
        + (1 - lam) * A_gaussian_abs(y - mu_clim, var_clim)
    )

    # Second term: 0.5 E|X - X'|
    term2 = 0.5 * (
        lam**2 * A_gaussian_abs(0.0, 2 * var_cp)
        + 2 * lam * (1 - lam) * A_gaussian_abs(mu_cp - mu_clim, var_cp + var_clim)
        + (1 - lam)**2 * A_gaussian_abs(0.0, 2 * var_clim)
    )

    return term1 - term2


def tune_lambda_crps(df_val, lambda_grid=None,
                     y_true_col="y_true",
                     mu_cp_col="y_pred",
                     sd_cp_col="sigma_cp",
                     mu_clim_col="hetgp_mean",
                     sd_clim_col="hetgp_sd"):
    if lambda_grid is None:
        lambda_grid = np.linspace(0, 1, 101)

    rows = []
    for lam in lambda_grid:
        crps = crps_gaussian_mixture_2(
            y=df_val[y_col],
            mu_cp=df_val[mu_cp_col],
            sd_cp=df_val[sd_cp_col],
            mu_clim=df_val[mu_clim_col],
            sd_clim=df_val[sd_clim_col],
            lam=lam,
        )
        rows.append({
            "lambda": lam,
            "mean_crps": np.nanmean(crps)
        })

    out = pd.DataFrame(rows)
    best = out.loc[out["mean_crps"].idxmin()]
    return float(best["lambda"]), out


def gaussian_mixture_cdf(x, mu_cp, sd_cp, mu_clim, sd_clim, lam):
    return (
        lam * norm.cdf((x - mu_cp) / sd_cp)
        + (1 - lam) * norm.cdf((x - mu_clim) / sd_clim)
    )


def gaussian_mixture_quantile(
    p,
    mu_cp,
    sd_cp,
    mu_clim,
    sd_clim,
    lam,
    bracket_width=12.0,
):
    """
    Invert two-component Gaussian mixture CDF.

    Returns q such that F_mix(q) = p.
    """
    sd_cp = max(float(sd_cp), 1e-8)
    sd_clim = max(float(sd_clim), 1e-8)

    sd_max = max(sd_cp, sd_clim)
    lo = min(mu_cp, mu_clim) - bracket_width * sd_max
    hi = max(mu_cp, mu_clim) + bracket_width * sd_max

    def root_fn(x):
        return gaussian_mixture_cdf(
            x,
            mu_cp=mu_cp,
            sd_cp=sd_cp,
            mu_clim=mu_clim,
            sd_clim=sd_clim,
            lam=lam,
        ) - p

    return brentq(root_fn, lo, hi)


def add_cp_clim_and_mixture_intervals(
    df,
    *,
    alpha,
    y_true_col="y_true_kelvin",
    mu_cp_col="y_pred_kelvin",
    sd_cp_col="sigma_cp",
    mu_clim_col="hetgp_mean_kelvin",
    sd_clim_col="hetgp_predSD",
    best_lambda_global,
    lambda_region_col="lambda_region",
):
    """
    Adds 90% intervals for:
      - CP-implied Gaussian density
      - hetGP climatology density
      - global-lambda Gaussian mixture
      - region-lambda Gaussian mixture

    Mixture intervals are exact central mixture-CDF quantiles.
    """
    out = df.copy()
    z = norm.ppf(1 - alpha / 2)

    # CP density interval
    out["cp_lower"] = out[mu_cp_col] - z * out[sd_cp_col]
    out["cp_upper"] = out[mu_cp_col] + z * out[sd_cp_col]
    out["cp_width"] = out["cp_upper"] - out["cp_lower"]

    # Climatology density interval
    out["clim_lower"] = out[mu_clim_col] - z * out[sd_clim_col]
    out["clim_upper"] = out[mu_clim_col] + z * out[sd_clim_col]
    out["clim_width"] = out["clim_upper"] - out["clim_lower"]

    # Global mixture intervals
    mix_global_lower = []
    mix_global_upper = []

    # Region mixture intervals
    mix_region_lower = []
    mix_region_upper = []

    for row in out.itertuples(index=False):
        mu_cp = getattr(row, mu_cp_col)
        sd_cp = getattr(row, sd_cp_col)
        mu_clim = getattr(row, mu_clim_col)
        sd_clim = getattr(row, sd_clim_col)

        # global lambda
        lg = best_lambda_global

        mix_global_lower.append(
            gaussian_mixture_quantile(
                alpha / 2,
                mu_cp,
                sd_cp,
                mu_clim,
                sd_clim,
                lg,
            )
        )
        mix_global_upper.append(
            gaussian_mixture_quantile(
                1 - alpha / 2,
                mu_cp,
                sd_cp,
                mu_clim,
                sd_clim,
                lg,
            )
        )

        # region lambda
        lr = getattr(row, lambda_region_col)

        mix_region_lower.append(
            gaussian_mixture_quantile(
                alpha / 2,
                mu_cp,
                sd_cp,
                mu_clim,
                sd_clim,
                lr,
            )
        )
        mix_region_upper.append(
            gaussian_mixture_quantile(
                1 - alpha / 2,
                mu_cp,
                sd_cp,
                mu_clim,
                sd_clim,
                lr,
            )
        )

    out["mix_global_lower"] = mix_global_lower
    out["mix_global_upper"] = mix_global_upper
    out["mix_global_width"] = out["mix_global_upper"] - out["mix_global_lower"]

    out["mix_region_lower"] = mix_region_lower
    out["mix_region_upper"] = mix_region_upper
    out["mix_region_width"] = out["mix_region_upper"] - out["mix_region_lower"]

    # coverage columns
    for prefix in ["cp", "clim", "mix_global", "mix_region"]:
        out[f"{prefix}_covered"] = (
            (out[y_true_col] >= out[f"{prefix}_lower"])
            & (out[y_true_col] <= out[f"{prefix}_upper"])
        )

    return out

def summarize_all_global_clim_mix_variants_by_group(
    df,
    *,
    horizon,
    alpha,
    group_cols=("Region",),
    y_true_col="y_true_kelvin",
):
    method_specs = [
        ("CP Gaussian proxy", "cp_lower", "cp_upper", "crps_cp"),
        ("hetGP climatology", "clim_lower", "clim_upper", "crps_clim"),
        ("Linear density mix: global lambda", "mix_global_lower", "mix_global_upper", "crps_mix_global_lambda"),
        ("Linear density mix: region lambda", "mix_region_lower", "mix_region_upper", "crps_mix_region_lambda"),
        ("Variance mix: global rho", "varmix_global_lower", "varmix_global_upper", "varmix_global_crps"),
        ("Variance mix: region rho", "varmix_region_lower", "varmix_region_upper", "varmix_region_crps"),
    ]

    rows = []

    for method, lower_col, upper_col, crps_col in method_specs:
        if lower_col not in df.columns or upper_col not in df.columns:
            continue

        tmp = df.copy()
        tmp["_covered"] = (
            (tmp[y_true_col] >= tmp[lower_col])
            & (tmp[y_true_col] <= tmp[upper_col])
        )
        tmp["_width"] = tmp[upper_col] - tmp[lower_col]
        tmp["_interval_score"] = interval_score_from_cols(
            tmp,
            alpha=alpha,
            y_true_col=y_true_col,
            lower_col=lower_col,
            upper_col=upper_col,
        )

        if crps_col in tmp.columns:
            tmp["_crps"] = tmp[crps_col]
        else:
            tmp["_crps"] = np.nan

        grouped = (
            tmp.groupby(list(group_cols), observed=False)
            .agg(
                coverage=("_covered", "mean"),
                mean_width=("_width", "mean"),
                median_width=("_width", "median"),
                mean_interval_score=("_interval_score", "mean"),
                median_interval_score=("_interval_score", "median"),
                mean_crps=("_crps", "mean"),
                median_crps=("_crps", "median"),
                n=("_covered", "size"),
            )
            .reset_index()
        )

        grouped["horizon"] = horizon
        grouped["method"] = method
        grouped["summary_level"] = "_".join(group_cols)

        rows.append(grouped)

    return pd.concat(rows, ignore_index=True)

def summarize_interval_coverage_width(
    df,
    *,
    horizon,
    y_true_col="y_true_kelvin",
    group_cols=("Region",),
):
    """
    Returns coverage and interval width summaries for each density method.
    """
    method_specs = [
        ("CP density", "cp_lower", "cp_upper", "cp_covered", "cp_width"),
        ("Climatology hetGP", "clim_lower", "clim_upper", "clim_covered", "clim_width"),
        ("Mixture global lambda", "mix_global_lower", "mix_global_upper", "mix_global_covered", "mix_global_width"),
        ("Mixture region lambda", "mix_region_lower", "mix_region_upper", "mix_region_covered", "mix_region_width"),
    
        # New variance-only methods
        ("CP mean + varmix global rho", "varmix_global_lower", "varmix_global_upper", "varmix_global_covered", "varmix_global_width"),
        ("CP mean + varmix region rho", "varmix_region_lower", "varmix_region_upper", "varmix_region_covered", "varmix_region_width"),
    ]

    rows = []

    for method, lower_col, upper_col, covered_col, width_col in method_specs:
        # overall
        rows.append({
            "horizon": horizon,
            "method": method,
            "summary_level": "overall",
            "Region": "ALL",
            "Season": "ALL",
            "coverage": float(df[covered_col].mean()),
            "mean_width": float(df[width_col].mean()),
            "median_width": float(df[width_col].median()),
            "n": int(len(df)),
        })

        # grouped summaries
        grouped = (
            df.groupby(list(group_cols), observed=False)
              .agg(
                  coverage=(covered_col, "mean"),
                  mean_width=(width_col, "mean"),
                  median_width=(width_col, "median"),
                  n=(covered_col, "size"),
              )
              .reset_index()
        )

        grouped["horizon"] = horizon
        grouped["method"] = method
        grouped["summary_level"] = "_".join(group_cols)

        # Make sure these exist for easier concatenation
        if "Region" not in grouped.columns:
            grouped["Region"] = "ALL"
        if "Season" not in grouped.columns:
            grouped["Season"] = "ALL"

        rows.append(grouped)

    return pd.concat(
        [pd.DataFrame([r]) if isinstance(r, dict) else r for r in rows],
        ignore_index=True,
    )


def crps_gaussian(y, mu, sd):
    """
    CRPS for N(mu, sd^2).
    Lower is better.
    """
    var = np.asarray(sd) ** 2
    return (
        A_gaussian_abs(np.asarray(y) - np.asarray(mu), var)
        - 0.5 * A_gaussian_abs(0.0, 2 * var)
    )

def add_variance_mix_density(
    df,
    *,
    rho,
    y_true_col="y_true_kelvin",
    mu_cp_col="y_pred_kelvin",
    sd_cp_col="sigma_cp",
    sd_clim_col="hetgp_predSD",
    prefix="varmix",
    mixing_type = "convex"
):
    """
    Keep CP/RF mean, blend CP and climatology variances.

    sigma_varmix^2 = (1-rho) sigma_cp^2 + rho sigma_clim^2

    rho = 0 -> pure CP variance
    rho = 1 -> pure climatology variance
    """
    out = df.copy()

    #varmix = (1 - rho) * sd_cp**2 + rho * sd_clim**2

    sd_cp = out[sd_cp_col].to_numpy()
    sd_clim = out[sd_clim_col].to_numpy()
    
    var_cp = sd_cp**2
    var_clim = sd_clim**2
    
    # convex or inflation only variance mixing
    if mixing_type == "convex":
        varmix = (1 - rho) * sd_cp**2 + rho * sd_clim**2
    else:
        varmix = var_cp + rho * np.maximum(var_clim - var_cp, 0.0)
    
    out[f"{prefix}_sd"] = np.sqrt(np.maximum(varmix, 1e-12))

    out[f"{prefix}_crps"] = crps_gaussian(
        y=out[y_true_col],
        mu=out[mu_cp_col],
        sd=out[f"{prefix}_sd"],
    )

    return out

def tune_rho_varmix_crps(
    df_val,
    *,
    rho_grid=None,
    y_true_col="y_true_kelvin",
    mu_cp_col="y_pred_kelvin",
    sd_cp_col="sigma_cp",
    sd_clim_col="hetgp_predSD",
):
    if rho_grid is None:
        rho_grid = np.linspace(0, 1, 101)

    rows = []

    for rho in rho_grid:
        tmp = add_variance_mix_density(
            df_val,
            rho=rho,
            y_true_col=y_true_col,
            mu_cp_col=mu_cp_col,
            sd_cp_col=sd_cp_col,
            sd_clim_col=sd_clim_col,
            prefix="varmix",
        )

        rows.append({
            "rho": rho,
            "mean_crps": float(np.nanmean(tmp["varmix_crps"])),
        })

    out = pd.DataFrame(rows)
    best = out.loc[out["mean_crps"].idxmin()]

    return float(best["rho"]), out

def tune_rho_varmix_by_region(
    df_val,
    *,
    rho_grid=None,
    region_col="Region",
    y_true_col="y_true_kelvin",
    mu_cp_col="y_pred_kelvin",
    sd_cp_col="sigma_cp",
    sd_clim_col="hetgp_predSD",
):
    if rho_grid is None:
        rho_grid = np.linspace(0, 1, 101)

    best_rows = []
    all_rows = []

    for region, sub in df_val.groupby(region_col):
        sub = sub.dropna(
            subset=[y_true_col, mu_cp_col, sd_cp_col, sd_clim_col]
        )

        if len(sub) == 0:
            continue

        rows = []

        for rho in rho_grid:
            tmp = add_variance_mix_density(
                sub,
                rho=rho,
                y_true_col=y_true_col,
                mu_cp_col=mu_cp_col,
                sd_cp_col=sd_cp_col,
                sd_clim_col=sd_clim_col,
                prefix="varmix",
            )

            rows.append({
                region_col: region,
                "rho": rho,
                "mean_crps": float(np.nanmean(tmp["varmix_crps"])),
                "n": len(sub),
            })

        region_results = pd.DataFrame(rows)
        all_rows.append(region_results)

        best = region_results.loc[region_results["mean_crps"].idxmin()]
        best_rows.append(best)

    best_df = pd.DataFrame(best_rows).reset_index(drop=True)
    all_df = pd.concat(all_rows, ignore_index=True)

    return best_df, all_df

def add_variance_mix_intervals(
    df,
    *,
    alpha,
    rho_global,
    rho_by_region=None,
    y_true_col="y_true_kelvin",
    mu_cp_col="y_pred_kelvin",
    sd_cp_col="sigma_cp",
    sd_clim_col="hetgp_predSD",
    mixing_type = "convex"
):
    out = df.copy()
    z = norm.ppf(1 - alpha / 2)

    # -------------------------
    # Global rho
    # -------------------------
    if mixing_type == "convex":
        var_global = (
            (1 - rho_global) * out[sd_cp_col].to_numpy()**2
            + rho_global * out[sd_clim_col].to_numpy()**2
        )
    else:
        var_global = var_cp + rho_global * np.maximum(var_clim - var_cp, 0.0)
        
    var_cp = out[sd_cp_col].to_numpy()**2
    var_clim = out[sd_clim_col].to_numpy()**2
    




    out["varmix_global_sd"] = np.sqrt(np.maximum(var_global, 1e-12))
    out["varmix_global_lower"] = out[mu_cp_col] - z * out["varmix_global_sd"]
    out["varmix_global_upper"] = out[mu_cp_col] + z * out["varmix_global_sd"]
    out["varmix_global_width"] = (
        out["varmix_global_upper"] - out["varmix_global_lower"]
    )
    out["varmix_global_covered"] = (
        (out[y_true_col] >= out["varmix_global_lower"])
        & (out[y_true_col] <= out["varmix_global_upper"])
    )
    out["varmix_global_crps"] = crps_gaussian(
        y=out[y_true_col],
        mu=out[mu_cp_col],
        sd=out["varmix_global_sd"],
    )

    # -------------------------
    # Region-specific rho
    # -------------------------
    if rho_by_region is not None:
        out["rho_region"] = out["Region"].map(rho_by_region)

        rho_arr = out["rho_region"].to_numpy()
        #var_region = (
        #    (1 - rho_arr) * out[sd_cp_col].to_numpy()**2
        #    + rho_arr * out[sd_clim_col].to_numpy()**2
        #)
        var_region = var_cp + rho_arr * np.maximum(var_clim - var_cp, 0.0)

        out["varmix_region_sd"] = np.sqrt(np.maximum(var_region, 1e-12))
        out["varmix_region_lower"] = out[mu_cp_col] - z * out["varmix_region_sd"]
        out["varmix_region_upper"] = out[mu_cp_col] + z * out["varmix_region_sd"]
        out["varmix_region_width"] = (
            out["varmix_region_upper"] - out["varmix_region_lower"]
        )
        out["varmix_region_covered"] = (
            (out[y_true_col] >= out["varmix_region_lower"])
            & (out[y_true_col] <= out["varmix_region_upper"])
        )
        out["varmix_region_crps"] = crps_gaussian(
            y=out[y_true_col],
            mu=out[mu_cp_col],
            sd=out["varmix_region_sd"],
        )

    return out