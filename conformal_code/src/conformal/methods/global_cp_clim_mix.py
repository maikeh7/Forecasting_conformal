# conformal/global_cp.py
import numpy as np
import pandas as pd
from pathlib import Path

from conformal.config import TUNING_DIR, CONFORMAL_UNPROCESSED_DIR, CONFORMAL_PROCESSED_DIR
from conformal.base import ConformalResult, validate_prediction_df
from conformal.methods.method_helpers import calculate_detrended_intervals
from conformal.utils import (
    add_week_num,
    add_season,
    apply_symmetric_intervals,
)

from conformal.data_helpers import CAL_FILES, TEST_FILES
from conformal.methods.global_cp import conformal_q
from conformal.metrics import add_interval_score, crps_gaussian_mixture_2, A_gaussian_abs, add_cp_clim_and_mixture_intervals, summarize_interval_coverage_width, add_variance_mix_intervals, tune_rho_varmix_crps, tune_rho_varmix_by_region, summarize_all_global_clim_mix_variants_by_group
from .method_helpers import parse_year_week_from_yyyww, add_season, calculate_detrended_intervals
from conformal.hetGP.HetGP_Fitting import run_hetGP_fitting
from conformal.base import ConformalResult, validate_prediction_df


Z90 = 1.6448536269514722  # central 90% normal interval

# this is for climatology mixing
def add_global_cp_sigma(df_eval, df_cal_train,
                        y_true_col,
                        y_pred_col,
                        alpha):
    scores = np.abs(df_cal_train[y_true_col] -  df_cal_train[y_pred_col])
    q = np.quantile(scores.dropna(), 1 - alpha)

    out = df_eval.copy()
    out["q_cp"] = q
    out["sigma_cp"] = q / Z90
    return out, q  

def tune_lambda_by_region(
    df_val,
    region_col="Region",
    y_true_col="y_true_kelvin",
    mu_cp_col="y_pred_kelvin",
    sd_cp_col="sigma_cp",
    mu_clim_col="hetgp_mean_kelvin",
    sd_clim_col="hetgp_predSD",
):

    lambda_grid = np.linspace(0, 1, 101)

    best_rows = []
    all_rows = []

    for region, sub in df_val.groupby(region_col):
        sub = sub.dropna(
            subset=[y_true_col, mu_cp_col, sd_cp_col, mu_clim_col, sd_clim_col]
        )

        if len(sub) == 0:
            continue

        rows = []

        for lam in lambda_grid:
            crps = crps_gaussian_mixture_2(
                y=sub[y_true_col],
                mu_cp=sub[mu_cp_col],
                sd_cp=sub[sd_cp_col],
                mu_clim=sub[mu_clim_col],
                sd_clim=sub[sd_clim_col],
                lam=lam,
            )

            mean_crps = np.nanmean(crps)

            rows.append({
                region_col: region,
                "lambda": lam,
                "mean_crps": mean_crps,
                "n": len(sub),
            })

        region_results = pd.DataFrame(rows)
        all_rows.append(region_results)

        best = region_results.loc[region_results["mean_crps"].idxmin()]
        best_rows.append(best)

    best_df = pd.DataFrame(best_rows).reset_index(drop=True)
    all_df = pd.concat(all_rows, ignore_index=True)

    return best_df, all_df
    

def tune_lambda_crps(df_val,
                     y_true_col,
                     mu_cp_col,
                     sd_cp_col="sigma_cp",
                     mu_clim_col="hetgp_mean_kelvin",
                     sd_clim_col="hetgp_predSD"):

    lambda_grid = np.linspace(0, 1, 101)

    rows = []
    for lam in lambda_grid:
        crps = crps_gaussian_mixture_2(
            y=df_val[y_true_col],
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
    



def crps_curve_over_lambda(
    df,
    *,
    lambda_grid=None,
    y_true_col="y_true_kelvin",
    mu_cp_col="y_pred_kelvin",
    sd_cp_col="sigma_cp",
    mu_clim_col="hetgp_mean_kelvin",
    sd_clim_col="hetgp_predSD",
    label="",
):
    if lambda_grid is None:
        lambda_grid = np.linspace(0, 1, 101)

    rows = []

    for lam in lambda_grid:
        crps = crps_gaussian_mixture_2(
            y=df[y_true_col],
            mu_cp=df[mu_cp_col],
            sd_cp=df[sd_cp_col],
            mu_clim=df[mu_clim_col],
            sd_clim=df[sd_clim_col],
            lam=lam,
        )

        rows.append({
            "lambda": lam,
            "mean_crps": np.nanmean(crps),
            "label": label,
        })

    return pd.DataFrame(rows)
    

def summarize_density_methods(
    df,
    *,
    best_lambda_global,
    lambda_by_region=None,
    y_true_col="y_true_kelvin",
    mu_cp_col="y_pred_kelvin",
    sd_cp_col="sigma_cp",
    mu_clim_col="hetgp_mean_kelvin",
    sd_clim_col="hetgp_predSD",
):
    out = df.copy()

    out["crps_clim"] = crps_gaussian_mixture_2(
        y=out[y_true_col],
        mu_cp=out[mu_cp_col],
        sd_cp=out[sd_cp_col],
        mu_clim=out[mu_clim_col],
        sd_clim=out[sd_clim_col],
        lam=0.0,
    )

    out["crps_cp"] = crps_gaussian_mixture_2(
        y=out[y_true_col],
        mu_cp=out[mu_cp_col],
        sd_cp=out[sd_cp_col],
        mu_clim=out[mu_clim_col],
        sd_clim=out[sd_clim_col],
        lam=1.0,
    )

    out["crps_mix_global_lambda"] = crps_gaussian_mixture_2(
        y=out[y_true_col],
        mu_cp=out[mu_cp_col],
        sd_cp=out[sd_cp_col],
        mu_clim=out[mu_clim_col],
        sd_clim=out[sd_clim_col],
        lam=best_lambda_global,
    )

    if lambda_by_region is not None:
        out["lambda_region"] = out["Region"].map(lambda_by_region)

        out["crps_mix_region_lambda"] = crps_gaussian_mixture_2(
            y=out[y_true_col],
            mu_cp=out[mu_cp_col],
            sd_cp=out[sd_cp_col],
            mu_clim=out[mu_clim_col],
            sd_clim=out[sd_clim_col],
            lam=out["lambda_region"].to_numpy(),
        )
    else:
        out["crps_mix_region_lambda"] = np.nan

    summary = out[
        [
            "crps_clim",
            "crps_cp",
            "crps_mix_global_lambda",
            "crps_mix_region_lambda",
        ]
    ].mean()

    return out, summary

def run_global_conformal_clim_mix(
    cal_files: dict[int, str],
    test_files: dict[int, str],
    *,
    alpha: float,
    y_true_col: str = "y_true_kelvin",
    y_pred_col: str = "y_pred_kelvin",
    forecast_date_col: str = "forecast_date",
    y_trend_col: str = "trend_pred_K",
    rho_mode: str = "global", # or region
    mixing_type: str = "convex", # or "inflation_only"
    lambda_grid=None
):
    if lambda_grid is None:
        lambda_grid = np.linspace(0, 1, 101)

    
    val_train_years= (x for x in range(1980, 2017)), 
    val_test_years=(2017, 2018, 2019, 2020),
    test_years=(2021, 2022, 2023, 2024)
    # always train climatological model on 1980-2016, just like the RF model
    # tune lambda using 2017-2020
    pred_df_val = run_hetGP_fitting(
        train_years=list((x for x in range(1980, 2017))),
        test_years=list((2017, 2018, 2019, 2020)), # test years doesn't matter here--we only train on 1980-2016
    )

    #pred_df_test, pred_df_long_test = run_hetGP_fitting(
    #    train_years=list(val_train_years) + list(val_test_years),
    #    test_years=list(test_years),
    #)

    diagnostics_rows = []
    lambda_curve_rows = []
    lambda_region_rows = []
    test_outputs = []
    summary_rows = []
    summary_season_rows = []

    keep_cols = [
        "reference_date",
        "forecast_date",
        "Region",
        "Season",
        "week_num",
        "year",
        "y",
        "y_pred_resid",
        "y_true_kelvin",
        "y_pred_kelvin",
        "trend_pred_K",
    ]

    interval_summary_rows = []
    predictions_by_horizon = {}
    preds_by_h = {}
    metadata_by_horizon = {}
    summary_season_rows = []

    for h in sorted(cal_files):
        print(f"\n================ HORIZON {h} ================")

        # IMPORTANT: use h, not 1
        df_cal = pd.read_csv(cal_files[h])
        df_test = pd.read_csv(test_files[h])

        df_cal = parse_year_week_from_yyyww(df_cal)
        df_cal = add_season(df_cal)

        df_test = parse_year_week_from_yyyww(df_test)
        df_test = add_season(df_test)

        df_cal = df_cal[keep_cols].copy()
        df_test = df_test[keep_cols].copy()

        # chronological split inside calibration
        cal_train = df_cal[df_cal["year"].isin([2017,2018,2019,2020])].copy()
        cal_val = df_cal[df_cal["year"].isin([2017, 2018,2019,2020])].copy()

        # -----------------------------
        # Validation: merge hetGP preds
        # -----------------------------
        cal_val = pd.merge(
            cal_val,
            pred_df_val,
            on=["Region", "week_num"],
            how="left",
        )

        cal_val["hetgp_mean_kelvin"] = cal_val["hetgp_mean"] + cal_val[y_trend_col]
        cal_val["hetgp_lower90_kelvin"] = cal_val["hetgp_lower90"] + cal_val[y_trend_col]
        cal_val["hetgp_upper90_kelvin"] = cal_val["hetgp_upper90"] + cal_val[y_trend_col]


        # Add CP sigma from cal_train
        cal_val, q_tune = add_global_cp_sigma(
            df_eval=cal_val,
            df_cal_train=cal_train,
            y_true_col=y_true_col,
            y_pred_col=y_pred_col,
            alpha=alpha,
        )

        # Tune global lambda on validation year
        best_lambda_global, lambda_global_results = tune_lambda_crps(
            cal_val,
            y_true_col=y_true_col,
            mu_cp_col=y_pred_col,
            sd_cp_col="sigma_cp",
            mu_clim_col="hetgp_mean_kelvin",
            sd_clim_col="hetgp_predSD",
        )

        # Tune region-specific lambda on validation year
        best_lambda_region, lambda_region_results = tune_lambda_by_region(
            cal_val,
            region_col="Region",
            y_true_col=y_true_col,
            mu_cp_col=y_pred_col,
            sd_cp_col="sigma_cp",
            mu_clim_col="hetgp_mean_kelvin",
            sd_clim_col="hetgp_predSD",
        )

        lambda_by_region = dict(
            zip(best_lambda_region["Region"], best_lambda_region["lambda"])
        )

        # Tune variance-only mixture rho on calibration/tuning period
        best_rho_global, rho_global_results = tune_rho_varmix_crps(
            cal_val,
            rho_grid=lambda_grid,  # same 0..1 grid is fine
            y_true_col=y_true_col,
            mu_cp_col=y_pred_col,
            sd_cp_col="sigma_cp",
            sd_clim_col="hetgp_predSD",
        )
        
        best_rho_region, rho_region_results = tune_rho_varmix_by_region(
            cal_val,
            rho_grid=lambda_grid,
            region_col="Region",
            y_true_col=y_true_col,
            mu_cp_col=y_pred_col,
            sd_cp_col="sigma_cp",
            sd_clim_col="hetgp_predSD",
        )
        
        rho_by_region = dict(zip(best_rho_region["Region"], best_rho_region["rho"]))



        # Validation CRPS curve
        val_curve = crps_curve_over_lambda(
            cal_val,
            lambda_grid=lambda_grid,
            y_true_col=y_true_col,
            mu_cp_col=y_pred_col,
            sd_cp_col="sigma_cp",
            mu_clim_col="hetgp_mean_kelvin",
            sd_clim_col="hetgp_predSD",
            label="validation",
        )
        val_curve["horizon"] = h
        lambda_curve_rows.append(val_curve)

        # Validation diagnostics
        cal_val_eval, val_summary = summarize_density_methods(
            cal_val,
            best_lambda_global=best_lambda_global,
            lambda_by_region=lambda_by_region,
            y_true_col=y_true_col,
            mu_cp_col=y_pred_col,
            sd_cp_col="sigma_cp",
            mu_clim_col="hetgp_mean_kelvin",
            sd_clim_col="hetgp_predSD",
        )

        # ------------------------------------
        # Test: merge hetGP preds to test data
        # ------------------------------------
        cal_full = df_cal.copy()

        df_test = pd.merge(
            df_test,
            pred_df_val,
            on=["Region", "week_num"],
            how="left",
        )

        df_test["hetgp_mean_kelvin"] = df_test["hetgp_mean"] + df_test[y_trend_col]
        df_test["hetgp_lower90_kelvin"] = df_test["hetgp_lower90"] + df_test[y_trend_col]
        df_test["hetgp_upper90_kelvin"] = df_test["hetgp_upper90"] + df_test[y_trend_col]

        # Add CP sigma using full calibration
        df_test, q_final = add_global_cp_sigma(
            df_eval=df_test,
            df_cal_train=cal_full,
            y_true_col=y_true_col,
            y_pred_col=y_pred_col,
            alpha=alpha,
        )

        # Test CRPS curve
        test_curve = crps_curve_over_lambda(
            df_test,
            lambda_grid=lambda_grid,
            y_true_col=y_true_col,
            mu_cp_col=y_pred_col,
            sd_cp_col="sigma_cp",
            mu_clim_col="hetgp_mean_kelvin",
            sd_clim_col="hetgp_predSD",
            label="test",
        )
        test_curve["horizon"] = h
        lambda_curve_rows.append(test_curve)

        # Test diagnostics
        df_test_eval, test_summary = summarize_density_methods(
            df_test,
            best_lambda_global=best_lambda_global,
            lambda_by_region=lambda_by_region,
            y_true_col=y_true_col,
            mu_cp_col=y_pred_col,
            sd_cp_col="sigma_cp",
            mu_clim_col="hetgp_mean_kelvin",
            sd_clim_col="hetgp_predSD",
        )

        # Add exact 90% density intervals and coverage indicators
        df_test_eval = add_cp_clim_and_mixture_intervals(
            df_test_eval,
            alpha=alpha,
            y_true_col=y_true_col,
            mu_cp_col=y_pred_col,
            sd_cp_col="sigma_cp",
            mu_clim_col="hetgp_mean_kelvin",
            sd_clim_col="hetgp_predSD",
            best_lambda_global=best_lambda_global,
            lambda_region_col="lambda_region"
        )

        df_test_eval = add_variance_mix_intervals(
            df_test_eval,
            alpha=alpha,
            rho_global=best_rho_global,
            rho_by_region=rho_by_region,
            y_true_col=y_true_col,
            mu_cp_col=y_pred_col,
            sd_cp_col="sigma_cp",
            sd_clim_col="hetgp_predSD",
            mixing_type = mixing_type
        )
      

        if rho_mode == "global":
            final_lower_col = "varmix_global_lower"
            final_upper_col = "varmix_global_upper"
            final_covered_col = "varmix_global_covered"
            final_width_col = "varmix_global_width"
            final_score_col = "varmix_global_interval_score"  # if available
            final_crps_col = "varmix_global_crps"
        elif rho_mode == "region":
            final_lower_col = "varmix_region_lower"
            final_upper_col = "varmix_region_upper"
            final_covered_col = "varmix_region_covered"
            final_width_col = "varmix_region_width"
            final_score_col = "varmix_region_interval_score"
            final_crps_col = "varmix_region_crps"
        else:
            raise ValueError("rho_mode must be 'global' or 'region'")

        region_summary_h = summarize_all_global_clim_mix_variants_by_group(
            df_test_eval,
            horizon=h,
            alpha=alpha,
            group_cols=("Region",),
            y_true_col=y_true_col,
        )
        
        region_season_summary_h = summarize_all_global_clim_mix_variants_by_group(
            df_test_eval,
            horizon=h,
            alpha=alpha,
            group_cols=("Region", "Season"),
            y_true_col=y_true_col,
        )
        
        summary_rows.append(region_summary_h)
        summary_season_rows.append(region_season_summary_h)

        
        df_final = df_test_eval.copy()
        df_final["lower"] = df_final[final_lower_col]
        df_final["upper"] = df_final[final_upper_col]
        df_final["covered"] = df_final[final_covered_col]
        df_final["width"] = df_final[final_width_col] 
        df_final["horizon"] = h
        df_final["y_true"] = df_final[y_true_col]
        df_final["y_pred"] = df_final[y_pred_col]
        df_final["q_hat"] = q_final
        #df_final["interval_score"] = df_final[final_score_col]
        df_final["crps"] = df_final[final_crps_col]
        
        df_final = add_interval_score(
            df=df_final,
            alpha=alpha,
            y_true_col=y_true_col,
        )
        df_final = calculate_detrended_intervals(
            df=df_final,
            y_true_col=y_true_col,
            y_pred_col=y_pred_col,
            y_trend_col=y_trend_col,
        )

        
        #validate_prediction_df(df_final)
        
        predictions_by_horizon[h] = df_final
        
        metadata_by_horizon[h] = {
            "q_final": q_final,
            "rho_mode": rho_mode,
            "rho_global": best_rho_global,
            "rho_by_region": rho_by_region,
            "mixing_type": mixing_type,
        }

        
        # Overall + by-region summaries
        interval_summary_region = summarize_interval_coverage_width(
            df_test_eval,
            horizon=h,
            y_true_col=y_true_col,
            group_cols=("Region",),
        )
        interval_summary_rows.append(interval_summary_region)
        
        # Region x season summaries
        interval_summary_region_season = summarize_interval_coverage_width(
            df_test_eval,
            horizon=h,
            y_true_col=y_true_col,
            group_cols=("Region", "Season"),
        )
        interval_summary_rows.append(interval_summary_region_season)

        df_test_eval = df_test_eval.copy()
        df_test_eval["Horizon"] = h
        test_outputs.append(df_test_eval)

        # Region lambda table
        tmp_lambda_region = best_lambda_region.copy()
        tmp_lambda_region["horizon"] = h
        lambda_region_rows.append(tmp_lambda_region)

        # Oracle test lambda for diagnosis only
        test_best = test_curve.loc[test_curve["mean_crps"].idxmin()]
        oracle_test_lambda = float(test_best["lambda"])
        oracle_test_crps = float(test_best["mean_crps"])

        diagnostics_rows.append({
            "horizon": h,
            "q_tune": q_tune,
            "q_final": q_final,
            "best_lambda_global_val": best_lambda_global,
            "oracle_lambda_test": oracle_test_lambda,
            "oracle_crps_test": oracle_test_crps,

            "val_crps_clim": val_summary["crps_clim"],
            "val_crps_cp": val_summary["crps_cp"],
            "val_crps_mix_global": val_summary["crps_mix_global_lambda"],
            "val_crps_mix_region": val_summary["crps_mix_region_lambda"],

            "test_crps_clim": test_summary["crps_clim"],
            "test_crps_cp": test_summary["crps_cp"],
            "test_crps_mix_global": test_summary["crps_mix_global_lambda"],
            "test_crps_mix_region": test_summary["crps_mix_region_lambda"],

            "best_rho_global_val": best_rho_global,
            "test_crps_varmix_global": df_test_eval["varmix_global_crps"].mean(),
            "test_crps_varmix_region": df_test_eval["varmix_region_crps"].mean(),
            "test_cov_varmix_global": df_test_eval["varmix_global_covered"].mean(),
            "test_cov_varmix_region": df_test_eval["varmix_region_covered"].mean(),
            "test_med_width_varmix_global": df_test_eval["varmix_global_width"].median(),
            "test_med_width_varmix_region": df_test_eval["varmix_region_width"].median(),

        })

        print(f"Best validation lambda: {best_lambda_global:.2f}")
        print(f"Oracle test lambda: {oracle_test_lambda:.2f}")
        print("Validation CRPS:")
        print(val_summary.round(4))
        print("Test CRPS:")
        print(test_summary.round(4))


    summary_rows_df = pd.concat(summary_rows, ignore_index=True)
    summary_season_rows_df = pd.concat(summary_season_rows, ignore_index=True)
    summary_rows_df.to_csv(Path(CONFORMAL_PROCESSED_DIR, "clim_mix_region_summary_all.csv"))
    summary_season_rows_df.to_csv(Path(CONFORMAL_PROCESSED_DIR, "clim_mix_region_seasson_summary_all.csv"))
    
    all_data_results = pd.concat(test_outputs, ignore_index=True)
    all_data_results.to_csv(Path(CONFORMAL_UNPROCESSED_DIR, "all_clim_mix_results_by_horizon.csv"))
    diagnostics_df = pd.DataFrame(diagnostics_rows)
    lambda_curves_df = pd.concat(lambda_curve_rows, ignore_index=True)
    lambda_region_df = pd.concat(lambda_region_rows, ignore_index=True)
    diagnostics_df.to_csv(Path(TUNING_DIR , "clim_mix" , "diagnostics_df.csv"))
    lambda_curves_df.to_csv(Path(TUNING_DIR , "clim_mix" , "lambda_curves.csv"))
    lambda_region_df.to_csv(Path(TUNING_DIR , "clim_mix" , "lambda_region.csv"))
    
    interval_summary_df = pd.concat(interval_summary_rows, ignore_index=True)
    
    interval_summary_df.to_csv(
        Path(TUNING_DIR, "clim_mix", "interval_summary.csv"),
        index=False,
    )
    

    return ConformalResult(
    method=f"global_cp_clim_varmix_{rho_mode}_{mixing_type}",
    predictions_by_horizon=predictions_by_horizon,
    metadata_by_horizon=metadata_by_horizon,
    config={
        "alpha": alpha,
        "y_true_col": y_true_col,
        "y_pred_col": y_pred_col,
        "date_col": forecast_date_col,
        "rho_mode": rho_mode,
        "mixing_type": mixing_type,
    },
)
    #return {
    #    "test_outputs": test_outputs,
    #    "diagnostics": diagnostics_df,
    #    "lambda_curves": lambda_curves_df,
    #    "lambda_region": lambda_region_df,
    #    "interval_summary": interval_summary_df
    #}