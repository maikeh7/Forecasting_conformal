import pandas as pd
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from conformal.methods.weighted_cp import run_weighted_region_conformal
from conformal.methods.stratified_cp import run_region_season_conformal
from conformal.methods.global_cp import run_global_conformal
from conformal.methods.dynamic_graph import run_dynamic_graph_cp
from conformal.methods.dynamic_graph_clim_mix import add_clim_variance_mix_to_dynamic_result
from conformal.config import CONFORMAL_UNPROCESSED_DIR, CONFORMAL_PROCESSED_DIR
from conformal.features import Y_TRUE_COL, Y_PRED_COL, ALPHA, FORECAST_DATE_COL, CURRENT_WEEK_COL, Y_TREND_COL
from conformal.metrics import summarize_predictions, summarize_dynamic_varmix_result
from conformal.data_helpers import CAL_FILES, TEST_FILES

rho_by_horizon = {
    1: 0.15,
    2: 0.35,
    3: 0.50,
    4: 0.60,
}

def main():
    alpha = 0.10
    #y_true_col = "y_true_kelvin"
    #y_pred_col = "y_pred_kelvin"

    global_result = run_global_conformal(
    CAL_FILES,
    TEST_FILES,
    alpha=ALPHA,
    y_true_col=Y_TRUE_COL,
    y_pred_col=Y_PRED_COL,
    y_trend_col=Y_TREND_COL
    )
    res_global = global_result.stacked_predictions()
    global_summary = summarize_predictions(res_global, group_cols=["horizon", "Region"])
    
    strat_result = run_region_season_conformal(
        CAL_FILES,
        TEST_FILES,
        alpha=ALPHA,
        y_true_col=Y_TRUE_COL,
        y_pred_col=Y_PRED_COL,
        y_trend_col=Y_TREND_COL
    )
    res_strat = strat_result.stacked_predictions()
    strat_summary = summarize_predictions(res_strat, group_cols=["horizon", "Region"])
    
    weighted_result = run_weighted_region_conformal(
        CAL_FILES,
        TEST_FILES,
        alpha=ALPHA,
        y_true_col=Y_TRUE_COL,
        y_pred_col=Y_PRED_COL,
        date_col=FORECAST_DATE_COL,
        tau_weeks=2.5,
        kernel="exp",
        region_offdiag=0.40,
        y_trend_col=Y_TREND_COL
    )
    res_weighted = weighted_result.stacked_predictions()
    weighted_summary = summarize_predictions(res_weighted, group_cols=["horizon", "Region"])

    dynamic_result = run_dynamic_graph_cp(
        CAL_FILES,
        TEST_FILES,
        week_kernel="exp",
        tau_weeks=2.5,
        y_true_col=Y_TRUE_COL,
        y_pred_col=Y_PRED_COL,
        alpha=ALPHA,
        forecast_date_col=FORECAST_DATE_COL,
        current_week_col=CURRENT_WEEK_COL,
        T_window=40,
        graph_method="corr",
        y_trend_col=Y_TREND_COL
    )

    dynamic_varmix_result = add_clim_variance_mix_to_dynamic_result(
        dynamic_result,
        rho_by_horizon=rho_by_horizon,
        alpha=ALPHA,
        y_true_col=Y_TRUE_COL,
        y_pred_col=Y_PRED_COL,
        sd_clim_col="hetgp_predSD",
        prefix="dynamic_varmix",
        mixing_type="convex"
    )

    dynamic_varmix_infl_result = add_clim_variance_mix_to_dynamic_result(
        dynamic_result,
        rho_by_horizon=rho_by_horizon,
        alpha=ALPHA,
        y_true_col=Y_TRUE_COL,
        y_pred_col=Y_PRED_COL,
        sd_clim_col="hetgp_predSD",
        prefix="dynamic_varmix_infl",
        mixing_type="inflation_only",
    )

    dynamic_varmix_result = dynamic_varmix.stacked_predictions()
    dynamic_varmix_infl_result = dynamic_varmix_infl_result.stacked_predictions()
    
    dynamic_varmix_summary = summarize_predictions(dynamic_varmix_result, group_cols=["horizon", "Region"])
    
    dynamic_varmix_summary_infl = summarize_predictions(dynamic_varmix_infl_result, group_cols=["horizon", "Region"])
    
    dynamic_varmix_summary.to_csv(CONFORMAL_PROCESSED_DIR / "dynamic_varmix_summary.csv", index=False)
    
    dynamic_varmix_summary_infl.to_csv(CONFORMAL_PROCESSED_DIR / "dynamic_varmix_summary_infl.csv", index=False)

    
    res_dynamic = dynamic_result.stacked_predictions()
    dynamic_summary = summarize_predictions(res_dynamic, group_cols=["horizon", "Region"])
    
    dynamic_varmix_result.to_csv(CONFORMAL_UNPROCESSED_DIR / "dyn_varmix_unprocessed_results.csv", index=False)
    dynamic_varmix_infl_result.to_csv(CONFORMAL_UNPROCESSED_DIR / "dyn_varmix_infl_unprocessed_results.csv", index=False)
    
    res_global.to_csv(CONFORMAL_UNPROCESSED_DIR / "global_unprocessed_results.csv", index=False)
    res_strat.to_csv(CONFORMAL_UNPROCESSED_DIR / "stratified_unprocessed_results.csv", index=False)
    res_weighted.to_csv(CONFORMAL_UNPROCESSED_DIR / "weighted_unprocessed_results.csv", index=False)
    res_dynamic.to_csv(CONFORMAL_UNPROCESSED_DIR / "dynamic_unprocessed_results.csv", index=False)

    global_summary.to_csv(CONFORMAL_PROCESSED_DIR / "global_summary.csv", index=False)
    strat_summary.to_csv(CONFORMAL_PROCESSED_DIR / "stratified_summary.csv", index=False)
    weighted_summary.to_csv(CONFORMAL_PROCESSED_DIR / "weighted_summary.csv", index=False)
    dynamic_summary.to_csv(CONFORMAL_PROCESSED_DIR / "dynamic_summary.csv", index=False)
    

if __name__ == "__main__":
    main()