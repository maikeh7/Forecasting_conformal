from .weighted_cp import run_weighted_region_conformal
from .stratified_cp import run_stratified_conformal
from .global_cp import run_global_cp
from .dynampic_graph_cp import run_dynamic_graph_cp

from .config import CAL_FILES, TEST_FILES




def main():
    alpha = 0.10
    y_true_col = "y_true_kelvin"
    y_pred_col = "y_pred_kelvin"

    global_result = run_global_conformal(
    CAL_FILES,
    TEST_FILES,
    alpha=alpha,
    y_true_col=y_true_col,
    y_pred_col=y_pred_col,
    )
    
    strat_result = run_region_season_conformal(
        CAL_FILES,
        TEST_FILES,
        alpha=alpha,
        y_true_col=y_true_col,
        y_pred_col=y_pred_col,
    )
    
    weighted_result = run_weighted_region_conformal(
        CAL_FILES,
        TEST_FILES,
        alpha=alpha,
        y_true_col=y_true_col,
        y_pred_col=y_pred_col,
        tau_weeks=4.0,
        kernel="exp",
        region_offdiag=0.10,
    )

    dynamic_graph_result = run_dynamic_graph_conformal(
        CAL_FILES,
        TEST_FILES,
        alpha=alpha,
        y_true_col=y_true_col,
        y_pred_col=y_pred_col,
        tau_weeks=4.0,
        kernel="exp"
    )

if __name__ == "main":
    main()