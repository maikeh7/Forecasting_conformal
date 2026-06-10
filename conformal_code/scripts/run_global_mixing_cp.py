import pandas as pd
from pathlib import Path
import sys
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from conformal.methods.global_cp_clim_mix import run_global_conformal_clim_mix
from conformal.features import Y_TRUE_COL, Y_PRED_COL, ALPHA, FORECAST_DATE_COL, CURRENT_WEEK_COL, Y_TREND_COL
from conformal.data_helpers import CAL_FILES, TEST_FILES
from conformal.config import CONFORMAL_UNPROCESSED_DIR, CONFORMAL_PROCESSED_DIR
from conformal.metrics import summarize_predictions

def main(mixing_type="convex"):
    var_mixing_type = mixing_type
    print(f"using Variance Mixing Type: {var_mixing_type}")
    results = run_global_conformal_clim_mix(
        cal_files=CAL_FILES,
        test_files=TEST_FILES,
        alpha=ALPHA,
        y_true_col=Y_TRUE_COL,
        y_pred_col=Y_PRED_COL,
        y_trend_col=Y_TREND_COL,
        rho_mode="global",
        mixing_type=var_mixing_type # convex or infation_only
        
    )
    
    results = results.stacked_predictions()
    results_summary = summarize_predictions(results, group_cols=["horizon", "Region"])
    results_summary.to_csv(CONFORMAL_PROCESSED_DIR / f"global_clim_mix_summary_{var_mixing_type}.csv", index=False)
    results.to_csv(CONFORMAL_UNPROCESSED_DIR / f"global_clim_mix_predictions_{var_mixing_type}.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mixing_type",
        choices=["convex", "inflation_only"],
        default="convex",
        help="Which type of variance mixing to use",
    )
    args = parser.parse_args()
    main(args.mixing_type)


