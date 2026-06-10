import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from conformal.data_helpers import CAL_FILES, TEST_FILES
from conformal.features import ALPHA, Y_TRUE_COL, Y_PRED_COL
from conformal.metrics import coverage_objective_region_season, interval_score, add_interval_score
from conformal.methods.tune_epsilon import run_kernel_regionmix_conformal, summarize_eval
from conformal.utils import parse_year_week_from_yyyww
from conformal.config import TUNING_DIR, HORIZONS
import argparse

def tuning_objective_combined_band(
    df_eval,
    alpha,
    y_true_col="y_true_bt",
    lower_col="lower",
    upper_col="upper",
    neff_col="n_eff",
    lambda_cov=50.0,
    lambda_neff=0.1,
    n_eff_min=30.0,
    coverage_tol=0.02,
):
    y = df_eval[y_true_col].to_numpy()
    l = df_eval[lower_col].to_numpy()
    u = df_eval[upper_col].to_numpy()

    width = u - l
    below = np.maximum(l - y, 0.0)
    above = np.maximum(y - u, 0.0)

    interval_score = np.mean(
        width + (2.0 / alpha) * below + (2.0 / alpha) * above
    )

    target = 1 - alpha
    coverage = float(df_eval["covered"].mean())
    diff = abs(coverage - target)
    coverage_penalty = lambda_cov * max(diff - coverage_tol, 0.0)

    if neff_col in df_eval.columns:
        med_neff = float(np.nanmedian(df_eval[neff_col]))
    else:
        med_neff = np.nan

    if np.isnan(med_neff):
        neff_penalty = 0.0
    else:
        neff_penalty = lambda_neff * max(n_eff_min - med_neff, 0.0)

    total_obj = interval_score + coverage_penalty + neff_penalty

    return {
        "objective": float(total_obj),
        "interval_score": float(interval_score),
        "coverage": coverage,
        "coverage_penalty": float(coverage_penalty),
        "median_neff": float(med_neff) if not np.isnan(med_neff) else np.nan,
        "neff_penalty": float(neff_penalty),
    }

def main(kernel_kind="exp", tau_weeks=4):
    region_prefix = "Region_"
    eps_grid = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.35, 0.50]
    best_eps = {}
    all_eps_results = []
    
    for h in HORIZONS:
        cal_all = pd.read_csv(CAL_FILES[h])
        cal_all = parse_year_week_from_yyyww(cal_all)
    
        cal_train = cal_all[cal_all["year"].isin([2017, 2018, 2019])].copy()
        cal_val   = cal_all[cal_all["year"].isin([2020])].copy()
    
        rows = []
        for eps in eps_grid:
            val_out = run_kernel_regionmix_conformal(
                cal_df=cal_train,
                eval_df=cal_val,
                eps=eps,
                alpha=ALPHA,
                tau_weeks=tau_weeks,
                kernel_kind=kernel_kind,
            )

            
            #obj = coverage_objective_region_season(val_out)
            cov = float(val_out["covered"].mean())
            medw = float(val_out["width"].median())
            med_neff = float(np.nanmedian(val_out["n_eff"]))
            obj = interval_score(
                    val_out,
                    alpha=ALPHA,
                    y_true_col=Y_TRUE_COL,
                    lower_col="lower",
                    upper_col="upper"
            )
            rows.append((eps, obj, cov, medw, med_neff, h))
            
            
            all_eps_results.append({"Horizon": h, "interval_score": obj, "coverage": cov,
                                 "Median_width": medw, "Median_neff": med_neff, "epsilon": eps})
      
        epsilon_df = pd.DataFrame(all_eps_results)
        epsilon_df.to_csv(TUNING_DIR / "epsilon_tuning" / f"epsilon_tuning_results_{kernel_kind}_{tau_weeks}.csv", index=False)
        
        rows.sort(key=lambda t: t[1])
        best = rows[0]
        best_eps[h] = best[0]
    
        print(f"\nH{h} tuning (tau={tau_weeks}, kernel={kernel_kind}):")
        for eps, obj, cov, medw, med_neff, h in rows:
            print(f"  eps={eps:>4} | obj={obj:.4f} | cov={cov:.3f} | medW={medw:.3f} | med n_eff={med_neff:.1f}")
        print(f"Best eps: {best[0]}")

    print("running full method on test data...")
    test_results = []
    for h in HORIZONS:
        cal_all = pd.read_csv(CAL_FILES[h])
        test = pd.read_csv(TEST_FILES[h])
    
        out_test = run_kernel_regionmix_conformal(
            cal_df=cal_all,      # full calibration period
            eval_df=test,
            eps=best_eps[h],
            alpha=ALPHA,
            tau_weeks=tau_weeks,
            kernel_kind=kernel_kind,
        )
    
        overall = float(out_test["covered"].mean())
        medw = float(out_test["width"].median())
        med_neff = float(np.nanmedian(out_test["n_eff"]))
        int_score = interval_score(
                    out_test,
                    alpha=ALPHA,
                    y_true_col=Y_TRUE_COL,
                    lower_col="lower",
                    upper_col="upper"
            )
        out_test = add_interval_score(out_test, alpha=ALPHA, y_true_col=Y_TRUE_COL)
    
        print(f"\nH{h} TEST | tuned eps={best_eps[h]} | cov={overall:.3f} | medW={medw:.3f} | med n_eff={med_neff:.1f} | int_score ={int_score:.1f}")
        tab = out_test.groupby(["Region","Season"], observed=False)["interval_score"].mean().unstack("Season")
        test_results.append(tab)
    final_df = pd.concat(test_results)
    final_df.to_csv(TUNING_DIR / "epsilon_tuning" / f"epsilon_tuning_test_{kernel_kind}_{tau_weeks}.csv", index=False)

if __name__ == "__main__":
    def positive_int(value):
        try:
            ivalue = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{value} is not an integer")
       
        if ivalue <= 0:
            raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
        return ivalue
        
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kernel_kind",
        choices=["exp", "gauss"],
        default="exp",
        help="Kernel to use for season correlation matrix",
    )
    parser.add_argument(
        "--tau_weeks",
        type=positive_int,
        default=4,
        help="Lengthscale for season correlation matrix"
    )
    args = parser.parse_args()
    main(args.kernel_kind, args.tau_weeks)
