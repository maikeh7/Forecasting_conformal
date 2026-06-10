#conformal_code/scripts/run_conformal_train.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os
from pathlib import Path
from conformal.config import SPLITS, DETREND_OUTPUTS, REGIONS, BASE_RESULTS_DIR, PROCESSED_DIR, RESULTS_DIR
from conformal.features import union_allowed_features_from_strings
from conformal.data_helpers import read_csv_with_date, make_long_panel
from conformal.metrics import weighted_rmse_step, weighted_rmse_smooth

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))



def main():
    spec_det = DETREND_OUTPUTS["dev"]
    spec = SPLITS["dev"]
    # preds from linear model on train set --  for backtransforming
    train_preds = read_csv_with_date(spec_det["train_preds_file"])

    # preds from linear model on test set -- for backtransforming
    test_preds = read_csv_with_date(spec_det["test_preds_file"))
    test_preds.set_index('Date', inplace=True)

    regional_inputs = read_csv_with_date(spec["inputs_file"])
    
    # has Z scores
    weekly_aves = read_csv_with_date(spec["data_file"])

    # This contains important variables from the 'per region' model
    importance_df = pd.read_csv(RESULTS_DIR / "RF_rmse_results_region_horizon_2021_2024.csv")

    mode = "standard"

    models_by_h = {}
    rmse_rows = []
    RF_preds = []
    
    for horizon in HORIZONS:
        train_long = make_long_panel(
            train_residuals, regional_inputs, horizon, REGIONS, weekly_aves_df=weekly_aves
        )
        test_long = make_long_panel(
            test_residuals, regional_inputs, horizon, REGIONS, weekly_aves_df=weekly_aves
        )
    
        allowed = union_allowed_features_from_strings(importance_df, horizon, REGIONS)
    
        region_dummy_cols = [c for c in train_long.columns if c.startswith("Region_")]
        lag_cols = ["current_week"] + [f"lag_{k}" for k in range(1, 6)]
    
        allowed_cols = [
            f for f in allowed
            if f in train_long.columns
            and f not in lag_cols
            and f not in region_dummy_cols
        ]
    
        feature_cols = lag_cols + allowed_cols + region_dummy_cols
    
        if len(feature_cols) != len(set(feature_cols)):
            raise ValueError("Duplicate columns detected in feature_cols")
    
        drop_cols = ["y"] + feature_cols
        train_long = train_long.dropna(subset=drop_cols)
        test_long = test_long.dropna(subset=drop_cols)
    
        X_train = train_long[feature_cols]
        y_train = train_long["y"]
        X_test = test_long[feature_cols]
        y_test = test_long["y"]
    
        final_dat = test_long.copy().reset_index()
        final_dat = final_dat[
            feature_cols + ["y", "Region", "horizon", "reference_date", "forecast_date", "zscore_true"]
        ]
        
           # fit
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        models_by_h[horizon] = model
    
        # predict residuals
        y_pred = model.predict(X_test)
        
        #final_dat["y_pred_resid"] = y_pred
        test_long["y_pred_resid"] = y_pred
        final_dat["y_pred_resid"] = y_pred
        #print(test_long.head())
        for region, df_r in test_long.groupby("Region"):
            rmse_all = rmse(df_r["y"], df_r["y_pred_resid"])
        
            extreme_mask = df_r["zscore_true"].abs() > 1
            rmse_ext = rmse(df_r.loc[extreme_mask, "y"], df_r.loc[extreme_mask, "y_pred_resid"]) if extreme_mask.any() else np.nan
        
            rmse_rows.append({
                "Horizon": horizon,
                "Region": region,
                "RMSE_resid": rmse_all,
                "RMSE_resid_extreme": rmse_ext,
                "N": len(df_r),
                "N_extreme": int(extreme_mask.sum())
            })
    
            print(f"horizon: {horizon} | Region: {region} | RMSE: {rmse_all} | RMSE extreme = {rmse_ext}")
    
        # trend at forecast_date aligned to reference_date
        test_preds_target = test_preds.shift(-horizon)
        
        trend_long = (
            test_preds_target
            .reset_index()
            .rename(columns={"Date": "reference_date"})  # or {"index": "reference_date"} depending on your index name
            .melt(id_vars="reference_date", var_name="Region", value_name="trend_pred_K")
        )
        
        # Merge into test_long (which now has reference_date as a column)
        test_long2 = final_dat.reset_index(drop=True)  # reference_date is already a column too
        test_long2 = final_dat.merge(trend_long, on=["reference_date", "Region"], how="left")
        
        # Kelvin prediction
        test_long2["y_pred_kelvin"] = test_long2["y_pred_resid"] + test_long2["trend_pred_K"]
        test_long2["y_true_kelvin"] = test_long2["y"] + test_long2["trend_pred_K"]
        test_long2.to_csv(Path(PROCESSED_DIR / f"/Calibration_long_{horizon}.csv"), index=False)
    
    
    
    rmse_df = pd.DataFrame(rmse_rows)
    rmse_df.to_csv(Path(BASE_RESULTS_DIR / "rmse_calib.csv"), index=False)

    ## Now predict onto test set
    spec_det = DETREND_OUTPUTS["final"]
    spec = SPLITS["final"]

    # preds from linear model on test set -- for backtransforming
    test_preds = read_csv_with_date(spec_det["test_preds_file"]).set_index('Date') # 2021-2024
    test_preds.set_index('Date', inplace=True)

    regional_inputs = read_csv_with_date(spec["inputs_file"))
    
    # has Z scores
    weekly_aves = read_csv_with_date(spec["data_file"])

    # TEST SET for conformal
    # we are just making predictions here
    for horizon in HORIZONS:
        
        # build long ttest
        test_long  = make_long_panel(test_residuals, regional_inputs, horizon, REGIONS, weekly_aves_df=weekly_aves)
    
        # features: lags + allowed + region dummies
        allowed = union_allowed_features_from_strings(importance_df, horizon, REGIONS)
    
        # region dummy columns in long df:
        region_dummy_cols = [c for c in train_long.columns if c.startswith("Region_")]
    
        lag_cols = ["current_week"] + [f"lag_{k}" for k in range(1, 6)]
        feature_cols = lag_cols + [f for f in allowed if f in train_long.columns] + region_dummy_cols
    
        drop_cols = ["y"] + feature_cols
        test_long = test_long.dropna(subset=drop_cols)
    
        X_test  = test_long[feature_cols]
        y_test  = test_long["y"]
        final_dat = test_long.copy()
        final_dat = final_dat.reset_index()
        final_dat = final_dat[feature_cols + ["y", "Region", "horizon", "reference_date", "forecast_date", "zscore_true"]]
    
        # fit
        model = models_by_h[horizon]
    
        # predict residuals
        y_pred = model.predict(X_test)
        test_long["y_pred_resid"] = y_pred
        final_dat["y_pred_resid"] = y_pred
        
        for region, df_r in test_long.groupby("Region"):
            rmse_all = rmse(df_r["y"], df_r["y_pred_resid"])
        
            extreme_mask = df_r["zscore_true"].abs() > 1
            z_data = df_r["zscore_true"]
            
            rmse_ext = rmse(df_r.loc[extreme_mask, "y"], df_r.loc[extreme_mask, "y_pred_resid"]) if extreme_mask.any() else np.nan
            
            y_true = df_r.loc[extreme_mask, "y"]
            y_pred = df_r.loc[extreme_mask, "y_pred_resid"]

            wrmse_step = weighted_rmse_step(y_true, y_pred, z_data)
            wrmse_smooth = weighted_rmse_smooth(y_true, y_pred, z_data, alpha=1.0)

            # Convert to numpy arrays and remove any potential NaNs jointly
            valid_mask = (~pd.isna(y_test.values)) & (~pd.isna(y_pred)) & (~pd.isna(z_data.values))
            
            y_true_valid = y_test.values[valid_mask]
            y_pred_valid = y_pred[valid_mask]
            z_valid = z_data.values[valid_mask]
            
            wrmse_step = weighted_rmse_step(y_true_valid, y_pred_valid, z_valid)
            wrmse_smooth = weighted_rmse_smooth(y_true_valid, y_pred_valid, z_valid, alpha=1.0)

        
            rmse_rows.append({
                "Horizon": horizon,
                "Region": region,
                "RMSE_resid": rmse_all,
                "RMSE_resid_extreme": rmse_ext,
                "Weighted_RMSE_step": wrmse_step,
                "Weighted_RMSE_smooth": wrmse_smooth,
                "N": len(df_r),
                "N_extreme": int(extreme_mask.sum())
            })
    
        # trend at forecast_date aligned to reference_date
        test_preds_target = test_preds.shift(-horizon)
        
        trend_long = (
            test_preds_target
            .reset_index()
            .rename(columns={"Date": "reference_date"})  # or {"index": "reference_date"} depending on your index name
            .melt(id_vars="reference_date", var_name="Region", value_name="trend_pred_K")
        )
        
        # Merge into test_long (which now has reference_date as a column)
        test_long2 = final_dat.reset_index(drop=True)  # reference_date is already a column too
        test_long2 = final_dat.merge(trend_long, on=["reference_date", "Region"], how="left")
        
        # Kelvin prediction
        test_long2["y_pred_kelvin"] = test_long2["y_pred_resid"] + test_long2["trend_pred_K"]
        test_long2["y_true_kelvin"] = test_long2["y"] + test_long2["trend_pred_K"]
        test_long2.to_csv(Path(PROCESSED_DIR / f"Test_long_{horizon}.csv"), index=False)

        rmse_df = pd.DataFrame(rmse_rows)
        rmse_df.to_csv(Path(BASE_RESULTS_DIR / "rmse_test.csv"), index=False)
        
if __name__ == "__main__":
    main()


