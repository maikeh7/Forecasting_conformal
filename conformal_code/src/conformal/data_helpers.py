# conformal/data_helpers.py
import numpy as np
import pandas as pd
from pathlib import Path
from .config import REDUNDANT_COLS, BASE_RESULTS_DIR


CAL_FILES = {
    1: BASE_RESULTS_DIR / "Calibration_long_1.csv",
    2: BASE_RESULTS_DIR / "Calibration_long_2.csv",
    3: BASE_RESULTS_DIR / "Calibration_long_3.csv",
    4: BASE_RESULTS_DIR / "Calibration_long_4.csv",
}

TEST_FILES = {
    1: BASE_RESULTS_DIR / "Test_long_1.csv",
    2: BASE_RESULTS_DIR / "Test_long_2.csv",
    3: BASE_RESULTS_DIR / "Test_long_3.csv",
    4: BASE_RESULTS_DIR / "Test_long_4.csv",
}


def read_csv_with_date(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
   
    # 1. Standardize 'date' to 'Date' if it exists
    if 'date' in df.columns:
        df = df.rename(columns={'date': 'Date'})
   
    # 2. Convert to string if 'Date' is now in the columns
    if 'Date' in df.columns:
        df['Date'] = df['Date'].astype(str)
       
    return df

def load_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_train_test_data_dev() -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = SPLITS["dev"]
    resid_spec = DETREND_OUTPUTS["dev"]
    regional_inputs = read_csv_with_date(spec["inputs_file"])
    
    # drop columns deemed to be redundant from correlation analysis
    regional_inputs = regional_inputs.drop(columns = REDUNDANT_COLS)
    
    train_residuals = read_csv_with_date(resid_spec["train_residuals_file"])
    test_residuals = read_csv_with_date(resid_spec["test_residuals_file"])
    # Merge inputs with residuals to get full training/testing sets
    train_data = pd.merge(train_residuals, regional_inputs, on="Date").set_index("Date")
    test_data = pd.merge(test_residuals, regional_inputs, on="Date").set_index("Date")
    return train_data, test_data


def make_long_panel(
    residuals_df,
    inputs_df,
    horizon,
    weekly_aves_df=None,
    keep_region_dummies=True,
):
    """
    Builds long-form panel where each row is (reference_date=t, Region=r).
    Target y corresponds to forecast_date=t+h.

    residuals_df: ['Date', W, SW, ...] residuals
    inputs_df: ['Date', ...features...]
    weekly_aves_df (optional): ['Date', W, W_Zscore, SW, SW_Zscore, ...] with W etc = Kelvin truth
    """

    # --- Base wide table on reference_date ---
    wide = pd.merge(residuals_df, inputs_df, on="Date", how="inner")
    wide = wide.sort_values("Date").set_index("Date")  # Date is reference timeline
    wide.index.name = "reference_date"

    # --- Merge Kelvin truth + zscores (rename to avoid collision with residual region names) ---
    if weekly_aves_df is not None:
        wk = weekly_aves_df.copy()
        rename_map = {}
        for r in regions:
            if r in wk.columns:
                rename_map[r] = f"{r}_K"
            if f"{r}_Zscore" in wk.columns:
                rename_map[f"{r}_Zscore"] = f"{r}_Z"
        wk = wk.rename(columns=rename_map).sort_values("Date").set_index("Date")
        wk.index.name = "reference_date"

        wide = wide.merge(wk, left_index=True, right_index=True, how="left")

    # --- Build long panel ---
    long_parts = []
    ref_dates = wide.index.to_series()

    for r in REGIONS:
        part = pd.DataFrame(index=wide.index.copy())
        part.index.name = "reference_date"
        part["Region"] = r
        part["horizon"] = horizon

        # Explicit reference_date column (nice for merges without relying on index)
        #part["reference_date"] = part.index

        # forecast_date: the Date code at t+h (NOT calendar arithmetic)
        part["forecast_date"] = ref_dates.shift(-horizon).values

        # residual target at forecast_date
        part["y"] = wide[r].shift(-horizon)

        # true kelvin and zscore at forecast_date (optional)
        if weekly_aves_df is not None:
            part["y_kelvin_true"] = wide[f"{r}_K"].shift(-horizon)
            part["zscore_true"] = wide.get(f"{r}_Z", np.nan).shift(-horizon)

        # predictors: current + lags of residuals (known at reference_date)
        part["current_week"] = wide[r]
        for lag in range(1, 6):
            part[f"lag_{lag}"] = wide[r].shift(lag)

        # attach other predictors (exclude residual region columns)
        other = wide.drop(columns=regions, errors="ignore")

        # keep kelvin/z columns out of predictors
        if weekly_aves_df is not None:
            eval_cols = [f"{rr}_K" for rr in regions] + [f"{rr}_Z" for rr in regions]
            other = other.drop(columns=eval_cols, errors="ignore")

        part = part.join(other)
        long_parts.append(part)

    long_df = pd.concat(long_parts, axis=0)

    # Drop rows with missing target (and missing kelvin truth if present)
    drop_cols = ["y", "forecast_date"]
    if weekly_aves_df is not None:
        drop_cols.append("y_kelvin_true")
    long_df = long_df.dropna(subset=drop_cols)

    # Keep region dummies but preserve Region string column
    if keep_region_dummies:
        dummies = pd.get_dummies(long_df["Region"], prefix="Region", drop_first=False)
        long_df = pd.concat([long_df, dummies], axis=1)
        
    long_df = long_df.reset_index()
    return long_df
