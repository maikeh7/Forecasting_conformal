# conformal/config.py
from dataclasses import dataclass
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class RegionModelConfig:
    y_true_col: str = "y_true_kelvin"
    y_pred_col: str = "y_pred_kelvin"
    region_col: str = "Region"
    date_col: str = "forecast_date"
    alpha: float = 0.10


DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
TUNING_DIR = OUTPUT_DIR / "tuning"

BASE_RESULTS_DIR = OUTPUT_DIR / "base_model"
CONFORMAL_RESULTS_DIR = OUTPUT_DIR / "conformal"
CONFORMAL_PROCESSED_DIR = CONFORMAL_RESULTS_DIR / "processed_results"
CONFORMAL_UNPROCESSED_DIR = CONFORMAL_RESULTS_DIR / "unprocessed_results"
PLOT_DIR = OUTPUT_DIR / "plotting" / "plots"

REGIONS = ["W", "SW", "MW", "SE", "NE"]
REGION_ZSCORES = [f"{region}_Zscore" for region in REGIONS]

HORIZONS = [1, 2, 3, 4]
MAX_LAG = 6 # produces lags 1-5


SPLITS = {
    "dev": {
        "name": "1980_2016_train",
        "data_file": RAW_DIR / "CONUS_Regions_1980_to_2020.csv",
        "time_file": RAW_DIR / "time_inputs_1980_to_2020.csv",
        "inputs_file": RAW_DIR / "weekly_aves_regional_inputs_1980_2020.csv",
        "train_end_year": 2016,
        "label": "dev",   # used in output paths
    },
    "final": {
        "name": "1980_2020_train",
        "data_file": RAW_DIR / "CONUS_Regions_1980_to_2024.csv",
        "time_file": RAW_DIR / "time_inputs_1980_to_2024.csv",
        "inputs_file": RAW_DIR / "weekly_aves_regional_inputs_1980_2024.csv",
        "train_end_year": 2020,
        "label": "final",
    },
}

DETREND_OUTPUTS = {
    "dev": {
        "train_residuals_file": OUTPUT_DIR / "detrending" / "dev" / "train_residuals.csv",
        "test_residuals_file": OUTPUT_DIR / "detrending" / "dev" / "test_residuals.csv",
        "test_preds_file": OUTPUT_DIR / "detrending" / "dev" / "test_preds.csv",
        "train_preds_file": OUTPUT_DIR / "detrending" / "dev" / "train_preds.csv",
        "label": "dev",
    },
    "final": {
        "train_residuals_file": OUTPUT_DIR / "detrending" / "final" / "train_residuals.csv",
        "test_residuals_file": OUTPUT_DIR / "detrending" / "final" / "test_residuals.csv",
        "test_preds_file": OUTPUT_DIR / "detrending" / "final" / "test_preds.csv",
        "train_preds_file": OUTPUT_DIR / "detrending" / "final" / "train_preds.csv",
        "label": "final",
    }
    
}

alpha = 0.10
y_true_col = "y_true_kelvin"
y_pred_col = "y_pred_kelvin"

REDUNDANT_COLS = ['slp_weekave_atlantic_ocean_mean', 'h_850_weekave_conus_mean', 'h_850_weekave_atlantic_ocean_pc4',
                  'h_850_weekave_mexico_gulf_mean', 'slp_weekave_mexico_gulf_mean', 'slp_weekave_conus_pc1', 'ts_weekave_conus_pc1', 
                  'slp_weekave_atlantic_trop_mean', 'ts_weekave_southern_canada_pc1', 't_850_weekave_southern_canada_mean', 
                  'h_850_weekave_atlantic_ocean_pc5', 'slp_weekave_atlantic_ocean_pc6', 'h_500_weekave_pacific_trop_mean', 
                  'h_850_weekave_atlantic_trop_mean', 't_500_weekave_pacific_trop_mean', 't_500_weekave_atlantic_trop_pc1',
                  'h_850_weekave_pacific_ocean_pc1', 't_850_weekave_conus_pc1', 'slp_weekave_atlantic_ocean_pc1', 
                  'slp_weekave_pacific_trop_mean', 't_850_weekave_southern_canada_pc1', 'h_850_weekave_arctic_pc1',
                  'h_500_weekave_atlantic_trop_mean', 'h_850_weekave_atlantic_ocean_pc2', 'slp_weekave_atlantic_ocean_pc3',
                  'h_850_weekave_atlantic_ocean_pc7', 't_500_weekave_mexico_gulf_mean', 't_500_weekave_southern_canada_pc1', 
                  'slp_weekave_pacific_trop_pc2', 'h_850_weekave_arctic_mean', 'ts_weekave_atlantic_trop_mean',
                  'h_850_weekave_pacific_trop_mean']


