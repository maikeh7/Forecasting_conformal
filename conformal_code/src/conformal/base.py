# conformal/base.py
from dataclasses import dataclass, field
from typing import Any
import pandas as pd
from .config import SPLITS, DETREND_OUTPUTS, REDUNDANT_COLS
from .data_helpers import read_csv_with_date

REQUIRED_PRED_COLS = [
    "horizon",
    "Region",
    "y_true",
    "y_pred",
    "lower",
    "upper",
    "covered",
    "width",
    "interval_score",
    "forecast_date",
    "reference_date",
    "q_hat",
    "trend_pred_K",
    "y_true_detrended",
    "y_pred_detrended",
    "lower_detrended",
    "upper_detrended"
]


OPTIONAL_STACK_COLS = [
    "Season",
]




@dataclass
class ConformalResult:
    method: str
    predictions_by_horizon: dict[int, pd.DataFrame]
    metadata_by_horizon: dict[int, dict[str, Any]] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    def stacked_predictions(self) -> pd.DataFrame:
        frames = []
        keep_cols_base = REQUIRED_PRED_COLS + OPTIONAL_STACK_COLS

        for h, df in self.predictions_by_horizon.items():
            keep_cols = [c for c in keep_cols_base if c in df.columns]
            out = df[keep_cols].copy()
            out["method"] = self.method
            out["horizon"] = h
            frames.append(out)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True, sort=False)



def validate_prediction_df(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_PRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Prediction dataframe missing columns: {missing}")

