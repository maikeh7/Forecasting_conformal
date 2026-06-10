import pandas as pd
from pathlib import Path
import sys
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from conformal.features import Y_TRUE_COL, Y_PRED_COL, ALPHA, FORECAST_DATE_COL, CURRENT_WEEK_COL, Y_TREND_COL
from conformal.data_helpers import CAL_FILES, TEST_FILES
from conformal.config import CONFORMAL_UNPROCESSED_DIR, CONFORMAL_PROCESSED_DIR, PLOT_DIR
from conformal.methods.dynamic_graph import make_dynamic_graph_snapshots, check_graph_row_sums, plot_dynamic_graph_heatmaps_by_season, plot_graph_edge_variability, plot_graph_row_entropy, plot_graph_entropy_by_season, extract_graph_row_over_time, plot_graph_row_weights_over_time, add_season_background, plot_graph_row_weights_over_time_with_seasons, summarize_graph_row_by_season


h = 1
T_WIN=12
cal_df = pd.read_csv(CAL_FILES[h])
test_df = pd.read_csv(TEST_FILES[h])

graph_df, graph_cache = make_dynamic_graph_snapshots(
    cal_df=cal_df,
    test_df=test_df,
    forecast_date_col="forecast_date",
    current_week_col="current_week",
    T_window=T_WIN,
    graph_method="corr",
)

region_order = ["MW", "NE", "SE", "SW", "W"]

w_row_df = extract_graph_row_over_time(
    graph_cache,
    target_region="W",
    region_order=region_order,
    graph_df=graph_df,
    forecast_date_col="forecast_date",
)

plot_graph_row_weights_over_time(
    w_row_df,
    T_window=T_WIN,
    horizon=h,
    target_region="W",
    region_order=region_order,
    forecast_date_col="forecast_date",
    use_timestep=True,
)

plot_graph_row_weights_over_time_with_seasons(
    w_row_df,
    T_window=T_WIN,
    horizon=h,
    target_region="W",
    region_order=["MW", "NE", "SE", "SW", "W"],
    forecast_date_col="forecast_date",
    use_timestep=True,
)

w_season_summary = summarize_graph_row_by_season(
    w_row_df,
    region_order=["MW", "NE", "SE", "SW", "W"],
)

print(w_season_summary)

'''

check_graph_row_sums(graph_df)

selected_dates = plot_dynamic_graph_heatmaps_by_season(
    graph_df,
    graph_cache,
    T_window=T_WIN,
    horizon=h,
    forecast_date_col="forecast_date",
    seasons=("Winter", "Spring", "Summer", "Fall"),
    n_dates_per_season=3,
    region_order=["MW", "NE", "SE", "SW", "W"],
    figsize=(13, 14),
    annot=True,
)

variability = plot_graph_edge_variability(
    graph_cache,
    T_window = T_WIN,
    horizon = h,
    region_order=["MW", "NE", "SE", "SW", "W"],
    stat="std",
)

entropy_df = plot_graph_row_entropy(
    graph_cache,
    graph_df=graph_df,
    T_window=T_WIN,
    horizon=h,
    region_order=["MW", "NE", "SE", "SW", "W"],
    normalize=True,
)

entropy_summary = (
    entropy_df.groupby("Region")["entropy"]
    .agg(["mean", "std", "min", "max"])
    .round(3)
)
print("entropy summary:")
entropy_summary.to_csv(PLOT_DIR / "entropy_summary.csv")
print(entropy_summary)

plot_graph_entropy_by_season(
    entropy_df,
    T_window=T_WIN,
    horizon=h,
    region_order=["MW", "NE", "SE", "SW", "W"],
)
'''
