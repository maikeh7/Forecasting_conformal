# conformal/plots.py
import matplotlib.pyplot as plt
from .utils import thin
from .base import ConformalResult


def plot_interval_grid(
    result: ConformalResult,
    regions: list[str],
    horizons: list[int] = [1, 2, 3, 4],
    max_points_per_panel: int = 500,
    x_col: str = None,
):
    nrows = len(horizons)
    ncols = len(regions)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4 * ncols, 2.8 * nrows),
        sharex=False,
        sharey=False,
    )

    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]

    for i, h in enumerate(horizons):
        dfh = result.predictions_by_horizon[h]

        for j, region in enumerate(regions):
            ax = axes[i][j]
            sub = dfh[dfh["Region"] == region].copy()
            sub = thin(sub, max_points_per_panel)

            if x_col is not None and x_col in sub.columns:
                x = sub[x_col].values
            else:
                x = sub.index.values

            ax.fill_between(x, sub["lower"].values, sub["upper"].values, alpha=0.25)
            ax.plot(x, sub["y_true"].values, linewidth=1)
            ax.plot(x, sub["y_pred"].values, linewidth=1)

            if i == 0:
                ax.set_title(f"{result.method} | {region}")
            if j == 0:
                ax.set_ylabel(f"h={h}")

    plt.tight_layout()
    return fig, axes