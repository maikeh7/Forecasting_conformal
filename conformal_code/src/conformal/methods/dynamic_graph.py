import numpy as np
import pandas as pd
from .method_helpers import add_season, weighted_quantile, circ_week_dist, Pi_shrinkage, kernel_fn, parse_year_week_from_yyyww, n_eff
from conformal.features import Y_PRED_COL, Y_TRUE_COL, ALPHA, FORECAST_DATE_COL, CURRENT_WEEK_COL
from conformal.config import HORIZONS
from conformal.utils import apply_symmetric_intervals
from conformal.metrics import add_interval_score
from conformal.base import ConformalResult, validate_prediction_df
from conformal.methods.method_helpers import calculate_detrended_intervals
import matplotlib.pyplot as plt

def build_region_panel(df, value_col="current_week"):
    """
    Build wide matrix:
      rows   = forecast_date (YYYYWW)
      cols   = Region
      values = current_week anomaly
    Assumes one row per region per date.
    """
    wide = (
        df.pivot_table(
            index=FORECAST_DATE_COL,
            columns="Region",
            values=value_col,
            aggfunc="first"
        )
        .sort_index()
    )
    return wide

    
def effective_n(weights):
    w = np.asarray(weights, dtype=float)
    s = w.sum()
    if s <= 0:
        return np.nan
    w = w / s
    return float(1.0 / np.sum(w ** 2))

    
def compute_dynamic_similarity_matrix(panel_wide, 
                                      current_date,
                                      T_window,
                                      method="corr"):
    """
    panel_wide: wide matrix indexed by forecast_date, columns=Region
    current_date: integer YYYYWW for the test point
    T_window: number of past rows to use, excluding current_date
    method: 'corr' or 'cosine'

    Returns:
      G: DataFrame of region-region similarities, row-normalized
    """
    if current_date not in panel_wide.index:
        raise ValueError(f"current_date {current_date} not found in region panel")

    idx = panel_wide.index.get_loc(current_date)

    # use only past rows, never current or future
    start = max(0, idx - T_window)
    window = panel_wide.iloc[start:idx].copy()

    # if too few rows, fallback later
    if len(window) < max(4, T_window // 3):
        return None

    # fill missing values with column means inside window
    window = window.apply(lambda col: col.fillna(col.mean()), axis=0)

    regions = list(window.columns)

    if method == "corr":
        G = window.corr()
    elif method == "cosine":
        X = window.to_numpy().T  # shape: regions x time
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Xn = X / norms
        G = pd.DataFrame(Xn @ Xn.T, index=regions, columns=regions)
    else:
        raise ValueError("method must be 'corr' or 'cosine'")

    # clip negatives to zero
    G = G.clip(lower=0.0)

    # add tiny diagonal stabilization
    for r in G.index:
        G.loc[r, r] = max(G.loc[r, r], 1e-8)

    # row normalize so each row sums to 1
    row_sums = G.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    G = G.div(row_sums, axis=0)

    return G

# ============================================================
# MAIN METHOD
# ============================================================
def dynamic_graph_conformal(cal_df,
                            test_df, 
                            y_true_col, 
                            y_pred_col, 
                            alpha, 
                            forecast_date_col,
                            current_week_col,
                            T_window=20, 
                            tau_weeks=4.0,
                            graph_method="corr", 
                            week_kernel="exp"):
    """
    Dynamic graph weighted conformal:
      w_j(x*) ∝ g_t(r_j, r*) * k(w_j, w*)
    """

    cal = parse_year_week_from_yyyww(cal_df)
    test = parse_year_week_from_yyyww(test_df)

    cal = add_season(cal)
    test = add_season(test)

    cal["score"] = (cal[y_true_col] - cal[y_pred_col]).abs()

    # build region panel from ALL available observations up through test period
    # for a real deployment, this should only contain information available up to each time.
    all_df = pd.concat(
        [
            cal[[forecast_date_col, "Region", current_week_col]],
            test[[forecast_date_col, "Region", current_week_col]],
        ],
        ignore_index=True
    ).drop_duplicates(subset=[forecast_date_col, "Region"])

    region_panel = build_region_panel(all_df, value_col=current_week_col)

    # calibration arrays
    cal_scores = cal["score"].to_numpy()
    cal_weeks = cal["week_num"].to_numpy()
    cal_regions = cal["Region"].to_numpy()

    qhat = np.empty(len(test), dtype=float)
    neff = np.empty(len(test), dtype=float)

    # precompute static fallback global quantile
    global_q = float(np.quantile(cal_scores, 1 - alpha))

    # cache graph by test date since many rows share same date
    graph_cache = {}

    for i, row in enumerate(test.itertuples(index=False)):
        test_date = int(getattr(row, forecast_date_col))
        r_star = getattr(row, "Region")
        w_star = int(getattr(row, "week_num"))

        # dynamic graph for this test date
        if test_date not in graph_cache:
            G_t = compute_dynamic_similarity_matrix(
                region_panel,
                current_date=test_date,
                T_window=T_window,
                method=graph_method
            )
            graph_cache[test_date] = G_t
        else:
            G_t = graph_cache[test_date]

        # fallback if insufficient history
        if G_t is None:
            qhat[i] = global_q
            neff[i] = np.nan
            continue

        # region similarity for each calibration point
        # similarity from test region r_star to cal region r_j
        # because G_t is row-normalized, use row r_star
        if r_star not in G_t.index:
            qhat[i] = global_q
            neff[i] = np.nan
            continue
            
        # Long vec of corrs of all rj w/ r_star
        g_vec = np.array([G_t.loc[r_star, rj] if rj in G_t.columns else 0.0 for rj in cal_regions])

        # week similarity
        # d_week is long vec of corrs between weeks in cal and test point week
        d_week = circ_week_dist(cal_weeks, w_star, period=52)
        k_vec = kernel_fn(d_week, tau=tau_weeks, kind=week_kernel)

        # combined weights
        w = g_vec * k_vec
        s = w.sum()

        if s <= 0:
            qhat[i] = global_q
            neff[i] = np.nan
        else:
            qhat[i] = weighted_quantile(cal_scores, w / s, 1 - alpha)
            neff[i] = effective_n(w)

    out = test.copy()
    out["q_hat"] = qhat
    out["n_eff"] = neff

    df_pi = apply_symmetric_intervals(
            out,
            q=out["q_hat"].values,
            y_true_col=y_true_col,
            y_pred_col=y_pred_col,
            q_col_name="q_hat",
    )

    return df_pi

    
def run_dynamic_graph_cp(cal_files,
                         test_files,
                         week_kernel,
                         tau_weeks,
                         y_true_col,
                         y_pred_col, 
                         alpha, 
                         forecast_date_col,
                         current_week_col,
                         T_window, 
                         graph_method,
                         y_trend_col="trend_pred_K",
                        ):
    
    predictions_by_horizon = {}
    metadata_by_horizon = {}
    
    for h in HORIZONS:
        cal_df = pd.read_csv(cal_files[h])
        test_df = pd.read_csv(test_files[h])
    
        out = dynamic_graph_conformal(
            cal_df=cal_df,
            test_df=test_df,
            y_true_col=y_true_col,
            y_pred_col=y_pred_col,
            alpha=alpha,
            forecast_date_col=forecast_date_col,
            current_week_col=current_week_col,
            T_window=T_window,
            tau_weeks=tau_weeks,
            graph_method="corr",   # try "cosine" too
            week_kernel=week_kernel
        )

        overall_cov = float(out["covered"].mean())
        median_width = float(out["width"].median())
        mean_width = float(out["width"].mean())
        median_neff = float(np.nanmedian(out["n_eff"]))
    
        print(f"\n--- Dynamic Graph Conformal | Horizon {h} ---")
        print(f"Overall coverage: {overall_cov:.3f}")
        print(f"Median width: {median_width:.3f}")
        print(f"Mean width: {mean_width:.3f}")
        print(f"Median n_eff: {median_neff:.1f}")
    
        print("\nCoverage by Region:")
        print(out.groupby("Region")["covered"].mean().round(3))
    
        print("\nCoverage by Region x Season:")
        print(
            out.groupby(["Region", "Season"])["covered"]
            .mean()
            .unstack("Season")
            .round(3)
        )

        df_pi = add_interval_score(df=out, alpha=alpha, y_true_col=y_true_col)
        df_pi = calculate_detrended_intervals(df=df_pi, y_true_col=y_true_col, y_pred_col=y_pred_col, y_trend_col=y_trend_col)
        
        validate_prediction_df(df_pi)

        predictions_by_horizon[h] = df_pi
        metadata_by_horizon[h] = {"kernel_kind": week_kernel, 
                                  "tau_weeks": tau_weeks,
                                  "T_window": T_window
                                 }
    
    return ConformalResult(
        method="dynamic_graph",
        predictions_by_horizon=predictions_by_horizon,
        metadata_by_horizon=metadata_by_horizon,
        config={
            "alpha": alpha,
            "y_true_col": y_true_col,
            "y_pred_col": y_pred_col,
            "date_col": forecast_date_col,
        },
    )


"""
These are functions for plotting heatmaps of the dynamic graph
"""
def make_dynamic_graph_snapshots(
    cal_df,
    test_df,
    *,
    forecast_date_col,
    current_week_col,
    T_window=20,
    graph_method="corr",
):
    """
    Builds dynamic graph matrices G_t for all unique test forecast dates.

    Returns
    -------
    graph_df : pd.DataFrame
        Metadata for each available graph date.
    graph_cache : dict
        Maps forecast_date -> G_t matrix.
    """

    cal = parse_year_week_from_yyyww(cal_df)
    test = parse_year_week_from_yyyww(test_df)

    cal = add_season(cal)
    test = add_season(test)

    # Build region panel using calibration + test current_week values.
    # compute_dynamic_similarity_matrix still uses only past rows before each current_date.
    all_df = pd.concat(
        [
            cal[[forecast_date_col, "Region", current_week_col]],
            test[[forecast_date_col, "Region", current_week_col]],
        ],
        ignore_index=True,
    ).drop_duplicates(subset=[forecast_date_col, "Region"])

    region_panel = build_region_panel(all_df, value_col=current_week_col)

    # One metadata row per test forecast date
    date_meta = (
        test[[forecast_date_col, "week_num", "Season", "year"]]
        .drop_duplicates(subset=[forecast_date_col])
        .sort_values(forecast_date_col)
        .reset_index(drop=True)
    )

    graph_cache = {}
    rows = []

    for _, row in date_meta.iterrows():
        date = int(row[forecast_date_col])

        G_t = compute_dynamic_similarity_matrix(
            region_panel,
            current_date=date,
            T_window=T_window,
            method=graph_method,
        )

        if G_t is None:
            continue

        graph_cache[date] = G_t

        row_sums = G_t.sum(axis=1)

        rows.append({
            forecast_date_col: date,
            "week_num": int(row["week_num"]),
            "Season": row["Season"],
            "year": int(row["year"]),
            "min_row_sum": row_sums.min(),
            "max_row_sum": row_sums.max(),
            "mean_row_sum": row_sums.mean(),
        })

    graph_df = pd.DataFrame(rows)

    return graph_df, graph_cache


def select_graph_dates_by_season(
    graph_df,
    *,
    forecast_date_col,
    seasons=("Winter", "Spring", "Summer", "Fall"),
    n_dates_per_season=3,
):
    """
    Select dates spread across the test period for each season.
    Uses quantiles of available dates within each season.
    """

    selected = {}

    for season in seasons:
        sub = (
            graph_df[graph_df["Season"] == season]
            .sort_values(forecast_date_col)
            .reset_index(drop=True)
        )

        if len(sub) == 0:
            selected[season] = []
            continue

        if len(sub) <= n_dates_per_season:
            dates = sub[forecast_date_col].tolist()
        else:
            idx = np.linspace(0, len(sub) - 1, n_dates_per_season)
            idx = np.round(idx).astype(int)
            dates = sub.loc[idx, forecast_date_col].tolist()

        selected[season] = dates

    return selected


def plot_dynamic_graph_heatmaps_by_season(
    graph_df,
    graph_cache,
    T_window,
    horizon,
    forecast_date_col,
    seasons=("Winter", "Spring", "Summer", "Fall"),
    n_dates_per_season=3,
    region_order=None,
    figsize=None,
    annot=True,
):
    """
    Plot dynamic graph matrices for selected dates across seasons.

    Rows = seasons.
    Columns = dates spread across the test period.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    selected_dates = select_graph_dates_by_season(
        graph_df,
        forecast_date_col=forecast_date_col,
        seasons=seasons,
        n_dates_per_season=n_dates_per_season,
    )

    if region_order is None:
        # infer from first available graph
        first_date = None
        for season in seasons:
            if len(selected_dates[season]) > 0:
                first_date = selected_dates[season][0]
                break

        if first_date is None:
            raise ValueError("No graph dates available to plot.")

        region_order = list(graph_cache[first_date].index)

    n_rows = len(seasons)
    n_cols = n_dates_per_season

    if figsize is None:
        figsize = (4.2 * n_cols, 3.5 * n_rows)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        sharex=True,
        sharey=True,
    )

    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    for i, season in enumerate(seasons):
        dates = selected_dates[season]

        for j in range(n_cols):
            ax = axes[i, j]

            if j >= len(dates):
                ax.axis("off")
                continue

            date = dates[j]
            G_t = graph_cache[date].copy()

            # consistent row/column ordering
            G_t = G_t.reindex(index=region_order, columns=region_order)

            meta = graph_df[graph_df[forecast_date_col] == date].iloc[0]
            week = int(meta["week_num"])
            year = int(meta["year"])

            sns.heatmap(
                G_t,
                ax=ax,
                annot=annot,
                fmt=".2f",
                cmap="viridis",
                vmin=0,
                vmax=1,
                cbar=(i == 0 and j == n_cols - 1),
                square=True,
            )

            ax.set_title(f"{season}: {date}\nYear={year}, week={week}", fontsize=11)
            ax.set_xlabel("Borrow from region")
            ax.set_ylabel("Test region" if j == 0 else "")

    fig.suptitle(
        "Dynamic region graph weights across test-period seasons",
        fontsize=16,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(f'/home/mfholth/subseasonal/weekly_data/conformal_UQ/conformal_code/outputs/plotting/plots/graph_by_season_TWIN{T_window}_h{horizon}.png')
    plt.show()

    return selected_dates

def check_graph_row_sums(graph_df, tol=1e-6):
    """
    Print row-sum diagnostics for dynamic graph matrices.
    """

    print("Graph row-sum diagnostics:")
    print(graph_df[["min_row_sum", "mean_row_sum", "max_row_sum"]].describe())

    bad = graph_df[
        (np.abs(graph_df["min_row_sum"] - 1.0) > tol)
        | (np.abs(graph_df["max_row_sum"] - 1.0) > tol)
    ]

    if len(bad) == 0:
        print("All graph row sums are approximately 1.")
    else:
        print(f"Warning: {len(bad)} dates have row sums outside tolerance.")
        print(bad.head())

def graph_edge_variability(graph_cache, region_order=None, stat="std"):
    """
    Compute edge-wise variability over time across dynamic graph matrices.

    Parameters
    ----------
    graph_cache : dict
        Maps forecast_date -> G_t DataFrame.
    region_order : list or None
        Region ordering for rows/columns.
    stat : {"std", "range", "iqr"}
        Variability statistic.

    Returns
    -------
    variability : pd.DataFrame
        Region x region matrix of edge variability.
    """

    if len(graph_cache) == 0:
        raise ValueError("graph_cache is empty.")

    if region_order is None:
        first_key = sorted(graph_cache.keys())[0]
        region_order = list(graph_cache[first_key].index)

    # Stack graphs into array: time x rows x cols
    dates = sorted(graph_cache.keys())
    mats = []

    for d in dates:
        G = graph_cache[d].reindex(index=region_order, columns=region_order)
        mats.append(G.to_numpy())

    arr = np.stack(mats, axis=0)

    if stat == "std":
        vals = np.nanstd(arr, axis=0)
    elif stat == "range":
        vals = np.nanmax(arr, axis=0) - np.nanmin(arr, axis=0)
    elif stat == "iqr":
        vals = np.nanpercentile(arr, 75, axis=0) - np.nanpercentile(arr, 25, axis=0)
    else:
        raise ValueError("stat must be 'std', 'range', or 'iqr'.")

    variability = pd.DataFrame(vals, index=region_order, columns=region_order)
    return variability


def plot_graph_edge_variability(
    graph_cache,
    T_window, 
    horizon,
    region_order=None,
    stat="std",
    figsize=(6, 5),
    annot=True,
):
    """
    Plot heatmap of dynamic graph edge variability over time.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    variability = graph_edge_variability(
        graph_cache,
        region_order=region_order,
        stat=stat,
    )

    plt.figure(figsize=figsize)

    sns.heatmap(
        variability,
        annot=annot,
        fmt=".3f",
        cmap="magma",
        square=True,
        cbar_kws={"label": f"Edge-weight {stat} over time"},
    )

    plt.title(f"Dynamic graph edge variability over test period ({stat})")
    plt.xlabel("Borrow from region")
    plt.ylabel("Test region")
    plt.tight_layout()
    plt.savefig(f'/home/mfholth/subseasonal/weekly_data/conformal_UQ/conformal_code/outputs/plotting/plots/graph_var_TWIN{T_window}_h{horizon}.png')
    plt.show()

    return variability

def compute_graph_row_entropy(graph_cache, region_order=None, normalize=True):
    """
    Compute row entropy of each G_t row over time.

    Parameters
    ----------
    graph_cache : dict
        Maps forecast_date -> G_t DataFrame.
    region_order : list or None
        Region ordering.
    normalize : bool
        If True, divide entropy by log(number of regions), so entropy is in [0, 1].

    Returns
    -------
    entropy_df : pd.DataFrame
        Columns: forecast_date, Region, entropy
    """

    if len(graph_cache) == 0:
        raise ValueError("graph_cache is empty.")

    if region_order is None:
        first_key = sorted(graph_cache.keys())[0]
        region_order = list(graph_cache[first_key].index)

    rows = []
    max_entropy = np.log(len(region_order))

    for date in sorted(graph_cache.keys()):
        G = graph_cache[date].reindex(index=region_order, columns=region_order)

        for region in region_order:
            p = G.loc[region].to_numpy(dtype=float)

            # avoid log(0)
            p_safe = np.where(p > 0, p, 1.0)
            entropy = -np.sum(np.where(p > 0, p * np.log(p_safe), 0.0))

            if normalize:
                entropy = entropy / max_entropy

            rows.append({
                "forecast_date": date,
                "Region": region,
                "entropy": entropy,
            })

    return pd.DataFrame(rows)


def plot_graph_row_entropy(
    graph_cache,
    T_window,
    horizon,
    graph_df=None,
    region_order=None,
    normalize=True,
    figsize=(12, 5),
):
    """
    Plot graph row entropy over time by region.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    entropy_df = compute_graph_row_entropy(
        graph_cache,
        region_order=region_order,
        normalize=normalize,
    )

    # Optionally merge year/week/season metadata
    if graph_df is not None:
        meta_cols = [c for c in ["forecast_date", "year", "week_num", "Season"] if c in graph_df.columns]
        entropy_df = entropy_df.merge(
            graph_df[meta_cols].drop_duplicates("forecast_date"),
            on="forecast_date",
            how="left",
        )

    plt.figure(figsize=figsize)

    for region in region_order or sorted(entropy_df["Region"].unique()):
        sub = entropy_df[entropy_df["Region"] == region].sort_values("forecast_date")
        x = np.arange(len(sub))

        plt.plot(
            x,
            sub["entropy"],
            linewidth=2,
            label=region,
        )

    ylabel = "Normalized row entropy" if normalize else "Row entropy"
    plt.ylabel(ylabel)
    plt.xlabel("Test timestep")
    plt.title("Dynamic graph row entropy over test period")
    plt.grid(alpha=0.3)
    plt.legend(title="Region", ncol=5)
    plt.tight_layout()
    plt.savefig(f'/home/mfholth/subseasonal/weekly_data/conformal_UQ/conformal_code/outputs/plotting/plots/graph_entropy_TWIN{T_window}_h{horizon}.png')
    plt.show()

    return entropy_df

def plot_graph_entropy_by_season(entropy_df, T_window, horizon, region_order=None):
    """
    Boxplot of row entropy by region and season.
    Requires entropy_df to contain Season.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    if "Season" not in entropy_df.columns:
        raise ValueError("entropy_df must contain a Season column. Pass graph_df to plot_graph_row_entropy first.")

    plt.figure(figsize=(10, 5))

    sns.boxplot(
        data=entropy_df,
        x="Region",
        y="entropy",
        hue="Season",
        order=region_order,
    )

    plt.title("Dynamic graph row entropy by region and season")
    plt.ylabel("Normalized row entropy")
    plt.xlabel("Region")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'/home/mfholth/subseasonal/weekly_data/conformal_UQ/conformal_code/outputs/plotting/plots/seas_entropy_TWIN{T_window}_h{horizon}.png')
    plt.show()


def extract_graph_row_over_time(
    graph_cache,
    *,
    target_region="W",
    region_order=None,
    graph_df=None,
    forecast_date_col="forecast_date",
):
    """
    Extract G_t(target_region, :) over time.

    Parameters
    ----------
    graph_cache : dict
        Maps forecast_date -> G_t DataFrame.
    target_region : str
        Row of the graph to extract, e.g. "W".
    region_order : list or None
        Column order for borrowed-from regions.
    graph_df : pd.DataFrame or None
        Optional metadata dataframe containing forecast_date, year, week_num, Season.
    forecast_date_col : str
        Forecast date column name.

    Returns
    -------
    row_df : pd.DataFrame
        One row per graph date, with columns for borrowed-from regions.
    """

    if len(graph_cache) == 0:
        raise ValueError("graph_cache is empty.")

    if region_order is None:
        first_key = sorted(graph_cache.keys())[0]
        region_order = list(graph_cache[first_key].columns)

    rows = []

    for date in sorted(graph_cache.keys()):
        G = graph_cache[date].reindex(index=region_order, columns=region_order)

        if target_region not in G.index:
            continue

        row = {
            forecast_date_col: date,
            "target_region": target_region,
        }

        for borrow_region in region_order:
            row[borrow_region] = G.loc[target_region, borrow_region]

        rows.append(row)

    row_df = pd.DataFrame(rows)

    if graph_df is not None:
        meta_cols = [
            c for c in [forecast_date_col, "year", "week_num", "Season"]
            if c in graph_df.columns
        ]

        row_df = row_df.merge(
            graph_df[meta_cols].drop_duplicates(subset=[forecast_date_col]),
            on=forecast_date_col,
            how="left",
        )

    return row_df

def plot_graph_row_weights_over_time(
    row_df,
    T_window,
    horizon,
    target_region="W",
    region_order=None,
    forecast_date_col="forecast_date",
    use_timestep=True,
    figsize=(14, 5),
):
    """
    Plot graph row weights over time as lines.
    """

    if region_order is None:
        exclude = {forecast_date_col, "target_region", "year", "week_num", "Season"}
        region_order = [c for c in row_df.columns if c not in exclude]

    plot_df = row_df.sort_values(forecast_date_col).reset_index(drop=True)

    if use_timestep:
        x = np.arange(len(plot_df))
        xlabel = "Test timestep"
    else:
        x = plot_df[forecast_date_col]
        xlabel = forecast_date_col

    plt.figure(figsize=figsize)

    for borrow_region in region_order:
        plt.plot(
            x,
            plot_df[borrow_region],
            linewidth=2,
            label=f"Borrow from {borrow_region}",
        )

    plt.title(f"Dynamic graph row over time: test region = {target_region}")
    plt.xlabel(xlabel)
    plt.ylabel("Graph weight")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(f'/home/mfholth/subseasonal/weekly_data/conformal_UQ/conformal_code/outputs/plotting/plots/row_weights_{T_window}_h{horizon}.png')
    plt.show()


def add_season_background(ax, plot_df, *, x_values, season_col="Season"):
    """
    Add light background shading for seasons.
    """

    if season_col not in plot_df.columns:
        return

    season_blocks = []
    current_season = None
    start_idx = None

    seasons = plot_df[season_col].to_numpy()

    for i, season in enumerate(seasons):
        if season != current_season:
            if current_season is not None:
                season_blocks.append((start_idx, i - 1, current_season))
            current_season = season
            start_idx = i

    season_blocks.append((start_idx, len(seasons) - 1, current_season))

    for start, end, season in season_blocks:
        if season == "Winter":
            alpha = 0.08
        elif season == "Spring":
            alpha = 0.05
        elif season == "Summer":
            alpha = 0.03
        elif season == "Fall":
            alpha = 0.06
        else:
            alpha = 0.03

        ax.axvspan(
            x_values[start],
            x_values[end],
            alpha=alpha,
            color="gray",
            linewidth=0,
        )


def plot_graph_row_weights_over_time_with_seasons(
    row_df,
    T_window,
    horizon,
    target_region="W",
    region_order=None,
    forecast_date_col="forecast_date",
    use_timestep=True,
    figsize=(14, 5),
):
    """
    Line plot of graph row weights over time with seasonal background shading.
    """

    if region_order is None:
        exclude = {forecast_date_col, "target_region", "year", "week_num", "Season"}
        region_order = [c for c in row_df.columns if c not in exclude]

    plot_df = row_df.sort_values(forecast_date_col).reset_index(drop=True)

    if use_timestep:
        x = np.arange(len(plot_df))
        xlabel = "Test timestep"
    else:
        x = plot_df[forecast_date_col].to_numpy()
        xlabel = forecast_date_col

    fig, ax = plt.subplots(figsize=figsize)

    add_season_background(ax, plot_df, x_values=x, season_col="Season")

    for borrow_region in region_order:
        ax.plot(
            x,
            plot_df[borrow_region],
            linewidth=2,
            label=f"Borrow from {borrow_region}",
        )

    ax.set_title(f"Dynamic graph row over time: test region = {target_region}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Graph weight")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(f'/home/mfholth/subseasonal/weekly_data/conformal_UQ/conformal_code/outputs/plotting/plots/row_weights_over_time_{T_window}_h{horizon}.png')
    plt.show()

def summarize_graph_row_by_season(
    row_df,
    *,
    region_order=None,
    season_col="Season",
    forecast_date_col="forecast_date",
):
    """
    Summarize average graph row weights by season.
    """

    if region_order is None:
        exclude = {forecast_date_col, "target_region", "year", "week_num", season_col}
        region_order = [c for c in row_df.columns if c not in exclude]

    summary = (
        row_df.groupby(season_col, observed=False)[region_order]
        .mean()
        .round(3)
    )

    return summary

