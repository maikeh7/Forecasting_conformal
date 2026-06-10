# conformal/features

ALPHA = 0.10
Y_TRUE_COL = "y_true_kelvin"
Y_PRED_COL = "y_pred_kelvin"
FORECAST_DATE_COL = "forecast_date"
CURRENT_WEEK_COL = "current_week"
Y_TREND_COL = "trend_pred_K"

def union_allowed_features_from_strings(df, horizon, regions=None, feature_col="Feature_names", sep=";"):
    """
    Returns sorted union of features for a given horizon across selected regions.

    df: DataFrame with columns ['Region','Horizon', feature_col]
    horizon: int (1-4)
    regions: optional list of regions to include (default: all in df for that horizon)
    """
    sub = df[df["Horizon"] == horizon].copy()
    if regions is not None:
        sub = sub[sub["Region"].isin(regions)]

    features = set()
    for s in sub[feature_col].dropna().astype(str):
        # split, strip whitespace, ignore empties
        for f in s.split(sep):
            f = f.strip()
            if f:
                features.add(f)

    return sorted(features)

# gets allowed features for combo of region-horizon
def union_allowed_features(importance_df, horizon, regions):
    feats = set()
    for r in regions:
        r_feats = importance_df[
            (importance_df["Region"] == r) & (importance_df["Horizon"] == horizon)
        ]["Feature"].tolist()
        feats.update(r_feats)
    return sorted(feats)
    
def rmse(a, b):
    return np.sqrt(mean_squared_error(a, b))