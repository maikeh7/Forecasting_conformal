import pandas as pd
import numpy as np
import os
import argparse
import sys
from pathlib import Path
from hetgpy import hetGP
from conformal.config import DETREND_OUTPUTS, REGIONS
from conformal.methods.method_helpers import add_season

    
def add_week_sincos(df, week_col="week_num"):
    df = df.copy()
    df["week_sin"] = np.sin(2 * np.pi * df[week_col] / 52.0)
    df["week_cos"] = np.cos(2 * np.pi * df[week_col] / 52.0)
    return df

# train GP on 1980-2016, eval on 2017-2020
def load_process_data(train_years):
    split_key = "dev"
    resid_spec = DETREND_OUTPUTS[split_key]
    df_train = pd.read_csv(resid_spec["train_residuals_file"])
    out = df_train.copy()
    fd = out["Date"].astype(int)
    out["year"] = (fd // 100).astype(int)
    out["week_num"] = (fd % 100).astype(int).clip(1, 53)
    
    out = add_week_sincos(out)
    out = out[out["year"].isin(train_years)].copy()
    return out 


'''
Load in seasonal + linear preds on dev test set from detrending model
These are on the Kelvin scale
Necessary for backtransforming preds from hetGP models

# 
def load_test_data(test_years):
    split_key="final"
    resid_spec = DETREND_OUTPUTS[split_key]
    train_preds = pd.read_csv(resid_spec["test_preds_file"])
    out = train_preds.copy()
    fd = out["Date"].astype(int)
    out["year"] = (fd // 100).astype(int)
    out["week_num"] = (fd % 100).astype(int).clip(1, 53)
    out = out[out["year"].isin(test_years)].copy()
    out_long = pd.melt(out, id_vars=['Date', 'week_num', 'year'], var_name='Region', value_name="trend_pred_K")
    return out_long
'''


def fit_hetgp_by_region(
    df_train,
    region_col="Region",
    week_col="week_num",
    y_col="y_detrended",
    covtype="Gaussian",
):
    #df_train = add_week_sincos(df_train, week_col)
    models = {}

    for region in REGIONS:
        sub = df_train[["week_sin", "week_cos", region]]
        sub = sub.dropna(subset=["week_sin", "week_cos", region])

        X = sub[["week_sin", "week_cos"]].to_numpy()
        Z = sub[region].to_numpy()

        gp = hetGP()
        gp.mle(
            X=X,
            Z=Z,
            covtype=covtype,
            lower=np.array([0.05, 0.05]),
            upper=np.array([5.0, 5.0]),
        )

        models[region] = gp
        print(f"\nRegion: {region}")

    return models

def predict_hetgp_var(df, models, region_col="Region", week_col="week_num"):
    df = add_week_sincos(df, week_col).copy()
    df["hetgp_mean"] = np.nan
    df["hetgp_lower90"] = np.nan
    df["hetgp_upper90"] = np.nan
    df["hetgp_predvar"] = np.nan
    
    for region, gp in models.items():
        mask = df[region_col] == region
        if mask.sum() == 0:
            continue

        Xnew = df.loc[mask, ["week_sin", "week_cos"]].to_numpy()

        preds = gp.predict(
            x=Xnew,
            interval="predictive",
            interval_lower=0.05,
            interval_upper=0.95,
        )
        preds2 = gp.predict(x=Xnew)
        my_var = preds2['sd2'] + preds2['nugs']

        df.loc[mask, "hetgp_mean"] = preds["mean"]
        df.loc[mask, "hetgp_lower90"] = preds["predictive_interval"]["lower"]
        df.loc[mask, "hetgp_upper90"] = preds["predictive_interval"]["upper"]
        df.loc[mask, "hetgp_predvar"] = my_var 

    return df

def run_hetGP_fitting(train_years, test_years):
    '''
    Returns a dataframe containing hetGP climatological preds from models trained on train_years (~2017-2019)
    and evaluated on test_year. The Kelvin trend is taken from linear+seasonal model preds on test_year
    Returned dataframe contains these columns:
    'Date', 'week_num', 'year', 'Region', 'trend_pred_K', 'hetgp_mean',
    'hetgp_lower90', 'hetgp_upper90', 'hetgp_predvar', 'hetgp_predSD',
    'year', 'hetgp_mean_kelvin', 'hetgp_lower90_kelvin',
    'hetgp_upper90_kelvin'
    '''
    # train on 1980-2016 of detrended temp data
    df_train = load_process_data(train_years)

    # fit hetGP models
    hetgp_models = fit_hetgp_by_region(df_train)
    # make predictions on a seasonal grid
    week_grid = np.arange(1, 53)
    all_preds = []
    for region in REGIONS:
        grid = pd.DataFrame({
            "Region": region,
            "week_num": week_grid,
        })
    
        pred = predict_hetgp_var(grid, hetgp_models)
        all_preds.append(pred)
    pred_df = pd.concat(all_preds)
    pred_df = pred_df[["Region", "week_num", "hetgp_mean", "hetgp_lower90", "hetgp_upper90", "hetgp_predvar"]]
    pred_df["hetgp_predSD"] = np.sqrt(pred_df["hetgp_predvar"])

    # We need the kelvin predicted trend from this dataset
    #test_preds_long = load_test_data(test_years)
    # merge these 2 datasets together so we can backtransform hetGP preds/var
    #test_preds_long = pd.merge(test_preds_long, pred_df, on=["Region", "week_num"], how="left")
    
    #test_preds_long['hetgp_mean_kelvin'] = test_preds_long['hetgp_mean'] + test_preds_long['trend_pred_K']
    #test_preds_long['hetgp_lower90_kelvin'] = test_preds_long['hetgp_lower90'] + test_preds_long['trend_pred_K'] 
    #test_preds_long['hetgp_upper90_kelvin'] = test_preds_long['hetgp_upper90'] + test_preds_long['trend_pred_K'] 

    return pred_df
    

    