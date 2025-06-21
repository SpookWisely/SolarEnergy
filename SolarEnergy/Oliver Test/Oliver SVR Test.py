#----# Libraries
import array
from ast import Str
from calendar import c
from ctypes import Array
from enum import Enum
from tkinter import CURRENT
from tokenize import String
import numpy as np
import pandas as pd
from pandas.core.indexes import multi
import shutil
import os
import scipy as sp
import xgboost as xgb
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from keras_tuner import RandomSearch, Hyperband, BayesianOptimization
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from deap import base, creator, tools, algorithms
import random
import multiprocessing
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#Olivers Results from his code file -
#MSE: 628.8253
#MAE: 12.3056
#RMSE: 25.0764
#R² Score: 0.9527

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
"""
Random seed for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
"""
oliver_Demand = pd.read_excel(r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\Oliver Test\Sakakah 2021 Demand dataset.xlsx")
oliver_Supply = pd.read_excel(r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\Oliver Test\Sakakah 2021 PV supply dataset.xlsx")
oliver_DemandWeather = pd.read_csv(r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\Oliver Test\Sakakah 2021 weather dataset Demand.csv",skiprows=16)
oliver_SupplyWeather = pd.read_csv(r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\Oliver Test\Sakakah 2021 weather dataset.csv",skiprows=16)
harry_WeatherDemand = pd.read_excel(r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\Oliver Test\Weather for demand 2018 - NEW.xlsx")
harry_WeatherSupply = pd.read_excel(r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\Oliver Test\weather for solar NEW 2021.xlsx")

oliver_Demand["Timestamp"] = pd.to_datetime(oliver_Demand["DATE-TIME"])
oliver_Supply["Timestamp"] = pd.to_datetime(oliver_Supply["Date & Time"])
oliver_DemandWeather["Timestamp"] = pd.to_datetime(
    oliver_DemandWeather[['YEAR', 'MO', 'DY', 'HR']].astype(str).agg('-'.join, axis=1),
    format="%Y-%m-%d-%H"
)
oliver_SupplyWeather["Timestamp"] = pd.to_datetime(
    oliver_SupplyWeather[['YEAR', 'MO', 'DY', 'HR']].astype(str).agg('-'.join, axis=1),
    format="%Y-%m-%d-%H"
)
harry_WeatherDemand["Timestamp"] = pd.to_datetime(
    harry_WeatherDemand[['YEAR', 'MO', 'DY', 'HR']].astype(str).agg('-'.join, axis=1),
    format="%Y-%m-%d-%H"
)
harry_WeatherSupply["Timestamp"] = pd.to_datetime(
    harry_WeatherSupply[['YEAR', 'MO', 'DY', 'HR']].astype(str).agg('-'.join, axis=1),
    format="%Y-%m-%d-%H"
)
# Drop unnecessary columns
oliver_Demand.drop(columns=["DATE-TIME"], inplace=True)
oliver_Supply.drop(columns=["Date & Time"], inplace=True)
oliver_DemandWeather.drop(columns=['YEAR', 'MO', 'DY', 'HR'], inplace=True)
oliver_SupplyWeather.drop(columns=['YEAR', 'MO', 'DY', 'HR'], inplace=True)
harry_WeatherDemand.drop(columns=['YEAR', 'MO', 'DY', 'HR'], inplace=True)
harry_WeatherSupply.drop(columns=['YEAR', 'MO', 'DY', 'HR'], inplace=True)


# Rename columns
oliver_Demand.rename(columns={"MW": "MW_demand"}, inplace=True)
oliver_Supply.rename(columns={"MW": "MW_supply"}, inplace=True)

mergedDs = oliver_Demand.merge(oliver_Supply, on="Timestamp").merge(oliver_SupplyWeather, on="Timestamp").merge(oliver_DemandWeather, on="Timestamp")
mergedNewDs = oliver_Demand.merge(oliver_Supply, on="Timestamp").merge(harry_WeatherSupply, on="Timestamp").merge(harry_WeatherDemand, on="Timestamp")
mergedNewDs.replace(-999, np.nan, inplace=True)
mergedNewDs.interpolate(method="linear", inplace=True)
mergedNewDs.dropna(inplace=True)
mergedDs.replace(-999, np.nan, inplace=True)
mergedDs.interpolate(method="linear", inplace=True)
mergedDs.dropna(inplace=True)

"""
This is the same as oliver has done his feature only difference is I used S and D insteaad of X and Y as the 
value to differentiate between the supply and demand data.
"ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","ALLSKY_KT_S","T2M_S","QV2M_S","PS_S","WS10M_S","WD10M_S",
"ALLSKY_SFC_SW_DWN_D","ALLSKY_KT_D","ALLSKY_SFC_UV_INDEX_D","T2M_D","QV2M_D","PS_D","WS10M_D","WD10M_D"
"""
print(mergedDs.head())

def create_sequence_with_time(df, feature_columns, target_columns, window_size=24):
    """
    Creates sequences of features and targets for time series models.
    Each X[i] contains the features for window_size time steps,
    and y[i] contains the target at the next time step.
    """
    X, y = [], []
    for i in range(len(df) - window_size):
        X.append(df[feature_columns].iloc[i:i+window_size].values.flatten())
        y.append(df[target_columns].iloc[i+window_size].values)
    return np.array(X), np.array(y)


def Oliver_SVR(merged:pd.DataFrame, id:int):
    print("\n","Entering Default Method")
    merged = merged.copy()

    merged["hour"] = merged["Timestamp"].dt.hour
    merged["day"] = merged["Timestamp"].dt.day
    merged["month"] = merged["Timestamp"].dt.month
    if id == 1: 
        feature_columns = [
        "ALLSKY_SFC_SW_DWN_S", "ALLSKY_SFC_UV_INDEX_S", "ALLSKY_KT_S", "T2M_S", "QV2M_S", "PS_S", "WS10M_S", "WD10M_S",
        "ALLSKY_SFC_SW_DWN_D", "ALLSKY_KT_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "QV2M_D", "PS_D", "WS10M_D", "WD10M_D","hour", "day", "month"
        ]
    elif id == 2:
        feature_columns = [
            "ALLSKY_SFC_SW_DWN_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "PRECTOTCORR_D", "ALLSKY_KT_D", "CLRSKY_SFC_PAR_TOT_D", "RH2M_D", "PS_D", "PSC_D", "WS10M_D", "WD10M_D",
            "ALLSKY_SFC_SW_DWN_S", "ALLSKY_SFC_UV_INDEX_S", "T2M_S", "PRECTOTCORR_S", "ALLSKY_KT_S", "CLRSKY_SFC_PAR_TOT_S", "RH2M_S", "PS_S", "PSC_S", "WS10M_S", "WD10M_S", "hour", "day", "month"
        ]
    else:
        print("Invalid ID provided. Please use 1 or 2.")
        return

    target_cols = ["MW_demand", "MW_supply"]
                    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X = scaler_X.fit_transform(merged[feature_columns])
    y = scaler_y.fit_transform(merged[target_cols])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    svr_base = SVR(kernel='rbf', C=100, epsilon=0.01)
    model = MultiOutputRegressor(svr_base)
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)

    X = scaler_X.fit_transform(merged[feature_columns])
    y = scaler_y.fit_transform(merged[target_cols])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    svr_base = SVR(kernel='rbf', C=100, epsilon=0.01)
    model = MultiOutputRegressor(svr_base)
    model.fit(X_train, y_train)

    # Inverse transform
    y_test_inv = scaler_y.inverse_transform(y_test)
    y_pred_inv = scaler_y.inverse_transform(y_pred)

    # Evaluation
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_inv, y_pred_inv)
    label = "Default Oliver code"
    results = [mse, mae, rmse, r2,label]
    return results

def Oliver_SVR_HyP(merged:pd.DataFrame, id:int):
    print("\n","Entering Method with Hyper Params")
    smerged = merged.copy()

    merged["hour"] = merged["Timestamp"].dt.hour
    merged["day"] = merged["Timestamp"].dt.day
    merged["month"] = merged["Timestamp"].dt.month
    if id == 1: 
        feature_columns = [
        "ALLSKY_SFC_SW_DWN_S", "ALLSKY_SFC_UV_INDEX_S", "ALLSKY_KT_S", "T2M_S", "QV2M_S", "PS_S", "WS10M_S", "WD10M_S",
        "ALLSKY_SFC_SW_DWN_D", "ALLSKY_KT_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "QV2M_D", "PS_D", "WS10M_D", "WD10M_D","hour", "day", "month"
        ]
    elif id == 2:
        feature_columns = [
            "ALLSKY_SFC_SW_DWN_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "PRECTOTCORR_D", "ALLSKY_KT_D", "CLRSKY_SFC_PAR_TOT_D", "RH2M_D", "PS_D", "PSC_D", "WS10M_D", "WD10M_D",
            "ALLSKY_SFC_SW_DWN_S", "ALLSKY_SFC_UV_INDEX_S", "T2M_S", "PRECTOTCORR_S", "ALLSKY_KT_S", "CLRSKY_SFC_PAR_TOT_S", "RH2M_S", "PS_S", "PSC_S", "WS10M_S", "WD10M_S", "hour", "day", "month"
        ]
    else:
        print("Invalid ID provided. Please use 1 or 2.")
        return
    target_cols = ["MW_demand", "MW_supply"]
                    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()


    X = scaler_X.fit_transform(merged[feature_columns])
    y = scaler_y.fit_transform(merged[target_cols])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'estimator__C': [1, 10, 100],
        'estimator__epsilon': [0.01, 0.1, 0.5],
        'estimator__kernel': ['rbf', 'linear']
    }

    svr_base = SVR()
    model = MultiOutputRegressor(svr_base)

    grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Inverse transform
    y_test_inv = scaler_y.inverse_transform(y_test)
    y_pred_inv = scaler_y.inverse_transform(y_pred)

    # Evaluation
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_inv, y_pred_inv)
    label = "SVR GridSearch HyP"
    results = [mse, mae, rmse, r2, label]
    return results

def Oliver_SVR_Lag(merged:pd.DataFrame, id:int):
    print("\n","Entering Method With 24 Lag")
    merged = merged.copy()

    merged["hour"] = merged["Timestamp"].dt.hour
    merged["day"] = merged["Timestamp"].dt.day
    merged["month"] = merged["Timestamp"].dt.month
    if id == 1: 
        feature_columns = [
        "ALLSKY_SFC_SW_DWN_S", "ALLSKY_SFC_UV_INDEX_S", "ALLSKY_KT_S", "T2M_S", "QV2M_S", "PS_S", "WS10M_S", "WD10M_S",
        "ALLSKY_SFC_SW_DWN_D", "ALLSKY_KT_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "QV2M_D", "PS_D", "WS10M_D", "WD10M_D","hour", "day", "month"
        ]
    elif id == 2:
        feature_columns = [
            "ALLSKY_SFC_SW_DWN_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "PRECTOTCORR_D", "ALLSKY_KT_D", "CLRSKY_SFC_PAR_TOT_D", "RH2M_D", "PS_D", "PSC_D", "WS10M_D", "WD10M_D",
            "ALLSKY_SFC_SW_DWN_S", "ALLSKY_SFC_UV_INDEX_S", "T2M_S", "PRECTOTCORR_S", "ALLSKY_KT_S", "CLRSKY_SFC_PAR_TOT_S", "RH2M_S", "PS_S", "PSC_S", "WS10M_S", "WD10M_S", "hour", "day", "month"
        ]
    else:
        print("Invalid ID provided. Please use 1 or 2.")
        return
    target_cols = ["MW_demand", "MW_supply"]

    # Create lagged sequences
    window_size = 24
    X, y = create_sequence_with_time(merged, feature_columns, target_cols, window_size=window_size)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    svr_base = SVR(kernel='rbf', C=100, epsilon=0.01)
    model = MultiOutputRegressor(svr_base)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Inverse transform
    y_test_inv = scaler_y.inverse_transform(y_test)
    y_pred_inv = scaler_y.inverse_transform(y_pred)

    # Evaluation
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_inv, y_pred_inv)
    label = "SVR 24hr Lag"
    results = [mse, mae, rmse, r2, label]
    return results

def Oliver_SVR_LagandHyp(merged:pd.DataFrame, id:int):
    print("\n","Entering Method with Both 24/Lag and Hyper Params")
    merged = merged.copy()

    merged["hour"] = merged["Timestamp"].dt.hour
    merged["day"] = merged["Timestamp"].dt.day
    merged["month"] = merged["Timestamp"].dt.month
    if id == 1: 
        feature_columns = [
        "ALLSKY_SFC_SW_DWN_S", "ALLSKY_SFC_UV_INDEX_S", "ALLSKY_KT_S", "T2M_S", "QV2M_S", "PS_S", "WS10M_S", "WD10M_S",
        "ALLSKY_SFC_SW_DWN_D", "ALLSKY_KT_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "QV2M_D", "PS_D", "WS10M_D", "WD10M_D","hour", "day", "month"
        ]
    elif id == 2:
        feature_columns = [
            "ALLSKY_SFC_SW_DWN_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "PRECTOTCORR_D", "ALLSKY_KT_D", "CLRSKY_SFC_PAR_TOT_D", "RH2M_D", "PS_D", "PSC_D", "WS10M_D", "WD10M_D",
            "ALLSKY_SFC_SW_DWN_S", "ALLSKY_SFC_UV_INDEX_S", "T2M_S", "PRECTOTCORR_S", "ALLSKY_KT_S", "CLRSKY_SFC_PAR_TOT_S", "RH2M_S", "PS_S", "PSC_S", "WS10M_S", "WD10M_S", "hour", "day", "month"
        ]
    else:
        print("Invalid ID provided. Please use 1 or 2.")
        return
    target_cols = ["MW_demand", "MW_supply"]

    # Create lagged sequences
    window_size = 24
    X, y = create_sequence_with_time(merged, feature_columns, target_cols, window_size=window_size)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
    param_grid = {
        'estimator__C': [1],
        'estimator__epsilon': [0.01],
        'estimator__kernel': ['rbf']
    }
    """
    Best SVR parameters (24hr Lag & HyP): {'estimator__C': 1, 'estimator__epsilon': 0.01, 'estimator__kernel': 'rbf'}

    param_grid = {
        'estimator__C': [1, 10, 100],
        'estimator__epsilon': [0.01, 0.1, 0.5],
        'estimator__kernel': ['rbf', 'linear']
    }
    """
    svr_base = SVR()
    model = MultiOutputRegressor(svr_base)

    grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    print("Best SVR parameters (24hr Lag & HyP):", grid_search.best_params_)

    best_model = grid_search.best_estimator_
   
    y_pred = best_model.predict(X_test)

    # Inverse transform
    y_test_inv = scaler_y.inverse_transform(y_test)
    y_pred_inv = scaler_y.inverse_transform(y_pred)

    # Evaluation
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_inv, y_pred_inv)
    label = "SVR 24hr Lag & HyP"
    results = [mse, mae, rmse, r2, label]
    return results
def BetterModelSelectionMethod(ModelArray: list):
    """
    Orders the models from best to worst based on MSE, MAE, RMSE, and R2.
    Lower MSE, MAE, RMSE are better; higher R2 is better.
    """
    def compare_results(res1, res2):
        score1 = 0
        score2 = 0
        # MSE
        if res1[0] < res2[0]:
            score1 += 1
        else:
            score2 += 1
        # MAE
        if res1[1] < res2[1]:
            score1 += 1
        else:
            score2 += 1
        # RMSE
        if res1[2] < res2[2]:
            score1 += 1
        else:
            score2 += 1
        # R2
        r2OneScore = 1 - res1[3]
        r2TwoScore = 1 - res2[3]
        if res1[3] < 0 and res2[3] >= 0:
            score2 += 1
        elif res2[3] < 0 and res1[3] >= 0:
            score1 += 1
        elif res1[3] >= 0 and res2[3] >= 0:
            if r2OneScore > r2TwoScore:
                score1 += 1
            else:
                score2 += 1
        return score1 > score2

    # Sort using a nested loop (selection sort style)
    n = len(ModelArray)
    ordered = ModelArray.copy()
    for i in range(n):
        best_idx = i
        for j in range(i + 1, n):
            if compare_results(ordered[j], ordered[best_idx]):
                best_idx = j
        if best_idx != i:
            ordered[i], ordered[best_idx] = ordered[best_idx], ordered[i]
    return ordered

MSE = 628.8253
MAE = 12.3056
RMSE = 25.0764
R2Score =  0.9527
label = "Prior Copied Results"
copiedResults = [MSE,MAE,RMSE,R2Score,label]
oliverDefault = Oliver_SVR(mergedDs,1)
oliverWithHyP = Oliver_SVR_HyP(mergedDs,1)
oliverWith24Lag = Oliver_SVR_Lag(mergedDs,1)
oliverWithHypandLag = Oliver_SVR_LagandHyp(mergedDs,1)

bestSVRResultsOrdered = BetterModelSelectionMethod([
    oliverDefault,
    oliverWithHyP,
    oliverWith24Lag,
    oliverWithHypandLag,
    copiedResults
])
    
harryDefault = Oliver_SVR(mergedNewDs,2)
harryWithHyP = Oliver_SVR_HyP(mergedNewDs,2)
harryWith24Lag = Oliver_SVR_Lag(mergedNewDs,2)
harryWithHypandLag = Oliver_SVR_LagandHyp(mergedNewDs,2)
bestSVRResultsOrderedHarry = BetterModelSelectionMethod([
    harryDefault,
    harryWithHyP,
    harryWith24Lag,
    harryWithHypandLag,
    
])
print("\nSVR Model Ranking Best to Worst - Olivers Dataset:")
print("{:<20} {:>12} {:>12} {:>12} {:>10}".format("Model", "MSE", "MAE", "RMSE", "R2"))
print("-" * 70)
for res in bestSVRResultsOrdered:
    print("{:<20} {:>12.4f} {:>12.4f} {:>12.4f} {:>10.4f}".format(
        res[4], res[0], res[1], res[2], res[3]
    ))

print("\nSVR Model Ranking Best to Worst - Harrys Dataset")
print("{:<20} {:>12} {:>12} {:>12} {:>10}".format("Model", "MSE", "MAE", "RMSE", "R2"))
print("-" * 70)
for res in bestSVRResultsOrderedHarry:
    print("{:<20} {:>12.4f} {:>12.4f} {:>12.4f} {:>10.4f}".format(
        res[4], res[0], res[1], res[2], res[3]
    ))