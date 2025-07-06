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
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, GRU, Conv1D, Flatten, MaxPooling1D,MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
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
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
##Questions to ask Tomorrow
"""
1. Should I look into adding lag into the supply dataset aspects of the method as its results for MSE specifically are wildly different
  to that of Demand dataset results.
2. Why when comparing the creation process for the merged dataset in olivers code does he
duplicate the weather data
with ALLSKY_SFC_SW_DWN ending up as ALLSKY_SFC_SW_DWN_x/ALLSKY_SFC_SW_DWN_y as an example.
"""

sp_DemandDef = pd.read_excel(r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\Sakakah 2021 Demand dataset.xlsx")
sp_SupplyDef = pd.read_excel(r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\Sakakah 2021 PV supply dataset.xlsx")
sp_WeatherDef = pd.read_excel(r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\weather for solar 2021.xlsx")
sp_WeatherDe = pd.read_excel(r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\Weather for demand 2018.xlsx")
sp_WeatherWind = pd.read_excel(
    r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\Sakakah 2021 weather dataset.xlsx",
    usecols=lambda col, c=pd.read_excel(
        r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\Sakakah 2021 weather dataset.xlsx", nrows=0
    ).columns: col in list(c)[-2:]
)
print(sp_WeatherWind.head())
##sp_WeatherDeDef = pd.read_excel
sp_DemandDef["TimeStamp"] = pd.to_datetime(sp_DemandDef["DATE-TIME"])
sp_SupplyDef["TimeStamp"] = pd.to_datetime(sp_SupplyDef["Date & Time"])
sp_WeatherDef["TimeStamp"] = pd.to_datetime(
    sp_WeatherDef[['YEAR', 'MO', 'DY', 'HR']].astype(str).agg('-'.join, axis=1),
    format="%Y-%m-%d-%H"
)
sp_WeatherDe["TimeStamp"] = pd.to_datetime(
    sp_WeatherDe[['YEAR', 'MO', 'DY', 'HR']].astype(str).agg('-'.join, axis=1),
    format="%Y-%m-%d-%H"
)
sp_DemandDef.rename(columns={"MW": "Demand_MW"}, inplace=True)
sp_SupplyDef.rename(columns={"MW": "Supply_MW"}, inplace=True)

sp_DemandDef.drop(columns=["DATE-TIME"], inplace=True)
sp_SupplyDef.drop(columns=["Date & Time"], inplace=True)
sp_WeatherDef.drop(columns=["YEAR", "MO", "DY", "HR"], inplace=True)

sp_FullMerg = pd.merge(sp_DemandDef, sp_WeatherDef, on="TimeStamp", how="inner")
sp_FullMerg = pd.merge(sp_SupplyDef, sp_FullMerg, on="TimeStamp", how="inner")
sp_FullMerg = pd.concat([sp_FullMerg, sp_WeatherWind.reset_index(drop=True)], axis=1)
print(sp_FullMerg.head(20))
##sp_FullMerg = pd.merge(sp_WeatherDe, sp_FullMerg, on="TimeStamp", how="inner")

##linear interpolation to smooth out the datasets
##So far has worked out great for supply but demand is getting bad results.
sp_FullMerg.interpolate(method='linear', inplace=True)
#print(sp_FullMerg.head())

#desave_loc = r"E:\AI Lecture Notes\Datasets\Merged\DemandxWeather.csv"
##susave_loc = r"E:\AI Lecture Notes\Datasets\Merged\SupplyxWeather.csv"
#sp_WeatherxDem.to_csv(desave_loc, index=False)
##sp_WeatherxSup.to_csv(susave_loc, index=False)
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
#------#

tuner_subdirs = [
    'tuner_dir/blstm_tuning',
    'tuner_dir/lstm_tuning',
    'tuner_dir/gru_tuning',
    'tuner_dir/mlp_tuning',
    'tuner_dir/cnn_tuning'
]

for subdir in tuner_subdirs:
    if os.path.exists(subdir):
        shutil.rmtree(subdir)

def create_sequences_with_time(data, targets, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length].flatten())
        y.append(targets[i + seq_length])
    return np.array(X), np.array(y)


def create_monthly_sequences(df, feature_cols, target_cols, pad_to_max=True):
    df = df.copy()
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    df['year'] = df['TimeStamp'].dt.year
    df['month'] = df['TimeStamp'].dt.month

    X, y = [], []
    grouped = df.groupby(['year', 'month'])
    months = sorted(grouped.groups.keys())

    max_len = max(len(grouped.get_group(m)[feature_cols]) for m in months[:-1]) if pad_to_max else None

    for i in range(len(months) - 1):
        this_month = grouped.get_group(months[i])
        next_month = grouped.get_group(months[i + 1])

        x_seq = this_month[feature_cols].values
        if pad_to_max and x_seq.shape[0] < max_len:
            pad_width = ((0, max_len - x_seq.shape[0]), (0, 0))
            x_seq = np.pad(x_seq, pad_width, mode='constant')
        X.append(x_seq)
        # Use mean for the next month as target
        y.append(next_month[target_cols].mean().values)

    X = np.array(X)
    y = np.array(y)
    return X, y

#----#

def decisionTreeModelDS(mergedDs: pd.DataFrame,plots:bool):
    mergedDs = mergedDs.copy()

    mergedDs.replace(-999, np.nan, inplace=True)
    mergedDs.dropna(inplace=True)

    ###minLength = min(len(demandDs))
    ###demandDs = demandDs.iloc[minLength]

    #Extraction of Data.

    mergedDs["hour"] = mergedDs["TimeStamp"].dt.hour
    mergedDs["day"] = mergedDs["TimeStamp"].dt.day
    mergedDs["month"] = mergedDs["TimeStamp"].dt.month
    """
    feature_cols = [
        "ALLSKY_SFC_SW_DWN_S", "ALLSKY_SFC_UV_INDEX_S", "T2M_S", "PRECTOTCORR_S", "ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S", "RH2M_S", "PS_S", "PSC_S", "ALLSKY_SFC_SW_DWN_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "PRECTOTCORR_D", "ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D", "RH2M_D", "PS_D", "PSC_D","WS10M","WD10M" "hour", "day", "month"
    ]
    """
    feature_cols = [
    "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
    "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","WS10M","WD10M","hour","day","month"
    ]
    """
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW', 'Supply_MW']
    X = scaler_X.fit_transform(mergedDs[feature_cols])
    y = scaler_y.fit_transform(mergedDs[target_cols])
    seq_length = 24
    X_seq, y_seq = create_sequences_with_time(X, y, seq_length)

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.3, shuffle=False
    )
    #X_seq, y_seq = create_seasonal_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
    #X_seq_flat = X_seq.reshape(X_seq.shape[0], -1)
    """
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW','Supply_MW']

    # Optionally scale the whole DataFrame first
    mergedDs[feature_cols] = scaler_X.fit_transform(mergedDs[feature_cols])
    mergedDs[target_cols] = scaler_y.fit_transform(mergedDs[target_cols])

    # Create monthly sequences
    X, y = create_monthly_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
    X = X.reshape(X.shape[0], -1)
    # Now split into train/test (e.g., 70% train, 30% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    mergedDs['year'] = mergedDs['TimeStamp'].dt.year
    mergedDs['month'] = mergedDs['TimeStamp'].dt.month
    months = sorted(mergedDs.groupby(['year', 'month']).groups.keys())
    month_labels = [f"{y}-{m:02d}" for (y, m) in months]

    months = sorted(mergedDs.groupby(['year', 'month']).groups.keys())
    month_labels = [f"{y}-{m:02d}" for (y, m) in months]
    split_idx = int(len(months) * 0.8)
    test_month_labels = month_labels[split_idx:]
 
    
    param_grid = {
        'estimator__max_depth': [3, 5, 10, None],
        'estimator__min_samples_split': [2, 5, 10],
        'estimator__min_samples_leaf': [1, 2, 4],
        'estimator__max_features': ['auto', 'sqrt', 'log2', None]
    }
    base_model = MultiOutputRegressor(DecisionTreeRegressor(random_state=42))
    grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print("Best parameters:", "\n", grid_search.best_params_)
    """
    base_model = MultiOutputRegressor(DecisionTreeRegressor(random_state=42))
    base_model.fit(X_train, y_train) 
    """
    y_pred = grid_search.predict(X_test)
    
    y_demand_pred = scaler_y.inverse_transform(y_pred)
    y_demand_true = scaler_y.inverse_transform(y_test)

    mse = mean_squared_error(y_demand_true, y_demand_pred)
    mae = mean_absolute_error(y_demand_true, y_demand_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_demand_true, y_demand_pred)
    modelName = "Decision Tree"
    results = [mse, mae, rmse, r2,modelName]
    #----#
    print('\n', "Merged Data Decision Tree Model Results:")
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))
    
     
    assert len(test_month_labels) == y_demand_true.shape[0] == y_demand_pred.shape[0]
    if plots == True:
        plt.figure(figsize=(12, 6))
        plt.plot(test_month_labels, y_demand_true[:, 0], label='Actual Demand', marker='o')
        plt.plot(test_month_labels, y_demand_pred[:, 0], label='Predicted Demand', marker='s')
        plt.title('Decision Tree Model: Actual vs Predicted Demand: Monthly')
        plt.xlabel('Month')
        plt.ylabel('MW')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.8)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(test_month_labels, y_demand_true[:, 1], label='Actual Supply', marker='o')
        plt.plot(test_month_labels, y_demand_pred[:, 1], label='Predicted Supply', marker='s')
        plt.title('Decision Tree Model: Actual vs Predicted Supply: Monthly')
        plt.xlabel('Month')
        plt.ylabel('MW')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.8)
        plt.legend()
        plt.tight_layout()
        plt.show()

        """
        #-Demand Plot-#
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true[:, 0], label='Actual Demand')
        plt.plot(y_demand_pred[:, 0], label='Predicted Demand')
        plt.title(f'Decision Tree Model: Actual vs Predicted Demand: Daily')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
        #-Supply Plot-#
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true[:, 1], label='Actual Supply')
        plt.plot(y_demand_pred[:, 1], label='Predicted Supply')
        plt.title(f'Decision Tree Model: Actual vs Predicted Supply: Daily')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
        """
    return results


def randomForestModelDS(mergedDs: pd.DataFrame,plots:bool):
    mergedDs = mergedDs.copy()


    mergedDs.replace(-999, np.nan, inplace=True)
    mergedDs.dropna(inplace=True)

    ###minLength = min(len(demandDs))
    ###demandDs = demandDs.iloc[minLength]

    #Extraction of Data.

    mergedDs["hour"] = mergedDs["TimeStamp"].dt.hour
    mergedDs["day"] = mergedDs["TimeStamp"].dt.day
    mergedDs["month"] = mergedDs["TimeStamp"].dt.month

    feature_cols = [
    "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
    "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","WS10M","WD10M","hour","day","month"
    ]
    """
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW', 'Supply_MW']
    X = scaler_X.fit_transform(mergedDs[feature_cols])
    y = scaler_y.fit_transform(mergedDs[target_cols])

    seq_length = 24
    X_seq, y_seq = create_sequences_with_time(X, y, seq_length)

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.3, shuffle=False
    )
    """
    #X_seq, y_seq = create_seasonal_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
    #X_seq_flat = X_seq.reshape(X_seq.shape[0], -1)

    """
    """
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW','Supply_MW']

    # Optionally scale the whole DataFrame first
    mergedDs[feature_cols] = scaler_X.fit_transform(mergedDs[feature_cols])
    mergedDs[target_cols] = scaler_y.fit_transform(mergedDs[target_cols])

    # Create monthly sequences
    X, y = create_monthly_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
    X = X.reshape(X.shape[0], -1)
    # Now split into train/test (e.g., 70% train, 30% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    mergedDs['year'] = mergedDs['TimeStamp'].dt.year
    mergedDs['month'] = mergedDs['TimeStamp'].dt.month
    months = sorted(mergedDs.groupby(['year', 'month']).groups.keys())
    month_labels = [f"{y}-{m:02d}" for (y, m) in months]

    months = sorted(mergedDs.groupby(['year', 'month']).groups.keys())
    month_labels = [f"{y}-{m:02d}" for (y, m) in months]
    split_idx = int(len(months) * 0.8)
    test_month_labels = month_labels[split_idx:]
    
    
    
    param_grid = {
        'estimator__n_estimators': [50],
        'estimator__max_depth': [10],
        'estimator__min_samples_split': [5],
        'estimator__min_samples_leaf': [2],
        'estimator__max_features': ['sqrt']
    }
    """
    param_grid = {
        'estimator__n_estimators': [50, 100, 200],
        'estimator__max_depth': [5, 10, 20, None],
        'estimator__min_samples_split': [2, 5, 10],
        'estimator__min_samples_leaf': [1, 2, 4],
        'estimator__max_features': ['auto', 'sqrt', 'log2', None]
    }
    """
    base_model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
    grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    ##print("Best parameters:\n", grid_search.best_params_)

    y_pred = grid_search.predict(X_test)
    """
    
    base_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))

    base_model.fit(X_train, y_train)

    y_pred = base_model.predict(X_test)
    """
    y_demand_pred = scaler_y.inverse_transform(y_pred)
    y_demand_true = scaler_y.inverse_transform(y_test)

    mse = mean_squared_error(y_demand_true, y_demand_pred)
    mae = mean_absolute_error(y_demand_true, y_demand_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_demand_true, y_demand_pred)
    modelName = "Random Forest"
    results = [mse, mae, rmse, r2,modelName]
    #----#
    print('\n', "Merged Random Forest Model Results:")
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))

    assert len(test_month_labels) == y_demand_true.shape[0] == y_demand_pred.shape[0]
    if plots == True:
        plt.figure(figsize=(12, 6))
        plt.plot(test_month_labels, y_demand_true[:, 0], label='Actual Demand', marker='o')
        plt.plot(test_month_labels, y_demand_pred[:, 0], label='Predicted Demand', marker='s')
        plt.title('Random Forest  Model: Actual vs Predicted Demand: Monthly')
        plt.xlabel('Month')
        plt.ylabel('MW')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.8)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(test_month_labels, y_demand_true[:, 1], label='Actual Supply', marker='o')
        plt.plot(test_month_labels, y_demand_pred[:, 1], label='Predicted Supply', marker='s')
        plt.title('Random Forest Model: Actual vs Predicted Supply: Monthly')
        plt.xlabel('Month')
        plt.ylabel('MW')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.8)
        plt.legend()
        plt.tight_layout()
        plt.show()

        """
        #-Demand Plot-#
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true[:, 0], label='Actual Demand')
        plt.plot(y_demand_pred[:, 0], label='Predicted Demand')
        plt.title(f'Random Forest Model: Actual vs Predicted Demand: Daily')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
        #-Supply Plot-#
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true[:, 1], label='Actual Supply')
        plt.plot(y_demand_pred[:, 1], label='Predicted Supply')
        plt.title(f'Random Forest Model: Actual vs Predicted Supply: Daily')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
        """
    return results


def xgbModelDS(mergedDs: pd.DataFrame,plots:bool):
    mergedDs = mergedDs.copy()


    #----#
    mergedDs.replace(-999, np.nan, inplace=True)
    mergedDs.dropna(inplace=True)

    ###minLength = min(len(demandDs))
    ###demandDs = demandDs.iloc[minLength]

    #Extraction of Data.

    mergedDs["hour"] = mergedDs["TimeStamp"].dt.hour
    mergedDs["day"] = mergedDs["TimeStamp"].dt.day
    mergedDs["month"] = mergedDs["TimeStamp"].dt.month

    mergedDs["hour"] = mergedDs["TimeStamp"].dt.hour
    mergedDs["day"] = mergedDs["TimeStamp"].dt.day
    mergedDs["month"] = mergedDs["TimeStamp"].dt.month

    feature_cols = [
    "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
    "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","WS10M","WD10M","hour","day","month"
    ]
    """
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW', 'Supply_MW']
    X = scaler_X.fit_transform(mergedDs[feature_cols])
    y = scaler_y.fit_transform(mergedDs[target_cols])

    seq_length = 24
    X_seq, y_seq = create_sequences_with_time(X, y, seq_length)

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.3, shuffle=False
    )
    #X_seq, y_seq = create_seasonal_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
    #X_seq_flat = X_seq.reshape(X_seq.shape[0], -1)

    """
 
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW','Supply_MW']

    # Optionally scale the whole DataFrame first
    mergedDs[feature_cols] = scaler_X.fit_transform(mergedDs[feature_cols])
    mergedDs[target_cols] = scaler_y.fit_transform(mergedDs[target_cols])

    # Create monthly sequences
    X, y = create_monthly_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
    X = X.reshape(X.shape[0], -1)
    # Now split into train/test (e.g., 70% train, 30% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    mergedDs['year'] = mergedDs['TimeStamp'].dt.year
    mergedDs['month'] = mergedDs['TimeStamp'].dt.month
    months = sorted(mergedDs.groupby(['year', 'month']).groups.keys())
    month_labels = [f"{y}-{m:02d}" for (y, m) in months]

    months = sorted(mergedDs.groupby(['year', 'month']).groups.keys())
    month_labels = [f"{y}-{m:02d}" for (y, m) in months]
    split_idx = int(len(months) * 0.8)
    test_month_labels = month_labels[split_idx:]
    
    param_grid = {
        'estimator__n_estimators': [50],
        'estimator__max_depth': [3],
        'estimator__learning_rate': [0.1],
        'estimator__subsample': [0.8],
        'estimator__colsample_bytree': [0.8]
    }
    """
    param_grid = {
    'estimator__n_estimators': [50, 100, 200],
    'estimator__max_depth': [3, 5, 7],
    'estimator__learning_rate': [0.01, 0.1, 0.2],
    'estimator__subsample': [0.8, 1.0],
    'estimator__colsample_bytree': [0.8, 1.0]
    """
    base_model = MultiOutputRegressor(xgb.XGBRegressor(random_state=42, verbosity=0))
    grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Best parameters:\n", grid_search.best_params_)

    y_pred = grid_search.predict(X_test)
    """
    base_model = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=100, random_state=42))
    model = MultiOutputRegressor(base_model)
    base_model.fit(X_train, y_train)
    y_pred = base_model.predict(X_test)
    """
    y_demand_pred = scaler_y.inverse_transform(y_pred)
    y_demand_true = scaler_y.inverse_transform(y_test)

    mse = mean_squared_error(y_demand_true, y_demand_pred)
    mae = mean_absolute_error(y_demand_true, y_demand_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_demand_true, y_demand_pred)
    modelName = "XGB"
    results = [mse, mae, rmse, r2,modelName]

    #----#
    print('\n', "Merged XGB Model Results:")
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))

    assert len(test_month_labels) == y_demand_true.shape[0] == y_demand_pred.shape[0]
    if plots == True:
        plt.figure(figsize=(12, 6))
        plt.plot(test_month_labels, y_demand_true[:, 0], label='Actual Demand', marker='o')
        plt.plot(test_month_labels, y_demand_pred[:, 0], label='Predicted Demand', marker='s')
        plt.title('XGB Model: Actual vs Predicted Demand: Monthly')
        plt.xlabel('Month')
        plt.ylabel('MW')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.8)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(test_month_labels, y_demand_true[:, 1], label='Actual Supply', marker='o')
        plt.plot(test_month_labels, y_demand_pred[:, 1], label='Predicted Supply', marker='s')
        plt.title('XGB Model: Actual vs Predicted Supply: Monthly')
        plt.xlabel('Month')
        plt.ylabel('MW')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.8)
        plt.legend()
        plt.tight_layout()
        plt.show()
        """
        #-Demand Plot-#
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true[:, 0], label='Actual Demand')
        plt.plot(y_demand_pred[:, 0], label='Predicted Demand')
        plt.title(f'XGB Model: Actual vs Predicted Demand: Daily')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
        #-Supply Plot-#
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true[:, 1], label='Actual Supply')
        plt.plot(y_demand_pred[:, 1], label='Predicted Supply')
        plt.title(f'XGB Model: Actual vs Predicted Supply: Daily')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
        """
    return results


def gbModelDS(mergedDs: pd.DataFrame,plots:bool):
    mergedDs = mergedDs.copy()


    mergedDs["hour"] = mergedDs["TimeStamp"].dt.hour
    mergedDs["day"] = mergedDs["TimeStamp"].dt.day
    mergedDs["month"] = mergedDs["TimeStamp"].dt.month
    feature_cols = [
    "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
    "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","WS10M","WD10M","hour","day","month"
    ]
    """
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW', 'Supply_MW']
    X = scaler_X.fit_transform(mergedDs[feature_cols])
    y = scaler_y.fit_transform(mergedDs[target_cols])

    seq_length = 24
    X_seq, y_seq = create_sequences_with_time(X, y, seq_length)

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.3, shuffle=False
    )
    #X_seq, y_seq = create_seasonal_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
    #X_seq_flat = X_seq.reshape(X_seq.shape[0], -1)

    #X_seq, y_seq = create_seasonal_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
    #X_seq_flat = X_seq.reshape(X_seq.shape[0], -1)

    """
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW','Supply_MW']

    # Optionally scale the whole DataFrame first
    mergedDs[feature_cols] = scaler_X.fit_transform(mergedDs[feature_cols])
    mergedDs[target_cols] = scaler_y.fit_transform(mergedDs[target_cols])

    # Create monthly sequences
    X, y = create_monthly_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
    X = X.reshape(X.shape[0], -1)
    # Now split into train/test (e.g., 70% train, 30% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    mergedDs['year'] = mergedDs['TimeStamp'].dt.year
    mergedDs['month'] = mergedDs['TimeStamp'].dt.month
    months = sorted(mergedDs.groupby(['year', 'month']).groups.keys())
    month_labels = [f"{y}-{m:02d}" for (y, m) in months]

    months = sorted(mergedDs.groupby(['year', 'month']).groups.keys())
    month_labels = [f"{y}-{m:02d}" for (y, m) in months]
    split_idx = int(len(months) * 0.8)
    test_month_labels = month_labels[split_idx:]
        
    param_grid = {
        'estimator__n_estimators': [100],
        'estimator__max_depth': [3],
        'estimator__learning_rate': [0.1],
        'estimator__min_samples_split': [10],
        'estimator__min_samples_leaf': [2],
        'estimator__max_features': ['sqrt']
    }
    """
    param_grid = {
        'estimator__n_estimators': [50, 100, 200],
        'estimator__max_depth': [3, 5, 10],
        'estimator__learning_rate': [0.01, 0.1, 0.2],
        'estimator__min_samples_split': [2, 5, 10],
        'estimator__min_samples_leaf': [1, 2, 4],
        'estimator__max_features': ['auto', 'sqrt', 'log2', None]
    }
    """
    base_model = MultiOutputRegressor(GradientBoostingRegressor(random_state=42))
    grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Best parameters:\n", grid_search.best_params_)

    y_pred = grid_search.predict(X_test)
    """
    base_model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, random_state=42))
    model = MultiOutputRegressor(base_model)
    base_model.fit(X_train, y_train)
    y_pred = base_model.predict(X_test)
    """
    y_demand_pred = scaler_y.inverse_transform(y_pred)
    y_demand_true = scaler_y.inverse_transform(y_test)

    mse = mean_squared_error(y_demand_true, y_demand_pred)
    mae = mean_absolute_error(y_demand_true, y_demand_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_demand_true, y_demand_pred)
    modelName = "Gradient Boost"
    results = [mse, mae, rmse, r2,modelName]
    
    #----#
    print('\n', "Merged GB Model Results:")
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))
    assert len(test_month_labels) == y_demand_true.shape[0] == y_demand_pred.shape[0]
    if plots == True:
        plt.figure(figsize=(12, 6))
        plt.plot(test_month_labels, y_demand_true[:, 0], label='Actual Demand', marker='o')
        plt.plot(test_month_labels, y_demand_pred[:, 0], label='Predicted Demand', marker='s')
        plt.title('Gradient Boosting Model: Actual vs Predicted Demand: Monthly')
        plt.xlabel('Month')
        plt.ylabel('MW')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.8)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(test_month_labels, y_demand_true[:, 1], label='Actual Supply', marker='o')
        plt.plot(test_month_labels, y_demand_pred[:, 1], label='Predicted Supply', marker='s')
        plt.title('Gradient Boosting Model: Actual vs Predicted Supply: Monthly')
        plt.xlabel('Month')
        plt.ylabel('MW')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.8)
        plt.legend()
        plt.tight_layout()
        plt.show()
        """
        #-Demand Plot-#
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true[:, 0], label='Actual Demand')
        plt.plot(y_demand_pred[:, 0], label='Predicted Demand')
        plt.title(f'Gradient Boosting Model: Actual vs Predicted Demand: Daily')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
        #-Supply Plot-#
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true[:, 1], label='Actual Supply')
        plt.plot(y_demand_pred[:, 1], label='Predicted Supply')
        plt.title(f'Gradient Boosting Model: Actual vs Predicted Supply: Daily')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
        """
    return results


def biDirectionalLSTMDS(mergedDs: pd.DataFrame,plots:bool):
    mergedDs = mergedDs.copy()



    #----#
    ##Basic transformation for the base dataset
    mergedDs["hour"] = mergedDs["TimeStamp"].dt.hour
    mergedDs["day"] = mergedDs["TimeStamp"].dt.day
    mergedDs["month"] = mergedDs["TimeStamp"].dt.month

    feature_cols = [
    "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
    "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","WS10M","WD10M","hour","day","month"
    ]
    """
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW', 'Supply_MW']
    X = scaler_X.fit_transform(mergedDs[feature_cols])
    y = scaler_y.fit_transform(mergedDs[target_cols])
    seq_length = 24
    X_seq, y_seq = create_sequences_with_time(X, y, seq_length)

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.3, shuffle=False
    )
    #X_seq, y_seq = create_seasonal_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
    #X_seq_flat = X_seq.reshape(X_seq.shape[0], -1)
    #
    num_features = len(feature_cols)
    X_train = X_train.reshape((-1, seq_length, num_features))
    X_test = X_test.reshape((-1, seq_length, num_features))
    """
   
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW','Supply_MW']

    # Optionally scale the whole DataFrame first
    mergedDs[feature_cols] = scaler_X.fit_transform(mergedDs[feature_cols])
    mergedDs[target_cols] = scaler_y.fit_transform(mergedDs[target_cols])

    # Create monthly sequences
    X, y = create_monthly_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
    #X = X.reshape(X.shape[0], -1) # Flattening
    # Now split into train/test (e.g., 70% train, 30% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    max_len = X.shape[1]
    n_features = X.shape[2]
    mergedDs['year'] = mergedDs['TimeStamp'].dt.year
    mergedDs['month'] = mergedDs['TimeStamp'].dt.month
    months = sorted(mergedDs.groupby(['year', 'month']).groups.keys())
    month_labels = [f"{y}-{m:02d}" for (y, m) in months]

    months = sorted(mergedDs.groupby(['year', 'month']).groups.keys())
    month_labels = [f"{y}-{m:02d}" for (y, m) in months]
    split_idx = int(len(months) * 0.8)
    test_month_labels = month_labels[split_idx:]
        
    tuner = Hyperband(
    lambda hp: (
        lambda model: (
            model.compile(
                optimizer=Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                loss='mse'
            ),
            model
        )[1]
    )(Sequential([
        Bidirectional(LSTM(
            units=hp.Int('units1', 32, 128, step=32),
            return_sequences=True,
            input_shape=(max_len, n_features)
        )),
        Dropout(hp.Float('dropout1', 0.1, 0.5, step=0.1)),
        Bidirectional(LSTM(
            units=hp.Int('units2', 32, 128, step=32),
            return_sequences=False
        )),
        Dense(hp.Int('dense_units', 32, 128, step=32), activation='relu'),
        Dense(2)
    ])),
    objective='val_loss',
    max_epochs=20,
    factor=3,
    directory='tuner_dir',
    project_name='blstm_tuning'
)
    tuner.search(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=16)
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best hyperparameters found for BLSTM:", best_hp.values)
    best_model = tuner.get_best_models(num_models=1)[0]

    y_pred = best_model.predict(X_test)

    y_demand_true = scaler_y.inverse_transform(y_test)
    y_demand_pred = scaler_y.inverse_transform(y_pred)

    mse = mean_squared_error(y_demand_true, y_demand_pred)
    mae = mean_absolute_error(y_demand_true, y_demand_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_demand_true, y_demand_pred)
    modelName = "BLSTM"
    results = [mse, mae, rmse, r2,modelName]

    #----#
    print('\n', "Merged BLSTM Model Results:")
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))
    
    assert len(test_month_labels) == y_demand_true.shape[0] == y_demand_pred.shape[0]
    if plots == True:
        plt.figure(figsize=(12, 6))
        plt.plot(test_month_labels, y_demand_true[:, 0], label='Actual Demand', marker='o')
        plt.plot(test_month_labels, y_demand_pred[:, 0], label='Predicted Demand', marker='s')
        plt.title('BLSTM Model: Actual vs Predicted Demand: Monthly')
        plt.xlabel('Month')
        plt.ylabel('MW')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.8)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(test_month_labels, y_demand_true[:, 1], label='Actual Supply', marker='o')
        plt.plot(test_month_labels, y_demand_pred[:, 1], label='Predicted Supply', marker='s')
        plt.title('BLSTM Model: Actual vs Predicted Supply: Monthly')
        plt.xlabel('Month')
        plt.ylabel('MW')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.8)
        plt.legend()
        plt.tight_layout()
        plt.show()
        """
        #-Demand Plot-#
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true[:, 0], label='Actual Demand')
        plt.plot(y_demand_pred[:, 0], label='Predicted Demand')
        plt.title(f'BLSTM Model: Actual vs Predicted Demand: Daily')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
        #-Supply Plot-#
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true[:, 1], label='Actual Supply')
        plt.plot(y_demand_pred[:, 1], label='Predicted Supply')
        plt.title(f'BLSTM Model: Actual vs Predicted Supply: Daily')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
        """
    return results


def LSTMModelDS(mergedDs: pd.DataFrame,plots:bool):
    mergedDs = mergedDs.copy()



    mergedDs["hour"] = mergedDs["TimeStamp"].dt.hour
    mergedDs["day"] = mergedDs["TimeStamp"].dt.day
    mergedDs["month"] = mergedDs["TimeStamp"].dt.month

    feature_cols = [
    "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
    "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","WS10M","WD10M","hour","day","month"
    ]
    """
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW', 'Supply_MW']
    X = scaler_X.fit_transform(mergedDs[feature_cols])
    y = scaler_y.fit_transform(mergedDs[target_cols])

    seq_length = 24
    X_seq, y_seq = create_sequences_with_time(X, y, seq_length)

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.3, shuffle=False
    )
    num_features = len(feature_cols)
    X_train = X_train.reshape((-1, seq_length, num_features))
    X_test = X_test.reshape((-1, seq_length, num_features))
    #X_seq, y_seq = create_seasonal_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
    #X_seq_flat = X_seq.reshape(X_seq.shape[0], -1)

    """
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW','Supply_MW']

    # Optionally scale the whole DataFrame first
    mergedDs[feature_cols] = scaler_X.fit_transform(mergedDs[feature_cols])
    mergedDs[target_cols] = scaler_y.fit_transform(mergedDs[target_cols])

    # Create monthly sequences
    X, y = create_monthly_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
    #X = X.reshape(X.shape[0], -1) # Flattening
    # Now split into train/test (e.g., 70% train, 30% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    mergedDs['year'] = mergedDs['TimeStamp'].dt.year
    mergedDs['month'] = mergedDs['TimeStamp'].dt.month
    max_len = X.shape[1]
    n_features = X.shape[2]
    months = sorted(mergedDs.groupby(['year', 'month']).groups.keys())
    month_labels = [f"{y}-{m:02d}" for (y, m) in months]

    months = sorted(mergedDs.groupby(['year', 'month']).groups.keys())
    month_labels = [f"{y}-{m:02d}" for (y, m) in months]
    split_idx = int(len(months) * 0.8)
    test_month_labels = month_labels[split_idx:]
        
    tuner = Hyperband(
        lambda hp: (
            lambda model: (
                model.compile(
                    optimizer=Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                    loss='mse'
                ),
                model
            )[1]
        )(Sequential([
            LSTM(
                units=hp.Int('units1', 32, 128, step=32),
                return_sequences=True,
                input_shape=(max_len, n_features)
            ),
            Dropout(hp.Float('dropout1', 0.1, 0.5, step=0.1)),
            LSTM(
                units=hp.Int('units2', 32, 128, step=32),
                return_sequences=False
            ),
            Dense(hp.Int('dense_units', 32, 128, step=32), activation='relu'),
            Dense(2)
        ])),
        objective='val_loss',
        max_epochs=20,
        factor=3,
        directory='tuner_dir',
        project_name='lstm_tuning'
    )
    tuner.search(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=16)
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best hyperparameters found for LSTM:", best_hp.values)
    best_model = tuner.get_best_models(num_models=1)[0]


    y_pred = best_model.predict(X_test)
    
    """
    ## Base Model Implementation
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_length, X_train.shape[2])),
        Dropout(0.2),
        LSTM(100, return_sequences=False),
        Dense(64),
        Dense(2)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test),
              callbacks=[early_stopping])
    
    y_pred = model.predict(X_test)
    """
    y_demand_pred = scaler_y.inverse_transform(y_pred)
    y_demand_true = scaler_y.inverse_transform(y_test)

    mse = mean_squared_error(y_demand_true, y_demand_pred)
    mae = mean_absolute_error(y_demand_true, y_demand_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_demand_true, y_demand_pred)
    modelName = "LSTM"
    results = [mse, mae, rmse, r2,modelName]

    #----#
    print('\n', "Merged LSTM Model Results:")
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))
    
    assert len(test_month_labels) == y_demand_true.shape[0] == y_demand_pred.shape[0]
    if plots == True:
        plt.figure(figsize=(12, 6))
        plt.plot(test_month_labels, y_demand_true[:, 0], label='Actual Demand', marker='o')
        plt.plot(test_month_labels, y_demand_pred[:, 0], label='Predicted Demand', marker='s')
        plt.title('LSTM Model: Actual vs Predicted Demand: Monthly')
        plt.xlabel('Month')
        plt.ylabel('MW')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.8)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(test_month_labels, y_demand_true[:, 1], label='Actual Supply', marker='o')
        plt.plot(test_month_labels, y_demand_pred[:, 1], label='Predicted Supply', marker='s')
        plt.title('LSTM Model: Actual vs Predicted Supply: Monthly')
        plt.xlabel('Month')
        plt.ylabel('MW')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.8)
        plt.legend()
        plt.tight_layout()
        plt.show()
        """
        #-Demand Plot-#
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true[:, 0], label='Actual Demand')
        plt.plot(y_demand_pred[:, 0], label='Predicted Demand')
        plt.title(f'LSTM Model: Actual vs Predicted Demand: Daily')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
        #-Supply Plot-#
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true[:, 1], label='Actual Supply')
        plt.plot(y_demand_pred[:, 1], label='Predicted Supply')
        plt.title(f'LSTM Tree Model: Actual vs Predicted Supply: Daily')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
        """
    return results


def GRUModelDS(mergedDs: pd.DataFrame,plots:bool):
    mergedDs = mergedDs.copy()


    #----#
    ##Basic transformation for the base dataset
    mergedDs["hour"] = mergedDs["TimeStamp"].dt.hour
    mergedDs["day"] = mergedDs["TimeStamp"].dt.day
    mergedDs["month"] = mergedDs["TimeStamp"].dt.month

    feature_cols = [
    "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
    "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","WS10M","WD10M","hour","day","month"
    ]
    """
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW', 'Supply_MW']
    X = scaler_X.fit_transform(mergedDs[feature_cols])
    y = scaler_y.fit_transform(mergedDs[target_cols])

    seq_length = 24
    X_seq, y_seq = create_sequences_with_time(X, y, seq_length)

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.3, shuffle=False
    )
    num_features = len(feature_cols)
    X_train = X_train.reshape((-1, seq_length, num_features))
    X_test = X_test.reshape((-1, seq_length, num_features))
    """
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW','Supply_MW']

    # Optionally scale the whole DataFrame first
    mergedDs[feature_cols] = scaler_X.fit_transform(mergedDs[feature_cols])
    mergedDs[target_cols] = scaler_y.fit_transform(mergedDs[target_cols])

    # Create monthly sequences
    X, y = create_monthly_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
    #X = X.reshape(X.shape[0], -1) # Flattening
    # Now split into train/test (e.g., 70% train, 30% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    mergedDs['year'] = mergedDs['TimeStamp'].dt.year
    mergedDs['month'] = mergedDs['TimeStamp'].dt.month
    max_len = X.shape[1]
    n_features = X.shape[2]
    months = sorted(mergedDs.groupby(['year', 'month']).groups.keys())
    month_labels = [f"{y}-{m:02d}" for (y, m) in months]

    months = sorted(mergedDs.groupby(['year', 'month']).groups.keys())
    month_labels = [f"{y}-{m:02d}" for (y, m) in months]
    split_idx = int(len(months) * 0.8)
    test_month_labels = month_labels[split_idx:]
        
    tuner = Hyperband(
        lambda hp: (
            lambda model: (
                model.compile(
                    optimizer=Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                    loss='mse'
                ),
                model
            )[1]
        )(Sequential([
            GRU(
                units=hp.Int('units1', 32, 128, step=32),
                return_sequences=True,
                input_shape=(max_len, n_features)
            ),
            Dropout(hp.Float('dropout1', 0.1, 0.5, step=0.1)),
            GRU(
                units=hp.Int('units2', 32, 128, step=32),
                return_sequences=False
            ),
            Dense(hp.Int('dense_units', 32, 128, step=32), activation='relu'),
            Dense(2)
        ])),
        objective='val_loss',
        max_epochs=20,
        factor=3,
        directory='tuner_dir',
        project_name='gru_tuning'
    )
    tuner.search(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=16)
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best hyperparameters found for GRU:", best_hp.values)
    best_model = tuner.get_best_models(num_models=1)[0]


    y_pred = best_model.predict(X_test)
    """
    ## Base Model Implementation
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=(seq_length, X_train.shape[2])),
        Dropout(0.2),
        GRU(100, return_sequences=False),
        Dense(64, activation='relu'),
        Dense(2)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test),
              callbacks=[early_stopping])

    y_pred = model.predict(X_test)
    """
    y_demand_pred = scaler_y.inverse_transform(y_pred)
    y_demand_true = scaler_y.inverse_transform(y_test)

    mse = mean_squared_error(y_demand_true, y_demand_pred)
    mae = mean_absolute_error(y_demand_true, y_demand_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_demand_true, y_demand_pred)
    modelName = "GRU"
    results = [mse, mae, rmse, r2,modelName]

    #----#
    print('\n', "Merged GRU Model Results:")
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))
    
    assert len(test_month_labels) == y_demand_true.shape[0] == y_demand_pred.shape[0]
    if plots == True:
        plt.figure(figsize=(12, 6))
        plt.plot(test_month_labels, y_demand_true[:, 0], label='Actual Demand', marker='o')
        plt.plot(test_month_labels, y_demand_pred[:, 0], label='Predicted Demand', marker='s')
        plt.title('GRU Model: Actual vs Predicted Demand: Monthly')
        plt.xlabel('Month')
        plt.ylabel('MW')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.8)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(test_month_labels, y_demand_true[:, 1], label='Actual Supply', marker='o')
        plt.plot(test_month_labels, y_demand_pred[:, 1], label='Predicted Supply', marker='s')
        plt.title('GRU Model: Actual vs Predicted Supply: Monthly')
        plt.xlabel('Month')
        plt.ylabel('MW')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.8)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
        """
        #-Demand Plot-#
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true[:, 0], label='Actual Demand')
        plt.plot(y_demand_pred[:, 0], label='Predicted Demand')
        plt.title(f'GRU Model: Actual vs Predicted Demand: Daily')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
        #-Supply Plot-#
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true[:, 1], label='Actual Supply')
        plt.plot(y_demand_pred[:, 1], label='Predicted Supply')
        plt.title(f'GRU Model: Actual vs Predicted Supply: Daily')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
        """
    return results


def SVRModelDS(mergedDs: pd.DataFrame,plots:bool):
    mergedDs = mergedDs.copy()




    mergedDs["hour"] = mergedDs["TimeStamp"].dt.hour
    mergedDs["day"] = mergedDs["TimeStamp"].dt.day
    mergedDs["month"] = mergedDs["TimeStamp"].dt.month

    feature_cols = [
    "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
    "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","WS10M","WD10M","hour","day","month"
    ]
    """
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW', 'Supply_MW']
    X = scaler_X.fit_transform(mergedDs[feature_cols])
    y = scaler_y.fit_transform(mergedDs[target_cols])

    seq_length = 24
    X_seq, y_seq = create_sequences_with_time(X, y, seq_length)

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.3, shuffle=False
    )
    #X_seq, y_seq = create_seasonal_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
    #X_seq_flat = X_seq.reshape(X_seq.shape[0], -1)

    """
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW','Supply_MW']

    # Optionally scale the whole DataFrame first
    mergedDs[feature_cols] = scaler_X.fit_transform(mergedDs[feature_cols])
    mergedDs[target_cols] = scaler_y.fit_transform(mergedDs[target_cols])

    # Create monthly sequences
    X, y = create_monthly_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
    X = X.reshape(X.shape[0], -1)
    # Now split into train/test (e.g., 70% train, 30% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    mergedDs['year'] = mergedDs['TimeStamp'].dt.year
    mergedDs['month'] = mergedDs['TimeStamp'].dt.month
    max_len = X.shape[1]
    months = sorted(mergedDs.groupby(['year', 'month']).groups.keys())
    month_labels = [f"{y}-{m:02d}" for (y, m) in months]

    months = sorted(mergedDs.groupby(['year', 'month']).groups.keys())
    month_labels = [f"{y}-{m:02d}" for (y, m) in months]
    split_idx = int(len(months) * 0.8)
    test_month_labels = month_labels[split_idx:]
        
    
    param_grid = {
        'estimator__C': [0.1],           # Regularization, similar to tree depth/complexity
        'estimator__gamma': [1],  # Kernel coefficient, like learning rate
        'estimator__epsilon': [0.01], # Insensitivity, similar to min_samples_split
        'estimator__kernel': ['rbf']                 # Keep kernel consistent
    }
    """
    param_grid = {
        'estimator__C': [0.1, 1, 10, 100],           # Regularization, similar to tree depth/complexity
        'estimator__gamma': ['scale', 'auto', 0.01, 0.1, 1],  # Kernel coefficient, like learning rate
        'estimator__epsilon': [0.01, 0.1, 0.5, 1.0], # Insensitivity, similar to min_samples_split
        'estimator__kernel': ['rbf']                 # Keep kernel consistent
    }
    """
    model = MultiOutputRegressor(SVR(kernel='rbf'))
    ## model = MultiOutputRegressor(SVR())

    grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    print("Best parameters:", grid.best_params_)
    """
    model = MultiOutputRegressor(SVR(kernel='rbf'))
    model.fit(X_train, y_train)
    """
    y_pred = grid.predict(X_test)
    
    y_demand_pred = scaler_y.inverse_transform(y_pred)
    y_demand_true = scaler_y.inverse_transform(y_test)

    mse = mean_squared_error(y_demand_true, y_demand_pred)
    mae = mean_absolute_error(y_demand_true, y_demand_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_demand_true, y_demand_pred)
    modelName = "SVR"
    results = [mse, mae, rmse, r2,modelName]

    #----#
    print('\n', "Merged SVR Model Results:")
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))
    
    assert len(test_month_labels) == y_demand_true.shape[0] == y_demand_pred.shape[0]
    if plots == True:
        plt.figure(figsize=(12, 6))
        plt.plot(test_month_labels, y_demand_true[:, 0], label='Actual Demand', marker='o')
        plt.plot(test_month_labels, y_demand_pred[:, 0], label='Predicted Demand', marker='s')
        plt.title('SVR Model: Actual vs Predicted Demand: Monthly')
        plt.xlabel('Month')
        plt.ylabel('MW')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.8)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(test_month_labels, y_demand_true[:, 1], label='Actual Supply', marker='o')
        plt.plot(test_month_labels, y_demand_pred[:, 1], label='Predicted Supply', marker='s')
        plt.title('SVR Model: Actual vs Predicted Supply: Monthly')
        plt.xlabel('Month')
        plt.ylabel('MW')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.8)
        plt.legend()
        plt.tight_layout()
        plt.show()
        """
        #-Demand Plot-#
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true[:, 0], label='Actual Demand')
        plt.plot(y_demand_pred[:, 0], label='Predicted Demand')
        plt.title(f'SVR Model: Actual vs Predicted Demand: Daily')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
        #-Supply Plot-#
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true[:, 1], label='Actual Supply')
        plt.plot(y_demand_pred[:, 1], label='Predicted Supply')
        plt.title(f'SVR Model: Actual vs Predicted Supply: Daily')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
        """
    return results


def MLPModelDS(mergedDs: pd.DataFrame,plots:bool):
    mergedDs = mergedDs.copy()


    # All code below is now at the same indentation as the inner function
    mergedDs["hour"] = mergedDs["TimeStamp"].dt.hour
    mergedDs["day"] = mergedDs["TimeStamp"].dt.day
    mergedDs["month"] = mergedDs["TimeStamp"].dt.month

    feature_cols = [
    "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
    "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","WS10M","WD10M","hour","day","month"
    ]
    """
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW', 'Supply_MW']
    X = scaler_X.fit_transform(mergedDs[feature_cols])
    y = scaler_y.fit_transform(mergedDs[target_cols])

    seq_length = 24
    X_seq, y_seq = create_sequences_with_time(X, y, seq_length)

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.3, shuffle=False
    )
    num_features = len(feature_cols)
    X_train = X_train.reshape((X_train.shape[0], seq_length * num_features))
    X_test = X_test.reshape((X_test.shape[0], seq_length * num_features))
    #X_seq, y_seq = create_seasonal_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
    #X_seq_flat = X_seq.reshape(X_seq.shape[0], -1)

    """
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW','Supply_MW']

    # Optionally scale the whole DataFrame first
    mergedDs[feature_cols] = scaler_X.fit_transform(mergedDs[feature_cols])
    mergedDs[target_cols] = scaler_y.fit_transform(mergedDs[target_cols])

    # Create monthly sequences
    X, y = create_monthly_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
    X = X.reshape(X.shape[0], -1)
    # Now split into train/test (e.g., 70% train, 30% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    mergedDs['year'] = mergedDs['TimeStamp'].dt.year
    mergedDs['month'] = mergedDs['TimeStamp'].dt.month
    max_len = X.shape[1]
    n_features = X.shape[1]
    months = sorted(mergedDs.groupby(['year', 'month']).groups.keys())
    month_labels = [f"{y}-{m:02d}" for (y, m) in months]

    months = sorted(mergedDs.groupby(['year', 'month']).groups.keys())
    month_labels = [f"{y}-{m:02d}" for (y, m) in months]
    split_idx = int(len(months) * 0.8)
    test_month_labels = month_labels[split_idx:]
        
    tuner = Hyperband(
        lambda hp: (
            lambda model: (
                model.compile(
                    optimizer=Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                    loss='mse'
                ),
                model
            )[1]
        )(Sequential([
            Dense(hp.Int('dense1', 64, 256, step=32), activation='relu', input_dim=X.shape[1]),
            Dropout(hp.Float('dropout1', 0.1, 0.5, step=0.1)),
            Dense(hp.Int('dense2', 32, 128, step=32), activation='relu'),
            Dropout(hp.Float('dropout2', 0.1, 0.5, step=0.1)),
            Dense(2)
        ])),
        objective='val_loss',
        max_epochs=20,
        factor=3,
        directory='tuner_dir',
        project_name='mlp_tuning'
    )
    tuner.search(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=16)
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best hyperparameters found for MLP:", best_hp.values)
    best_model = tuner.get_best_models(num_models=1)[0]

    y_pred = best_model.predict(X_test)


    """
    ## Base Implementation 
    model = Sequential([
        Dense(128, activation='relu', input_dim=X_train.shape[1]),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(2)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))

    y_pred = model.predict(X_test)
    """
    y_demand_pred = scaler_y.inverse_transform(y_pred)
    y_demand_true = scaler_y.inverse_transform(y_test)

    mse = mean_squared_error(y_demand_true, y_demand_pred)
    mae = mean_absolute_error(y_demand_true, y_demand_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_demand_true, y_demand_pred)
    modelName = "MLP"
    results = [mse, mae, rmse, r2,modelName]

    print('\n', "Merged MLP Model Results:")
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))

    assert len(test_month_labels) == y_demand_true.shape[0] == y_demand_pred.shape[0]
    if plots == True:
        plt.figure(figsize=(12, 6))
        plt.plot(test_month_labels, y_demand_true[:, 0], label='Actual Demand', marker='o')
        plt.plot(test_month_labels, y_demand_pred[:, 0], label='Predicted Demand', marker='s')
        plt.title('MLP Model: Actual vs Predicted Demand: Monthly')
        plt.xlabel('Month')
        plt.ylabel('MW')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.8)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(test_month_labels, y_demand_true[:, 1], label='Actual Supply', marker='o')
        plt.plot(test_month_labels, y_demand_pred[:, 1], label='Predicted Supply', marker='s')
        plt.title('MLP Model: Actual vs Predicted Supply: Monthly')
        plt.xlabel('Month')
        plt.ylabel('MW')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.8)
        plt.legend()
        plt.tight_layout()
        plt.show()
        """
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true[:, 0], label='Actual Demand')
        plt.plot(y_demand_pred[:, 0], label='Predicted Demand')
        plt.title(f'MLP Model: Actual vs Predicted Demand: Daily')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true[:, 1], label='Actual Supply')
        plt.plot(y_demand_pred[:, 1], label='Predicted Supply')
        plt.title(f'MLP Model: Actual vs Predicted Supply: Daily')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
        """

    return results


def CNNModelDS(mergedDs: pd.DataFrame,plots:bool):
    mergedDs = mergedDs.copy()


    mergedDs["hour"] = mergedDs["TimeStamp"].dt.hour
    mergedDs["day"] = mergedDs["TimeStamp"].dt.day
    mergedDs["month"] = mergedDs["TimeStamp"].dt.month

    feature_cols = [
    "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
    "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","WS10M","WD10M","hour","day","month"
    ]
    """
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW', 'Supply_MW']
    X = scaler_X.fit_transform(mergedDs[feature_cols])
    y = scaler_y.fit_transform(mergedDs[target_cols])

    seq_length = 24
    X_seq, y_seq = create_sequences_with_time(X, y, seq_length)

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.3, shuffle=False
    )
    #X_seq, y_seq = create_seasonal_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
    #X_seq_flat = X_seq.reshape(X_seq.shape[0], -1)
    num_features = len(feature_cols)
    X_train = X_train.reshape((-1, seq_length, num_features))
    X_test = X_test.reshape((-1, seq_length, num_features))
    """
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW','Supply_MW']

    # Optionally scale the whole DataFrame first
    mergedDs[feature_cols] = scaler_X.fit_transform(mergedDs[feature_cols])
    mergedDs[target_cols] = scaler_y.fit_transform(mergedDs[target_cols])

    # Create monthly sequences
    X, y = create_monthly_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
    #X = X.reshape(X.shape[0], -1) # Flattening
    # Now split into train/test (e.g., 70% train, 30% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    mergedDs['year'] = mergedDs['TimeStamp'].dt.year
    mergedDs['month'] = mergedDs['TimeStamp'].dt.month
    max_len = X.shape[1]
    n_features = X.shape[2]
    months = sorted(mergedDs.groupby(['year', 'month']).groups.keys())
    month_labels = [f"{y}-{m:02d}" for (y, m) in months]

    months = sorted(mergedDs.groupby(['year', 'month']).groups.keys())
    month_labels = [f"{y}-{m:02d}" for (y, m) in months]
    split_idx = int(len(months) * 0.8)
    test_month_labels = month_labels[split_idx:]
        
    tuner = Hyperband(
        lambda hp: (
            lambda model: (
                model.compile(
                    optimizer=Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                    loss='mse'
                ),
                model
            )[1]
        )(Sequential([
            Conv1D(
                filters=hp.Int('filters1', 32, 128, step=32),
                kernel_size=hp.Choice('kernel_size1', [2, 3, 5]),
                activation='relu',
                input_shape=(max_len, n_features)
            ),
            MaxPooling1D(pool_size=2),
            Dropout(hp.Float('dropout1', 0.1, 0.5, step=0.1)),
            Conv1D(
                filters=hp.Int('filters2', 32, 128, step=32),
                kernel_size=hp.Choice('kernel_size2', [2, 3, 5]),
                activation='relu'
            ),
            Dropout(hp.Float('dropout2', 0.1, 0.5, step=0.1)),
            Flatten(),
            Dense(hp.Int('dense_units', 32, 128, step=32), activation='relu'),
            Dense(2)
        ])),
        objective='val_loss',
        max_epochs=20,
        factor=3,
        directory='tuner_dir',
        project_name='cnn_tuning'
    )
    tuner.search(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=16)
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best hyperparameters found for CNN:", best_hp.values)
    best_model = tuner.get_best_models(num_models=1)[0]


    y_pred = best_model.predict(X_test)


    """
    ## Base Model Implementation 
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(seq_length, X_train.shape[2])),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Conv1D(128, kernel_size=3, activation='relu'),
        Dropout(0.2),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(2)
    ])

    model.compile(optimizer='adam', loss='mse')

    model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))

    y_pred = model.predict(X_test)
    """
    y_demand_pred = scaler_y.inverse_transform(y_pred)
    y_demand_true = scaler_y.inverse_transform(y_test)

    mse = mean_squared_error(y_demand_true, y_demand_pred)
    mae = mean_absolute_error(y_demand_true, y_demand_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_demand_true, y_demand_pred)
    modelName = "CNN"
    results = [mse, mae, rmse, r2,modelName]

    #---#
    print('\n', "Demand CNN Model Results:")
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))
    assert len(test_month_labels) == y_demand_true.shape[0] == y_demand_pred.shape[0]
    if plots == True:
        plt.figure(figsize=(12, 6))
        plt.plot(test_month_labels, y_demand_true[:, 0], label='Actual Demand', marker='o')
        plt.plot(test_month_labels, y_demand_pred[:, 0], label='Predicted Demand', marker='s')
        plt.title('CNN Model: Actual vs Predicted Demand: Monthly')
        plt.xlabel('Month')
        plt.ylabel('MW')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.8)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(test_month_labels, y_demand_true[:, 1], label='Actual Supply', marker='o')
        plt.plot(test_month_labels, y_demand_pred[:, 1], label='Predicted Supply', marker='s')
        plt.title('CNN Model: Actual vs Predicted Supply: Monthly')
        plt.xlabel('Month')
        plt.ylabel('MW')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.8)
        plt.legend()
        plt.tight_layout()
        plt.show()    
        """
        #-Demand Plot-#
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true[:, 0], label='Actual Demand')
        plt.plot(y_demand_pred[:, 0], label='Predicted Demand')
        plt.title(f'CNN Model: Actual vs Predicted Demand: Daily')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
        #-Supply Plot-#
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true[:, 1], label='Actual Supply')
        plt.plot(y_demand_pred[:, 1], label='Predicted Supply')
        plt.title(f'CNN Model: Actual vs Predicted Supply: Daily')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
        """
    return results


def GBDTModelDS(mergedDs: pd.DataFrame,plots:bool):
    mergedDs = mergedDs.copy()


    mergedDs["hour"] = mergedDs["TimeStamp"].dt.hour
    mergedDs["day"] = mergedDs["TimeStamp"].dt.day
    mergedDs["month"] = mergedDs["TimeStamp"].dt.month

    feature_cols = [
    "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
    "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","WS10M","WD10M","hour","day","month"
    ]
    """
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW', 'Supply_MW']
    X = scaler_X.fit_transform(mergedDs[feature_cols])
    y = scaler_y.fit_transform(mergedDs[target_cols])

    seq_length = 24
    X_seq, y_seq = create_sequences_with_time(X, y, seq_length)

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.3, shuffle=False
    )
    #X_seq, y_seq = create_seasonal_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
    #X_seq_flat = X_seq.reshape(X_seq.shape[0], -1)
    """
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW','Supply_MW']

    # Optionally scale the whole DataFrame first
    mergedDs[feature_cols] = scaler_X.fit_transform(mergedDs[feature_cols])
    mergedDs[target_cols] = scaler_y.fit_transform(mergedDs[target_cols])

    # Create monthly sequences
    X, y = create_monthly_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
    X = X.reshape(X.shape[0], -1) # Flattening
    # Now split into train/test (e.g., 70% train, 30% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    mergedDs['year'] = mergedDs['TimeStamp'].dt.year
    mergedDs['month'] = mergedDs['TimeStamp'].dt.month
    max_len = X.shape[1]
    months = sorted(mergedDs.groupby(['year', 'month']).groups.keys())
    month_labels = [f"{y}-{m:02d}" for (y, m) in months]

    months = sorted(mergedDs.groupby(['year', 'month']).groups.keys())
    month_labels = [f"{y}-{m:02d}" for (y, m) in months]
    split_idx = int(len(months) * 0.8)
    test_month_labels = month_labels[split_idx:]
        
    param_grid = {
        'estimator__n_estimators': [50],
        'estimator__max_depth': [5],
        'estimator__learning_rate': [0.1],
        'estimator__min_samples_split': [10],
        'estimator__min_samples_leaf': [4],
        'estimator__max_features': ['sqrt']
    }
    """
    param_grid = {
        'estimator__n_estimators': [50, 100, 200],
        'estimator__max_depth': [3, 5, 10],
        'estimator__learning_rate': [0.01, 0.1, 0.2],
        'estimator__min_samples_split': [2, 5, 10],
        'estimator__min_samples_leaf': [1, 2, 4],
        'estimator__max_features': ['auto', 'sqrt', 'log2', None]
    }
    """
    base_model = MultiOutputRegressor(GradientBoostingRegressor(random_state=42))
    base_model.fit(X_train,y_train)
    grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Best parameters:\n", grid_search.best_params_)
    ##base_model.fit(X_train, y_train) 

    y_pred = grid_search.predict(X_test)

    y_demand_pred = scaler_y.inverse_transform(y_pred)
    y_demand_true = scaler_y.inverse_transform(y_test)

    mse = mean_squared_error(y_demand_true, y_demand_pred)
    mae = mean_absolute_error(y_demand_true, y_demand_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_demand_true, y_demand_pred)
    modelName = "GBDT"
    results = [mse, mae, rmse, r2,modelName]

    #---#

    print('\n', "Merged GBDT Model Results:")
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))
    
    assert len(test_month_labels) == y_demand_true.shape[0] == y_demand_pred.shape[0]
    if plots == True:
        plt.figure(figsize=(12, 6))
        plt.plot(test_month_labels, y_demand_true[:, 0], label='Actual Demand', marker='o')
        plt.plot(test_month_labels, y_demand_pred[:, 0], label='Predicted Demand', marker='s')
        plt.title('GBDT Model: Actual vs Predicted Demand: Monthly')
        plt.xlabel('Month')
        plt.ylabel('MW')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.8)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(test_month_labels, y_demand_true[:, 1], label='Actual Supply', marker='o')
        plt.plot(test_month_labels, y_demand_pred[:, 1], label='Predicted Supply', marker='s')
        plt.title('GBDT Model: Actual vs Predicted Supply: Monthly')
        plt.xlabel('Month')
        plt.ylabel('MW')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.8)
        plt.legend()
        plt.tight_layout()
        plt.show()
        """
        #-Demand Plot-#
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true[:, 0], label='Actual Demand')
        plt.plot(y_demand_pred[:, 0], label='Predicted Demand')
        plt.title(f'GBDT Model: Actual vs Predicted Demand: Daily')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
        #-Supply Plot-#
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true[:, 1], label='Actual Supply')
        plt.plot(y_demand_pred[:, 1], label='Predicted Supply')
        plt.title(f'GBDT Model: Actual vs Predicted Supply: Daily')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
        """
    return results
def custom_mutate(individual, indpb):
    if random.random() < indpb:
        individual[0] = random.choice([32, 64, 128])  # filters1
    if random.random() < indpb:
        individual[1] = random.choice([2, 3, 5])  # kernel_size1
    if random.random() < indpb:
        individual[2] = random.uniform(0.1, 0.5)  # dropout1
    if random.random() < indpb:
        individual[3] = random.choice([32, 64, 128])  # dense_units
    if random.random() < indpb:
        individual[4] = random.choice([1e-2, 1e-3, 1e-4])  # learning_rate
    return (individual,)
def ensure_real_result(results):
    # Only convert the first 4 metrics, leave the model name as is
    return [float(np.real(x)) if isinstance(x, (complex, np.complexfloating, np.floating, float)) else x for x in results]
def NSGA2_CNN_ModelDS(
    mergedDs: pd.DataFrame,
    plots:bool,
    pop_size=3,
    ngen=3,
    cxpb=0.7,
    mutpb=0.3,
    seq_length=24,
    epochs=20,
    batch_size=16
):

    mergedDs = mergedDs.copy()
    mergedDs.replace(-999, np.nan, inplace=True)
    mergedDs.dropna(inplace=True)
    mergedDs["hour"] = mergedDs["TimeStamp"].dt.hour
    mergedDs["day"] = mergedDs["TimeStamp"].dt.day
    mergedDs["month"] = mergedDs["TimeStamp"].dt.month

    feature_cols = [
    "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
    "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","WS10M","WD10M","hour","day","month"
    ]
    target_cols = ['Demand_MW', 'Supply_MW']
    """
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X = scaler_X.fit_transform(mergedDs[feature_cols])
    y = scaler_y.fit_transform(mergedDs[target_cols])

    X_seq, y_seq = create_sequences_with_time(X, y, seq_length)
    num_features = len(feature_cols)
    X_seq = X_seq.reshape((-1, seq_length, num_features))
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.3, shuffle=False
    )
    """
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW','Supply_MW']

    # Optionally scale the whole DataFrame first
    mergedDs[feature_cols] = scaler_X.fit_transform(mergedDs[feature_cols])
    mergedDs[target_cols] = scaler_y.fit_transform(mergedDs[target_cols])

    # Create monthly sequences
    X, y = create_monthly_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
    ##X = X.reshape(X.shape[0], -1) # Flattening
    # Now split into train/test (e.g., 70% train, 30% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    mergedDs['year'] = mergedDs['TimeStamp'].dt.year
    mergedDs['month'] = mergedDs['TimeStamp'].dt.month
    max_len = X.shape[1]
    n_features = X.shape[2]
    months = sorted(mergedDs.groupby(['year', 'month']).groups.keys())
    month_labels = [f"{y}-{m:02d}" for (y, m) in months]

    months = sorted(mergedDs.groupby(['year', 'month']).groups.keys())
    month_labels = [f"{y}-{m:02d}" for (y, m) in months]
    split_idx = int(len(months) * 0.8)
    test_month_labels = month_labels[split_idx:]
        

    scaler_demand = MinMaxScaler()
    scaler_supply = MinMaxScaler()
    scaler_demand.fit(mergedDs[['Demand_MW']])
    scaler_supply.fit(mergedDs[['Supply_MW']])

    if not hasattr(NSGA2_CNN_ModelDS, "creator_built"):
        creator.create("FitnessMultiCNN", base.Fitness, weights=(-1.0, -1.0))
        creator.create("IndividualCNN", list, fitness=creator.FitnessMultiCNN)
        NSGA2_CNN_ModelDS.creator_built = True

    toolbox = base.Toolbox()
    toolbox.register("filters1", random.choice, [32, 64, 128])
    toolbox.register("kernel_size1", random.choice, [2, 3, 5])
    toolbox.register("dropout1", random.uniform, 0.1, 0.5)
    toolbox.register("dense_units", random.choice, [32, 64, 128])
    toolbox.register("learning_rate", random.choice, [1e-2, 1e-3, 1e-4])
    toolbox.register("individual", tools.initCycle, creator.IndividualCNN,
                     (toolbox.filters1, toolbox.kernel_size1, toolbox.dropout1, toolbox.dense_units, toolbox.learning_rate), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def compute_full_metrics(y_true, y_pred):
        y_demand_pred = scaler_demand.inverse_transform(y_pred[:, 0].reshape(-1, 1))
        y_supply_pred = scaler_supply.inverse_transform(y_pred[:, 1].reshape(-1, 1))
        y_demand_true = scaler_demand.inverse_transform(y_true[:, 0].reshape(-1, 1))
        y_supply_true = scaler_supply.inverse_transform(y_true[:, 1].reshape(-1, 1))
        # Ensure all are real
        y_demand_pred = np.real(y_demand_pred)
        y_supply_pred = np.real(y_supply_pred)
        y_demand_true = np.real(y_demand_true)
        y_supply_true = np.real(y_supply_true)
        mse_demand = mean_squared_error(y_demand_true, y_demand_pred)
        mse_supply = mean_squared_error(y_supply_true, y_supply_pred)
        # Force to real float (in case metrics return complex)
        mse_demand = float(np.real(mse_demand))
        mse_supply = float(np.real(mse_supply))
        return mse_demand, mse_supply

    def evaluate(individual):
        try:
            filters1 = int(individual[0])
            kernel_size1 = int(individual[1])
            dropout1 = float(individual[2])
            dense_units = int(individual[3])
            learning_rate = float(individual[4])

            tf.keras.backend.clear_session()
            model = Sequential([
                Conv1D(filters=filters1, kernel_size=kernel_size1, activation='relu', input_shape=(max_len, n_features)),
                MaxPooling1D(pool_size=2),
                Dropout(dropout1),
                Flatten(),
                Dense(dense_units, activation='relu'),
                Dense(2)
            ])
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
            model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping],
                verbose=0
            )
            y_pred = model.predict(X_test, verbose=0)
            mse_demand, mse_supply = compute_full_metrics(y_test, y_pred)
            # Ensure fitness values are real floats
            mse_demand = float(np.real(mse_demand))
            mse_supply = float(np.real(mse_supply))
            return mse_demand, mse_supply
        except Exception as e:
            print(f"Error evaluating individual {individual}: {e}")
            return (float('inf'), float('inf'))

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", custom_mutate, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=pop_size)
    NGEN = ngen

    for gen in range(NGEN):
        print(f"===== Generation {gen + 1} =====")
        offspring = algorithms.varAnd(pop, toolbox, cxpb=cxpb, mutpb=mutpb)
        fits = list(map(toolbox.evaluate, offspring))
        for fit, ind in zip(fits, offspring):
            if isinstance(fit, tuple) and len(fit) == 2:
                ind.fitness.values = fit
            else:
                ind.fitness.values = (float('inf'), float('inf'))
        valid_offspring = [ind for ind in offspring if ind.fitness.valid and len(ind.fitness.values) == 2]
        valid_parents = [ind for ind in pop if ind.fitness.valid and len(ind.fitness.values) == 2]
        total_valid = valid_offspring + valid_parents
        if len(total_valid) < len(pop):
            extra = toolbox.population(n=(len(pop) - len(total_valid)))
            for ind in extra:
                fit = toolbox.evaluate(ind)
                if isinstance(fit, tuple) and len(fit) == 2:
                    ind.fitness.values = fit
                else:
                    ind.fitness.values = (float('inf'), float('inf'))
            total_valid += extra
        pop = toolbox.select(total_valid, k=len(pop))

    valid_inds = [ind for ind in pop if ind.fitness.valid]
    if not valid_inds:
        raise ValueError("No valid individuals found in the final population.")

    best_front = tools.sortNondominated(valid_inds, len(valid_inds), first_front_only=True)[0]
    best = best_front[0]
    filters1, kernel_size1, dropout1, dense_units, learning_rate = best

    tf.keras.backend.clear_session()
    final_model = Sequential([
        Conv1D(filters=int(filters1), kernel_size=int(kernel_size1), activation='relu', input_shape=(max_len, n_features)),
        MaxPooling1D(pool_size=2),
        Dropout(float(dropout1)),
        Flatten(),
        Dense(int(dense_units), activation='relu'),
        Dense(2)
    ])
    final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=float(learning_rate)), loss='mse')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    final_model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=0
    )
    final_pred = final_model.predict(X_test, verbose=0)

    y_demand_pred = scaler_demand.inverse_transform(final_pred[:, 0].reshape(-1, 1))
    y_supply_pred = scaler_supply.inverse_transform(final_pred[:, 1].reshape(-1, 1))
    y_demand_true = scaler_demand.inverse_transform(y_test[:, 0].reshape(-1, 1))
    y_supply_true = scaler_supply.inverse_transform(y_test[:, 1].reshape(-1, 1))

    # Ensure all are real
    y_demand_pred = np.real(y_demand_pred)
    y_supply_pred = np.real(y_supply_pred)
    y_demand_true = np.real(y_demand_true)
    y_supply_true = np.real(y_supply_true)

    
    
    all_true = np.concatenate((y_demand_true, y_supply_true), axis=0)
    all_pred = np.concatenate((y_demand_pred, y_supply_pred), axis=0)
    all_true = np.real(all_true)
    all_pred = np.real(all_pred)

    mse = mean_squared_error(all_true, all_pred)
    mae = mean_absolute_error(all_true, all_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_true, all_pred)
    modelName = "NSGA-II CNN"
    results = [mse, mae, rmse, r2, modelName]


    print('\n', "NSGA-II CNN Model Results:")
    print(f"Best Params: filters1={int(filters1)}, kernel_size1={int(kernel_size1)}, dropout1={float(dropout1):.2f}, dense_units={int(dense_units)}, learning_rate={float(learning_rate)}")
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))

    assert len(test_month_labels) == y_demand_true.shape[0] == y_demand_pred.shape[0]
    if plots == True:
        plt.figure(figsize=(12, 6))
        plt.plot(test_month_labels, y_demand_true, label='Actual Demand', marker='o')
        plt.plot(test_month_labels, y_demand_pred, label='Predicted Demand', marker='s')
        plt.title('NSGA-2 CNN Model: Actual vs Predicted Demand: Monthly')
        plt.xlabel('Month')
        plt.ylabel('MW')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.8)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(test_month_labels, y_supply_true, label='Actual Supply', marker='o')
        plt.plot(test_month_labels, y_supply_pred, label='Predicted Supply', marker='s')
        plt.title('NSGA-2 CNN Model: Actual vs Predicted Supply: Monthly')
        plt.xlabel('Month')
        plt.ylabel('MW')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.8)
        plt.legend()
        plt.tight_layout()
        plt.show()

        """
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true, label='Actual Demand')
        plt.plot(y_demand_pred, label='Predicted Demand')
        plt.title('NSGA-II CNN: Actual vs Predicted Demand')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(y_supply_true, label='Actual Supply')
        plt.plot(y_supply_pred, label='Predicted Supply')
        plt.title('NSGA-II CNN: Actual vs Predicted Supply')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
        """
    return results    


def NSGA3_CNN_ModelDS(
    mergedDs: pd.DataFrame,
    plots:bool,
    pop_size=5,
    ngen=3,
    cxpb=0.7,
    mutpb=0.3,
    seq_length=24,
    epochs=20,
    batch_size=16
):


    mergedDs = mergedDs.copy()
    mergedDs.replace(-999, np.nan, inplace=True)
    mergedDs.dropna(inplace=True)
    mergedDs["hour"] = mergedDs["TimeStamp"].dt.hour
    mergedDs["day"] = mergedDs["TimeStamp"].dt.day
    mergedDs["month"] = mergedDs["TimeStamp"].dt.month

    feature_cols = [
    "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
    "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","WS10M","WD10M","hour","day","month"
    ]
    target_cols = ['Demand_MW', 'Supply_MW']
    """
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X = scaler_X.fit_transform(mergedDs[feature_cols])
    y = scaler_y.fit_transform(mergedDs[target_cols])

    X_seq, y_seq = create_sequences_with_time(X, y, seq_length)
    num_features = len(feature_cols)
    X_seq = X_seq.reshape((-1, seq_length, num_features))
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.3, shuffle=False
    )
    """
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW','Supply_MW']

    # Optionally scale the whole DataFrame first
    mergedDs[feature_cols] = scaler_X.fit_transform(mergedDs[feature_cols])
    mergedDs[target_cols] = scaler_y.fit_transform(mergedDs[target_cols])

    # Create monthly sequences
    X, y = create_monthly_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
    #X = X.reshape(X.shape[0], -1) # Flattening
    # Now split into train/test (e.g., 70% train, 30% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    mergedDs['year'] = mergedDs['TimeStamp'].dt.year
    mergedDs['month'] = mergedDs['TimeStamp'].dt.month
    max_len = X.shape[1]
    n_features = X.shape[2]
    months = sorted(mergedDs.groupby(['year', 'month']).groups.keys())
    month_labels = [f"{y}-{m:02d}" for (y, m) in months]

    months = sorted(mergedDs.groupby(['year', 'month']).groups.keys())
    month_labels = [f"{y}-{m:02d}" for (y, m) in months]
    split_idx = int(len(months) * 0.8)
    test_month_labels = month_labels[split_idx:]
        
    scaler_demand = MinMaxScaler()
    scaler_supply = MinMaxScaler()
    scaler_demand.fit(mergedDs[['Demand_MW']])
    scaler_supply.fit(mergedDs[['Supply_MW']])

    # Reference points for NSGA-III (for 2 objectives, just [1,0] and [0,1])
    ref_points = tools.uniform_reference_points(nobj=2, p=12)

    if not hasattr(NSGA3_CNN_ModelDS, "creator_built"):
        creator.create("FitnessMultiCNN3", base.Fitness, weights=(-1.0, -1.0))
        creator.create("IndividualCNN3", list, fitness=creator.FitnessMultiCNN3)
        NSGA3_CNN_ModelDS.creator_built = True

    toolbox = base.Toolbox()
    toolbox.register("filters1", random.choice, [32, 64, 128])
    toolbox.register("kernel_size1", random.choice, [2, 3, 5])
    toolbox.register("dropout1", random.uniform, 0.1, 0.5)
    toolbox.register("dense_units", random.choice, [32, 64, 128])
    toolbox.register("learning_rate", random.choice, [1e-2, 1e-3, 1e-4])
    toolbox.register("individual", tools.initCycle, creator.IndividualCNN3,
                     (toolbox.filters1, toolbox.kernel_size1, toolbox.dropout1, toolbox.dense_units, toolbox.learning_rate), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def compute_full_metrics(y_true, y_pred):
        y_demand_pred = scaler_demand.inverse_transform(y_pred[:, 0].reshape(-1, 1))
        y_supply_pred = scaler_supply.inverse_transform(y_pred[:, 1].reshape(-1, 1))
        y_demand_true = scaler_demand.inverse_transform(y_true[:, 0].reshape(-1, 1))
        y_supply_true = scaler_supply.inverse_transform(y_true[:, 1].reshape(-1, 1))
        # Ensure all are real
        y_demand_pred = np.real(y_demand_pred)
        y_supply_pred = np.real(y_supply_pred)
        y_demand_true = np.real(y_demand_true)
        y_supply_true = np.real(y_supply_true)
        mse_demand = mean_squared_error(y_demand_true, y_demand_pred)
        mse_supply = mean_squared_error(y_supply_true, y_supply_pred)
        # Force to real float (in case metrics return complex)
        mse_demand = float(np.real(mse_demand))
        mse_supply = float(np.real(mse_supply))
        return mse_demand, mse_supply

    def evaluate(individual):
        try:
            filters1 = int(individual[0])
            kernel_size1 = int(individual[1])
            dropout1 = float(individual[2])
            dense_units = int(individual[3])
            learning_rate = float(individual[4])

            tf.keras.backend.clear_session()
            model = Sequential([
                Conv1D(filters=filters1, kernel_size=kernel_size1, activation='relu', input_shape=(max_len, n_features)),
                MaxPooling1D(pool_size=2),
                Dropout(dropout1),
                Flatten(),
                Dense(dense_units, activation='relu'),
                Dense(2)
            ])
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
            model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping],
                verbose=0
            )
            y_pred = model.predict(X_test, verbose=0)
            mse_demand, mse_supply = compute_full_metrics(y_test, y_pred)
            # Ensure fitness values are real floats
            mse_demand = float(np.real(mse_demand))
            mse_supply = float(np.real(mse_supply))
            return mse_demand, mse_supply
        except Exception as e:
            print(f"Error evaluating individual {individual}: {e}")
            return (float('inf'), float('inf'))

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", custom_mutate, indpb=0.2)
    toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

    pop = toolbox.population(n=pop_size)
    NGEN = ngen

    for gen in range(NGEN):
        print(f"===== Generation {gen + 1} =====")
        offspring = algorithms.varAnd(pop, toolbox, cxpb=cxpb, mutpb=mutpb)
        fits = list(map(toolbox.evaluate, offspring))
        for fit, ind in zip(fits, offspring):
            if isinstance(fit, tuple) and len(fit) == 2:
                ind.fitness.values = fit
            else:
                ind.fitness.values = (float('inf'), float('inf'))
        valid_offspring = [ind for ind in offspring if ind.fitness.valid and len(ind.fitness.values) == 2]
        valid_parents = [ind for ind in pop if ind.fitness.valid and len(ind.fitness.values) == 2]
        total_valid = valid_offspring + valid_parents
        if len(total_valid) < len(pop):
            extra = toolbox.population(n=(len(pop) - len(total_valid)))
            for ind in extra:
                fit = toolbox.evaluate(ind)
                if isinstance(fit, tuple) and len(fit) == 2:
                    ind.fitness.values = fit
                else:
                    ind.fitness.values = (float('inf'), float('inf'))
            total_valid += extra
        pop = toolbox.select(total_valid, k=len(pop))

    valid_inds = [ind for ind in pop if ind.fitness.valid]
    if not valid_inds:
        raise ValueError("No valid individuals found in the final population.")

    best_front = tools.sortNondominated(valid_inds, len(valid_inds), first_front_only=True)[0]
    best = best_front[0]
    filters1, kernel_size1, dropout1, dense_units, learning_rate = best

    tf.keras.backend.clear_session()
    final_model = Sequential([
        Conv1D(filters=int(filters1), kernel_size=int(kernel_size1), activation='relu', input_shape=(max_len, n_features)),
        MaxPooling1D(pool_size=2),
        Dropout(float(dropout1)),
        Flatten(),
        Dense(int(dense_units), activation='relu'),
        Dense(2)
    ])
    final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=float(learning_rate)), loss='mse')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    final_model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=0
    )
    final_pred = final_model.predict(X_test, verbose=0)

    y_demand_pred = scaler_demand.inverse_transform(final_pred[:, 0].reshape(-1, 1))
    y_supply_pred = scaler_supply.inverse_transform(final_pred[:, 1].reshape(-1, 1))
    y_demand_true = scaler_demand.inverse_transform(y_test[:, 0].reshape(-1, 1))
    y_supply_true = scaler_supply.inverse_transform(y_test[:, 1].reshape(-1, 1))

    # Ensure all are real
    y_demand_pred = np.real(y_demand_pred)
    y_supply_pred = np.real(y_supply_pred)
    y_demand_true = np.real(y_demand_true)
    y_supply_true = np.real(y_supply_true)
         
    all_true = np.concatenate((y_demand_true, y_supply_true), axis=0)
    all_pred = np.concatenate((y_demand_pred, y_supply_pred), axis=0)
    all_true = np.real(all_true)
    all_pred = np.real(all_pred)

    mse = mean_squared_error(all_true, all_pred)
    mae = mean_absolute_error(all_true, all_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_true, all_pred)
    modelName = "NSGA-III CNN"
    results = [mse, mae, rmse, r2, modelName]

    print('\n', "NSGA-III CNN Model Results:")
    print(f"Best Params: filters1={int(filters1)}, kernel_size1={int(kernel_size1)}, dropout1={float(dropout1):.2f}, dense_units={int(dense_units)}, learning_rate={float(learning_rate)}")
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))
    assert len(test_month_labels) == y_demand_true.shape[0] == y_demand_pred.shape[0]
    if plots == True:
        plt.figure(figsize=(12, 6))
        plt.plot(test_month_labels, y_demand_true, label='Actual Demand', marker='o')
        plt.plot(test_month_labels, y_demand_pred, label='Predicted Demand', marker='s')
        plt.title('NSGA-3 CNN Model: Actual vs Predicted Demand: Monthly')
        plt.xlabel('Month')
        plt.ylabel('MW')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.8)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(test_month_labels, y_supply_true, label='Actual Supply', marker='o')
        plt.plot(test_month_labels, y_supply_pred, label='Predicted Supply', marker='s')
        plt.title('NSGA-3 CNNModel: Actual vs Predicted Supply: Monthly')
        plt.xlabel('Month')
        plt.ylabel('MW')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.8)
        plt.legend()
        plt.tight_layout()
        plt.show()
        """
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true, label='Actual Demand')
        plt.plot(y_demand_pred, label='Predicted Demand')
        plt.title('NSGA-III CNN: Actual vs Predicted Demand')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(y_supply_true, label='Actual Supply')
        plt.plot(y_supply_pred, label='Predicted Supply')
        plt.title('NSGA-III CNN: Actual vs Predicted Supply')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
        """
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


DecTree = decisionTreeModelDS(sp_FullMerg,False)

tf.keras.backend.clear_session()

randForest = randomForestModelDS(sp_FullMerg, False)
tf.keras.backend.clear_session()

xgb_results= xgbModelDS(sp_FullMerg, False)
tf.keras.backend.clear_session()

gb = gbModelDS(sp_FullMerg, False)
tf.keras.backend.clear_session()

gbdt = GBDTModelDS(sp_FullMerg, False)
tf.keras.backend.clear_session()

blstm = biDirectionalLSTMDS(sp_FullMerg, False)
tf.keras.backend.clear_session()

lstm = LSTMModelDS(sp_FullMerg, False)
tf.keras.backend.clear_session()

gru = GRUModelDS(sp_FullMerg, False)
tf.keras.backend.clear_session()

svr = SVRModelDS(sp_FullMerg, False)
tf.keras.backend.clear_session()

mlp = MLPModelDS(sp_FullMerg, False)
tf.keras.backend.clear_session()

cnn = CNNModelDS(sp_FullMerg, False)
tf.keras.backend.clear_session()

nsga2cnn = NSGA2_CNN_ModelDS(sp_FullMerg, False)
tf.keras.backend.clear_session()
nsga3cnn = NSGA3_CNN_ModelDS(sp_FullMerg,False)
modelresults = [DecTree,randForest,xgb_results,gbdt,blstm,lstm,gru,svr,mlp,cnn,nsga2cnn,nsga3cnn,]

BestResultsOrdered = BetterModelSelectionMethod(modelresults)

print("\nModel Ranking (Best to Worst) Monthly:")
print("{:<20} {:>12} {:>12} {:>12} {:>10}".format("Model", "MSE", "MAE", "RMSE", "R2"))
print("-" * 70)
for res in BestResultsOrdered:
    print("{:<20} {:>12.4f} {:>12.4f} {:>12.4f} {:>10.4f}".format(
        res[4], res[0], res[1], res[2], res[3]
    ))

