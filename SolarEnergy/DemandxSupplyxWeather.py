#----# Libraries
import array
from ast import Str
from calendar import c
from ctypes import Array
from enum import Enum
from re import M
from tkinter import CURRENT
from tokenize import String
import numpy as np
import pandas as pd
from pandas.core.indexes import multi
import scipy as sp
import xgboost as xgb
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, GRU, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from keras_tuner import RandomSearch, Hyperband, BayesianOptimization
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from deap import base, creator, tools, algorithms

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
sp_FullMerg = pd.merge(sp_WeatherDe, sp_FullMerg, on="TimeStamp", how="inner")

##linear interpolation to smooth out the datasets
##So far has worked out great for supply but demand is getting bad results.
sp_FullMerg.interpolate(method='linear', inplace=True)
print(sp_FullMerg.head())

#desave_loc = r"E:\AI Lecture Notes\Datasets\Merged\DemandxWeather.csv"
##susave_loc = r"E:\AI Lecture Notes\Datasets\Merged\SupplyxWeather.csv"
#sp_WeatherxDem.to_csv(desave_loc, index=False)
##sp_WeatherxSup.to_csv(susave_loc, index=False)

#------#
def decisionTreeModelDS(mergedDs: pd.DataFrame):
    mergedDs = mergedDs.copy()

    """
    Decision Tree Results for Merged Dataset -
    MSE: 9269.3319
    MAE: 78.9769
    RMSE: 96.2774
    R2: -0.8230
    --
    With Hyper Params -  {'estimator__max_depth': 5, 'estimator__max_features': 'sqrt', 'estimator__min_samples_leaf': 4, 'estimator__min_samples_split': 2} Best Params -                                                                                                                                                  
    MSE: 7073.9328
    MAE: 66.5279
    RMSE: 84.1067
    R2: -0.6290
    --
    """

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
            # Pad with zeros if needed
            if pad_to_max and x_seq.shape[0] < max_len:
                pad_width = ((0, max_len - x_seq.shape[0]), (0, 0))
                x_seq = np.pad(x_seq, pad_width, mode='constant')
            X.append(x_seq)
            y.append(next_month[target_cols].values[0])  # or customize as needed

        X = np.array(X)
        y = np.array(y)
        return X, y

    #----#

    mergedDs.replace(-999, np.nan, inplace=True)
    mergedDs.dropna(inplace=True)

    ###minLength = min(len(demandDs))
    ###demandDs = demandDs.iloc[minLength]

    #Extraction of Data.

    mergedDs["hour"] = mergedDs["TimeStamp"].dt.hour
    mergedDs["day"] = mergedDs["TimeStamp"].dt.day
    mergedDs["month"] = mergedDs["TimeStamp"].dt.month

    feature_cols = [
        "ALLSKY_SFC_SW_DWN_S", "ALLSKY_SFC_UV_INDEX_S", "T2M_S", "PRECTOTCORR_S", "ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S", "RH2M_S", "PS_S", "PSC_S", "ALLSKY_SFC_SW_DWN_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "PRECTOTCORR_D", "ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D", "RH2M_D", "PS_D", "PSC_D", "hour", "day", "month"
    ]
    """
    feature_cols = [
    "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
    "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","hour","day","month"
]

    """

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW', 'Supply_MW']
    X = scaler_X.fit_transform(mergedDs[feature_cols])
    y = scaler_y.fit_transform(mergedDs[target_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False
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
    X = X.reshape(X.shape[0], -1)
    # Now split into train/test (e.g., 70% train, 30% test)
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    """
    
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
    results = [mse, mae, rmse, r2]
    #----#
    print('\n', "Merged Data Decision Tree Model Results:")
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))
    """
    #-Demand Plot-#
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:, 0], label='Actual Demand')
    plt.plot(y_demand_pred[:, 0], label='Predicted Demand')
    plt.title(f'Decision Tree Model: Actual vs Predicted Demand')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()
    #-Supply Plot-#
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:, 1], label='Actual Supply')
    plt.plot(y_demand_pred[:, 1], label='Predicted Supply')
    plt.title(f'Decision Tree Model: Actual vs Predicted Supply')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()
    """
    return results


def randomForestModelDS(mergedDs: pd.DataFrame):
    mergedDs = mergedDs.copy()

    """
    Random Forest Results for Merged Dataset -
    MSE: 6698.6218
    MAE: 64.0826
    RMSE: 81.8451
    R2: -0.6079
    --
    With Hyper Params -  {'estimator__max_depth': 10, 'estimator__max_features': 'sqrt', 'estimator__min_samples_leaf': 2, 'estimator__min_samples_split': 5, 'estimator__n_estimators': 50} - Best Params
    MSE: 3083.7902
    MAE: 44.4229
    RMSE: 55.5319
    R2: 0.3781
    --
    """

    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length].flatten())
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)

    mergedDs.replace(-999, np.nan, inplace=True)
    mergedDs.dropna(inplace=True)

    ###minLength = min(len(demandDs))
    ###demandDs = demandDs.iloc[minLength]

    #Extraction of Data.

    mergedDs["hour"] = mergedDs["TimeStamp"].dt.hour
    mergedDs["day"] = mergedDs["TimeStamp"].dt.day
    mergedDs["month"] = mergedDs["TimeStamp"].dt.month

    feature_cols = [
        "ALLSKY_SFC_SW_DWN_S", "ALLSKY_SFC_UV_INDEX_S", "T2M_S", "PRECTOTCORR_S", "ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S", "RH2M_S", "PS_S", "PSC_S", "ALLSKY_SFC_SW_DWN_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "PRECTOTCORR_D", "ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D", "RH2M_D", "PS_D", "PSC_D", "hour", "day", "month"
    ]
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW', 'Supply_MW']
    X = scaler_X.fit_transform(mergedDs[feature_cols])
    y = scaler_y.fit_transform(mergedDs[target_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )
    
    param_grid = {
        'estimator__n_estimators': [50, 100, 200],
        'estimator__max_depth': [5, 10, 20, None],
        'estimator__min_samples_split': [2, 5, 10],
        'estimator__min_samples_leaf': [1, 2, 4],
        'estimator__max_features': ['auto', 'sqrt', 'log2', None]
    }
    base_model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
    grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print("Best parameters:\n", grid_search.best_params_)

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
    results = [mse, mae, rmse, r2]
    #----#
    print('\n', "Merged Random Forest Model Results:")
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))
    """
    #-Demand Plot-#
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:, 0], label='Actual Demand')
    plt.plot(y_demand_pred[:, 0], label='Predicted Demand')
    plt.title(f'Random Forest Model: Actual vs Predicted Demand')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()
    #-Supply Plot-#
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:, 1], label='Actual Supply')
    plt.plot(y_demand_pred[:, 1], label='Predicted Supply')
    plt.title(f'Random Forest Model: Actual vs Predicted Supply')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()
    """
    return results


def xgbModelDS(mergedDs: pd.DataFrame):
    mergedDs = mergedDs.copy()

    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length].flatten())
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)

    """
    XGB for Merged Dataset -  {'estimator__colsample_bytree': 0.8, 'estimator__learning_rate': 0.1, 'estimator__max_depth': 3, 'estimator__n_estimators': 50, 'estimator__subsample': 0.8} - Best Params
    MSE: 6849.3777
    MAE: 69.5737
    RMSE: 82.7610
    R2: -0.4499
    -- 
    With Hyper Params - 
    MSE: 4750.1360
    MAE: 54.7361
    RMSE: 68.9212
    R2: -0.0966

    """
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
        "ALLSKY_SFC_SW_DWN_S", "ALLSKY_SFC_UV_INDEX_S", "T2M_S", "PRECTOTCORR_S", "ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S", "RH2M_S", "PS_S", "PSC_S", "ALLSKY_SFC_SW_DWN_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "PRECTOTCORR_D", "ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D", "RH2M_D", "PS_D", "PSC_D", "hour", "day", "month"
    ]
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW', 'Supply_MW']
    X = scaler_X.fit_transform(mergedDs[feature_cols])
    y = scaler_y.fit_transform(mergedDs[target_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )
    
    param_grid = {
        'estimator__n_estimators': [50, 100, 200],
        'estimator__max_depth': [3, 5, 10],
        'estimator__learning_rate': [0.01, 0.1, 0.2],
        'estimator__subsample': [0.8, 1.0],
        'estimator__colsample_bytree': [0.8, 1.0]
    }
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
    results = [mse, mae, rmse, r2]

    #----#
    print('\n', "Merged XGB Model Results:")
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))
    """
    #-Demand Plot-#
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:, 0], label='Actual Demand')
    plt.plot(y_demand_pred[:, 0], label='Predicted Demand')
    plt.title(f'XGB Model: Actual vs Predicted Demand')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()
    #-Supply Plot-#
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:, 1], label='Actual Supply')
    plt.plot(y_demand_pred[:, 1], label='Predicted Supply')
    plt.title(f'XGB Model: Actual vs Predicted Supply')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()
    """
    return results


def gbModelDS(mergedDs: pd.DataFrame):
    mergedDs = mergedDs.copy()

    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length].flatten())
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)

    """
    GradientBoosting for Merged Dataset -
    MSE: 5705.4124
    MAE: 62.9679
    RMSE: 75.5342
    R2: -0.2487
    --
    With hyper params -  {'estimator__learning_rate': 0.1, 'estimator__max_depth': 3, 'estimator__max_features': 'sqrt', 'estimator__min_samples_leaf': 2, 'estimator__min_samples_split': 10, 'estimator__n_estimators': 100} - Best params
    MSE: 4380.3926
    MAE: 56.7835
    RMSE: 66.1845
    R2: 0.1103
    --
    """

    mergedDs["hour"] = mergedDs["TimeStamp"].dt.hour
    mergedDs["day"] = mergedDs["TimeStamp"].dt.day
    mergedDs["month"] = mergedDs["TimeStamp"].dt.month

    feature_cols = [
        "ALLSKY_SFC_SW_DWN_S", "ALLSKY_SFC_UV_INDEX_S", "T2M_S", "PRECTOTCORR_S", "ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S", "RH2M_S", "PS_S", "PSC_S", "ALLSKY_SFC_SW_DWN_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "PRECTOTCORR_D", "ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D", "RH2M_D", "PS_D", "PSC_D", "hour", "day", "month"
    ]
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW', 'Supply_MW']
    X = scaler_X.fit_transform(mergedDs[feature_cols])
    y = scaler_y.fit_transform(mergedDs[target_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )
    
    param_grid = {
        'estimator__n_estimators': [50, 100, 200],
        'estimator__max_depth': [3, 5, 10],
        'estimator__learning_rate': [0.01, 0.1, 0.2],
        'estimator__min_samples_split': [2, 5, 10],
        'estimator__min_samples_leaf': [1, 2, 4],
        'estimator__max_features': ['auto', 'sqrt', 'log2', None]
    }
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
    results = [mse, mae, rmse, r2]
    """
    #----#
    print('\n', "Merged GB Model Results:")
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))
    #-Demand Plot-#
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:, 0], label='Actual Demand')
    plt.plot(y_demand_pred[:, 0], label='Predicted Demand')
    plt.title(f'Gradient Boosting Model: Actual vs Predicted Demand')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()
    #-Supply Plot-#
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:, 1], label='Actual Supply')
    plt.plot(y_demand_pred[:, 1], label='Predicted Supply')
    plt.title(f'Gradient Boosting Model: Actual vs Predicted Supply')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()
    """
    return results


def biDirectionalLSTMDS(mergedDs: pd.DataFrame):
    mergedDs = mergedDs.copy()

    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)

    """
    Bidirectional LSTM for Merged Dataset -
    MSE: 2366.0192
    MAE: 37.8718
    RMSE: 48.6417   
    R2: 0.2168
    --
    With Hyper Params -  {'units1': 32, 'dropout1': 0.1, 'units2': 128, 'dense_units': 32, 'learning_rate': 0.0001, 'tuner/epochs': 7, 'tuner/initial_epoch': 3, 'tuner/bracket': 2, 'tuner/round': 1, 'tuner/trial_id': '0008'} - Best Params
    MSE: 1729.6055
    MAE: 32.9587
    RMSE: 41.5885
    R2: 0.4663
    """

    #----#
    ##Basic transformation for the base dataset
    mergedDs["hour"] = mergedDs["TimeStamp"].dt.hour
    mergedDs["day"] = mergedDs["TimeStamp"].dt.day
    mergedDs["month"] = mergedDs["TimeStamp"].dt.month

    feature_cols = [
        "ALLSKY_SFC_SW_DWN_S", "ALLSKY_SFC_UV_INDEX_S", "T2M_S", "PRECTOTCORR_S", "ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S", "RH2M_S", "PS_S", "PSC_S", "ALLSKY_SFC_SW_DWN_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "PRECTOTCORR_D", "ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D", "RH2M_D", "PS_D", "PSC_D", "hour", "day", "month"
    ]
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
            input_shape=(seq_length, X_train.shape[2])
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
    """
    
    ## Base model implementation
    model = Sequential([
        Bidirectional(LSTM(100, return_sequences=True, input_shape=(seq_length, X_train.shape[2]))),
        Dropout(0.2),
        Bidirectional(LSTM(100, return_sequences=False)),
        Dense(64, activation='relu'),
        Dense(2)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test),
              callbacks=[early_stopping])

    y_pred = model.predict(X_test)
    """
    y_demand_true = scaler_y.inverse_transform(y_test)
    y_demand_pred = scaler_y.inverse_transform(y_pred)

    mse = mean_squared_error(y_demand_true, y_demand_pred)
    mae = mean_absolute_error(y_demand_true, y_demand_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_demand_true, y_demand_pred)
    results = [mse, mae, rmse, r2]

    #----#
    print('\n', "Merged BLSTM Model Results:")
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))
    """
    #-Demand Plot-#
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:, 0], label='Actual Demand')
    plt.plot(y_demand_pred[:, 0], label='Predicted Demand')
    plt.title(f'BLSTM Model: Actual vs Predicted Demand')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()
    #-Supply Plot-#
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:, 1], label='Actual Supply')
    plt.plot(y_demand_pred[:, 1], label='Predicted Supply')
    plt.title(f'BLSTM Model: Actual vs Predicted Supply')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()
    """
    return results


def LSTMModelDS(mergedDs: pd.DataFrame):
    mergedDs = mergedDs.copy()

    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)

    """
    LSTM for Merged Dataset -
    MSE: 2843.6818
    MAE: 42.5091
    RMSE: 53.3262
    R2: 0.3143

    --
    With Hyper Params - {'units1': 32, 'dropout1': 0.5, 'units2': 128, 'dense_units': 32, 'learning_rate': 0.0001, 'tuner/epochs': 20, 'tuner/initial_epoch': 7, 'tuner/bracket': 2, 'tuner/round': 2, 'tuner/trial_id': '0013'} - Best params
    MSE: 2679.7903
    MAE: 37.0395
    RMSE: 51.7667
    R2: 0.3878
    """

    mergedDs["hour"] = mergedDs["TimeStamp"].dt.hour
    mergedDs["day"] = mergedDs["TimeStamp"].dt.day
    mergedDs["month"] = mergedDs["TimeStamp"].dt.month

    feature_cols = [
        "ALLSKY_SFC_SW_DWN_S", "ALLSKY_SFC_UV_INDEX_S", "T2M_S", "PRECTOTCORR_S", "ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S", "RH2M_S", "PS_S", "PSC_S", "ALLSKY_SFC_SW_DWN_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "PRECTOTCORR_D", "ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D", "RH2M_D", "PS_D", "PSC_D", "hour", "day", "month"
    ]
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
                input_shape=(seq_length, X_train.shape[2])
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
    results = [mse, mae, rmse, r2]

    #----#
    print('\n', "Merged LSTM Model Results:")
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))
    """
    #-Demand Plot-#
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:, 0], label='Actual Demand')
    plt.plot(y_demand_pred[:, 0], label='Predicted Demand')
    plt.title(f'LSTM Model: Actual vs Predicted Demand')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()
    #-Supply Plot-#
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:, 1], label='Actual Supply')
    plt.plot(y_demand_pred[:, 1], label='Predicted Supply')
    plt.title(f'LSTM Tree Model: Actual vs Predicted Supply')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()
    """
    return results


def GRUModelDS(mergedDs: pd.DataFrame):
    mergedDs = mergedDs.copy()

    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)

    """
    GRU for Merged Dataset -
    MSE: 2876.9410
    MAE: 41.2654
    RMSE: 53.6371
    R2: 0.0266
    --
    With Hyper Params - {'units1': 32, 'dropout1': 0.4, 'units2': 32, 'dense_units': 128, 'learning_rate': 0.001, 'tuner/epochs': 7, 'tuner/initial_epoch': 0, 'tuner/bracket': 1, 'tuner/round': 0} - Best Params

    MSE: 2108.9465
    MAE: 33.3279
    RMSE: 45.9233
    R2: 0.3420
    --
    """

    #----#
    ##Basic transformation for the base dataset
    mergedDs["hour"] = mergedDs["TimeStamp"].dt.hour
    mergedDs["day"] = mergedDs["TimeStamp"].dt.day
    mergedDs["month"] = mergedDs["TimeStamp"].dt.month

    feature_cols = [
        "ALLSKY_SFC_SW_DWN_S", "ALLSKY_SFC_UV_INDEX_S", "T2M_S", "PRECTOTCORR_S", "ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S", "RH2M_S", "PS_S", "PSC_S", "ALLSKY_SFC_SW_DWN_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "PRECTOTCORR_D", "ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D", "RH2M_D", "PS_D", "PSC_D", "hour", "day", "month"
    ]
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
                input_shape=(seq_length, X_train.shape[2])
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
    results = [mse, mae, rmse, r2]

    #----#
    print('\n', "Merged GRU Model Results:")
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))
    """
    #-Demand Plot-#
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:, 0], label='Actual Demand')
    plt.plot(y_demand_pred[:, 0], label='Predicted Demand')
    plt.title(f'GRU Model: Actual vs Predicted Demand')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()
    #-Supply Plot-#
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:, 1], label='Actual Supply')
    plt.plot(y_demand_pred[:, 1], label='Predicted Supply')
    plt.title(f'GRU Model: Actual vs Predicted Supply')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()
    """
    return results


def SVRModelDS(mergedDs: pd.DataFrame):
    mergedDs = mergedDs.copy()

    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)

    """
    Additonal Note the base results for SVR very bad to the point I thought they were broken I made a basic hyper param and cycled
    through features to see if I could get better results but will need clarification if what I've done is correct in anyway. Though have
    kept the base implementation for comparisons sake.
    
    SVR for Merged Dataset -
    MSE: 1682.6278
    MAE: 33.0451
    RMSE: 41.0198
    R2: -0.2768   
    --
    With Hyper Params -
    {'estimator__C': 0.1, 'estimator__epsilon': 0.01, 'estimator__gamma': 1, 'estimator__kernel': 'rbf'} - Best params
    MSE: 1166.4916
    MAE: 25.9020
    RMSE: 34.1539
    R2: 0.3737
    --

    """

    mergedDs["hour"] = mergedDs["TimeStamp"].dt.hour
    mergedDs["day"] = mergedDs["TimeStamp"].dt.day
    mergedDs["month"] = mergedDs["TimeStamp"].dt.month

    feature_cols = [
        "ALLSKY_SFC_SW_DWN_S", "ALLSKY_SFC_UV_INDEX_S", "T2M_S", "PRECTOTCORR_S", "ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S", "RH2M_S", "PS_S", "PSC_S", "ALLSKY_SFC_SW_DWN_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "PRECTOTCORR_D", "ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D", "RH2M_D", "PS_D", "PSC_D", "hour", "day", "month"
    ]
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW', 'Supply_MW']
    X = scaler_X.fit_transform(mergedDs[feature_cols])
    y = scaler_y.fit_transform(mergedDs[target_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    param_grid = {
        'estimator__C': [0.1, 1, 10, 100],           # Regularization, similar to tree depth/complexity
        'estimator__gamma': ['scale', 'auto', 0.01, 0.1, 1],  # Kernel coefficient, like learning rate
        'estimator__epsilon': [0.01, 0.1, 0.5, 1.0], # Insensitivity, similar to min_samples_split
        'estimator__kernel': ['rbf']                 # Keep kernel consistent
    }
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
    results = [mse, mae, rmse, r2]

    #----#
    print('\n', "Merged SVR Model Results:")
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))
    """
    #-Demand Plot-#
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:, 0], label='Actual Demand')
    plt.plot(y_demand_pred[:, 0], label='Predicted Demand')
    plt.title(f'SVR Model: Actual vs Predicted Demand')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()
    #-Supply Plot-#
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:, 1], label='Actual Supply')
    plt.plot(y_demand_pred[:, 1], label='Predicted Supply')
    plt.title(f'SVR Model: Actual vs Predicted Supply')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()
    """
    return results


def MLPModelDS(mergedDs: pd.DataFrame):
    mergedDs = mergedDs.copy()

    """
    MLP Modle Merged Dataset Results -
    MSE: 5693.7312
    MAE: 58.2081
    RMSE: 75.4568
    R2: -1.1991
    --
    With Hyper Params - 
    - {'dense1': 128, 'dropout1': 0.30000000000000004, 'dense2': 32, 'dropout2': 0.5, 'learning_rate': 0.0001, 'tuner/epochs': 7, 'tuner/initial_epoch': 3, 'tuner/bracket': 2, 'tuner/round': 1, 'tuner/trial_id': '0006'} - Best Params
    MSE: 2081.2024
    MAE: 32.2146
    RMSE: 45.6202
    R2: 0.5852
    ---
    """
    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length].flatten())
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)

    # All code below is now at the same indentation as the inner function
    mergedDs["hour"] = mergedDs["TimeStamp"].dt.hour
    mergedDs["day"] = mergedDs["TimeStamp"].dt.day
    mergedDs["month"] = mergedDs["TimeStamp"].dt.month

    feature_cols = [
        "ALLSKY_SFC_SW_DWN_S", "ALLSKY_SFC_UV_INDEX_S", "T2M_S", "PRECTOTCORR_S", "ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S", "RH2M_S", "PS_S", "PSC_S", "ALLSKY_SFC_SW_DWN_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "PRECTOTCORR_D", "ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D", "RH2M_D", "PS_D", "PSC_D", "hour", "day", "month"
    ]
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW', 'Supply_MW']
    X = scaler_X.fit_transform(mergedDs[feature_cols])
    y = scaler_y.fit_transform(mergedDs[target_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )
    
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
            Dense(hp.Int('dense1', 64, 256, step=32), activation='relu', input_dim=X_train.shape[1]),
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
    results = [mse, mae, rmse, r2]

    print('\n', "Merged MLP Model Results:")
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))
    """
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:, 0], label='Actual Demand')
    plt.plot(y_demand_pred[:, 0], label='Predicted Demand')
    plt.title(f'MLP Model: Actual vs Predicted Demand')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:, 1], label='Actual Supply')
    plt.plot(y_demand_pred[:, 1], label='Predicted Supply')
    plt.title(f'MLP Model: Actual vs Predicted Supply')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()
    """

    return results


def CNNModelDS(mergedDs: pd.DataFrame):
    mergedDs = mergedDs.copy()

    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)

    """
    CNN for Merged Dataset -
    MSE: 5767.0716
    MAE: 59.7123
    RMSE: 75.9412
    R2: -0.8570
    --
    With Hyper Params - 
    {'filters1': 32, 'kernel_size1': 2, 'dropout1': 0.5, 'filters2': 64, 'kernel_size2': 3, 'dropout2': 0.5, 'dense_units': 32, 'learning_rate': 0.001, 'tuner/epochs': 3, 'tuner/initial_epoch': 0, 'tuner/bracket': 2, 'tuner/round': 0} - Best Parms
    MSE: 1454.0116
    MAE: 26.8473
    RMSE: 38.1315
    R2: 0.7116
    """
    mergedDs["hour"] = mergedDs["TimeStamp"].dt.hour
    mergedDs["day"] = mergedDs["TimeStamp"].dt.day
    mergedDs["month"] = mergedDs["TimeStamp"].dt.month

    feature_cols = [
        "ALLSKY_SFC_SW_DWN_S", "ALLSKY_SFC_UV_INDEX_S", "T2M_S", "PRECTOTCORR_S", "ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S", "RH2M_S", "PS_S", "PSC_S", "ALLSKY_SFC_SW_DWN_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "PRECTOTCORR_D", "ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D", "RH2M_D", "PS_D", "PSC_D", "hour", "day", "month"
    ]
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
                input_shape=(seq_length, X_train.shape[2])
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
    results = [mse, mae, rmse, r2]

    #---#
    print('\n', "Demand CNN Model Results:")
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))
    """
    #-Demand Plot-#
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:, 0], label='Actual Demand')
    plt.plot(y_demand_pred[:, 0], label='Predicted Demand')
    plt.title(f'CNN Model: Actual vs Predicted Demand')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()
    #-Supply Plot-#
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:, 1], label='Actual Supply')
    plt.plot(y_demand_pred[:, 1], label='Predicted Supply')
    plt.title(f'CNN Model: Actual vs Predicted Supply')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()
    """
    return results


def GBDTModelDS(mergedDs: pd.DataFrame):
    mergedDs = mergedDs.copy()

    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length].flatten())
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)

    """
    GBDT for Merged Dataset -
    MSE: 5280.6335
    MAE: 54.7121
    RMSE: 72.6680
    R2: -1.2277
        
    --
    With Hyper Params -  {'estimator__learning_rate': 0.1, 'estimator__max_depth': 5, 'estimator__max_features': 'sqrt', 'estimator__min_samples_leaf': 4, 'estimator__min_samples_split': 10, 'estimator__n_estimators': 50} - Best Params
    MSE: 2868.0986
    MAE: 40.2360
    RMSE: 53.5546
    R2: -0.0580
    --
    """

    mergedDs["hour"] = mergedDs["TimeStamp"].dt.hour
    mergedDs["day"] = mergedDs["TimeStamp"].dt.day
    mergedDs["month"] = mergedDs["TimeStamp"].dt.month

    feature_cols = [
        "ALLSKY_SFC_SW_DWN_S", "ALLSKY_SFC_UV_INDEX_S", "T2M_S", "PRECTOTCORR_S", "ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S", "RH2M_S", "PS_S", "PSC_S", "ALLSKY_SFC_SW_DWN_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "PRECTOTCORR_D", "ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D", "RH2M_D", "PS_D", "PSC_D", "hour", "day", "month"
    ]
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW', 'Supply_MW']
    X = scaler_X.fit_transform(mergedDs[feature_cols])
    y = scaler_y.fit_transform(mergedDs[target_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )
    
    param_grid = {
        'estimator__n_estimators': [50, 100, 200],
        'estimator__max_depth': [3, 5, 10],
        'estimator__learning_rate': [0.01, 0.1, 0.2],
        'estimator__min_samples_split': [2, 5, 10],
        'estimator__min_samples_leaf': [1, 2, 4],
        'estimator__max_features': ['auto', 'sqrt', 'log2', None]
    }
   
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
    results = [mse, mae, rmse, r2]

    #---#

    print('\n', "Merged GBDT Model Results:")
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))
    """
    #-Demand Plot-#
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:, 0], label='Actual Demand')
    plt.plot(y_demand_pred[:, 0], label='Predicted Demand')
    plt.title(f'GBDT Model: Actual vs Predicted Demand')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()
    #-Supply Plot-#
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:, 1], label='Actual Supply')
    plt.plot(y_demand_pred[:, 1], label='Predicted Supply')
    plt.title(f'GBDT Model: Actual vs Predicted Supply')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()
    """
    return results


def BestModelChoice(setOne: Array, setTwo: Array, modelNameOne: str, modelNameTwo: str):
    scoreOne = 0
    scoreTwo = 0
    bestresult = []
    bestModelName = ''
    if (setOne[0] < setTwo[0]):  # MSE
        scoreOne += 1
    else:
        scoreTwo += 1
    ##
    if (setOne[1] < setTwo[1]):  # MAE
        scoreOne += 1
    else:
        scoreTwo += 1
    ##
    if (setOne[2] < setTwo[2]):  # RMSE
        scoreOne += 1
    else:
        scoreTwo += 1
    ##
    r2OneScore = 1 - setOne[3]
    r2TwoScore = 1 - setTwo[3]
    if setOne[3] < 0 and setTwo[3] >= 0:
        scoreTwo += 1
    elif setTwo[3] < 0 and setOne[3] >= 0:
        scoreOne += 1
    elif setOne[3] >= 0 and setTwo[3] >= 0:
        if r2OneScore > r2TwoScore:
            scoreOne += 1
        else:
            scoreTwo += 1
    
    if (scoreOne > scoreTwo):
        bestresult = setOne
        bestModelName = modelNameOne
        return [bestresult, bestModelName]

    else:
        bestresult = setTwo
        bestModelName = modelNameTwo
        return [bestresult, bestModelName]


##Jury Rigged implementation for now ,
## will look at creting an array of result arrays and then doing it in a loop.

##decisionTreeModelDS(sp_FullMerg)

##randomForestModelDS(sp_FullMerg)
##xgbModelDS(sp_FullMerg)
##gbModelDS(sp_FullMerg)
##GBDTModelDS(sp_FullMerg)

##biDirectionalLSTMDS(sp_FullMerg)
#LSTMModelDS(sp_FullMerg)
#GRUModelDS(sp_FullMerg)
SVRModelDS(sp_FullMerg)
##MLPModelDS(sp_FullMerg)
##CNNModelDS(sp_FullMerg)



CurrentTopResult, CurrentTopMLName = BestModelChoice(
  decisionTreeModelDS(sp_FullMerg),
  randomForestModelDS(sp_FullMerg),
 'DecisionTree','RandomForest'
 )
##xgbModelDS(sp_DemandDef,sp_SupplyDef,1)

CurrentTopResult, CurrentTopMLName = BestModelChoice(
 CurrentTopResult,
 xgbModelDS(sp_FullMerg),
 CurrentTopMLName,'XGB'
 )

##gbModelDS(sp_DemandDef,sp_SupplyDef,1)

CurrentTopResult, CurrentTopMLName = BestModelChoice(
 CurrentTopResult,
 gbModelDS(sp_FullMerg),
 CurrentTopMLName,'GB'
 )

##biDirectionalLSTMDS(sp_DemandDef,sp_SupplyDef,1)

CurrentTopResult, CurrentTopMLName = BestModelChoice(
CurrentTopResult,
biDirectionalLSTMDS(sp_FullMerg),
CurrentTopMLName,'BSTMD'
)

##LSTMModelDS(sp_DemandDef,sp_SupplyDef,1)

CurrentTopResult, CurrentTopMLName = BestModelChoice(
CurrentTopResult,
LSTMModelDS(sp_FullMerg),
CurrentTopMLName,'LSTM'
 )

##GRUModelDS(sp_DemandDef,sp_SupplyDef,1)

CurrentTopResult, CurrentTopMLName = BestModelChoice(
CurrentTopResult,
GRUModelDS(sp_FullMerg),
CurrentTopMLName,'GRU'
 )

##SVRModelDS(sp_DemandDef,sp_SupplyDef,1)
svr_result = SVRModelDS(sp_FullMerg)
print("SVRModelDS result:", svr_result)
CurrentTopResult, CurrentTopMLName = BestModelChoice(CurrentTopResult, svr_result, CurrentTopMLName, 'SVR')

mlp_result = MLPModelDS(sp_FullMerg)
print("MLPModelDS result:", mlp_result)
CurrentTopResult, CurrentTopMLName = BestModelChoice(CurrentTopResult, mlp_result, CurrentTopMLName, 'MLP')
##CNNModelDS(sp_DemandDef,sp_SupplyDef,1)

CurrentTopResult, CurrentTopMLName = BestModelChoice(CurrentTopResult,
CNNModelDS(sp_FullMerg),
CurrentTopMLName,'CNN'
)

##GBDTModelDS(sp_DemandDef,sp_SupplyDef,1) 

CurrentTopResult, CurrentTopMLName = BestModelChoice(
CurrentTopResult,
GBDTModelDS(sp_FullMerg),
CurrentTopMLName,'GBDT'
 )


print('\n', "Best Model for Demand&Supply&Weather DataSet is : " ,CurrentTopMLName)
print("MSE: {:.4f}".format(CurrentTopResult[0]))
print("MAE: {:.4f}".format(CurrentTopResult[1]))
print("RMSE: {:.4f}".format(CurrentTopResult[2]))
print("R2: {:.4f}".format(CurrentTopResult[3]))


"""
Best Result - No Hyper Params or Lag - SVR
MSE: 1682.6278
MAE: 33.0451
RMSE: 41.0198
R2: -0.2768
--
Best Result - No Hyper Params - SVR
MSE: 1166.4916
MAE: 25.9020
RMSE: 34.1539
R2: 0.3737


--
Best Result - No Lag but with Hyper Params -



--
Best Result - With Both Hyper Params & Lag -



--
"""