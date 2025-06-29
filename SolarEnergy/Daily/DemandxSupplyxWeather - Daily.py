#----# Libraries
import array
from ast import Str
from calendar import c
from ctypes import Array
from enum import Enum
from tkinter import CURRENT
from tokenize import String
from xmlrpc.client import boolean
import numpy as np
import pandas as pd
from pandas.core.indexes import multi
import shutil
import os
import scipy as sp
import xgboost as xgb
import shap
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
sp_WeatherDef = pd.read_excel(r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\Oliver Test\weather for solar NEW 2021.xlsx")
sp_WeatherDe = pd.read_excel(r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\Weather for demand 2018.xlsx")
##sp_WeatherDe = pd.read_excel(r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\Sakakah 2021 weather dataset Demand.xlsx")

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
sp_WeatherDe.drop(columns=["YEAR", "MO", "DY", "HR"], inplace=True)

sp_FullMerg = pd.merge(sp_DemandDef, sp_WeatherDef, on="TimeStamp", how="inner")
sp_FullMerg = pd.merge(sp_SupplyDef, sp_FullMerg, on="TimeStamp", how="inner")
##sp_FullMerg = pd.concat([sp_FullMerg, sp_WeatherWind.reset_index(drop=True)], axis=1)
sp_FullMerg = pd.merge(sp_WeatherDe, sp_FullMerg, on="TimeStamp", how="inner")

##linear interpolation to smooth out the datasets
##So far has worked out great for supply but demand is getting bad results.
#print(sp_FullMerg.head())
sp_FullMerg.replace(-999, np.nan, inplace=True)
sp_FullMerg.interpolate(method='linear', inplace=True)
sp_FullMerg.dropna(inplace=True)
#desave_loc = r"E:\AI Lecture Notes\Datasets\Merged\DemandxWeather.csv"
##susave_loc = r"E:\AI Lecture Notes\Datasets\Merged\SupplyxWeather.csv"
#sp_WeatherxDem.to_csv(desave_loc, index=False)
##sp_WeatherxSup.to_csv(susave_loc, index=False)

#------#
"""
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
"""
def create_sequences_with_time(data, targets, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length].flatten())
        y.append(targets[i + seq_length])
    return np.array(X), np.array(y)

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

def decisionTreeModelDS(mergedDs: pd.DataFrame, plots:boolean):
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



    ###minLength = min(len(demandDs))
    ###demandDs = demandDs.iloc[minLength]

    #Extraction of Data.

    mergedDs["hour"] = mergedDs["TimeStamp"].dt.hour
    mergedDs["day"] = mergedDs["TimeStamp"].dt.day
    mergedDs["month"] = mergedDs["TimeStamp"].dt.month
    
    feature_cols = [
        "ALLSKY_SFC_SW_DWN_S", "ALLSKY_SFC_UV_INDEX_S", "T2M_S", "PRECTOTCORR_S", "ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S", "RH2M_S", "PS_S", "PSC_S","WS10M_S","WD10M_S", 
        "ALLSKY_SFC_SW_DWN_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "PRECTOTCORR_D", "ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D", "RH2M_D", "PS_D", "PSC_D","WS10M_D","WD10M_D", "hour", "day", "month"
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
    #X_seq, y_seq = create_seasonal_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
    #X_seq_flat = X_seq.reshape(X_seq.shape[0], -1)

    
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
    # Extract individual regressors from MultiOutputRegressor
    best_tree_model = grid_search.best_estimator_

    # Initialize a dictionary to store SHAP values for each output
    shap_values_dict = {}

    # Loop through each output's regressor
    for i, estimator in enumerate(best_tree_model.estimators_):
        print(f"\nComputing SHAP values for output {i + 1}...")
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_test)
        shap_values_dict[f"Output_{i + 1}"] = shap_values

        # Compute mean absolute SHAP values for global feature importance
        mean_shap_values = np.abs(shap_values).mean(axis=0)

        # Print feature importance for this output
        print(f"\nFeature Importance (SHAP Values) for Output {i + 1}:")
        for feature, importance in sorted(zip(feature_cols, mean_shap_values), key=lambda x: x[1], reverse=True):
            print(f"{feature:<30} {importance:.4f}")

    # Prepare SHAP values for return
    shap_importance = {
        f"Output_{i + 1}": sorted(
            zip(feature_cols, np.abs(shap_values_dict[f"Output_{i + 1}"]).mean(axis=0)),
            key=lambda x: x[1],
            reverse=True
        )
        for i in range(len(best_tree_model.estimators_))
    }
    expanded_feature_cols = [
        f"{feature}_t-{i}" for i in range(seq_length, 0, -1) for feature in feature_cols
    ]

    # Debugging prints
    print(f"Number of expanded feature columns: {len(expanded_feature_cols)}")
    print(f"Shape of X_test: {X_test.shape}")
    print(f"Shape of shap_values: {shap_values.shape}")

    # Ensure compatibility
    assert len(expanded_feature_cols) == X_test.shape[1], "Mismatch between expanded_feature_cols and X_test!"

    # Plot SHAP summary
    shap.summary_plot(shap_values, X_test, feature_names=expanded_feature_cols)
                      
    """
    for feature, importance in sorted_features:
        print("{:<30} {:>10.4f}".format(feature, importance))  
    """
    if plots == True:
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
        
    return results


def randomForestModelDS(mergedDs: pd.DataFrame, plots:boolean):
    mergedDs = mergedDs.copy()

    """
    Random Forest Results for Merged Dataset -
    MSE: 6698.6218
    MAE: 64.0826
    RMSE: 81.8451
    R2: -0.6079
    --
    With Hyper Params -  {'estimator__max_depth': 10, 'estimator__max_features': 'sqrt',
   'estimator__min_samples_leaf': 2, 'estimator__min_samples_split': 5, 'estimator__n_estimators': 50} - Best Params
    MSE: 3083.7902
    MAE: 44.4229
    RMSE: 55.5319
    R2: 0.3781
    --
    """


    ###minLength = min(len(demandDs))
    ###demandDs = demandDs.iloc[minLength]

    #Extraction of Data.

    mergedDs["hour"] = mergedDs["TimeStamp"].dt.hour
    mergedDs["day"] = mergedDs["TimeStamp"].dt.day
    mergedDs["month"] = mergedDs["TimeStamp"].dt.month

    feature_cols = [
        "ALLSKY_SFC_SW_DWN_S", "ALLSKY_SFC_UV_INDEX_S", "T2M_S", "PRECTOTCORR_S", "ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S", "RH2M_S", "PS_S", "PSC_S","WS10M_S","WD10M_S", 
        "ALLSKY_SFC_SW_DWN_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "PRECTOTCORR_D", "ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D", "RH2M_D", "PS_D", "PSC_D","WS10M_D","WD10M_D", "hour", "day", "month"
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
    
    #X_seq, y_seq = create_seasonal_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
    #X_seq_flat = X_seq.reshape(X_seq.shape[0], -1)



    
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
    best_tree_model = grid_search.best_estimator_

    # Initialize a dictionary to store SHAP values for each output
    shap_values_dict = {}

    # Loop through each output's regressor
    for i, estimator in enumerate(best_tree_model.estimators_):
        print(f"\nComputing SHAP values for output {i + 1}...")
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_test)
        shap_values_dict[f"Output_{i + 1}"] = shap_values

        # Compute mean absolute SHAP values for global feature importance
        mean_shap_values = np.abs(shap_values).mean(axis=0)

        # Print feature importance for this output
        print(f"\nFeature Importance (SHAP Values) for Output {i + 1}:")
        for feature, importance in sorted(zip(feature_cols, mean_shap_values), key=lambda x: x[1], reverse=True):
            print(f"{feature:<30} {importance:.4f}")

    # Prepare SHAP values for return
    shap_importance = {
        f"Output_{i + 1}": sorted(
            zip(feature_cols, np.abs(shap_values_dict[f"Output_{i + 1}"]).mean(axis=0)),
            key=lambda x: x[1],
            reverse=True
        )
        for i in range(len(best_tree_model.estimators_))
    }
    expanded_feature_cols = [
        f"{feature}_t-{i}" for i in range(seq_length, 0, -1) for feature in feature_cols
    ]

    # Debugging prints
    print(f"Number of expanded feature columns: {len(expanded_feature_cols)}")
    print(f"Shape of X_test: {X_test.shape}")
    print(f"Shape of shap_values: {shap_values.shape}")

    # Ensure compatibility
    assert len(expanded_feature_cols) == X_test.shape[1], "Mismatch between expanded_feature_cols and X_test!"

    # Plot SHAP summary
    shap.summary_plot(shap_values, X_test, feature_names=expanded_feature_cols)
    #----#
    print('\n', "Merged Random Forest Model Results:")
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))


    if plots == True:
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
    
    return results


def xgbModelDS(mergedDs: pd.DataFrame, plots:boolean):
    mergedDs = mergedDs.copy()

    """
    XGB for Merged Dataset -  {'estimator__colsample_bytree': 0.8, 
    'estimator__learning_rate': 0.1, 'estimator__max_depth': 3, 'estimator__n_estimators': 50,
   'estimator__subsample': 0.8} - Best Params
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


    ###minLength = min(len(demandDs))
    ###demandDs = demandDs.iloc[minLength]

    #Extraction of Data.

    mergedDs["hour"] = mergedDs["TimeStamp"].dt.hour
    mergedDs["day"] = mergedDs["TimeStamp"].dt.day
    mergedDs["month"] = mergedDs["TimeStamp"].dt.month

    feature_cols = [
        "ALLSKY_SFC_SW_DWN_S", "ALLSKY_SFC_UV_INDEX_S", "T2M_S", "PRECTOTCORR_S", "ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S", "RH2M_S", "PS_S", "PSC_S","WS10M_S","WD10M_S", 
        "ALLSKY_SFC_SW_DWN_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "PRECTOTCORR_D", "ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D", "RH2M_D", "PS_D", "PSC_D","WS10M_D","WD10M_D", "hour", "day", "month"
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
    #X_seq, y_seq = create_seasonal_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
    #X_seq_flat = X_seq.reshape(X_seq.shape[0], -1)

    
    
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
    best_tree_model = grid_search.best_estimator_

    # Initialize a dictionary to store SHAP values for each output
    shap_values_dict = {}

    # Loop through each output's regressor
    for i, estimator in enumerate(best_tree_model.estimators_):
        print(f"\nComputing SHAP values for output {i + 1}...")
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_test)
        shap_values_dict[f"Output_{i + 1}"] = shap_values

        # Compute mean absolute SHAP values for global feature importance
        mean_shap_values = np.abs(shap_values).mean(axis=0)

        # Print feature importance for this output
        print(f"\nFeature Importance (SHAP Values) for Output {i + 1}:")
        for feature, importance in sorted(zip(feature_cols, mean_shap_values), key=lambda x: x[1], reverse=True):
            print(f"{feature:<30} {importance:.4f}")

    # Prepare SHAP values for return
    shap_importance = {
        f"Output_{i + 1}": sorted(
            zip(feature_cols, np.abs(shap_values_dict[f"Output_{i + 1}"]).mean(axis=0)),
            key=lambda x: x[1],
            reverse=True
        )
        for i in range(len(best_tree_model.estimators_))
    }
    expanded_feature_cols = [
        f"{feature}_t-{i}" for i in range(seq_length, 0, -1) for feature in feature_cols
    ]

    # Debugging prints
    print(f"Number of expanded feature columns: {len(expanded_feature_cols)}")
    print(f"Shape of X_test: {X_test.shape}")
    print(f"Shape of shap_values: {shap_values.shape}")

    # Ensure compatibility
    assert len(expanded_feature_cols) == X_test.shape[1], "Mismatch between expanded_feature_cols and X_test!"

    # Plot SHAP summary
    shap.summary_plot(shap_values, X_test, feature_names=expanded_feature_cols)
    #----#
    print('\n', "Merged XGB Model Results:")
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))
    if plots == True:

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
    
    return results


def gbModelDS(mergedDs: pd.DataFrame, plots:boolean):
    mergedDs = mergedDs.copy()
    """
    GradientBoosting for Merged Dataset -
    MSE: 5705.4124
    MAE: 62.9679
    RMSE: 75.5342
    R2: -0.2487
    --
    With hyper params -  {'estimator__learning_rate': 0.1,
    'estimator__max_depth': 3, 'estimator__max_features': 'sqrt',
    'estimator__min_samples_leaf': 2, 'estimator__min_samples_split': 10,
    'estimator__n_estimators': 100} - Best params
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
        "CLRSKY_SFC_PAR_TOT_S", "RH2M_S", "PS_S", "PSC_S","WS10M_S","WD10M_S", 
        "ALLSKY_SFC_SW_DWN_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "PRECTOTCORR_D", "ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D", "RH2M_D", "PS_D", "PSC_D","WS10M_D","WD10M_D", "hour", "day", "month"
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
    #X_seq, y_seq = create_seasonal_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
    #X_seq_flat = X_seq.reshape(X_seq.shape[0], -1)

    #X_seq, y_seq = create_seasonal_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
    #X_seq_flat = X_seq.reshape(X_seq.shape[0], -1)

    
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
    best_tree_model = grid_search.best_estimator_

    # Initialize a dictionary to store SHAP values for each output
    shap_values_dict = {}

    # Loop through each output's regressor
    for i, estimator in enumerate(best_tree_model.estimators_):
        print(f"\nComputing SHAP values for output {i + 1}...")
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_test)
        shap_values_dict[f"Output_{i + 1}"] = shap_values

        # Compute mean absolute SHAP values for global feature importance
        mean_shap_values = np.abs(shap_values).mean(axis=0)

        # Print feature importance for this output
        print(f"\nFeature Importance (SHAP Values) for Output {i + 1}:")
        for feature, importance in sorted(zip(feature_cols, mean_shap_values), key=lambda x: x[1], reverse=True):
            print(f"{feature:<30} {importance:.4f}")

    # Prepare SHAP values for return
    shap_importance = {
        f"Output_{i + 1}": sorted(
            zip(feature_cols, np.abs(shap_values_dict[f"Output_{i + 1}"]).mean(axis=0)),
            key=lambda x: x[1],
            reverse=True
        )
        for i in range(len(best_tree_model.estimators_))
    }
    expanded_feature_cols = [
        f"{feature}_t-{i}" for i in range(seq_length, 0, -1) for feature in feature_cols
    ]

    # Debugging prints
    print(f"Number of expanded feature columns: {len(expanded_feature_cols)}")
    print(f"Shape of X_test: {X_test.shape}")
    print(f"Shape of shap_values: {shap_values.shape}")

    # Ensure compatibility
    assert len(expanded_feature_cols) == X_test.shape[1], "Mismatch between expanded_feature_cols and X_test!"

    # Plot SHAP summary
    shap.summary_plot(shap_values, X_test, feature_names=expanded_feature_cols)
    #----#
    print('\n', "Merged GB Model Results:")
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))
    if plots == True:

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
    
    return results


def biDirectionalLSTMDS(mergedDs: pd.DataFrame, plots:boolean):
    mergedDs = mergedDs.copy()

    """
    Bidirectional LSTM for Merged Dataset -
    MSE: 2366.0192
    MAE: 37.8718
    RMSE: 48.6417   
    R2: 0.2168
    --
    With Hyper Params -  {'units1': 32, 'dropout1': 0.1, 'units2': 128, 'dense_units': 32,
   'learning_rate': 0.0001, 'tuner/epochs': 7, 'tuner/initial_epoch': 3, 'tuner/bracket': 2, 
   'tuner/round': 1, 'tuner/trial_id': '0008'} - Best Params
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
        "CLRSKY_SFC_PAR_TOT_S", "RH2M_S", "PS_S", "PSC_S","WS10M_S","WD10M_S", 
        "ALLSKY_SFC_SW_DWN_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "PRECTOTCORR_D", "ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D", "RH2M_D", "PS_D", "PSC_D","WS10M_D","WD10M_D", "hour", "day", "month"
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
    #X_seq, y_seq = create_seasonal_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
    #X_seq_flat = X_seq.reshape(X_seq.shape[0], -1)
    #
    num_features = len(feature_cols)
    X_train = X_train.reshape((-1, seq_length, num_features))
    X_test = X_test.reshape((-1, seq_length, num_features))
    
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
            input_shape=(seq_length, num_features)
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
    shap_values_dict = {}

    explainer = shap.GradientExplainer(best_model, X_train)
    shap_values = explainer.shap_values(X_test)
    expanded_feature_cols = [
       f"{feature}_t-{i}" for i in range(seq_length, 0, -1) for feature in feature_cols
   ]
    # Loop through each output (e.g., Demand and Supply)
    for i, shap_value in enumerate(shap_values):
        print(f"\nComputing SHAP values for output {i + 1}...")
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        # Store SHAP values in the dictionary
        shap_values_dict[f"Output_{i + 1}"] = shap_value
    """
    print("feature_cols:", feature_cols)
    print("mean_shap_values:", mean_shap_values)
    print("Length of feature_cols:", len(feature_cols))
    print("Length of mean_shap_values:", len(mean_shap_values))
    
    # If mean_shap_values is a NumPy array, ensure it's flattened
    if isinstance(mean_shap_values, np.ndarray):
        print("Shape of mean_shap_values:", mean_shap_values.shape)
    """
    # Prepare SHAP values for return
    X_test_flat = X_test.reshape(X_test.shape[0], -1)  # Shape: (2619, 600)

    # Adjust shap_values to match the flattened structure
    shap_values_flat = shap_values.reshape(shap_values.shape[0], -1, shap_values.shape[-1])  # Shape: (2619, 600, 2)

    # Select the SHAP values for the first output (e.g., Demand)
    shap_values_demand = shap_values_flat[:, :, 0]  # Shape: (2619, 600)

    # Prepare SHAP importance with adjusted SHAP values
    shap_importance = {
        f"Output_{i + 1}": sorted(
            zip(expanded_feature_cols, np.abs(shap_values_dict[f"Output_{i + 1}"][:, :-1]).mean(axis=0).tolist()),
            key=lambda x: x[1],
            reverse=True
        )
        for i in range(len(shap_values_dict))
}

    print("Shape of X_test_flat:", X_test_flat.shape)
    print("Shape of shap_values_flat:", shap_values_flat.shape)
    print("Shape of shap_values_demand:", shap_values_demand.shape)

    
    assert shap_values_demand.shape[1] == X_test_flat.shape[1], "Mismatch between shap_values and X_test!"
    assert len(expanded_feature_cols) == X_test_flat.shape[1], "Mismatch between expanded_feature_cols and X_test!"

    # Plot SHAP summary for the first output
    shap.summary_plot(shap_values_demand, X_test_flat, feature_names=expanded_feature_cols)
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
    if plots == True:

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
    
    return results


def LSTMModelDS(mergedDs: pd.DataFrame, plots:boolean):
    mergedDs = mergedDs.copy()

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
        "CLRSKY_SFC_PAR_TOT_S", "RH2M_S", "PS_S", "PSC_S","WS10M_S","WD10M_S", 
        "ALLSKY_SFC_SW_DWN_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "PRECTOTCORR_D", "ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D", "RH2M_D", "PS_D", "PSC_D","WS10M_D","WD10M_D", "hour", "day", "month"
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
    num_features = len(feature_cols)
    X_train = X_train.reshape((-1, seq_length, num_features))
    X_test = X_test.reshape((-1, seq_length, num_features))
    #X_seq, y_seq = create_seasonal_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
    #X_seq_flat = X_seq.reshape(X_seq.shape[0], -1)


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
                input_shape=(seq_length, num_features)
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
    shap_values_dict = {}

    explainer = shap.GradientExplainer(best_model, X_train)
    shap_values = explainer.shap_values(X_test)
    expanded_feature_cols = [
       f"{feature}_t-{i}" for i in range(seq_length, 0, -1) for feature in feature_cols
   ]
    # Loop through each output (e.g., Demand and Supply)
    for i, shap_value in enumerate(shap_values):
        print(f"\nComputing SHAP values for output {i + 1}...")
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        # Store SHAP values in the dictionary
        shap_values_dict[f"Output_{i + 1}"] = shap_value
    """
    print("feature_cols:", feature_cols)
    print("mean_shap_values:", mean_shap_values)
    print("Length of feature_cols:", len(feature_cols))
    print("Length of mean_shap_values:", len(mean_shap_values))
    
    # If mean_shap_values is a NumPy array, ensure it's flattened
    if isinstance(mean_shap_values, np.ndarray):
        print("Shape of mean_shap_values:", mean_shap_values.shape)
    """
    # Prepare SHAP values for return
    X_test_flat = X_test.reshape(X_test.shape[0], -1)  # Shape: (2619, 600)

    # Adjust shap_values to match the flattened structure
    shap_values_flat = shap_values.reshape(shap_values.shape[0], -1, shap_values.shape[-1])  # Shape: (2619, 600, 2)

    # Select the SHAP values for the first output (e.g., Demand)
    shap_values_demand = shap_values_flat[:, :, 0]  # Shape: (2619, 600)

    # Prepare SHAP importance with adjusted SHAP values
    shap_importance = {
        f"Output_{i + 1}": sorted(
            zip(expanded_feature_cols, np.abs(shap_values_dict[f"Output_{i + 1}"][:, :-1]).mean(axis=0).tolist()),
            key=lambda x: x[1],
            reverse=True
        )
        for i in range(len(shap_values_dict))
}

    print("Shape of X_test_flat:", X_test_flat.shape)
    print("Shape of shap_values_flat:", shap_values_flat.shape)
    print("Shape of shap_values_demand:", shap_values_demand.shape)

    
    assert shap_values_demand.shape[1] == X_test_flat.shape[1], "Mismatch between shap_values and X_test!"
    assert len(expanded_feature_cols) == X_test_flat.shape[1], "Mismatch between expanded_feature_cols and X_test!"

    # Plot SHAP summary for the first output
    shap.summary_plot(shap_values_demand, X_test_flat, feature_names=expanded_feature_cols)
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
    if plots == True:

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
    
    return results


def GRUModelDS(mergedDs: pd.DataFrame, plots:boolean):
    mergedDs = mergedDs.copy()
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
        "CLRSKY_SFC_PAR_TOT_S", "RH2M_S", "PS_S", "PSC_S","WS10M_S","WD10M_S", 
        "ALLSKY_SFC_SW_DWN_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "PRECTOTCORR_D", "ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D", "RH2M_D", "PS_D", "PSC_D","WS10M_D","WD10M_D", "hour", "day", "month"
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
    num_features = len(feature_cols)
    X_train = X_train.reshape((-1, seq_length, num_features))
    X_test = X_test.reshape((-1, seq_length, num_features))

        
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
                input_shape=(seq_length, num_features)
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
    shap_values_dict = {}

    explainer = shap.GradientExplainer(best_model, X_train)
    shap_values = explainer.shap_values(X_test)
    expanded_feature_cols = [
       f"{feature}_t-{i}" for i in range(seq_length, 0, -1) for feature in feature_cols
   ]
    # Loop through each output (e.g., Demand and Supply)
    for i, shap_value in enumerate(shap_values):
        print(f"\nComputing SHAP values for output {i + 1}...")
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        # Store SHAP values in the dictionary
        shap_values_dict[f"Output_{i + 1}"] = shap_value
    """
    print("feature_cols:", feature_cols)
    print("mean_shap_values:", mean_shap_values)
    print("Length of feature_cols:", len(feature_cols))
    print("Length of mean_shap_values:", len(mean_shap_values))
    
    # If mean_shap_values is a NumPy array, ensure it's flattened
    if isinstance(mean_shap_values, np.ndarray):
        print("Shape of mean_shap_values:", mean_shap_values.shape)
    """
    # Prepare SHAP values for return
    X_test_flat = X_test.reshape(X_test.shape[0], -1)  # Shape: (2619, 600)

    # Adjust shap_values to match the flattened structure
    shap_values_flat = shap_values.reshape(shap_values.shape[0], -1, shap_values.shape[-1])  # Shape: (2619, 600, 2)

    # Select the SHAP values for the first output (e.g., Demand)
    shap_values_demand = shap_values_flat[:, :, 0]  # Shape: (2619, 600)

    # Prepare SHAP importance with adjusted SHAP values
    shap_importance = {
        f"Output_{i + 1}": sorted(
            zip(expanded_feature_cols, np.abs(shap_values_dict[f"Output_{i + 1}"][:, :-1]).mean(axis=0).tolist()),
            key=lambda x: x[1],
            reverse=True
        )
        for i in range(len(shap_values_dict))
}

    print("Shape of X_test_flat:", X_test_flat.shape)
    print("Shape of shap_values_flat:", shap_values_flat.shape)
    print("Shape of shap_values_demand:", shap_values_demand.shape)

    
    assert shap_values_demand.shape[1] == X_test_flat.shape[1], "Mismatch between shap_values and X_test!"
    assert len(expanded_feature_cols) == X_test_flat.shape[1], "Mismatch between expanded_feature_cols and X_test!"

    # Plot SHAP summary for the first output
    shap.summary_plot(shap_values_demand, X_test_flat, feature_names=expanded_feature_cols)
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
   
    if plots == True:

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
    
    return results


def SVRModelDS(mergedDs: pd.DataFrame, plots:boolean):
    mergedDs = mergedDs.copy()


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
    {'estimator__C': 0.1, 'estimator__epsilon': 0.01, 
    'estimator__gamma': 1, 'estimator__kernel': 'rbf'} - Best params
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
        "CLRSKY_SFC_PAR_TOT_S", "RH2M_S", "PS_S", "PSC_S","WS10M_S","WD10M_S", 
        "ALLSKY_SFC_SW_DWN_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "PRECTOTCORR_D", "ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D", "RH2M_D", "PS_D", "PSC_D","WS10M_D","WD10M_D", "hour", "day", "month"
    ]
    """
    Merged SVR Model Results: - With below sequence creation method
    MSE: 3695.3976
    MAE: 48.1701
    RMSE: 60.7898
    R2: -0.4673
    """
    target_cols = ['Demand_MW', 'Supply_MW']
    seq_length = 24
    X_seq, y_seq = create_sequence_with_time(mergedDs,feature_cols,target_cols, seq_length)
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X = scaler_X.fit_transform(X_seq)
    y = scaler_y.fit_transform(y_seq)

    

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3,shuffle=False
    )
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
    """
    Merged SVR Model Results: - With Above Method of sequence creation
    MSE: 3695.3976
    MAE: 48.1701
    RMSE: 60.7898
    R2: -0.4673

    """
    param_grid = {
        'estimator__C': [1],           # Regularization, similar to tree depth/complexity
        #'estimator__gamma': [1],  # Kernel coefficient, like learning rate
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
    best_tree_model = grid.best_estimator_

    # Initialize a dictionary to store SHAP values for each output
    shap_values_dict = {}
    def model_predict(X):
        return estimator.predict(X)
    # Loop through each output's regressor
    for i, estimator in enumerate(best_tree_model.estimators_):
        print(f"\nComputing SHAP values for output {i + 1}...")
        explainer = shap.KernelExplainer(model_predict, X_train[:100])        
        shap_values = explainer.shap_values(X_test[:100])
        shap_values_dict[f"Output_{i + 1}"] = shap_values

        # Compute mean absolute SHAP values for global feature importance
        mean_shap_values = np.abs(shap_values).mean(axis=0)

        # Print feature importance for this output
        print(f"\nFeature Importance (SHAP Values) for Output {i + 1}:")
        for feature, importance in sorted(zip(feature_cols, mean_shap_values), key=lambda x: x[1], reverse=True):
            print(f"{feature:<30} {importance:.4f}")

    # Prepare SHAP values for return
    shap_importance = {
        f"Output_{i + 1}": sorted(
            zip(feature_cols, np.abs(shap_values_dict[f"Output_{i + 1}"]).mean(axis=0)),
            key=lambda x: x[1],
            reverse=True
        )
        for i in range(len(best_tree_model.estimators_))
    }
    expanded_feature_cols = [
        f"{feature}_t-{i}" for i in range(seq_length, 0, -1) for feature in feature_cols
    ]

    # Debugging prints
    print(f"Number of expanded feature columns: {len(expanded_feature_cols)}")
    print(f"Shape of X_test: {X_test.shape}")
    print(f"Shape of shap_values: {shap_values.shape}")

    # Ensure compatibility
    assert len(expanded_feature_cols) == X_test.shape[1], "Mismatch between expanded_feature_cols and X_test!"

    # Plot SHAP summary
    shap.summary_plot(shap_values, X_test, feature_names=expanded_feature_cols)
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
    if plots == True:

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
    
    return results, sorted_features


def MLPModelDS(mergedDs: pd.DataFrame, plots:boolean):
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

    # All code below is now at the same indentation as the inner function
    mergedDs["hour"] = mergedDs["TimeStamp"].dt.hour
    mergedDs["day"] = mergedDs["TimeStamp"].dt.day
    mergedDs["month"] = mergedDs["TimeStamp"].dt.month

    feature_cols = [
        "ALLSKY_SFC_SW_DWN_S", "ALLSKY_SFC_UV_INDEX_S", "T2M_S", "PRECTOTCORR_S", "ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S", "RH2M_S", "PS_S", "PSC_S","WS10M_S","WD10M_S", 
        "ALLSKY_SFC_SW_DWN_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "PRECTOTCORR_D", "ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D", "RH2M_D", "PS_D", "PSC_D","WS10M_D","WD10M_D", "hour", "day", "month"
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

    num_features = len(feature_cols)
    X_train = X_train.reshape((X_train.shape[0], seq_length * num_features))
    X_test = X_test.reshape((X_test.shape[0], seq_length * num_features))
    #X_seq, y_seq = create_seasonal_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
    #X_seq_flat = X_seq.reshape(X_seq.shape[0], -1)

    
  
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
            Dense(hp.Int('dense1', 64, 256, step=32), activation='relu', input_dim=seq_length * num_features),
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
    shap_values_dict = {}

    explainer = shap.GradientExplainer(best_model, X_train)
    shap_values = explainer.shap_values(X_test)
    expanded_feature_cols = [
       f"{feature}_t-{i}" for i in range(seq_length, 0, -1) for feature in feature_cols
   ]
    # Loop through each output (e.g., Demand and Supply)
    for i, shap_value in enumerate(shap_values):
        print(f"\nComputing SHAP values for output {i + 1}...")
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        # Store SHAP values in the dictionary
        shap_values_dict[f"Output_{i + 1}"] = shap_value
    """
    print("feature_cols:", feature_cols)
    print("mean_shap_values:", mean_shap_values)
    print("Length of feature_cols:", len(feature_cols))
    print("Length of mean_shap_values:", len(mean_shap_values))
    
    # If mean_shap_values is a NumPy array, ensure it's flattened
    if isinstance(mean_shap_values, np.ndarray):
        print("Shape of mean_shap_values:", mean_shap_values.shape)
    """
    # Prepare SHAP values for return
    X_test_flat = X_test.reshape(X_test.shape[0], -1)  # Shape: (2619, 600)

    # Adjust shap_values to match the flattened structure
    shap_values_flat = shap_values.reshape(shap_values.shape[0], -1, shap_values.shape[-1])  # Shape: (2619, 600, 2)

    # Select the SHAP values for the first output (e.g., Demand)
    shap_values_demand = shap_values_flat[:, :, 0]  # Shape: (2619, 600)

    # Prepare SHAP importance with adjusted SHAP values
    shap_importance = {
        f"Output_{i + 1}": sorted(
            zip(expanded_feature_cols, np.abs(shap_values_dict[f"Output_{i + 1}"][:, :-1]).mean(axis=0).tolist()),
            key=lambda x: x[1],
            reverse=True
        )
        for i in range(len(shap_values_dict))
}

    print("Shape of X_test_flat:", X_test_flat.shape)
    print("Shape of shap_values_flat:", shap_values_flat.shape)
    print("Shape of shap_values_demand:", shap_values_demand.shape)

    
    assert shap_values_demand.shape[1] == X_test_flat.shape[1], "Mismatch between shap_values and X_test!"
    assert len(expanded_feature_cols) == X_test_flat.shape[1], "Mismatch between expanded_feature_cols and X_test!"

    # Plot SHAP summary for the first output
    shap.summary_plot(shap_values_demand, X_test_flat, feature_names=expanded_feature_cols)
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
    if plots == True:

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
    
    return results


def CNNModelDS(mergedDs: pd.DataFrame, plots:boolean):
    mergedDs = mergedDs.copy()


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
        "CLRSKY_SFC_PAR_TOT_S", "RH2M_S", "PS_S", "PSC_S","WS10M_S","WD10M_S", 
        "ALLSKY_SFC_SW_DWN_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "PRECTOTCORR_D", "ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D", "RH2M_D", "PS_D", "PSC_D","WS10M_D","WD10M_D", "hour", "day", "month"
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
    #X_seq, y_seq = create_seasonal_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
    #X_seq_flat = X_seq.reshape(X_seq.shape[0], -1)
    num_features = len(feature_cols)
    X_train = X_train.reshape((-1, seq_length, num_features))
    X_test = X_test.reshape((-1, seq_length, num_features))
    
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
                input_shape=(seq_length, num_features)
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
    shap_values_dict = {}

    explainer = shap.GradientExplainer(best_model, X_train)
    shap_values = explainer.shap_values(X_test)
    expanded_feature_cols = [
       f"{feature}_t-{i}" for i in range(seq_length, 0, -1) for feature in feature_cols
   ]
    # Loop through each output (e.g., Demand and Supply)
    for i, shap_value in enumerate(shap_values):
        print(f"\nComputing SHAP values for output {i + 1}...")
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        # Store SHAP values in the dictionary
        shap_values_dict[f"Output_{i + 1}"] = shap_value
    """
    print("feature_cols:", feature_cols)
    print("mean_shap_values:", mean_shap_values)
    print("Length of feature_cols:", len(feature_cols))
    print("Length of mean_shap_values:", len(mean_shap_values))
    
    # If mean_shap_values is a NumPy array, ensure it's flattened
    if isinstance(mean_shap_values, np.ndarray):
        print("Shape of mean_shap_values:", mean_shap_values.shape)
    """
    # Prepare SHAP values for return
    X_test_flat = X_test.reshape(X_test.shape[0], -1)  # Shape: (2619, 600)

    # Adjust shap_values to match the flattened structure
    shap_values_flat = shap_values.reshape(shap_values.shape[0], -1, shap_values.shape[-1])  # Shape: (2619, 600, 2)

    # Select the SHAP values for the first output (e.g., Demand)
    shap_values_demand = shap_values_flat[:, :, 0]  # Shape: (2619, 600)

    # Prepare SHAP importance with adjusted SHAP values
    shap_importance = {
        f"Output_{i + 1}": sorted(
            zip(expanded_feature_cols, np.abs(shap_values_dict[f"Output_{i + 1}"][:, :-1]).mean(axis=0).tolist()),
            key=lambda x: x[1],
            reverse=True
        )
        for i in range(len(shap_values_dict))
}

    print("Shape of X_test_flat:", X_test_flat.shape)
    print("Shape of shap_values_flat:", shap_values_flat.shape)
    print("Shape of shap_values_demand:", shap_values_demand.shape)

    
    assert shap_values_demand.shape[1] == X_test_flat.shape[1], "Mismatch between shap_values and X_test!"
    assert len(expanded_feature_cols) == X_test_flat.shape[1], "Mismatch between expanded_feature_cols and X_test!"

    # Plot SHAP summary for the first output
    shap.summary_plot(shap_values_demand, X_test_flat, feature_names=expanded_feature_cols)
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
    if plots == True:

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
    
    return results


def GBDTModelDS(mergedDs: pd.DataFrame, plots:boolean):
    mergedDs = mergedDs.copy()

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
        "CLRSKY_SFC_PAR_TOT_S", "RH2M_S", "PS_S", "PSC_S","WS10M_S","WD10M_S", 
        "ALLSKY_SFC_SW_DWN_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "PRECTOTCORR_D", "ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D", "RH2M_D", "PS_D", "PSC_D","WS10M_D","WD10M_D", "hour", "day", "month"
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
    #X_seq, y_seq = create_seasonal_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
    #X_seq_flat = X_seq.reshape(X_seq.shape[0], -1)

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
    best_tree_model = grid_search.best_estimator_

    # Initialize a dictionary to store SHAP values for each output
    shap_values_dict = {}

    # Loop through each output's regressor
    for i, estimator in enumerate(best_tree_model.estimators_):
        print(f"\nComputing SHAP values for output {i + 1}...")
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_test)
        shap_values_dict[f"Output_{i + 1}"] = shap_values

        # Compute mean absolute SHAP values for global feature importance
        mean_shap_values = np.abs(shap_values).mean(axis=0)

        # Print feature importance for this output
        print(f"\nFeature Importance (SHAP Values) for Output {i + 1}:")
        for feature, importance in sorted(zip(feature_cols, mean_shap_values), key=lambda x: x[1], reverse=True):
            print(f"{feature:<30} {importance:.4f}")

    # Prepare SHAP values for return
    shap_importance = {
        f"Output_{i + 1}": sorted(
            zip(feature_cols, np.abs(shap_values_dict[f"Output_{i + 1}"]).mean(axis=0)),
            key=lambda x: x[1],
            reverse=True
        )
        for i in range(len(best_tree_model.estimators_))
    }
    expanded_feature_cols = [
        f"{feature}_t-{i}" for i in range(seq_length, 0, -1) for feature in feature_cols
    ]

    # Debugging prints
    print(f"Number of expanded feature columns: {len(expanded_feature_cols)}")
    print(f"Shape of X_test: {X_test.shape}")
    print(f"Shape of shap_values: {shap_values.shape}")

    # Ensure compatibility
    assert len(expanded_feature_cols) == X_test.shape[1], "Mismatch between expanded_feature_cols and X_test!"

    # Plot SHAP summary
    shap.summary_plot(shap_values, X_test, feature_names=expanded_feature_cols)
    #---#

    print('\n', "Merged GBDT Model Results:")
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))
    
    if plots == True:

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
    
    return results, shap_importance

def custom_mutate(individual, indpb):
    # filters1: int, kernel_size1: int, dropout1: float, dense_units: int, learning_rate: float
    if random.random() < indpb:
        individual[0] = random.choice([32, 64, 128])
    if random.random() < indpb:
        individual[1] = random.choice([2, 3, 5])
    if random.random() < indpb:
        individual[2] = float(random.uniform(0.1, 0.5))
    if random.random() < indpb:
        individual[3] = random.choice([32, 64, 128])
    if random.random() < indpb:
        individual[4] = float(random.choice([1e-2, 1e-3, 1e-4]))
    return (individual,)
def ensure_real_result(results):
    # Only convert the first 4 metrics, leave the model name as is
    return [float(np.real(x)) if isinstance(x, (complex, np.complexfloating, np.floating, float)) else x for x in results]
def NSGA2_CNN_ModelDS(
    mergedDs: pd.DataFrame,
    plots: boolean,
    pop_size=5,
    ngen=3,
    cxpb=0.7,
    mutpb=0.3,
    seq_length=24,
    epochs=20,
    batch_size=16
):

    mergedDs = mergedDs.copy()

    mergedDs["hour"] = mergedDs["TimeStamp"].dt.hour
    mergedDs["day"] = mergedDs["TimeStamp"].dt.day
    mergedDs["month"] = mergedDs["TimeStamp"].dt.month

    feature_cols = [
        "ALLSKY_SFC_SW_DWN_S", "ALLSKY_SFC_UV_INDEX_S", "T2M_S", "PRECTOTCORR_S", "ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S", "RH2M_S", "PS_S", "PSC_S","WS10M_S","WD10M_S", 
        "ALLSKY_SFC_SW_DWN_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "PRECTOTCORR_D", "ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D", "RH2M_D", "PS_D", "PSC_D","WS10M_D","WD10M_D", "hour", "day", "month"
    ]
    target_cols = ['Demand_MW', 'Supply_MW']
    
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
                Conv1D(filters=filters1, kernel_size=kernel_size1, activation='relu', input_shape=(seq_length, num_features)),
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
        Conv1D(filters=int(filters1), kernel_size=int(kernel_size1), activation='relu', input_shape=(seq_length, num_features)),
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
    if plots == True:

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
    
    return results    

def NSGA3_CNN_ModelDS(
    mergedDs: pd.DataFrame,
    plots: boolean,
    pop_size=5,
    ngen=3,
    cxpb=0.7,
    mutpb=0.3,
    seq_length=24,
    epochs=20,
    batch_size=16
):

    mergedDs = mergedDs.copy()
    mergedDs["hour"] = mergedDs["TimeStamp"].dt.hour
    mergedDs["day"] = mergedDs["TimeStamp"].dt.day
    mergedDs["month"] = mergedDs["TimeStamp"].dt.month

    feature_cols = [
        "ALLSKY_SFC_SW_DWN_S", "ALLSKY_SFC_UV_INDEX_S", "T2M_S", "PRECTOTCORR_S", "ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S", "RH2M_S", "PS_S", "PSC_S","WS10M_S","WD10M_S", 
        "ALLSKY_SFC_SW_DWN_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "PRECTOTCORR_D", "ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D", "RH2M_D", "PS_D", "PSC_D","WS10M_D","WD10M_D", "hour", "day", "month"
    ]
    target_cols = ['Demand_MW', 'Supply_MW']
    
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

    scaler_demand = MinMaxScaler()
    scaler_supply = MinMaxScaler()
    scaler_demand.fit(mergedDs[['Demand_MW']])
    scaler_supply.fit(mergedDs[['Supply_MW']])

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
                Conv1D(filters=filters1, kernel_size=kernel_size1, activation='relu', input_shape=(seq_length, num_features)),
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
        Conv1D(filters=int(filters1), kernel_size=int(kernel_size1), activation='relu', input_shape=(seq_length, num_features)),
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
    if plots == True:

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
"""
DecTree = ensure_real_result(decisionTreeModelDS(sp_FullMerg,False))

randForest = ensure_real_result(randomForestModelDS(sp_FullMerg,False))
xgb  = ensure_real_result(xgbModelDS(sp_FullMerg,False))
gb = ensure_real_result(gbModelDS(sp_FullMerg,False))
gbdt = ensure_real_result(GBDTModelDS(sp_FullMerg,False))

blstm = ensure_real_result(biDirectionalLSTMDS(sp_FullMerg,False))

lstm  = ensure_real_result(LSTMModelDS(sp_FullMerg,False))
gru  = ensure_real_result(GRUModelDS(sp_FullMerg,False))

svr  = ensure_real_result(SVRModelDS(sp_FullMerg,False))
"""
mlp  = ensure_real_result(MLPModelDS(sp_FullMerg,False))
cnn  = ensure_real_result(CNNModelDS(sp_FullMerg,False))
nsga2cnn  = ensure_real_result(NSGA2_CNN_ModelDS(sp_FullMerg,False))
nsga3cnn  = ensure_real_result(NSGA3_CNN_ModelDS(sp_FullMerg,False))
modelresults = [DecTree, randForest, xgb,gb, gbdt, blstm, lstm, gru, svr, mlp, cnn, nsga2cnn, nsga3cnn]


"""
DecTree = decisionTreeModelDS(sp_FullMerg)
randForest = randomForestModelDS(sp_FullMerg)
xgb = xgbModelDS(sp_FullMerg)
gb = gbModelDS(sp_FullMerg)
gbdt = GBDTModelDS(sp_FullMerg)
blstm = biDirectionalLSTMDS(sp_FullMerg)
lstm = LSTMModelDS(sp_FullMerg)
gru =GRUModelDS(sp_FullMerg)
svr = SVRModelDS(sp_FullMerg)
mlp =MLPModelDS(sp_FullMerg)
cnn = CNNModelDS(sp_FullMerg) 

nsga2cnn = NSGA2_CNN_ModelDS(sp_FullMerg)
nsga3cnn = NSGA3_CNN_ModelDS(sp_FullMerg)
modelresults = [DecTree,randForest,xgb,gbdt,blstm,lstm,gru,svr,mlp,cnn,nsga2cnn,nsga3cnn,]
"""
BestResultsOrdered = BetterModelSelectionMethod(modelresults)

print("\nModel Ranking (Best to Worst) Hourly:")
print("{:<20} {:>12} {:>12} {:>12} {:>10}".format("Model", "MSE", "MAE", "RMSE", "R2"))
print("-" * 70)
for res in BestResultsOrdered:
    print("{:<20} {:>12.4f} {:>12.4f} {:>12.4f} {:>10.4f}".format(
        res[4], res[0], res[1], res[2], res[3]
    ))


