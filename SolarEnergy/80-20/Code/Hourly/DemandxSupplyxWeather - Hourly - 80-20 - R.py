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
import keras_tuner as kt
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
def create_sequences_with_time_noflatten(data, targets, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])  # Remove .flatten()
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
def compute_shap_values_Trees(model_type, best_tree_model, X_test, feature_cols, seq_length):
    """
    Computes SHAP values for the given model type and visualizes feature importance.
    """
    shap_values_dict = {}
    expanded_feature_cols = [
        f"{feature}_t-{i}" for i in range(seq_length, 0, -1) for feature in feature_cols
    ]

    # Ensure compatibility
    assert len(expanded_feature_cols) == X_test.shape[1], "Mismatch between expanded_feature_cols and X_test!"

    # Handle Decision Tree separately
    if model_type == "Decision Tree":
        print("\nComputing SHAP values for Decision Tree...")
        explainer = shap.TreeExplainer(best_tree_model)
        shap_values = explainer.shap_values(X_test)
        shap_values_dict["Output_1"] = shap_values  # Decision Tree has a single output
    elif model_type in ["GBDT", "Random Forest", "MultiOutputRegressor"]:
        # Loop through each output's regressor for ensemble models
        for i, estimator in enumerate(best_tree_model.estimators_):
            print(f"\nComputing SHAP values for output {i + 1}...")
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(X_test)
            shap_values_dict[f"Output_{i + 1}"] = shap_values
    elif model_type == "XGB":
        print("\nComputing SHAP values for XGB...")
        explainer = shap.TreeExplainer(best_tree_model)
        shap_values = explainer.shap_values(X_test)
        shap_values_dict["Output_1"] = shap_values  # XGB typically has a single output
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Extract SHAP values for demand and supply
    shap_values_demand = shap_values_dict.get("Output_1", None)
    shap_values_supply = shap_values_dict.get("Output_2", None)

    # Aggregate SHAP values across outputs
    aggregated_shap_values = {}
    for output, shap_values in shap_values_dict.items():
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        for feature, importance in zip(expanded_feature_cols, mean_shap_values):
            condensed_feature = feature.split("_t-")[0]  # Extract condensed feature name
            if condensed_feature not in aggregated_shap_values:
                aggregated_shap_values[condensed_feature] = 0
            aggregated_shap_values[condensed_feature] += importance

    # Sort aggregated SHAP values by importance
    sorted_shap_values = sorted(aggregated_shap_values.items(), key=lambda x: x[1], reverse=True)

    # Convert to dictionary for return
    shap_importance = {
        "Sorted Feature Importance": sorted_shap_values
    }

    # Print sorted feature importance
    print("\nFeature Importance (Ordered):")
    print("{:<30} {:>12}".format("Feature", "Importance"))
    print("-" * 42)
    for feature, importance in sorted_shap_values:
        print("{:<30} {:>12.4f}".format(feature, importance))

    # Plot SHAP summary for demand
    if shap_values_demand is not None:
        print("\nPlotting SHAP summary for Demand...")
        shap.summary_plot(shap_values_demand, X_test, plot_type="bar", feature_names=expanded_feature_cols)

    # Plot SHAP summary for supply
    if shap_values_supply is not None:
        print("\nPlotting SHAP summary for Supply...")
        shap.summary_plot(shap_values_supply, X_test, plot_type="bar", feature_names=expanded_feature_cols)

    # Feature-to-lagged mapping for dependence plots
    feature_to_lagged_mapping = {
        feature: [f"{feature}_t-{i}" for i in range(seq_length, 0, -1)]
        for feature in feature_cols
    }

    # Loop through top features for dependence plots (Demand)
    for feature, _ in sorted_shap_values[:3]:  # Top 3 features
        if feature in feature_to_lagged_mapping:
            lagged_feature = feature_to_lagged_mapping[feature][0]
            if lagged_feature in expanded_feature_cols:
                shap.dependence_plot(
                    lagged_feature, shap_values_demand, X_test, feature_names=expanded_feature_cols
                )
            else:
                print(f"Time-lagged feature {lagged_feature} not found in expanded_feature_cols.")
        else:
            print(f"Feature {feature} not found in feature_to_lagged_mapping.")

    # Loop through top features for dependence plots (Supply)
    for feature, _ in sorted_shap_values[:3]:  # Top 3 features
        if feature in feature_to_lagged_mapping:
            lagged_feature = feature_to_lagged_mapping[feature][0]
            if lagged_feature in expanded_feature_cols:
                shap.dependence_plot(
                    lagged_feature, shap_values_supply, X_test, feature_names=expanded_feature_cols
                )
            else:
                print(f"Time-lagged feature {lagged_feature} not found in expanded_feature_cols.")
        else:
            print(f"Feature {feature} not found in feature_to_lagged_mapping.")

    return shap_importance
def decisionTreeModelDS(mergedDs: pd.DataFrame, plots:bool,features:bool):
    mergedDs = mergedDs.copy()
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
    seq_length = 1
    X_seq, y_seq = create_sequences_with_time(X, y, seq_length)

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, shuffle=False
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
   
    if plots == True:
    #-Demand Plot-#
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true[:, 0], label='Actual Demand')
        plt.plot(y_demand_pred[:, 0], label='Predicted Demand')
        plt.title(f'Decision Tree Model: Actual vs Predicted Demand:  Hourly')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
        #-Supply Plot-#
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true[:, 1], label='Actual Supply')
        plt.plot(y_demand_pred[:, 1], label='Predicted Supply')
        plt.title(f'Decision Tree Model: Actual vs Predicted Supply:  Hourly')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
    if features == True:
        shap_importance = compute_shap_values_Trees(
            model_type="MultiOutputRegressor",
            best_tree_model=grid_search.best_estimator_,
            X_test=X_test,
            feature_cols=feature_cols,
            seq_length=seq_length
        )
        return results        
    return results


def randomForestModelDS(mergedDs: pd.DataFrame, plots:bool,features:bool):
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

    seq_length = 1
    X_seq, y_seq = create_sequences_with_time(X, y, seq_length)

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, shuffle=False
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
        plt.title(f'Random Forest Model: Actual vs Predicted Demand:  Hourly')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
        #-Supply Plot-#
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true[:, 1], label='Actual Supply')
        plt.plot(y_demand_pred[:, 1], label='Predicted Supply')
        plt.title(f'Random Forest Model: Actual vs Predicted Supply:  Hourly')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
    if features == True:
        shap_importance = compute_shap_values_Trees(
            model_type="Random Forest",
            best_tree_model=grid_search.best_estimator_,
            X_test=X_test,
            feature_cols=feature_cols,
            seq_length=seq_length
    )
        return results       
    return results


def xgbModelDS(mergedDs: pd.DataFrame, plots:bool,features:bool):
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

    seq_length = 1
    X_seq, y_seq = create_sequences_with_time(X, y, seq_length)

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, shuffle=False
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
    """
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
        plt.title(f'XGB Model: Actual vs Predicted Demand:  Hourly')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
        #-Supply Plot-#
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true[:, 1], label='Actual Supply')
        plt.plot(y_demand_pred[:, 1], label='Predicted Supply')
        plt.title(f'XGB Model: Actual vs Predicted Supply:  Hourly')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
    if features == True:
        shap_importance = compute_shap_values_Trees(
            model_type="MultiOutputRegressor",
            best_tree_model=grid_search.best_estimator_,
            X_test=X_test,
            feature_cols=feature_cols,
            seq_length=seq_length
    )
        return results     
    return results

def biDirectionalLSTMDS(mergedDs: pd.DataFrame, plots:bool,features:bool):
    mergedDs = mergedDs.copy()

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
    seq_length = 1
    X_seq, y_seq = create_sequences_with_time(X, y, seq_length)

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, shuffle=False
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
        plt.title(f'BLSTM Model: Actual vs Predicted Demand:  Hourly')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
        #-Supply Plot-#
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true[:, 1], label='Actual Supply')
        plt.plot(y_demand_pred[:, 1], label='Predicted Supply')
        plt.title(f'BLSTM Model: Actual vs Predicted Supply:  Hourly')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
    if features == True:
        shap_values_dict = {}
        explainer = shap.GradientExplainer(best_model, X_train)
        shap_values = explainer.shap_values(X_test)
        expanded_feature_cols = [
        f"{feature}_t-{i}" for i in range(seq_length, 0, -1) for feature in feature_cols
    ]


        # Compute mean absolute SHAP values for global feature importance
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        mean_shap_values = np.mean(mean_shap_values, axis=0)  # Example: Reduce along the first axis
        mean_shap_values = mean_shap_values.flatten()
        # Identify top 3 features
        top_features = sorted(
                zip(feature_cols, mean_shap_values),
                key=lambda x: x[1],
                reverse=True
            )[:3]
        print("\nTop Features:")
        for feature, importance in top_features:
            print(f"{feature:<30} {importance:.4f}")
        # Print feature importance for this output
        for feature, importance in sorted(zip(feature_cols, mean_shap_values), key=lambda x: x[1], reverse=True):
            print(f"{feature:<30} {importance:.4f}")

        # Prepare SHAP importance with adjusted SHAP values
        shap_importance = {
            f"Output_{i + 1}": sorted(
                zip(expanded_feature_cols, np.abs(shap_values_dict[f"Output_{i + 1}"][:, :-1]).mean(axis=0).tolist()),
                key=lambda x: x[1],
                reverse=True
            )
            for i in range(len(shap_values_dict))
    }

        # Loop through each output (e.g., Demand and Supply)
        for i, shap_value in enumerate(shap_values):
            print(f"\nComputing SHAP values for output {i + 1}...")
            mean_shap_values = np.abs(shap_values).mean(axis=0)
            # Store SHAP values in the dictionary
            shap_values_dict[f"Output_{i + 1}"] = shap_value

        # Prepare SHAP values for return
        X_test_flat = X_test.reshape(X_test.shape[0], -1)  # Shape: (2619, 600)

        # Adjust shap_values to match the flattened structure
        shap_values_flat = shap_values.reshape(shap_values.shape[0], -1, shap_values.shape[-1])  # Shape: (2619, 600, 2)

        # Select the SHAP values for the first output (e.g., Demand)
        shap_values_demand = shap_values_flat[:, :, 0]  # Shape: (2619, 600)
        ##Select the SHAP values for the second output (e.g., Supply)
        shap_values_supply = shap_values_flat[:, :, 1] 

        # Prepare SHAP importance with adjusted SHAP values
        shap_importance = {
            f"Output_{i + 1}": sorted(
                zip(expanded_feature_cols, np.abs(shap_values_dict[f"Output_{i + 1}"][:, :-1]).mean(axis=0).tolist()),
                key=lambda x: x[1],
                reverse=True
            )
            for i in range(len(shap_values_dict))
    }
        """
        print("Shape of X_test_flat:", X_test_flat.shape)
        print("Shape of shap_values_flat:", shap_values_flat.shape)
        print("Shape of shap_values_demand:", shap_values_demand.shape)
        """
        assert shap_values_supply.shape[1] == X_test_flat.shape[1], "Mismatch between shap_values and X_test!"    
        assert shap_values_demand.shape[1] == X_test_flat.shape[1], "Mismatch between shap_values and X_test!"
        assert len(expanded_feature_cols) == X_test_flat.shape[1], "Mismatch between expanded_feature_cols and X_test!"
        mean_shap_values = np.array(mean_shap_values).flatten()
        # Sort features by importance in descending order
        sorted_features = sorted(zip(feature_cols, mean_shap_values), key=lambda x: x[1], reverse=True)

        # Print each feature and its importance
        for feature, importance in sorted_features:
            print("{:<30} {:>12.4f}".format(feature, importance))
        # Plot SHAP summary for the first output
        shap.summary_plot(shap_values_demand, X_test_flat, plot_type="bar", feature_names=expanded_feature_cols)
        shap.summary_plot(shap_values_supply, X_test_flat, plot_type="bar", feature_names=expanded_feature_cols)

        # Ensure compatibility
        print("Expanded Feature Columns:", expanded_feature_cols)
        feature_to_lagged_mapping = {
        feature: [f"{feature}_t-{i}" for i in range(seq_length, 0, -1)]
        for feature in feature_cols
    }   
        """
        # Loop through top features
        for feature, _ in top_features:
            if feature in feature_to_lagged_mapping:
                # Use the first time-lagged version of the feature for the dependence plot
                lagged_feature = feature_to_lagged_mapping[feature][0]
                if lagged_feature in expanded_feature_cols:
                    shap.dependence_plot(
                    lagged_feature, shap_values_demand, X_test_flat, feature_names=expanded_feature_cols
                )
            else:
                print(f"Time-lagged feature {lagged_feature} not found in expanded_feature_cols.")
        else:
            print(f"Feature {feature} not found in feature_to_lagged_mapping.")
        """
        return results  
    return results


def LSTMModelDS(mergedDs: pd.DataFrame, plots:bool,features:bool):
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

    seq_length = 1
    X_seq, y_seq = create_sequences_with_time(X, y, seq_length)

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, shuffle=False
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
        plt.title(f'LSTM Model: Actual vs Predicted Demand:  Hourly')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
        #-Supply Plot-#
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true[:, 1], label='Actual Supply')
        plt.plot(y_demand_pred[:, 1], label='Predicted Supply')
        plt.title(f'LSTM Tree Model: Actual vs Predicted Supply:  Hourly')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
    if features == True:
        shap_values_dict = {}
        explainer = shap.GradientExplainer(best_model, X_train)
        shap_values = explainer.shap_values(X_test)
        expanded_feature_cols = [
        f"{feature}_t-{i}" for i in range(seq_length, 0, -1) for feature in feature_cols
    ]


        # Compute mean absolute SHAP values for global feature importance
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        mean_shap_values = np.mean(mean_shap_values, axis=0)  # Example: Reduce along the first axis
        mean_shap_values = mean_shap_values.flatten()
        # Identify top 3 features
        top_features = sorted(
                zip(feature_cols, mean_shap_values),
                key=lambda x: x[1],
                reverse=True
            )[:3]
        # Print feature importance for this output
        for feature, importance in sorted(zip(feature_cols, mean_shap_values), key=lambda x: x[1], reverse=True):
            print(f"{feature:<30} {importance:.4f}")

        # Prepare SHAP importance with adjusted SHAP values
        shap_importance = {
            f"Output_{i + 1}": sorted(
                zip(expanded_feature_cols, np.abs(shap_values_dict[f"Output_{i + 1}"][:, :-1]).mean(axis=0).tolist()),
                key=lambda x: x[1],
                reverse=True
            )
            for i in range(len(shap_values_dict))
    }

        # Loop through each output (e.g., Demand and Supply)
        for i, shap_value in enumerate(shap_values):
            print(f"\nComputing SHAP values for output {i + 1}...")
            mean_shap_values = np.abs(shap_values).mean(axis=0)
            # Store SHAP values in the dictionary
            shap_values_dict[f"Output_{i + 1}"] = shap_value

        # Prepare SHAP values for return
        X_test_flat = X_test.reshape(X_test.shape[0], -1)  # Shape: (2619, 600)

        # Adjust shap_values to match the flattened structure
        shap_values_flat = shap_values.reshape(shap_values.shape[0], -1, shap_values.shape[-1])  # Shape: (2619, 600, 2)

        # Select the SHAP values for the first output (e.g., Demand)
        shap_values_demand = shap_values_flat[:, :, 0]  # Shape: (2619, 600)
        shap_values_supply = shap_values_flat[:, :, 1] 

        # Prepare SHAP importance with adjusted SHAP values
        shap_importance = {
            f"Output_{i + 1}": sorted(
                zip(expanded_feature_cols, np.abs(shap_values_dict[f"Output_{i + 1}"][:, :-1]).mean(axis=0).tolist()),
                key=lambda x: x[1],
                reverse=True
            )
            for i in range(len(shap_values_dict))
    }
        """
        print("Shape of X_test_flat:", X_test_flat.shape)
        print("Shape of shap_values_flat:", shap_values_flat.shape)
        print("Shape of shap_values_demand:", shap_values_demand.shape)
        """
    
        assert shap_values_demand.shape[1] == X_test_flat.shape[1], "Mismatch between shap_values and X_test!"
        assert shap_values_supply.shape[1] == X_test_flat.shape[1], "Mismatch between shap_values and X_test!"

        assert len(expanded_feature_cols) == X_test_flat.shape[1], "Mismatch between expanded_feature_cols and X_test!"
        mean_shap_values = np.array(mean_shap_values).flatten()
        # Sort features by importance in descending order
        sorted_features = sorted(zip(feature_cols, mean_shap_values), key=lambda x: x[1], reverse=True)

        # Print each feature and its importance
        for feature, importance in sorted_features:
            print("{:<30} {:>12.4f}".format(feature, importance))
        # Plot SHAP summary for the first output
        shap.summary_plot(shap_values_demand, X_test_flat, plot_type="bar", feature_names=expanded_feature_cols)
        shap.summary_plot(shap_values_supply, X_test_flat, plot_type="bar", feature_names=expanded_feature_cols)

        # Ensure compatibility
        print("Expanded Feature Columns:", expanded_feature_cols)
        feature_to_lagged_mapping = {
        feature: [f"{feature}_t-{i}" for i in range(seq_length, 0, -1)]
        for feature in feature_cols
    }
        # Loop through top features
        """
        for feature, _ in top_features:
            if feature in feature_to_lagged_mapping:
                # Use the first time-lagged version of the feature for the dependence plot
                lagged_feature = feature_to_lagged_mapping[feature][0]
                if lagged_feature in expanded_feature_cols:
                    shap.dependence_plot(
                    lagged_feature, shap_values_demand, X_test_flat, feature_names=expanded_feature_cols
                )
            else:
                print(f"Time-lagged feature {lagged_feature} not found in expanded_feature_cols.")
        else:
            print(f"Feature {feature} not found in feature_to_lagged_mapping.")
        
        for feature, _ in top_features:
            if feature in feature_to_lagged_mapping:
                # Use the first time-lagged version of the feature for the dependence plot
                lagged_feature = feature_to_lagged_mapping[feature][0]
                if lagged_feature in expanded_feature_cols:
                    shap.dependence_plot(
                    lagged_feature, shap_values_supply, X_test_flat, feature_names=expanded_feature_cols
                )
            else:
                print(f"Time-lagged feature {lagged_feature} not found in expanded_feature_cols.")
        else:
            print(f"Feature {feature} not found in feature_to_lagged_mapping.")
        """
        return results  
    return results


def GRUModelDS(mergedDs: pd.DataFrame, plots:bool,features:bool):
    mergedDs = mergedDs.copy()

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

    seq_length = 1
    X_seq, y_seq = create_sequences_with_time(X, y, seq_length)

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, shuffle=False
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
        plt.title(f'GRU Model: Actual vs Predicted Demand:  Hourly')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
        #-Supply Plot-#
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true[:, 1], label='Actual Supply')
        plt.plot(y_demand_pred[:, 1], label='Predicted Supply')
        plt.title(f'GRU Model: Actual vs Predicted Supply:  Hourly')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
    if features == True:
        shap_values_dict = {}
        explainer = shap.GradientExplainer(best_model, X_train)
        shap_values = explainer.shap_values(X_test)
        expanded_feature_cols = [
        f"{feature}_t-{i}" for i in range(seq_length, 0, -1) for feature in feature_cols
    ]


        # Compute mean absolute SHAP values for global feature importance
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        mean_shap_values = np.mean(mean_shap_values, axis=0)  # Example: Reduce along the first axis
        mean_shap_values = mean_shap_values.flatten()
        # Identify top 3 features
        top_features = sorted(
                zip(feature_cols, mean_shap_values),
                key=lambda x: x[1],
                reverse=True
            )[:3]
        # Print feature importance for this output
        for feature, importance in sorted(zip(feature_cols, mean_shap_values), key=lambda x: x[1], reverse=True):
            print(f"{feature:<30} {importance:.4f}")

        # Prepare SHAP importance with adjusted SHAP values
        shap_importance = {
            f"Output_{i + 1}": sorted(
                zip(expanded_feature_cols, np.abs(shap_values_dict[f"Output_{i + 1}"][:, :-1]).mean(axis=0).tolist()),
                key=lambda x: x[1],
                reverse=True
            )
            for i in range(len(shap_values_dict))
}

        # Loop through each output (e.g., Demand and Supply)
        for i, shap_value in enumerate(shap_values):
            print(f"\nComputing SHAP values for output {i + 1}...")
            mean_shap_values = np.abs(shap_values).mean(axis=0)
            # Store SHAP values in the dictionary
            shap_values_dict[f"Output_{i + 1}"] = shap_value

        # Prepare SHAP values for return
        X_test_flat = X_test.reshape(X_test.shape[0], -1)  # Shape: (2619, 600)

        # Adjust shap_values to match the flattened structure
        shap_values_flat = shap_values.reshape(shap_values.shape[0], -1, shap_values.shape[-1])  # Shape: (2619, 600, 2)

        # Select the SHAP values for the first output (e.g., Demand)
        shap_values_demand = shap_values_flat[:, :, 0]  # Shape: (2619, 600)

        ##Select the SHAP values for the second output (e.g., Supply)
        shap_values_supply = shap_values_flat[:, :, 1] 
        # Prepare SHAP importance with adjusted SHAP values
        shap_importance = {
            f"Output_{i + 1}": sorted(
                zip(expanded_feature_cols, np.abs(shap_values_dict[f"Output_{i + 1}"][:, :-1]).mean(axis=0).tolist()),
                key=lambda x: x[1],
                reverse=True
            )
            for i in range(len(shap_values_dict))
    }
        """
        print("Shape of X_test_flat:", X_test_flat.shape)
        print("Shape of shap_values_flat:", shap_values_flat.shape)
        print("Shape of shap_values_demand:", shap_values_demand.shape)
        """
        assert shap_values_supply.shape[1] == X_test_flat.shape[1], "Mismatch between shap_values and X_test!"
    
        assert shap_values_demand.shape[1] == X_test_flat.shape[1], "Mismatch between shap_values and X_test!"
        assert len(expanded_feature_cols) == X_test_flat.shape[1], "Mismatch between expanded_feature_cols and X_test!"
        mean_shap_values = np.array(mean_shap_values).flatten()
        # Sort features by importance in descending order
        sorted_features = sorted(zip(feature_cols, mean_shap_values), key=lambda x: x[1], reverse=True)

        # Print each feature and its importance
        for feature, importance in sorted_features:
            print("{:<30} {:>12.4f}".format(feature, importance))
        # Plot SHAP summary for the first output
        # Plot SHAP summary for the first output
        shap.summary_plot(shap_values_demand, X_test_flat, plot_type="bar", feature_names=expanded_feature_cols)
        shap.summary_plot(shap_values_supply, X_test_flat, plot_type="bar", feature_names=expanded_feature_cols)

        # Ensure compatibility
        print("Expanded Feature Columns:", expanded_feature_cols)
        feature_to_lagged_mapping = {
        feature: [f"{feature}_t-{i}" for i in range(seq_length, 0, -1)]
        for feature in feature_cols
    }
        # Loop through top features
        for feature, _ in top_features:
            if feature in feature_to_lagged_mapping:
                # Use the first time-lagged version of the feature for the dependence plot
                lagged_feature = feature_to_lagged_mapping[feature][0]
                if lagged_feature in expanded_feature_cols:
                    shap.dependence_plot(
                    lagged_feature, shap_values_demand, X_test_flat, feature_names=expanded_feature_cols
                )
            else:
                print(f"Time-lagged feature {lagged_feature} not found in expanded_feature_cols.")
        else:
            print(f"Feature {feature} not found in feature_to_lagged_mapping.")
        
        return results, sorted_features        
    return results


def SVRModelDS(mergedDs: pd.DataFrame, plots:bool,features:bool):
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
    seq_length = 1
    X_seq, y_seq = create_sequence_with_time(mergedDs,feature_cols,target_cols, seq_length)
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X = scaler_X.fit_transform(X_seq)
    y = scaler_y.fit_transform(y_seq)

    

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2,shuffle=False
    )
    """
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW', 'Supply_MW']
    X = scaler_X.fit_transform(mergedDs[feature_cols])
    y = scaler_y.fit_transform(mergedDs[target_cols])

    seq_length = 1
    X_seq, y_seq = create_sequences_with_time(X, y, seq_length)

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, shuffle=False
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
        plt.title(f'SVR Model: Actual vs Predicted Demand:  Hourly')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
        #-Supply Plot-#
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true[:, 1], label='Actual Supply')
        plt.plot(y_demand_pred[:, 1], label='Predicted Supply')
        plt.title(f'SVR Model: Actual vs Predicted Supply:  Hourly')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
    if features == True:
        shap_values_dict = {}
        # Use KernelExplainer for SVR
        explainer = shap.KernelExplainer(grid.predict, X_train[:10])  # Use a subset of X_train for efficiency
        shap_values = explainer.shap_values(X_test[:10])  # Compute SHAP values for a subset of X_test

        expanded_feature_cols = [
            f"{feature}_t-{i}" for i in range(seq_length, 0, -1) for feature in feature_cols
        ]

        # Compute mean absolute SHAP values for global feature importance
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        mean_shap_values = mean_shap_values.flatten()  # Ensure it's a flat array
        # Identify top 3 features
        top_features = sorted(
            zip(feature_cols, mean_shap_values),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        # Print feature importance for this output
        for feature, importance in sorted(zip(feature_cols, mean_shap_values), key=lambda x: x[1], reverse=True):
            print(f"{feature:<30} {importance:.4f}")

        # Prepare SHAP importance with adjusted SHAP values
        shap_importance = {
            f"Output_{i + 1}": sorted(
                zip(expanded_feature_cols, np.abs(shap_values_dict[f"Output_{i + 1}"]).mean(axis=0).tolist()),
                key=lambda x: x[1],
                reverse=True
            )
            for i in range(len(shap_values_dict))
        }

        # Loop through each output (e.g., Demand and Supply)
        for i, shap_value in enumerate(shap_values):
            print(f"\nComputing SHAP values for output {i + 1}...")
            mean_shap_values = np.abs(shap_values).mean(axis=0)
            # Store SHAP values in the dictionary
            shap_values_dict[f"Output_{i + 1}"] = shap_value

        # Prepare SHAP values for return
        X_test_flat = X_test[:10].reshape(X_test[:10].shape[0], -1)  # Flatten the subset of X_test

        # Adjust shap_values to match the flattened structure
        shap_values_flat = np.array(shap_values).reshape(len(shap_values), -1)  # Flatten SHAP values

        # Select the SHAP values for the first output (e.g., Demand)
        shap_values_demand = shap_values_flat[:, :len(expanded_feature_cols)]  # Adjust to match feature columns

        ##Select the SHAP values for the second output (e.g., Supply)
        shap_values_supply = shap_values_flat[:, :, 1] 
        # Prepare SHAP importance with adjusted SHAP values
        shap_importance = {
            f"Output_{i + 1}": sorted(
                zip(expanded_feature_cols, np.abs(shap_values_dict[f"Output_{i + 1}"]).mean(axis=0).tolist()),
                key=lambda x: x[1],
                reverse=True
            )
            for i in range(len(shap_values_dict))
        }

        # Debugging prints
        print("Shape of X_test_flat:", X_test_flat.shape)
        print("Shape of shap_values_flat:", shap_values_flat.shape)
        print("Shape of shap_values_demand:", shap_values_demand.shape)
        assert shap_values_supply.shape[1] == X_test_flat.shape[1], "Mismatch between shap_values and X_test!"

        assert shap_values_demand.shape[1] == X_test_flat.shape[1], "Mismatch between shap_values and X_test!"
        assert len(expanded_feature_cols) == X_test_flat.shape[1], "Mismatch between expanded_feature_cols and X_test!"
        mean_shap_values = np.array(mean_shap_values).flatten()
        # Sort features by importance in descending order
        sorted_features = sorted(zip(feature_cols, mean_shap_values), key=lambda x: x[1], reverse=True)

        # Print each feature and its importance
        for feature, importance in sorted_features:
            print("{:<30} {:>12.4f}".format(feature, importance))
        # Plot SHAP summary for the first output
        shap.summary_plot(shap_values_demand, X_test_flat, plot_type="bar", feature_names=expanded_feature_cols)
        # Plot SHAP summary for the seccond output
        shap.summary_plot(shap_values_supply, X_test_flat, plot_type="bar", feature_names=expanded_feature_cols)

        # Ensure compatibility
        print("Expanded Feature Columns:", expanded_feature_cols)
        feature_to_lagged_mapping = {
            feature: [f"{feature}_t-{i}" for i in range(seq_length, 0, -1)]
            for feature in feature_cols
        }

        # Loop through top features
        for feature, _ in top_features:
            if feature in feature_to_lagged_mapping:
                # Use the first time-lagged version of the feature for the dependence plot
                lagged_feature = feature_to_lagged_mapping[feature][0]
                if lagged_feature in expanded_feature_cols:
                    shap.dependence_plot(
                        lagged_feature, shap_values_demand, X_test_flat, feature_names=expanded_feature_cols
                    )
                else:
                    print(f"Time-lagged feature {lagged_feature} not found in expanded_feature_cols.")
            else:
                print(f"Feature {feature} not found in feature_to_lagged_mapping.")

        return results, sorted_features        
    return results


def MLPModelDS(mergedDs: pd.DataFrame, plots:bool,features:bool):
    mergedDs = mergedDs.copy()


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

    seq_length = 1
    X_seq, y_seq = create_sequences_with_time(X, y, seq_length)

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, shuffle=False
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
        plt.title(f'MLP Model: Actual vs Predicted Demand:  Hourly')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true[:, 1], label='Actual Supply')
        plt.plot(y_demand_pred[:, 1], label='Predicted Supply')
        plt.title(f'MLP Model: Actual vs Predicted Supply:  Hourly')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
    if features == True:
        shap_values_dict = {}
        explainer = shap.GradientExplainer(best_model, X_train)
        shap_values = explainer.shap_values(X_test)
        expanded_feature_cols = [
        f"{feature}_t-{i}" for i in range(seq_length, 0, -1) for feature in feature_cols
        ]


        # Compute mean absolute SHAP values for global feature importance
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        mean_shap_values = np.mean(mean_shap_values, axis=0)  # Example: Reduce along the first axis
        mean_shap_values = mean_shap_values.flatten()
        # Identify top 3 features
        top_features = sorted(
                zip(feature_cols, mean_shap_values),
                key=lambda x: x[1],
                reverse=True
            )[:3]
        # Print feature importance for this output
        for feature, importance in sorted(zip(feature_cols, mean_shap_values), key=lambda x: x[1], reverse=True):
            print(f"{feature:<30} {importance:.4f}")

        # Prepare SHAP importance with adjusted SHAP values
        shap_importance = {
            f"Output_{i + 1}": sorted(
                zip(expanded_feature_cols, np.abs(shap_values_dict[f"Output_{i + 1}"][:, :-1]).mean(axis=0).tolist()),
                key=lambda x: x[1],
                reverse=True
            )
            for i in range(len(shap_values_dict))
    }

        # Loop through each output (e.g., Demand and Supply)
        for i, shap_value in enumerate(shap_values):
            print(f"\nComputing SHAP values for output {i + 1}...")
            mean_shap_values = np.abs(shap_values).mean(axis=0)
            # Store SHAP values in the dictionary
            shap_values_dict[f"Output_{i + 1}"] = shap_value

        # Prepare SHAP values for return
        X_test_flat = X_test.reshape(X_test.shape[0], -1)  # Shape: (2619, 600)

        # Adjust shap_values to match the flattened structure
        shap_values_flat = shap_values.reshape(shap_values.shape[0], -1, shap_values.shape[-1])  # Shape: (2619, 600, 2)

        # Select the SHAP values for the first output (e.g., Demand)
        shap_values_demand = shap_values_flat[:, :, 0]  # Shape: (2619, 600)
        ##Select the SHAP values for the second output (e.g., Supply)
        shap_values_supply = shap_values_flat[:, :, 1] 
        # Prepare SHAP importance with adjusted SHAP values
        shap_importance = {
            f"Output_{i + 1}": sorted(
                zip(expanded_feature_cols, np.abs(shap_values_dict[f"Output_{i + 1}"][:, :-1]).mean(axis=0).tolist()),
                key=lambda x: x[1],
                reverse=True
            )
            for i in range(len(shap_values_dict))
        }
        """
        print("Shape of X_test_flat:", X_test_flat.shape)
        print("Shape of shap_values_flat:", shap_values_flat.shape)
        print("Shape of shap_values_demand:", shap_values_demand.shape)
        """
        assert shap_values_supply.shape[1] == X_test_flat.shape[1], "Mismatch between shap_values and X_test!"
    
        assert shap_values_demand.shape[1] == X_test_flat.shape[1], "Mismatch between shap_values and X_test!"
        assert len(expanded_feature_cols) == X_test_flat.shape[1], "Mismatch between expanded_feature_cols and X_test!"

    
        mean_shap_values = np.array(mean_shap_values).flatten()
        # Sort features by importance in descending order
        sorted_features = sorted(zip(feature_cols, mean_shap_values), key=lambda x: x[1], reverse=True)

        # Print each feature and its importance
        for feature, importance in sorted_features:
            print("{:<30} {:>12.4f}".format(feature, importance))
        # Plot SHAP summary for the first output
        shap.summary_plot(shap_values_demand, X_test_flat, plot_type="bar", feature_names=expanded_feature_cols)
        shap.summary_plot(shap_values_supply, X_test_flat, plot_type="bar", feature_names=expanded_feature_cols)

        # Ensure compatibility
        print("Expanded Feature Columns:", expanded_feature_cols)
        feature_to_lagged_mapping = {
        feature: [f"{feature}_t-{i}" for i in range(seq_length, 0, -1)]
        for feature in feature_cols
        }
        # Loop through top features
        for feature, _ in top_features:
            if feature in feature_to_lagged_mapping:
                # Use the first time-lagged version of the feature for the dependence plot
                lagged_feature = feature_to_lagged_mapping[feature][0]
                if lagged_feature in expanded_feature_cols:
                    shap.dependence_plot(
                    lagged_feature, shap_values_demand, X_test_flat, feature_names=expanded_feature_cols
                )
            else:
                print(f"Time-lagged feature {lagged_feature} not found in expanded_feature_cols.")
        else:
            print(f"Feature {feature} not found in feature_to_lagged_mapping.")
        
        return results       
    return results


def CNNModelDS(mergedDs: pd.DataFrame, plots:bool,features:bool):
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

    seq_length = 1
    X_seq, y_seq = create_sequences_with_time(X, y, seq_length)
    print(f"Shape of X_seq: {X_seq.shape}")  # Debugging output
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, shuffle=False
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
                kernel_size = min(hp.Choice('kernel_size1', [2, 3, 5]), seq_length),
                activation='relu',
                input_shape=(seq_length, num_features)
            ),
            MaxPooling1D(pool_size=2) if seq_length > 1 else tf.keras.layers.Lambda(lambda x: x),
            Dropout(hp.Float('dropout1', 0.1, 0.5, step=0.1)),
            Conv1D(
                filters=hp.Int('filters2', 32, 128, step=32),
                kernel_size = min(hp.Choice('kernel_size2', [2, 3, 5]), seq_length),
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
    if plots == True:

        #-Demand Plot-#
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true[:, 0], label='Actual Demand')
        plt.plot(y_demand_pred[:, 0], label='Predicted Demand')
        plt.title(f'CNN Model: Actual vs Predicted Demand: Hourly')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
        #-Supply Plot-#
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true[:, 1], label='Actual Supply')
        plt.plot(y_demand_pred[:, 1], label='Predicted Supply')
        plt.title(f'CNN Model: Actual vs Predicted Supply: Hourly')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
    if features == True:
        shap_values_dict = {}
        explainer = shap.GradientExplainer(best_model, X_train)
        shap_values = explainer.shap_values(X_test)
        expanded_feature_cols = [
        f"{feature}_t-{i}" for i in range(seq_length, 0, -1) for feature in feature_cols
    ]


        # Compute mean absolute SHAP values for global feature importance
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        mean_shap_values = np.mean(mean_shap_values, axis=0)  # Example: Reduce along the first axis
        mean_shap_values = mean_shap_values.flatten()
        # Identify top 3 features
        top_features = sorted(
                zip(feature_cols, mean_shap_values),
                key=lambda x: x[1],
                reverse=True
            )[:3]
        # Print feature importance for this output
        for feature, importance in sorted(zip(feature_cols, mean_shap_values), key=lambda x: x[1], reverse=True):
            print(f"{feature:<30} {importance:.4f}")

        # Prepare SHAP importance with adjusted SHAP values
        shap_importance = {
            f"Output_{i + 1}": sorted(
                zip(expanded_feature_cols, np.abs(shap_values_dict[f"Output_{i + 1}"][:, :-1]).mean(axis=0).tolist()),
                key=lambda x: x[1],
                reverse=True
            )
            for i in range(len(shap_values_dict))
}

        # Loop through each output (e.g., Demand and Supply)
        for i, shap_value in enumerate(shap_values):
            print(f"\nComputing SHAP values for output {i + 1}...")
            mean_shap_values = np.abs(shap_values).mean(axis=0)
            # Store SHAP values in the dictionary
            shap_values_dict[f"Output_{i + 1}"] = shap_value

        # Prepare SHAP values for return
        X_test_flat = X_test.reshape(X_test.shape[0], -1)  # Shape: (2619, 600)

        # Adjust shap_values to match the flattened structure
        shap_values_flat = shap_values.reshape(shap_values.shape[0], -1, shap_values.shape[-1])  # Shape: (2619, 600, 2)

        # Select the SHAP values for the first output (e.g., Demand)
        shap_values_demand = shap_values_flat[:, :, 0]  # Shape: (2619, 600)
        ##Select the SHAP values for the second output (e.g., Supply)
        shap_values_supply = shap_values_flat[:, :, 1] 
        # Prepare SHAP importance with adjusted SHAP values
        shap_importance = {
            f"Output_{i + 1}": sorted(
                zip(expanded_feature_cols, np.abs(shap_values_dict[f"Output_{i + 1}"][:, :-1]).mean(axis=0).tolist()),
                key=lambda x: x[1],
                reverse=True
            )
            for i in range(len(shap_values_dict))
    }
        """
        print("Shape of X_test_flat:", X_test_flat.shape)
        print("Shape of shap_values_flat:", shap_values_flat.shape)
        print("Shape of shap_values_demand:", shap_values_demand.shape)
        """
        assert shap_values_supply.shape[1] == X_test_flat.shape[1], "Mismatch between shap_values and X_test!"
        assert shap_values_demand.shape[1] == X_test_flat.shape[1], "Mismatch between shap_values and X_test!"
        assert len(expanded_feature_cols) == X_test_flat.shape[1], "Mismatch between expanded_feature_cols and X_test!"

        mean_shap_values = np.array(mean_shap_values).flatten()
        # Sort features by importance in descending order
        sorted_features = sorted(zip(feature_cols, mean_shap_values), key=lambda x: x[1], reverse=True)

        # Print each feature and its importance
        for feature, importance in sorted_features:
            print("{:<30} {:>12.4f}".format(feature, importance))
        # Plot SHAP summary for the first output
        shap.summary_plot(shap_values_demand, X_test_flat, plot_type="bar", feature_names=expanded_feature_cols)
        shap.summary_plot(shap_values_supply, X_test_flat, plot_type="bar", feature_names=expanded_feature_cols)

        # Ensure compatibility
        print("Expanded Feature Columns:", expanded_feature_cols)
        feature_to_lagged_mapping = {
        feature: [f"{feature}_t-{i}" for i in range(seq_length, 0, -1)]
        for feature in feature_cols
    }
        # Loop through top features
        for feature, _ in top_features:
            if feature in feature_to_lagged_mapping:
                # Use the first time-lagged version of the feature for the dependence plot
                lagged_feature = feature_to_lagged_mapping[feature][0]
                if lagged_feature in expanded_feature_cols:
                    shap.dependence_plot(
                    lagged_feature, shap_values_demand, X_test_flat, feature_names=expanded_feature_cols
                )
            else:
                print(f"Time-lagged feature {lagged_feature} not found in expanded_feature_cols.")
        else:
            print(f"Feature {feature} not found in feature_to_lagged_mapping.")
        
        return results, sorted_features        
    return results


def GBDTModelDS(mergedDs: pd.DataFrame, plots:bool,features:bool):
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

    seq_length = 1
    X_seq, y_seq = create_sequences_with_time(X, y, seq_length)

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, shuffle=False
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
        plt.title(f'GBDT Model: Actual vs Predicted Demand: Hourly')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
        #-Supply Plot-#
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true[:, 1], label='Actual Supply')
        plt.plot(y_demand_pred[:, 1], label='Predicted Supply')
        plt.title(f'GBDT Model: Actual vs Predicted Supply: Hourly')
        plt.xlabel('Time')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()
    if features == True:
        shap_importance = compute_shap_values_Trees(
            model_type="GBDT",
            best_tree_model=grid_search.best_estimator_,
            X_test=X_test,
            feature_cols=feature_cols,
            seq_length=seq_length
        )
        return results    
    return results

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
    plots: bool,
    features: bool,
    pop_size=5,
    ngen=3,
    cxpb=0.7,
    mutpb=0.3,
    seq_length = 1,
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
        X_seq, y_seq, test_size=0.2, shuffle=False
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
    toolbox.register("kernel_size1", random.choice, [k for k in [1, 2, 3, 5] if k <= seq_length] or [1])
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
            kernel_size1 = min(kernel_size1, seq_length)  # Ensure kernel size is valid
            pool_size = min(2, seq_length)  # Dynamically adjust pool size

            model = Sequential([
                Conv1D(filters=filters1, kernel_size=kernel_size1, activation='relu', input_shape=(seq_length, num_features)),
                MaxPooling1D(pool_size=pool_size) if seq_length > 1 else tf.keras.layers.Lambda(lambda x: x),  # Skip pooling if seq_length == 1
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
    pool_size = min(2, seq_length)  # Dynamically adjust pool size

    final_model = Sequential([
        Conv1D(filters=int(filters1), kernel_size=int(kernel_size1), activation='relu', input_shape=(seq_length, num_features)),
        MaxPooling1D(pool_size=pool_size) if seq_length > 1 else tf.keras.layers.Lambda(lambda x: x),  # Skip pooling if seq_length == 1
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
    if features == True:
        shap_values_dict = {}
        explainer = shap.GradientExplainer(final_model, X_train)
        shap_values = explainer.shap_values(X_test)
        expanded_feature_cols = [
            f"{feature}_t-{i}" for i in range(seq_length, 0, -1) for feature in feature_cols
        ]

        # Compute mean absolute SHAP values for global feature importance
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        mean_shap_values = np.mean(mean_shap_values, axis=0)  # Reduce along the first axis
        mean_shap_values = mean_shap_values.flatten()

        # Identify top 3 features
        top_features = sorted(
            zip(feature_cols, mean_shap_values),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        # Print feature importance for this output
        for feature, importance in sorted(zip(feature_cols, mean_shap_values), key=lambda x: x[1], reverse=True):
            print(f"{feature:<30} {importance:.4f}")

        # Prepare SHAP importance with adjusted SHAP values
        shap_importance = {
            f"Output_{i + 1}": sorted(
                zip(expanded_feature_cols, np.abs(shap_values_dict[f"Output_{i + 1}"][:, :-1]).mean(axis=0).tolist()),
                key=lambda x: x[1],
                reverse=True
            )
            for i in range(len(shap_values_dict))
        }

        # Loop through each output (e.g., Demand and Supply)
        for i, shap_value in enumerate(shap_values):
            print(f"\nComputing SHAP values for output {i + 1}...")
            mean_shap_values = np.abs(shap_values).mean(axis=0)
            # Store SHAP values in the dictionary
            shap_values_dict[f"Output_{i + 1}"] = shap_value

        # Prepare SHAP values for return
        X_test_flat = X_test.reshape(X_test.shape[0], -1)  # Flatten X_test

        # Adjust shap_values to match the flattened structure
        shap_values_flat = shap_values.reshape(shap_values.shape[0], -1, shap_values.shape[-1])  # Flatten SHAP values

        # Select the SHAP values for the first output (e.g., Demand)
        shap_values_demand = shap_values_flat[:, :, 0]  # Adjust to match feature columns
        
        ##Select the SHAP values for the second output (e.g., Supply)
        shap_values_supply = shap_values_flat[:, :, 1] 
        print(f"Type of expanded_feature_cols: {type(expanded_feature_cols)}")
        print(f"Type of X_test_flat: {type(X_test_flat)}")
        print(f"Length of expanded_feature_cols: {len(expanded_feature_cols)}")
        print(f"Shape of X_test_flat: {X_test_flat.shape}")
        # Ensure compatibility
        assert shap_values_supply.shape[1] == X_test_flat.shape[1], "Mismatch between shap_values and X_test!"
        assert shap_values_demand.shape[1] == X_test_flat.shape[1], "Mismatch between shap_values and X_test!"
        assert len(expanded_feature_cols) == X_test_flat.shape[1], (
        f"Mismatch between expanded_feature_cols and X_test! "
        f"expanded_feature_cols length: {len(expanded_feature_cols)}, "
        f"X_test_flat shape[1]: {X_test_flat.shape[1]}"
        )
        # Sort features by importance in descending order
        sorted_features = sorted(zip(feature_cols, mean_shap_values), key=lambda x: x[1], reverse=True)

        # Print each feature and its importance
        for feature, importance in sorted_features:
            feature = str(feature)  # Ensure feature is a string
            if isinstance(importance, np.ndarray):
                if importance.size == 1:
                    importance = importance.item()  # Extract scalar value
                else:
                    importance = importance.mean()  # Aggregate values (e.g., mean)
            importance = float(importance)  # Convert to float
            print("{:<30} {:>12.4f}".format(feature, importance))

        # Plot SHAP summary for the first output
        shap.summary_plot(shap_values_demand, X_test_flat, plot_type="bar", feature_names=expanded_feature_cols)
        shap.summary_plot(shap_values_supply, X_test_flat, plot_type="bar", feature_names=expanded_feature_cols)

        """
        # Feature-to-lagged mapping for dependence plots
        feature_to_lagged_mapping = {
            feature: [f"{feature}_t-{i}" for i in range(seq_length, 0, -1)]
            for feature in feature_cols
        }

        # Loop through top features
        for feature, _ in top_features:
            if feature in feature_to_lagged_mapping:
                lagged_feature = feature_to_lagged_mapping[feature][0]
                if lagged_feature in expanded_feature_cols:
                    shap.dependence_plot(
                        lagged_feature, shap_values_demand, X_test_flat, feature_names=expanded_feature_cols
                    )
                else:
                    print(f"Time-lagged feature {lagged_feature} not found in expanded_feature_cols.")
            else:
                print(f"Feature {feature} not found in feature_to_lagged_mapping.")
        """
        return results, sorted_features    
    return results    

def NSGA3_CNN_ModelDS(
    mergedDs: pd.DataFrame,
    plots: bool,
    features:bool,
       pop_size=5,
    ngen=3,
    cxpb=0.7,
    mutpb=0.3,
    seq_length = 1,
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
        X_seq, y_seq, test_size=0.2, shuffle=False
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
    toolbox.register("kernel_size1", random.choice, [k for k in [1, 2, 3, 5] if k <= seq_length] or [1])
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
            kernel_size1 = min(kernel_size1, seq_length)  # Ensure kernel size is valid
            pool_size = min(2, seq_length)  # Dynamically adjust pool size

            model = Sequential([
                Conv1D(filters=filters1, kernel_size=kernel_size1, activation='relu', input_shape=(seq_length, num_features)),
                MaxPooling1D(pool_size=pool_size) if seq_length > 1 else tf.keras.layers.Lambda(lambda x: x),  # Skip pooling if seq_length == 1
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
    pool_size = min(2, seq_length)  # Dynamically adjust pool size

    final_model = Sequential([
        Conv1D(filters=int(filters1), kernel_size=min(int(kernel_size1), seq_length), activation='relu', input_shape=(seq_length, num_features)),
        MaxPooling1D(pool_size=min(2, seq_length)) if seq_length > 1 else tf.keras.layers.Lambda(lambda x: x),  # Skip pooling if seq_length == 1
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
    if features == True:
        shap_values_dict = {}
        explainer = shap.GradientExplainer(final_model, X_train)
        shap_values = explainer.shap_values(X_test)
        expanded_feature_cols = [
            f"{feature}_t-{i}" for i in range(seq_length, 0, -1) for feature in feature_cols
        ]

        # Compute mean absolute SHAP values for global feature importance
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        mean_shap_values = np.mean(mean_shap_values, axis=0)  # Reduce along the first axis
        mean_shap_values = mean_shap_values.flatten()

        # Identify top 3 features
        top_features = sorted(
            zip(feature_cols, mean_shap_values),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        # Print feature importance for this output
        for feature, importance in sorted(zip(feature_cols, mean_shap_values), key=lambda x: x[1], reverse=True):
            print(f"{feature:<30} {importance:.4f}")
        # Print feature importance for this output


        # Prepare SHAP importance with adjusted SHAP values
        shap_importance = {
            f"Output_{i + 1}": sorted(
                zip(expanded_feature_cols, np.abs(shap_values_dict[f"Output_{i + 1}"][:, :-1]).mean(axis=0).tolist()),
                key=lambda x: x[1],
                reverse=True
            )
            for i in range(len(shap_values_dict))
        }

        # Loop through each output (e.g., Demand and Supply)
        for i, shap_value in enumerate(shap_values):
            print(f"\nComputing SHAP values for output {i + 1}...")
            mean_shap_values = np.abs(shap_values).mean(axis=0)
            # Store SHAP values in the dictionary
            shap_values_dict[f"Output_{i + 1}"] = shap_value

        # Prepare SHAP values for return
        X_test_flat = X_test.reshape(X_test.shape[0], -1)  # Flatten X_test

        # Adjust shap_values to match the flattened structure
        shap_values_flat = shap_values.reshape(shap_values.shape[0], -1, shap_values.shape[-1])  # Flatten SHAP values

        # Select the SHAP values for the first output (e.g., Demand)
        shap_values_demand = shap_values_flat[:, :, 0]  # Adjust to match feature columns
        ##Select the SHAP values for the second output (e.g., Supply)
        shap_values_supply = shap_values_flat[:, :, 1] 

        # Ensure compatibility
        assert shap_values_supply.shape[1] == X_test_flat.shape[1], "Mismatch between shap_values and X_test!"
        assert shap_values_demand.shape[1] == X_test_flat.shape[1], "Mismatch between shap_values and X_test!"
        assert len(expanded_feature_cols) == X_test_flat.shape[1], (
        f"Mismatch between expanded_feature_cols and X_test! "
        f"expanded_feature_cols length: {len(expanded_feature_cols)}, "
        f"X_test_flat shape[1]: {X_test_flat.shape[1]}"
        )

        # Sort features by importance in descending order
        sorted_features = sorted(zip(feature_cols, mean_shap_values), key=lambda x: x[1], reverse=True)

        # Print each feature and its importance
        for feature, importance in sorted_features:
            feature = str(feature)  # Ensure feature is a string
            if isinstance(importance, np.ndarray):
                if importance.size == 1:
                    importance = importance.item()  # Extract scalar value
                else:
                    importance = importance.mean()  # Aggregate values (e.g., mean)
            importance = float(importance)  # Convert to float
            print("{:<30} {:>12.4f}".format(feature, importance))
        # Plot SHAP summary for the first output
        shap.summary_plot(shap_values_demand, X_test_flat, plot_type="bar", feature_names=expanded_feature_cols)
        shap.summary_plot(shap_values_supply, X_test_flat, plot_type="bar", feature_names=expanded_feature_cols)
        """
        # Feature-to-lagged mapping for dependence plots
        feature_to_lagged_mapping = {
            feature: [f"{feature}_t-{i}" for i in range(seq_length, 0, -1)]
            for feature in feature_cols
        }
        
        # Loop through top features
        for feature, _ in top_features:
            if feature in feature_to_lagged_mapping:
                lagged_feature = feature_to_lagged_mapping[feature][0]
                if lagged_feature in expanded_feature_cols:
                    shap.dependence_plot(
                        lagged_feature, shap_values_demand, X_test_flat, feature_names=expanded_feature_cols
                    )
                else:
                    print(f"Time-lagged feature {lagged_feature} not found in expanded_feature_cols.")
            else:
                print(f"Feature {feature} not found in feature_to_lagged_mapping.")
        """
        return results, sorted_features    
    return results



def compile_shap_values(models_with_shap):
    """
    Compiles sorted SHAP features from multiple models into a dictionary.
    
    Args:
        models_with_shap (list): A list of tuples where each tuple contains:
            - model_name (str): Name of the model.
            - sorted_features (list): Sorted feature importance from the model.
    
    Returns:
        dict: A dictionary where keys are model names and values are DataFrames
              containing feature names and their importance.
    """
    compiled_tables = {}
    for model_name, sorted_features in models_with_shap:
        shap_df = pd.DataFrame(sorted_features, columns=["Feature", "Importance"])
        compiled_tables[model_name] = shap_df
    return compiled_tables

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

##DecTree = ensure_real_result(decisionTreeModelDS(sp_FullMerg,False))
"""
DecTree,Shap_DecTree = ensure_real_result(decisionTreeModelDS(sp_FullMerg,False,False))
randForest ,Shap_RandForest= ensure_real_result(randomForestModelDS(sp_FullMerg,False,False))
xgb,Shap_XGB  = ensure_real_result(xgbModelDS(sp_FullMerg,False,False))
gbdt,Shap_GBDT = ensure_real_result(GBDTModelDS(sp_FullMerg,False,False))
blstm,Shap_BLSTM = ensure_real_result(biDirectionalLSTMDS(sp_FullMerg,True,True))
lstm,Shap_LSTM= ensure_real_result(LSTMModelDS(sp_FullMerg,True,True))
gru,Shap_GRU  = ensure_real_result(GRUModelDS(sp_FullMerg,True,True))
svr,Shap_SVR  = ensure_real_result(SVRModelDS(sp_FullMerg,True,False))
mlp,Shap_MLP  = ensure_real_result(MLPModelDS(sp_FullMerg,True,True))
cnn,Shap_CNN  = ensure_real_result(CNNModelDS(sp_FullMerg,True,True))
nsga2cnn,Shap_NSGA2CNN  = ensure_real_result(NSGA2_CNN_ModelDS(sp_FullMerg,True,True))
nsga3cnn,Shap_NSGA3CNN = ensure_real_result(NSGA3_CNN_ModelDS(sp_FullMerg,True,True))
"""
"""
DecTree = ensure_real_result(decisionTreeModelDS(sp_FullMerg,False,True))
randForest = ensure_real_result(randomForestModelDS(sp_FullMerg,False,True))
xgb  = ensure_real_result(xgbModelDS(sp_FullMerg,False,True))
gbdt = ensure_real_result(GBDTModelDS(sp_FullMerg,False,True))

blstm = ensure_real_result(biDirectionalLSTMDS(sp_FullMerg,False,True))
lstm= ensure_real_result(LSTMModelDS(sp_FullMerg,False,True))
gru  = ensure_real_result(GRUModelDS(sp_FullMerg,False,True))
#svr = ensure_real_result(SVRModelDS(sp_FullMerg,False,False))
mlp  = ensure_real_result(MLPModelDS(sp_FullMerg,False,True))
"""
cnn  = ensure_real_result(CNNModelDS(sp_FullMerg,False,True))
nsga2cnn  = ensure_real_result(NSGA2_CNN_ModelDS(sp_FullMerg,False,True))
nsga3cnn = ensure_real_result(NSGA3_CNN_ModelDS(sp_FullMerg,True,True))
#modelresults = [DecTree, randForest, xgb,gb, gbdt, blstm, lstm, gru, svr, mlp, cnn, nsga2cnn, nsga3cnn]
#for feature, importance in Shap_DecTree["Sorted Feature Importance"]:
#    print(f"{feature:<30} {importance:.4f}"
"""
models_with_shap = [
    ("Decision Tree", Shap_DecTree),
    ("Random Forest", Shap_RandForest),
    ("XGB", Shap_XGB),
    ("GBDT", Shap_GBDT),
    ("BLSTM", Shap_BLSTM),
    ("LSTM", Shap_LSTM),
    ("GRU", Shap_GRU),
    #("SVR", Shap_SVR),
    ("MLP", Shap_MLP),
    ("CNN", Shap_CNN),
    ("NSGA-II CNN", Shap_NSGA2CNN),
    ("NSGA-III CNN", Shap_NSGA3CNN),
]

compiled_shap_tables = compile_shap_values(models_with_shap)
# Print tables for each model
for model_name, shap_table in compiled_shap_tables.items():
    print(f"\nSHAP Values for {model_name}:")
    print(shap_table)
"""
"""
BestResultsOrdered = BetterModelSelectionMethod(modelresults)

print("\nModel Ranking (Best to Worst) Hourly:")
print("{:<20} {:>12} {:>12} {:>12} {:>10}".format("Model", "MSE", "MAE", "RMSE", "R2"))
print("-" * 70)
for res in BestResultsOrdered:
    print("{:<20} {:>12.4f} {:>12.4f} {:>12.4f} {:>10.4f}".format(
        res[4], res[0], res[1], res[2], res[3]
    ))
"""

