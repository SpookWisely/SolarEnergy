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
import traceback
import keras_tuner as kt
##Questions to ask Tomorrow
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None) 
"""
1. Should I look into adding lag into the supply dataset aspects of the method as its results for MSE specifically are wildly different
  to that of Demand dataset results.
2. Why when comparing the creation process for the merged dataset in olivers code does he
duplicate the weather data
with ALLSKY_SFC_SW_DWN ending up as ALLSKY_SFC_SW_DWN_x/ALLSKY_SFC_SW_DWN_y as an example.
"""
# For clearing cache for when the shape is changed in the tuner directory

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

for subdir in tuner_subdirs:
    if os.path.exists(subdir):
        shutil.rmtree(subdir)

sp_DemandDef = pd.read_excel(r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\Sakakah 2021 Demand dataset.xlsx")
sp_SupplyDef = pd.read_excel(r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\Sakakah 2021 PV supply dataset.xlsx")
sp_WeatherDe = pd.read_excel(r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\Weather for demand 2018.xlsx")
sp_WeatherSu = pd.read_excel(r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\Oliver Test\weather for solar NEW 2021.xlsx")

##sp_WeatherDeDef = pd.read_excel
sp_DemandDef["TimeStamp"] = pd.to_datetime(sp_DemandDef["DATE-TIME"])
sp_SupplyDef["TimeStamp"] = pd.to_datetime(sp_SupplyDef["Date & Time"])
sp_WeatherDe["TimeStamp"] = pd.to_datetime(
    sp_WeatherDe[['YEAR', 'MO', 'DY', 'HR']].astype(str).agg('-'.join, axis=1),
    format="%Y-%m-%d-%H"
)
sp_WeatherSu["TimeStamp"] = pd.to_datetime(
    sp_WeatherDe[['YEAR', 'MO', 'DY', 'HR']].astype(str).agg('-'.join, axis=1),
    format="%Y-%m-%d-%H"
)
sp_WeatherWind = pd.read_excel(
    r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\Sakakah 2021 weather dataset.xlsx",
    usecols=lambda col, c=pd.read_excel(
        r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\Sakakah 2021 weather dataset.xlsx", nrows=0
    ).columns: col in list(c)[-2:]
)

sp_DemandDef.drop(columns=["DATE-TIME"], inplace=True)
sp_SupplyDef.drop(columns=["Date & Time"], inplace=True)
sp_WeatherDe.drop(columns=["YEAR", "MO", "DY", "HR"], inplace=True)
#Thought maybe that the demand dataset was specifically to be tied with demand
#but I'm still getting bad results so can uncomment
# this to turn dataset back to just using 2021 weather DS
print(sp_WeatherDe.head())
sp_WeatherxDem = pd.merge(sp_DemandDef,sp_WeatherDe, on="TimeStamp", how="inner")
sp_WeatherxSup = pd.merge(sp_SupplyDef,sp_WeatherSu, on="TimeStamp", how="inner")

sp_WeatherxDem.replace(-999, np.nan, inplace=True)
sp_WeatherxSup.replace(-999, np.nan, inplace=True)
sp_WeatherxDem.interpolate(method='linear', inplace=True)
sp_WeatherxSup.interpolate(method='linear', inplace=True)
sp_WeatherxDem.dropna(inplace=True)
sp_WeatherxSup.dropna(inplace=True)
##linear interpolation to smooth out the datasets 
##So far has worked out great for supply but demand is getting bad results.
sp_WeatherxDem.interpolate(method='linear', inplace=True)
sp_WeatherxSup.interpolate(method='linear', inplace=True)
#print(sp_WeatherxDem.head())
#print("\n", sp_WeatherxSup.head())
#print(sp_WeatherxDem.head())
#print(sp_WeatherxSup.head())

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
def create_sequences_3d(data, targets, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(targets[i + seq_length])
    return np.array(X), np.array(y)

def create_sequences_2d(data, targets, seq_length):
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
def compute_shap_values_Trees_demand(model_type, best_tree_model, X_test, feature_cols, seq_length):
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

    return shap_importance
def compute_shap_values_Trees_supply(model_type, best_tree_model, X_test, feature_cols, seq_length):
    """
    Computes SHAP values for the given model type and visualizes feature importance for supply.
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

    # Extract SHAP values for supply
    shap_values_supply = shap_values_dict.get("Output_1", None)
    # Debugging: Print model type and SHAP values dictionary
    print(f"Model Type: {model_type}")
    print(f"SHAP Values Dictionary Keys: {shap_values_dict.keys()}")
    # Check if SHAP values for supply are computed
    if shap_values_supply is None:
        print("SHAP values for supply could not be computed. Please check the model and data.")
        return None

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

    # Plot SHAP summary for supply
    print("\nPlotting SHAP summary for Supply...")
    shap.summary_plot(shap_values_supply, X_test, plot_type="bar", feature_names=expanded_feature_cols)

    # Feature-to-lagged mapping for dependence plots
    feature_to_lagged_mapping = {
        feature: [f"{feature}_t-{i}" for i in range(seq_length, 0, -1)]
        for feature in feature_cols
    }


    # Check if SHAP values for supply are computed
    if shap_values_supply is None:
        print("SHAP values for supply could not be computed. Please check the model and data.")
        print("Available SHAP values outputs:", shap_values_dict.keys())
        return None
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
#----#

##print(sp_SupplyDef.head())
##print(sp_DemandDef.head())
##Note to self can't remember if saeed wanted me to do confusion martrixs aswell as the MSE,MAE,RMSE & R2
def decisionTreeModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int, plots:bool,features:bool):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy()

    """
    Decision Tree Results for Demand Dataset -

    --- 
    Decision Tree  Results for Supply Dataset -

    """
    if (ident == 1):
    #----#        

        demandDs.replace(-999,np.nan, inplace=True)
        demandDs.dropna(inplace=True)

        ###minLength = min(len(demandDs))demandDs = demandDs.iloc[minLength]

        #Extraction of Data.
  
        demandDs["hour"] = demandDs["TimeStamp"].dt.hour   
        demandDs["day"] = demandDs["TimeStamp"].dt.day        
        demandDs["month"] = demandDs["TimeStamp"].dt.month        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "PRECTOTCORR_D", "ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D", "RH2M_D", "PS_D", "PSC_D","WS10M_D","WD10M_D", "hour", "day", "month"
        ]

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        target_cols = ['MW']
        X = scaler_X.fit_transform(demandDs[feature_cols])
        y = scaler_y.fit_transform(demandDs[target_cols])

        seq_length = 1
        X_seq, y_seq = create_sequences_2d(X, y, seq_length)

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
        print('\n', "Decision Tree Model for Demand Data:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
        if plots == True:
            plt.figure(figsize=(10, 5))
            plt.plot(y_demand_true, label='Actual')
            plt.title(f'Decision Tree Model: Actual Demand')
            plt.xlabel('Time')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()

            def plot_series(y_true, y_pred, label):
                plt.figure(figsize=(10, 5))
                plt.plot(y_true, label='Actual')
                plt.plot(y_pred, label='Predicted')
                plt.title(f'Decision Tree Model: Actual vs Predicted {label}')
                plt.xlabel('Time')
                plt.ylabel('MW')
                plt.legend()
                plt.tight_layout()
                plt.show()

            plot_series(y_demand_true, y_demand_pred, 'Demand')
        if features == True:
            shap_importance = compute_shap_values_Trees_demand(
                model_type="MultiOutputRegressor",
                best_tree_model=grid_search.best_estimator_,
                X_test=X_test,
                feature_cols=feature_cols,
                seq_length=seq_length
        )
            return results       
        return results
    elif (ident == 2):

        supplyDs.replace(-999,np.nan, inplace=True)
        supplyDs.dropna(inplace=True)
        supplyDs["hour"] = supplyDs["TimeStamp"].dt.hour   
        supplyDs["day"] = supplyDs["TimeStamp"].dt.day        
        supplyDs["month"] = supplyDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_S", "ALLSKY_SFC_UV_INDEX_S", "T2M_S", "PRECTOTCORR_S", "ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S", "RH2M_S", "PS_S", "PSC_S","WS10M_S","WD10M_S","hour","day","month"
        ]

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        target_cols = ['MW']
        X = scaler_X.fit_transform(supplyDs[feature_cols])
        y = scaler_y.fit_transform(supplyDs[target_cols])

        seq_length = 1
        X_seq, y_seq = create_sequences_2d(X, y, seq_length)

        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=0.2, shuffle=False
        )
        #X_seq, y_seq = create_seasonal_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
        #X_seq_flat = X_seq.reshape(X_seq.shape[0], -1)


        """
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        target_cols = ['MW','MW']

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
        modelName = "Decision Tree"
        results = [mse, mae, rmse, r2,modelName]
        print('\n', "Decison Tree Model for Supply Data:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
        if plots == True:
            plt.figure(figsize=(10, 5))
            plt.plot(y_demand_true, label='Actual')
            plt.title(f'Decision Tree Model: Actual Supply')
            plt.xlabel('Time')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()


            def plot_series(y_true, y_pred, label):
                plt.figure(figsize=(10, 5))
                plt.plot(y_true, label='Actual')
                plt.plot(y_pred, label='Predicted')
                plt.title(f'Decision Tree Model: Actual vs Predicted {label}')
                plt.xlabel('Time')
                plt.ylabel('MW')
                plt.legend()
                plt.tight_layout()
                plt.show()

            plot_series(y_demand_true, y_demand_pred, 'Supply')
        if features == True:
            shap_importance = compute_shap_values_Trees_supply(
                model_type="MultiOutputRegressor",
                best_tree_model=grid_search.best_estimator_,
                X_test=X_test,
                feature_cols=feature_cols,
                seq_length=seq_length
        )
            return results   
        return results
    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return
  

def randomForestModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int, plots:bool,features:bool):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy()     
    """
    Demand Random Forest Results -

    -----
    Supply Random Forest Results - 

    """
    if (ident == 1):
    #----#        
        demandDs.replace(-999,np.nan, inplace=True)
        demandDs.dropna(inplace=True)

        ###minLength = min(len(demandDs))
        ###demandDs = demandDs.iloc[minLength]

        #Extraction of Data.
  
        demandDs["hour"] = demandDs["TimeStamp"].dt.hour   
        demandDs["day"] = demandDs["TimeStamp"].dt.day        
        demandDs["month"] = demandDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "PRECTOTCORR_D", "ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D", "RH2M_D", "PS_D", "PSC_D","WS10M_D","WD10M_D", "hour", "day", "month"
        ]
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        target_cols = ['MW']
        X = scaler_X.fit_transform(demandDs[feature_cols])
        y = scaler_y.fit_transform(demandDs[target_cols])

        seq_length = 1
        X_seq, y_seq = create_sequences_2d(X, y, seq_length)

        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=0.2, shuffle=False
        )
        #X_seq, y_seq = create_seasonal_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
        #X_seq_flat = X_seq.reshape(X_seq.shape[0], -1)

        """
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        target_cols = ['MW','MW']

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
        print('\n', "Random Forest Model for Demand Data:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
        if plots == True:
            plt.figure(figsize=(10, 5))
            plt.plot(y_demand_true, label='Actual')
            plt.title(f'Random Forest Model: Actual Demand')
            plt.xlabel('Time')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()


            def plot_series(y_true, y_pred, label):
                 plt.figure(figsize=(10, 5))
                 plt.plot(y_true, label='Actual')
                 plt.plot(y_pred, label='Predicted')
                 plt.title(f'Random Forest Model: Actual vs Predicted {label}')
                 plt.xlabel('Time')
                 plt.ylabel('MW')
                 plt.legend()
                 plt.tight_layout()
                 plt.show()

            plot_series(y_demand_true, y_demand_pred, 'Demand')
        if features == True:
            shap_importance = compute_shap_values_Trees_demand(
                model_type="Random Forest",
                best_tree_model=grid_search.best_estimator_,
                X_test=X_test,
                feature_cols=feature_cols,
                seq_length=seq_length
        )
            return results 
        return results
    elif (ident == 2):
        supplyDs.replace(-999,np.nan, inplace=True)
        supplyDs.dropna(inplace=True)
        supplyDs["hour"] = supplyDs["TimeStamp"].dt.hour   
        supplyDs["day"] = supplyDs["TimeStamp"].dt.day        
        supplyDs["month"] = supplyDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_S", "ALLSKY_SFC_UV_INDEX_S", "T2M_S", "PRECTOTCORR_S", "ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S", "RH2M_S", "PS_S", "PSC_S","WS10M_S","WD10M_S","hour","day","month"
        ]
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        target_cols = ['MW']
        X = scaler_X.fit_transform(supplyDs[feature_cols])
        y = scaler_y.fit_transform(supplyDs[target_cols])

        seq_length = 1
        X_seq, y_seq = create_sequences_2d(X, y, seq_length)

        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=0.2, shuffle=False
        )
        #X_seq, y_seq = create_seasonal_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
        #X_seq_flat = X_seq.reshape(X_seq.shape[0], -1)

        """
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        target_cols = ['MW','MW']

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
        print('\n', "Random Forest Model for Supply Data:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
        if plots == True:
            plt.figure(figsize=(10, 5))
            plt.plot(y_demand_true, label='Actual')
            plt.title(f'Random Forest Model: Actual Supply')
            plt.xlabel('Time')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()


            def plot_series(y_true, y_pred, label):
                plt.figure(figsize=(10, 5))
                plt.plot(y_true, label='Actual')
                plt.plot(y_pred, label='Predicted')
                plt.title(f'Random Forest Model: Actual vs Predicted {label}')
                plt.xlabel('Time')
                plt.ylabel('MW')
                plt.legend()
                plt.tight_layout()
                plt.show()

            plot_series(y_demand_true, y_demand_pred, 'Supply')
        if features == True:
            shap_importance = compute_shap_values_Trees_supply(
                model_type="Random Forest",
                best_tree_model=grid_search.best_estimator_,
                X_test=X_test,
                feature_cols=feature_cols,
                seq_length=seq_length
        )
            return results     
        return results
    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return


def xgbModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int, plots:bool,features:bool):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy()
    """
    Demand XGB Results -

    -----
    Supply XGB Results - 

    """
    if (ident == 1):
    #----#        
        demandDs.replace(-999,np.nan, inplace=True)
        demandDs.dropna(inplace=True)

        ###minLength = min(len(demandDs))
        ###demandDs = demandDs.iloc[minLength]

        #Extraction of Data.
  
        demandDs["hour"] = demandDs["TimeStamp"].dt.hour   
        demandDs["day"] = demandDs["TimeStamp"].dt.day        
        demandDs["month"] = demandDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "PRECTOTCORR_D", "ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D", "RH2M_D", "PS_D", "PSC_D","WS10M_D","WD10M_D", "hour", "day", "month"
        ]
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        target_cols = ['MW',]
        X = scaler_X.fit_transform(demandDs[feature_cols])
        y = scaler_y.fit_transform(demandDs[target_cols])

        seq_length = 1
        X_seq, y_seq = create_sequences_2d(X, y, seq_length)


        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=0.2, shuffle=False
        )
        #X_seq, y_seq = create_seasonal_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
        #X_seq_flat = X_seq.reshape(X_seq.shape[0], -1)

        """
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        target_cols = ['MW','MW']

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
        print('\n',"Demand XGB Model Results:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
        if plots == True:
            plt.figure(figsize=(10, 5))
            plt.plot(y_demand_true, label='Actual')
            plt.title(f'XGB Model: Actual Demand')
            plt.xlabel('Time')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()


            def plot_series(y_true, y_pred, label):
                plt.figure(figsize=(10, 5))
                plt.plot(y_true, label='Actual')
                plt.plot(y_pred, label='Predicted')
                plt.title(f'XGB Model: Actual vs Predicted {label}')
                plt.xlabel('Time')
                plt.ylabel('MW')
                plt.legend()
                plt.tight_layout()
                plt.show()

            plot_series(y_demand_true, y_demand_pred, 'Demand')
        if features == True:
            shap_importance = compute_shap_values_Trees_demand(
            model_type="MultiOutputRegressor",
            best_tree_model=grid_search.best_estimator_,
            X_test=X_test,
            feature_cols=feature_cols,
            seq_length=seq_length
        )
        return results  

        return results
    elif (ident == 2):
       
        supplyDs.replace(-999,np.nan, inplace=True)
        supplyDs.dropna(inplace=True)
        supplyDs["hour"] = supplyDs["TimeStamp"].dt.hour   
        supplyDs["day"] = supplyDs["TimeStamp"].dt.day        
        supplyDs["month"] = supplyDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_S", "ALLSKY_SFC_UV_INDEX_S", "T2M_S", "PRECTOTCORR_S", "ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S", "RH2M_S", "PS_S", "PSC_S","WS10M_S","WD10M_S","hour","day","month"
        ]
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        target_cols = ['MW']
        X = scaler_X.fit_transform(supplyDs[feature_cols])
        y = scaler_y.fit_transform(supplyDs[target_cols])

        seq_length = 1
        X_seq, y_seq = create_sequences_2d(X, y, seq_length)


        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=0.2, shuffle=False
        )
        #X_seq, y_seq = create_seasonal_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
        #X_seq_flat = X_seq.reshape(X_seq.shape[0], -1)

        """
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        target_cols = ['MW','MW']

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
        print('\n', "XGB Model for Supply Data:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
        if plots == True:
            plt.figure(figsize=(10, 5))
            plt.plot(y_demand_true, label='Actual')
            plt.title(f'XGB Model: Actual Supply')
            plt.xlabel('Time')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()


            def plot_series(y_true, y_pred, label):
                plt.figure(figsize=(10, 5))
                plt.plot(y_true, label='Actual')
                plt.plot(y_pred, label='Predicted')
                plt.title(f'XGB Model: Actual vs Predicted {label}')
                plt.xlabel('Time')
                plt.ylabel('MW')
                plt.legend()
                plt.tight_layout()
                plt.show()

            plot_series(y_demand_true, y_demand_pred, 'Supply')
        if features == True:
            shap_importance = compute_shap_values_Trees_supply(
            model_type="MultiOutputRegressor",
            best_tree_model=grid_search.best_estimator_,
            X_test=X_test,
            feature_cols=feature_cols,
            seq_length=seq_length
        )
            return results          
        return results
    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return

def biDirectionalLSTMDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int, plots:bool,features:bool):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy()

   
    """
    Demand Bidirectional LSTM Results -

    -----
    Supply Bidirectional LSTM Results - 

    """
    if (ident == 1):
    #----#        
    ##Basic transformation for the base dataset
        demandDs["hour"] = demandDs["TimeStamp"].dt.hour   
        demandDs["day"] = demandDs["TimeStamp"].dt.day        
        demandDs["month"] = demandDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "PRECTOTCORR_D", "ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D", "RH2M_D", "PS_D", "PSC_D","WS10M_D","WD10M_D", "hour", "day", "month"
        ]
        demandx = demandDs[feature_cols].values
        demandy = demandDs['MW'].values.reshape(-1, 1)
        
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(demandx)
        y_scaled = scaler_y.fit_transform(demandy)
        

        seq_length = 1
        X_seq, y_seq = create_sequences_3d(X_scaled, y_scaled, seq_length)


        X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, shuffle=False)

        def build_blstm_model(hp):
            model = Sequential()
            model.add(Bidirectional(LSTM(
                units=hp.Int('units1', 32, 128, step=32),
                return_sequences=True
            ), input_shape=(X_train.shape[1], X_train.shape[2])))
            model.add(Dropout(hp.Float('dropout1', 0.1, 0.5, step=0.1)))
            model.add(Bidirectional(LSTM(
                units=hp.Int('units2', 32, 128, step=32),
                return_sequences=False
            )))
            model.add(Dense(hp.Int('dense_units', 32, 128, step=32), activation='relu'))
            model.add(Dense(1))
            model.compile(
                optimizer=Adam(learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                loss='mse'
            )
            return model

        tuner = kt.RandomSearch(
            build_blstm_model,
            objective='val_loss',
            max_trials=5,
            executions_per_trial=1,
            directory='blstm_tuning',
            project_name='blstm'
        )
        tuner.search(X_train, y_train, epochs=20, validation_data=(X_test, y_test), verbose=0)
        best_model = tuner.get_best_models(num_models=1)[0]
        y_pred = best_model.predict(X_test)

        print("y_pred shape:", y_pred.shape)
        print("Any NaN in y_pred:", np.isnan(y_pred).any())
        print("Any Inf in y_pred:", np.isinf(y_pred).any())

        y_demand_pred = scaler_y.inverse_transform(y_pred)
        y_demand_true = scaler_y.inverse_transform(y_test)


        mse = mean_squared_error(y_demand_true, y_demand_pred)
        mae = mean_absolute_error(y_demand_true, y_demand_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_demand_true, y_demand_pred)
        modelName = "BLSTM"
        results = [mse, mae, rmse, r2, modelName]

        #----#
        print('\n',"Demand BLSTM Model Results:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
        if plots == True:
            plt.figure(figsize=(10, 5))
            plt.plot(y_demand_true, label='Actual')
            plt.title(f'BLSTM Model: Actual Demand')
            plt.xlabel('Time')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()


            def plot_series(y_true, y_pred, label):
                plt.figure(figsize=(10, 5))
                plt.plot(y_true, label='Actual')
                plt.plot(y_pred, label='Predicted')
                plt.title(f'BLSTM Model: Actual vs Predicted {label}')
                plt.xlabel('Time')
                plt.ylabel('MW')
                plt.legend()
                plt.tight_layout()
                plt.show()

            plot_series(y_demand_true, y_demand_pred, 'Demand')
        if features == True:
                shap_values_dict = {}
                print("Computing SHAP values for Bidirectional LSTM model...")
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
                #shap_values_supply = shap_values_flat[:, :, 0] 

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
               # assert shap_values_supply.shape[1] == X_test_flat.shape[1], "Mismatch between shap_values and X_test!"    
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
                #shap.summary_plot(shap_values_supply, X_test_flat, plot_type="bar", feature_names=expanded_feature_cols)

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
    elif (ident == 2):

        supplyDs["hour"] = supplyDs["TimeStamp"].dt.hour   
        supplyDs["day"] = supplyDs["TimeStamp"].dt.day        
        supplyDs["month"] = supplyDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_S", "ALLSKY_SFC_UV_INDEX_S", "T2M_S", "PRECTOTCORR_S", "ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S", "RH2M_S", "PS_S", "PSC_S","WS10M_S","WD10M_S","hour","day","month"
        ]
        supplyx = supplyDs[feature_cols].values
        supplyy = supplyDs['MW'].values.reshape(-1, 1)
        
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(supplyx)
        y_scaled = scaler_y.fit_transform(supplyy)
        

        seq_length = 1
        X_seq, y_seq = create_sequences_3d(X_scaled, y_scaled, seq_length)

        X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, shuffle=False)

        
        def build_blstm_model(hp):
            model = Sequential()
            model.add(Bidirectional(LSTM(
                units=hp.Int('units1', 32, 128, step=32),
                return_sequences=True
            ), input_shape=(X_train.shape[1], X_train.shape[2])))
            model.add(Dropout(hp.Float('dropout1', 0.1, 0.5, step=0.1)))
            model.add(Bidirectional(LSTM(
                units=hp.Int('units2', 32, 128, step=32),
                return_sequences=False
            )))
            model.add(Dense(hp.Int('dense_units', 32, 128, step=32), activation='relu'))
            model.add(Dense(1))
            model.compile(
                optimizer=Adam(learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                loss='mse'
            )
            return model

        tuner = kt.RandomSearch(
            build_blstm_model,
            objective='val_loss',
            max_trials=5,
            executions_per_trial=1,
            directory='blstm_tuning',
            project_name='blstm'
        )
        tuner.search(X_train, y_train, epochs=20, validation_data=(X_test, y_test), verbose=0)
        best_model = tuner.get_best_models(num_models=1)[0]
        y_pred = best_model.predict(X_test)

        print("y_pred shape:", y_pred.shape)
        print("Any NaN in y_pred:", np.isnan(y_pred).any())
        print("Any Inf in y_pred:", np.isinf(y_pred).any())

        y_demand_pred = scaler_y.inverse_transform(y_pred)
        y_demand_true = scaler_y.inverse_transform(y_test)


        mse = mean_squared_error(y_demand_true, y_demand_pred)
        mae = mean_absolute_error(y_demand_true, y_demand_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_demand_true, y_demand_pred)
        modelName = "BLSTM"
        results = [mse, mae, rmse, r2, modelName]

        #----#
        print('\n', "BLSTM Model for Supply Data:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
        if plots == True:
            plt.figure(figsize=(10, 5))
            plt.plot(y_demand_true, label='Actual')
            plt.title(f'BLSTM Model: Actual Supply')
            plt.xlabel('Time')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()


            def plot_series(y_true, y_pred, label):
                plt.figure(figsize=(10, 5))
                plt.plot(y_true, label='Actual')
                plt.plot(y_pred, label='Predicted')
                plt.title(f'BLSTM Model: Actual vs Predicted {label}')
                plt.xlabel('Time')
                plt.ylabel('MW')
                plt.legend()
                plt.tight_layout()
                plt.show()
            
            plot_series(y_demand_true, y_demand_pred, 'Supply')
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

                ##Select the SHAP values for the second output (e.g., Supply)
                shap_values_supply = shap_values_flat[:, :, 0] 

                # Prepare SHAP importance with adjusted SHAP values
                shap_importance = {
                    f"Output_{i + 1}": sorted(
                        zip(expanded_feature_cols, np.abs(shap_values_dict[f"Output_{i + 1}"][:, :-1]).mean(axis=0).tolist()),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    for i in range(len(shap_values_dict))
            }

                assert shap_values_supply.shape[1] == X_test_flat.shape[1], "Mismatch between shap_values and X_test!"    
                #assert shap_values_demand.shape[1] == X_test_flat.shape[1], "Mismatch between shap_values and X_test!"
                assert len(expanded_feature_cols) == X_test_flat.shape[1], "Mismatch between expanded_feature_cols and X_test!"
                mean_shap_values = np.array(mean_shap_values).flatten()
                # Sort features by importance in descending order
                sorted_features = sorted(zip(feature_cols, mean_shap_values), key=lambda x: x[1], reverse=True)

                # Print each feature and its importance
                for feature, importance in sorted_features:
                    print("{:<30} {:>12.4f}".format(feature, importance))
                # Plot SHAP summary for the first output
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
                            lagged_feature, shap_values_supply, X_test_flat, feature_names=expanded_feature_cols
                        )
                    else:
                        print(f"Time-lagged feature {lagged_feature} not found in expanded_feature_cols.")
                else:
                    print(f"Feature {feature} not found in feature_to_lagged_mapping.")       
                
                return results  
        return results
    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return

def LSTMModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int, plots:bool,features:bool):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy()

   
    """
    Demand LSTM Results -

    -----
    Supply LSTM Results - 

    """
    if (ident == 1):
    #----#        
    ##Basic transformation for the base dataset
        demandDs["hour"] = demandDs["TimeStamp"].dt.hour   
        demandDs["day"] = demandDs["TimeStamp"].dt.day        
        demandDs["month"] = demandDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "PRECTOTCORR_D", "ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D", "RH2M_D", "PS_D", "PSC_D","WS10M_D","WD10M_D", "hour", "day", "month"
        ]
        demandx = demandDs[feature_cols].values
        demandy = demandDs['MW'].values.reshape(-1, 1)
        
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(demandx)
        y_scaled = scaler_y.fit_transform(demandy)
        

        seq_length = 1
        X_seq, y_seq = create_sequences_3d(X_scaled, y_scaled, seq_length)

        X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, shuffle=False)

        def build_lstm_model(hp):
            model = Sequential()
            model.add(LSTM(
                units=hp.Int('units1', 32, 128, step=32),
                return_sequences=True,
                input_shape=(X_train.shape[1], X_train.shape[2])
            ))
            model.add(Dropout(hp.Float('dropout1', 0.1, 0.5, step=0.1)))
            model.add(LSTM(
                units=hp.Int('units2', 32, 128, step=32),
                return_sequences=False
            ))
            model.add(Dense(hp.Int('dense_units', 32, 128, step=32)))
            model.add(Dense(1))
            model.compile(
                optimizer=Adam(learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                loss='mse'
            )
            return model

        tuner = kt.RandomSearch(
            build_lstm_model,
            objective='val_loss',
            max_trials=5,
            executions_per_trial=1,
            directory='lstm_tuning',
            project_name='lstm'
        )
        tuner.search(X_train, y_train, epochs=20, validation_data=(X_test, y_test), verbose=0)
        best_model = tuner.get_best_models(num_models=1)[0]
        y_pred = best_model.predict(X_test)

        y_demand_pred = scaler_y.inverse_transform(y_pred)
        y_demand_true = scaler_y.inverse_transform(y_test)


        mse = mean_squared_error(y_demand_true, y_demand_pred)
        mae = mean_absolute_error(y_demand_true, y_demand_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_demand_true, y_demand_pred)
        modelName = "LSTM"
        results = [mse, mae, rmse, r2, modelName]
        
        #----#
        print('\n',"Demand LSTM Model Results:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
        if plots == True:
            plt.figure(figsize=(10, 5))
            plt.plot(y_demand_true, label='Actual')
            plt.title(f'LSTM Model: Actual Demand')
            plt.xlabel('Time')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()


            def plot_series(y_true, y_pred, label):
                plt.figure(figsize=(10, 5))
                plt.plot(y_true, label='Actual')
                plt.plot(y_pred, label='Predicted')
                plt.title(f'LSTM Model: Actual vs Predicted {label}')
                plt.xlabel('Time')
                plt.ylabel('MW')
                plt.legend()
                plt.tight_layout()
                plt.show()

            plot_series(y_demand_true, y_demand_pred, 'Demand')
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
                #shap_values_supply = shap_values_flat[:, :, 0] 

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
                #assert shap_values_supply.shape[1] == X_test_flat.shape[1], "Mismatch between shap_values and X_test!"  
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
                #shap.summary_plot(shap_values_supply, X_test_flat, plot_type="bar", feature_names=expanded_feature_cols)

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
    elif (ident == 2):
        supplyDs["hour"] = supplyDs["TimeStamp"].dt.hour   
        supplyDs["day"] = supplyDs["TimeStamp"].dt.day        
        supplyDs["month"] = supplyDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_S", "ALLSKY_SFC_UV_INDEX_S", "T2M_S", "PRECTOTCORR_S", "ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S", "RH2M_S", "PS_S", "PSC_S","WS10M_S","WD10M_S","hour","day","month"
        ]
        supplyx = supplyDs[feature_cols].values
        supplyy = supplyDs['MW'].values.reshape(-1, 1)
        
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(supplyx)
        y_scaled = scaler_y.fit_transform(supplyy)
        

        seq_length = 1
        X_seq, y_seq = create_sequences_3d(X_scaled, y_scaled, seq_length)

        X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, shuffle=False)
        def build_lstm_model(hp):
            model = Sequential()
            model.add(LSTM(
                units=hp.Int('units1', 32, 128, step=32),
                return_sequences=True,
                input_shape=(X_train.shape[1], X_train.shape[2])
            ))
            model.add(Dropout(hp.Float('dropout1', 0.1, 0.5, step=0.1)))
            model.add(LSTM(
                units=hp.Int('units2', 32, 128, step=32),
                return_sequences=False
            ))
            model.add(Dense(hp.Int('dense_units', 32, 128, step=32)))
            model.add(Dense(1))
            model.compile(
                optimizer=Adam(learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                loss='mse'
            )
            return model

        tuner = kt.RandomSearch(
            build_lstm_model,
            objective='val_loss',
            max_trials=5,
            executions_per_trial=1,
            directory='lstm_tuning',
            project_name='lstm'
        )
        tuner.search(X_train, y_train, epochs=20, validation_data=(X_test, y_test), verbose=0)
        best_model = tuner.get_best_models(num_models=1)[0]
        y_pred = best_model.predict(X_test)

        y_demand_pred = scaler_y.inverse_transform(y_pred)
        y_demand_true = scaler_y.inverse_transform(y_test)


        mse = mean_squared_error(y_demand_true, y_demand_pred)
        mae = mean_absolute_error(y_demand_true, y_demand_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_demand_true, y_demand_pred)
        modelName = "LSTM"
        results = [mse, mae, rmse, r2, modelName]

        #----#
        print('\n', "LSTM Model for Supply Data:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
        if plots == True:
            plt.figure(figsize=(10, 5))
            plt.plot(y_demand_true, label='Actual')
            plt.title(f'LSTM Model: Actual Supply')
            plt.xlabel('Time')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()


            def plot_series(y_true, y_pred, label):
                plt.figure(figsize=(10, 5))
                plt.plot(y_true, label='Actual')
                plt.plot(y_pred, label='Predicted')
                plt.title(f'LSTM Model: Actual vs Predicted {label}')
                plt.xlabel('Time')
                plt.ylabel('MW')
                plt.legend()
                plt.tight_layout()
                plt.show()

            plot_series(y_demand_true, y_demand_pred, 'Supply')
        
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
                #shap_values_demand = shap_values_flat[:, :, 0]  # Shape: (2619, 600)
                ##Select the SHAP values for the second output (e.g., Supply)
                shap_values_supply = shap_values_flat[:, :, 0] 

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
                #assert shap_values_demand.shape[1] == X_test_flat.shape[1], "Mismatch between shap_values and X_test!"
                assert len(expanded_feature_cols) == X_test_flat.shape[1], "Mismatch between expanded_feature_cols and X_test!"
                mean_shap_values = np.array(mean_shap_values).flatten()
                # Sort features by importance in descending order
                sorted_features = sorted(zip(feature_cols, mean_shap_values), key=lambda x: x[1], reverse=True)

                # Print each feature and its importance
                for feature, importance in sorted_features:
                    print("{:<30} {:>12.4f}".format(feature, importance))
                # Plot SHAP summary for the first output
                #shap.summary_plot(shap_values_demand, X_test_flat, plot_type="bar", feature_names=expanded_feature_cols)
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
                            lagged_feature, shap_values_supply, X_test_flat, feature_names=expanded_feature_cols
                        )
                    else:
                        print(f"Time-lagged feature {lagged_feature} not found in expanded_feature_cols.")
                else:
                    print(f"Feature {feature} not found in feature_to_lagged_mapping.")       
                return results  
        return results
    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return

def GRUModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int, plots:bool,features:bool):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy()

   
    if (ident == 1):
    #----#        
    ##Basic transformation for the base dataset
        demandDs["hour"] = demandDs["TimeStamp"].dt.hour   
        demandDs["day"] = demandDs["TimeStamp"].dt.day        
        demandDs["month"] = demandDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "PRECTOTCORR_D", "ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D", "RH2M_D", "PS_D", "PSC_D","WS10M_D","WD10M_D", "hour", "day", "month"
        ]
        demandx = demandDs[feature_cols].values
        demandy = demandDs['MW'].values.reshape(-1, 1)
        
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(demandx)
        y_scaled = scaler_y.fit_transform(demandy)
        

        seq_length = 1
        X_seq, y_seq = create_sequences_3d(X_scaled, y_scaled, seq_length)

        X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, shuffle=False)
        def build_gru_model(hp):
            model = Sequential()
            model.add(GRU(
                units=hp.Int('units1', 32, 128, step=32),
                return_sequences=True,
                input_shape=(X_train.shape[1], X_train.shape[2])
            ))
            model.add(Dropout(hp.Float('dropout1', 0.1, 0.5, step=0.1)))
            model.add(GRU(
                units=hp.Int('units2', 32, 128, step=32),
                return_sequences=False
            ))
            model.add(Dense(hp.Int('dense_units', 32, 128, step=32), activation='relu'))
            model.add(Dense(1))
            model.compile(
                optimizer=Adam(learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                loss='mse'
            )
            return model

        tuner = kt.RandomSearch(
            build_gru_model,
            objective='val_loss',
            max_trials=5,
            executions_per_trial=1,
            directory='gru_tuning',
            project_name='gru'
        )
        tuner.search(X_train, y_train, epochs=20, validation_data=(X_test, y_test), verbose=0)
        best_model = tuner.get_best_models(num_models=1)[0]
        y_pred = best_model.predict(X_test)

        y_demand_pred = scaler_y.inverse_transform(y_pred)
        y_demand_true = scaler_y.inverse_transform(y_test)


        mse = mean_squared_error(y_demand_true, y_demand_pred)
        mae = mean_absolute_error(y_demand_true, y_demand_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_demand_true, y_demand_pred)
        modelName = "GRU"
        results = [mse, mae, rmse, r2, modelName]
        
        #----#
        print('\n',"Demand GRU Model Results:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
        if plots == True:
            plt.figure(figsize=(10, 5))
            plt.plot(y_demand_true, label='Actual')
            plt.title(f'GRU Model: Actual Demand')
            plt.xlabel('Time')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()


            def plot_series(y_true, y_pred, label):
                plt.figure(figsize=(10, 5))
                plt.plot(y_true, label='Actual')
                plt.plot(y_pred, label='Predicted')
                plt.title(f'GRU Model: Actual vs Predicted {label}')
                plt.xlabel('Time')
                plt.ylabel('MW')
                plt.legend()
                plt.tight_layout()
                plt.show()

            plot_series(y_demand_true, y_demand_pred, 'Demand')
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
                #shap_values_supply = shap_values_flat[:, :, 0] 
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
                # assert shap_values_supply.shape[1] == X_test_flat.shape[1], "Mismatch between shap_values and X_test!"
    
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
                # shap.summary_plot(shap_values_supply, X_test_flat, plot_type="bar", feature_names=expanded_feature_cols)
                
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
    elif (ident == 2):
        supplyDs["hour"] = supplyDs["TimeStamp"].dt.hour   
        supplyDs["day"] = supplyDs["TimeStamp"].dt.day        
        supplyDs["month"] = supplyDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_S", "ALLSKY_SFC_UV_INDEX_S", "T2M_S", "PRECTOTCORR_S", "ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S", "RH2M_S", "PS_S", "PSC_S","WS10M_S","WD10M_S","hour","day","month"
        ]
        supplyx = supplyDs[feature_cols].values
        supplyy = supplyDs['MW'].values.reshape(-1, 1)
        
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(supplyx)
        y_scaled = scaler_y.fit_transform(supplyy)
        

        seq_length = 1
        X_seq, y_seq = create_sequences_3d(X_scaled, y_scaled, seq_length)

        X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, shuffle=False)
        def build_gru_model(hp):
            model = Sequential()
            model.add(GRU(
                units=hp.Int('units1', 32, 128, step=32),
                return_sequences=True,
                input_shape=(X_train.shape[1], X_train.shape[2])
            ))
            model.add(Dropout(hp.Float('dropout1', 0.1, 0.5, step=0.1)))
            model.add(GRU(
                units=hp.Int('units2', 32, 128, step=32),
                return_sequences=False
            ))
            model.add(Dense(hp.Int('dense_units', 32, 128, step=32), activation='relu'))
            model.add(Dense(1))
            model.compile(
                optimizer=Adam(learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                loss='mse'
            )
            return model

        tuner = kt.RandomSearch(
            build_gru_model,
            objective='val_loss',
            max_trials=5,
            executions_per_trial=1,
            directory='gru_tuning',
            project_name='gru'
        )
        tuner.search(X_train, y_train, epochs=20, validation_data=(X_test, y_test), verbose=0)
        best_model = tuner.get_best_models(num_models=1)[0]
        y_pred = best_model.predict(X_test)

        y_demand_pred = scaler_y.inverse_transform(y_pred)
        y_demand_true = scaler_y.inverse_transform(y_test)


        mse = mean_squared_error(y_demand_true, y_demand_pred)
        mae = mean_absolute_error(y_demand_true, y_demand_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_demand_true, y_demand_pred)
        modelName = "GRU"
        results = [mse, mae, rmse, r2, modelName]

        #----#
        print('\n', "GRU Model for Supply Data:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
        if plots == True:
            plt.figure(figsize=(10, 5))
            plt.plot(y_demand_true, label='Actual')
            plt.title(f'GRU Model: Actual Supply')
            plt.xlabel('Time')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()


            def plot_series(y_true, y_pred, label):
                plt.figure(figsize=(10, 5))
                plt.plot(y_true, label='Actual')
                plt.plot(y_pred, label='Predicted')
                plt.title(f'GRU Model: Actual vs Predicted {label}')
                plt.xlabel('Time')
                plt.ylabel('MW')
                plt.legend()
                plt.tight_layout()
                plt.show()

            plot_series(y_demand_true, y_demand_pred, 'Supply')
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
                #shap_values_demand = shap_values_flat[:, :, 0]  # Shape: (2619, 600)

                ##Select the SHAP values for the second output (e.g., Supply)
                shap_values_supply = shap_values_flat[:, :, 0] 
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
    
                #assert shap_values_demand.shape[1] == X_test_flat.shape[1], "Mismatch between shap_values and X_test!"
                assert len(expanded_feature_cols) == X_test_flat.shape[1], "Mismatch between expanded_feature_cols and X_test!"
                mean_shap_values = np.array(mean_shap_values).flatten()
                # Sort features by importance in descending order
                sorted_features = sorted(zip(feature_cols, mean_shap_values), key=lambda x: x[1], reverse=True)

                # Print each feature and its importance
                for feature, importance in sorted_features:
                    print("{:<30} {:>12.4f}".format(feature, importance))
                # Plot SHAP summary for the first output
                # Plot SHAP summary for the first output
                #shap.summary_plot(shap_values_demand, X_test_flat, plot_type="bar", feature_names=expanded_feature_cols)
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
                            lagged_feature, shap_values_supply, X_test_flat, feature_names=expanded_feature_cols
                        )
                    else:
                        print(f"Time-lagged feature {lagged_feature} not found in expanded_feature_cols.")
                else:
                    print(f"Feature {feature} not found in feature_to_lagged_mapping.")       
                
                return results    
        return results
    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return

def SVRModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int, plots:bool,features:bool):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy()

   
    """
    Additonal Note the base results for SVR very bad to the point I thought they were broken I made a basic hyper param and cycled
    through features to see if I could get better results but will need clarification if what I've done is correct in anyway. Though have
    kept the base implementation for comparisons sake.
    
    WITH HYPERPARAMS RESULTS -
    Demand SVR Results -

    -----
    Supply GRU Results - 

        -----
    WITHOUT HYPER PARAMS
    Demand ---

    ----
    Supply ---

    
    """


    if (ident == 1):
    #----#        
    ##Basic transformation for the base dataset
        demandDs["hour"] = demandDs["TimeStamp"].dt.hour   
        demandDs["day"] = demandDs["TimeStamp"].dt.day        
        demandDs["month"] = demandDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "PRECTOTCORR_D", "ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D", "RH2M_D", "PS_D", "PSC_D","WS10M_D","WD10M_D", "hour", "day", "month"
        ]
        demandx = demandDs[feature_cols].values
        demandy = demandDs['MW'].values.reshape(-1, 1)
        
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(demandx)
        y_scaled = scaler_y.fit_transform(demandy)
        
        
        seq_length = 1
        X_seq, y_seq = create_sequences_2d(X_scaled, y_scaled, seq_length)
        X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, shuffle=False)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
        param_grid = {
        'estimator__C': [0.1, 1, 10, 100],           # Regularization, similar to tree depth/complexity
        'estimator__gamma': ['scale', 'auto', 0.01, 0.1, 1],  # Kernel coefficient, like learning rate
        'estimator__epsilon': [0.01, 0.1, 0.5, 1.0], # Insensitivity, similar to min_samples_split
        'estimator__kernel': ['rbf']                 # Keep kernel consistent
        }
        model = MultiOutputRegressor(SVR(kernel='rbf'))
       ## model = MultiOutputRegressor(SVR())
        
        ##grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        ##grid.fit(X_train, y_train)
        ##print("Best parameters:", grid.best_params_)

        model.fit(X_train,y_train)

        y_pred = model.predict(X_test)

        y_demand_pred = scaler_y.inverse_transform(y_pred)
        y_demand_true = scaler_y.inverse_transform(y_test)


        mse = mean_squared_error(y_demand_true, y_demand_pred)
        mae = mean_absolute_error(y_demand_true, y_demand_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_demand_true, y_demand_pred)
        modelName = "SVR"
        results = [mse, mae, rmse, r2, modelName]

        #----#
        print('\n',"Demand SVR Model Results:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
        if plots == True:
            plt.figure(figsize=(10, 5))
            plt.plot(y_demand_true, label='Actual')
            plt.title(f'SVR Model: Actual Demand')
            plt.xlabel('Time')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()


            def plot_series(y_true, y_pred, label):
                plt.figure(figsize=(10, 5))
                plt.plot(y_true, label='Actual')
                plt.plot(y_pred, label='Predicted')
                plt.title(f'SVR Model: Actual vs Predicted {label}')
                plt.xlabel('Time')
                plt.ylabel('MW')
                plt.legend()
                plt.tight_layout()
                plt.show()

            plot_series(y_demand_true, y_demand_pred, 'Demand')
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
            #shap_values_supply = shap_values_flat[:, :, 0] 
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
            #assert shap_values_supply.shape[1] == X_test_flat.shape[1], "Mismatch between shap_values and X_test!"

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
            #shap.summary_plot(shap_values_supply, X_test_flat, plot_type="bar", feature_names=expanded_feature_cols)

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
    elif (ident == 2):
        supplyDs["hour"] = supplyDs["TimeStamp"].dt.hour   
        supplyDs["day"] = supplyDs["TimeStamp"].dt.day        
        supplyDs["month"] = supplyDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_S", "ALLSKY_SFC_UV_INDEX_S", "T2M_S", "PRECTOTCORR_S", "ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S", "RH2M_S", "PS_S", "PSC_S","WS10M_S","WD10M_S","hour","day","month"
        ]
        supplyx = supplyDs[feature_cols].values
        supplyy = supplyDs['MW'].values.reshape(-1, 1)
        
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(supplyx)
        y_scaled = scaler_y.fit_transform(supplyy)
        
        
        seq_length = 1
        X_seq, y_seq = create_sequences_2d(X_scaled, y_scaled, seq_length)
        X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, shuffle=False)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

        param_grid = {
        'estimator__C': [0.1, 1, 10, 100],           # Regularization, similar to tree depth/complexity
        'estimator__gamma': ['scale', 'auto', 0.01, 0.1, 1],  # Kernel coefficient, like learning rate
        'estimator__epsilon': [0.01, 0.1, 0.5, 1.0], # Insensitivity, similar to min_samples_split
        'estimator__kernel': ['rbf']                 # Keep kernel consistent
        }
        ##model = MultiOutputRegressor(SVR(kernel='rbf'))
       ## model = MultiOutputRegressor(SVR())
        model = MultiOutputRegressor(SVR(kernel='rbf'))
       ## model = MultiOutputRegressor(SVR())
        
        ##grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        ##grid.fit(X_train, y_train)
        ##print("Best parameters:", grid.best_params_)

        model.fit(X_train,y_train)
        """
        grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid.fit(X_train, y_train)
        print('\n',"Best parameters:", grid.best_params_)

        grid.fit(X_train,y_train)
        """
        y_pred = model.predict(X_test)

        y_pred = model.predict(X_test)

        y_demand_pred = scaler_y.inverse_transform(y_pred)
        y_demand_true = scaler_y.inverse_transform(y_test)


        mse = mean_squared_error(y_demand_true, y_demand_pred)
        mae = mean_absolute_error(y_demand_true, y_demand_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_demand_true, y_demand_pred)
        modelName = "SVR"
        results = [mse, mae, rmse, r2, modelName]

        #----#
        print('\n', "SVR Model for Supply Data:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
        if plots == True:
            plt.figure(figsize=(10, 5))
            plt.plot(y_demand_true, label='Actual')
            plt.title(f'SVR Model: Actual Supply')
            plt.xlabel('Time')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()


            def plot_series(y_true, y_pred, label):
                plt.figure(figsize=(10, 5))
                plt.plot(y_true, label='Actual')
                plt.plot(y_pred, label='Predicted')
                plt.title(f'SVR Model: Actual vs Predicted {label}')
                plt.xlabel('Time')
                plt.ylabel('MW')
                plt.legend()
                plt.tight_layout()
                plt.show()

            plot_series(y_demand_true, y_demand_pred, 'Supply')
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
        #shap_values_demand = shap_values_flat[:, :len(expanded_feature_cols)]  # Adjust to match feature columns

        ##Select the SHAP values for the second output (e.g., Supply)
        shap_values_supply = shap_values_flat[:, :, 0] 
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

        #assert shap_values_demand.shape[1] == X_test_flat.shape[1], "Mismatch between shap_values and X_test!"
        assert len(expanded_feature_cols) == X_test_flat.shape[1], "Mismatch between expanded_feature_cols and X_test!"
        mean_shap_values = np.array(mean_shap_values).flatten()
        # Sort features by importance in descending order
        sorted_features = sorted(zip(feature_cols, mean_shap_values), key=lambda x: x[1], reverse=True)

        # Print each feature and its importance
        for feature, importance in sorted_features:
            print("{:<30} {:>12.4f}".format(feature, importance))
        # Plot SHAP summary for the first output
        #shap.summary_plot(shap_values_demand, X_test_flat, plot_type="bar", feature_names=expanded_feature_cols)
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

            return results       
        return results
    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return

def MLPModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int, plots:bool,features:bool):
        demandDs = demandDs.copy()
        supplyDs = supplyDs.copy()
        if (ident == 1):
            #----#        
            ##Basic transformation for the base dataset
                demandDs["hour"] = demandDs["TimeStamp"].dt.hour   
                demandDs["day"] = demandDs["TimeStamp"].dt.day        
                demandDs["month"] = demandDs["TimeStamp"].dt.month        
                feature_cols = [
                "ALLSKY_SFC_SW_DWN_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "PRECTOTCORR_D", "ALLSKY_KT_D",
                "CLRSKY_SFC_PAR_TOT_D", "RH2M_D", "PS_D", "PSC_D","WS10M_D","WD10M_D", "hour", "day", "month"
                ]
                demandx = demandDs[feature_cols].values
                demandy = demandDs['MW'].values.reshape(-1, 1)
        
                scaler_X = MinMaxScaler()
                scaler_y = MinMaxScaler()
                X_scaled = scaler_X.fit_transform(demandx)
                y_scaled = scaler_y.fit_transform(demandy)
        

                seq_length = 1
                X_seq, y_seq = create_sequences_2d(X_scaled, y_scaled, seq_length)

                X_train, X_test, y_train, y_test = train_test_split(
                X_seq, y_seq, test_size=0.2, shuffle=False)

        
                def build_mlp_model(hp):
                    model = Sequential()
                    model.add(Dense(
                        units=hp.Int('units1', 64, 256, step=64),
                        activation='relu',
                        input_dim=X_train.shape[1]
                    ))
                    model.add(Dropout(hp.Float('dropout1', 0.1, 0.5, step=0.1)))
                    model.add(Dense(
                        units=hp.Int('units2', 32, 128, step=32),
                        activation='relu'
                    ))
                    model.add(Dropout(hp.Float('dropout2', 0.1, 0.5, step=0.1)))
                    model.add(Dense(1))
                    model.compile(
                        optimizer=Adam(learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                        loss='mse'
                    )
                    return model

                tuner = kt.RandomSearch(
                    build_mlp_model,
                    objective='val_loss',
                    max_trials=5,
                    executions_per_trial=1,
                    directory='mlp_tuning',
                    project_name='mlp'
                )
                tuner.search(X_train, y_train, epochs=20, validation_data=(X_test, y_test), verbose=0)
                best_model = tuner.get_best_models(num_models=1)[0]
                y_pred = best_model.predict(X_test)

                y_demand_pred = scaler_y.inverse_transform(y_pred)
                y_demand_true = scaler_y.inverse_transform(y_test)


                mse = mean_squared_error(y_demand_true, y_demand_pred)
                mae = mean_absolute_error(y_demand_true, y_demand_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_demand_true, y_demand_pred)
                modelName = "MLP"
                results = [mse, mae, rmse, r2, modelName]

                #----#
                print('\n',"Demand MLP Model Results:")
                print("MSE: {:.4f}".format(mse))
                print("MAE: {:.4f}".format(mae))
                print("RMSE: {:.4f}".format(rmse))
                print("R2: {:.4f}".format(r2))
                if plots == True:
                    plt.figure(figsize=(10, 5))
                    plt.plot(y_demand_true, label='Actual')
                    plt.title(f'MLP Model: Actual Demand')
                    plt.xlabel('Time')
                    plt.ylabel('MW')
                    plt.legend()
                    plt.tight_layout()
                    plt.show()


                    def plot_series(y_true, y_pred, label):
                        plt.figure(figsize=(10, 5))
                        plt.plot(y_true, label='Actual')
                        plt.plot(y_pred, label='Predicted')
                        plt.title(f'MLP Model: Actual vs Predicted {label}')
                        plt.xlabel('Time')
                        plt.ylabel('MW')
                        plt.legend()
                        plt.tight_layout()
                        plt.show()

                    plot_series(y_demand_true, y_demand_pred, 'Demand')
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
                        #shap_values_supply = shap_values_flat[:, :, 0] 
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
                        #assert shap_values_supply.shape[1] == X_test_flat.shape[1], "Mismatch between shap_values and X_test!"
    
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
                        #shap.summary_plot(shap_values_supply, X_test_flat, plot_type="bar", feature_names=expanded_feature_cols)
                        
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
        elif (ident == 2):
                supplyDs["hour"] = supplyDs["TimeStamp"].dt.hour   
                supplyDs["day"] = supplyDs["TimeStamp"].dt.day        
                supplyDs["month"] = supplyDs["TimeStamp"].dt.month        
        
                feature_cols = [
                "ALLSKY_SFC_SW_DWN_S", "ALLSKY_SFC_UV_INDEX_S", "T2M_S", "PRECTOTCORR_S", "ALLSKY_KT_S",
                "CLRSKY_SFC_PAR_TOT_S", "RH2M_S", "PS_S", "PSC_S","WS10M_S","WD10M_S","hour","day","month"
                ]
                supplyx = supplyDs[feature_cols].values
                supplyy = supplyDs['MW'].values.reshape(-1, 1)
        
                scaler_X = MinMaxScaler()
                scaler_y = MinMaxScaler()
                X_scaled = scaler_X.fit_transform(supplyx)
                y_scaled = scaler_y.fit_transform(supplyy)
        

                seq_length = 1
                X_seq, y_seq = create_sequences_2d(X_scaled, y_scaled, seq_length)

                X_train, X_test, y_train, y_test = train_test_split(
                X_seq, y_seq, test_size=0.2, shuffle=False)
                def build_mlp_model(hp):
                    model = Sequential()
                    model.add(Dense(
                        units=hp.Int('units1', 64, 256, step=64),
                        activation='relu',
                        input_dim=X_train.shape[1]
                    ))
                    model.add(Dropout(hp.Float('dropout1', 0.1, 0.5, step=0.1)))
                    model.add(Dense(
                        units=hp.Int('units2', 32, 128, step=32),
                        activation='relu'
                    ))
                    model.add(Dropout(hp.Float('dropout2', 0.1, 0.5, step=0.1)))
                    model.add(Dense(1))
                    model.compile(
                        optimizer=Adam(learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                        loss='mse'
                    )
                    return model

                tuner = kt.RandomSearch(
                    build_mlp_model,
                    objective='val_loss',
                    max_trials=5,
                    executions_per_trial=1,
                    directory='mlp_tuning',
                    project_name='mlp'
                )
                tuner.search(X_train, y_train, epochs=20, validation_data=(X_test, y_test), verbose=0)
                best_model = tuner.get_best_models(num_models=1)[0]
                y_pred = best_model.predict(X_test)

                y_demand_pred = scaler_y.inverse_transform(y_pred)
                y_demand_true = scaler_y.inverse_transform(y_test)


                mse = mean_squared_error(y_demand_true, y_demand_pred)
                mae = mean_absolute_error(y_demand_true, y_demand_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_demand_true, y_demand_pred)
                modelName = "MLP"
                results = [mse, mae, rmse, r2, modelName]

                #----#
                print('\n', "MLP Model for Supply Data:")
                print("MSE: {:.4f}".format(mse))
                print("MAE: {:.4f}".format(mae))
                print("RMSE: {:.4f}".format(rmse))
                print("R2: {:.4f}".format(r2))
                if plots == True:
                    plt.figure(figsize=(10, 5))
                    plt.plot(y_demand_true, label='Actual')
                    plt.title(f'MLP Model: Actual Supply')
                    plt.xlabel('Time')
                    plt.ylabel('MW')
                    plt.legend()
                    plt.tight_layout()
                    plt.show()


                    def plot_series(y_true, y_pred, label):
                        plt.figure(figsize=(10, 5))
                        plt.plot(y_true, label='Actual')
                        plt.plot(y_pred, label='Predicted')
                        plt.title(f'MLP Model: Actual vs Predicted {label}')
                        plt.xlabel('Time')
                        plt.ylabel('MW')
                        plt.legend()
                        plt.tight_layout()
                        plt.show()

                    plot_series(y_demand_true, y_demand_pred, 'Supply')
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

                    ##Select the SHAP values for the second output (e.g., Supply)
                    shap_values_supply = shap_values_flat[:, :, 0] 
                    # Prepare SHAP importance with adjusted SHAP values
                    shap_importance = {
                        f"Output_{i + 1}": sorted(
                            zip(expanded_feature_cols, np.abs(shap_values_dict[f"Output_{i + 1}"][:, :-1]).mean(axis=0).tolist()),
                            key=lambda x: x[1],
                            reverse=True
                        )
                        for i in range(len(shap_values_dict))
                    }

                    assert shap_values_supply.shape[1] == X_test_flat.shape[1], "Mismatch between shap_values and X_test!"
    
                    assert len(expanded_feature_cols) == X_test_flat.shape[1], "Mismatch between expanded_feature_cols and X_test!"

    
                    mean_shap_values = np.array(mean_shap_values).flatten()
                    # Sort features by importance in descending order
                    sorted_features = sorted(zip(feature_cols, mean_shap_values), key=lambda x: x[1], reverse=True)

                    # Print each feature and its importance
                    for feature, importance in sorted_features:
                        print("{:<30} {:>12.4f}".format(feature, importance))
                    # Plot SHAP summary for the first output
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
                                lagged_feature, shap_values_supply, X_test_flat, feature_names=expanded_feature_cols
                            )
                        else:
                            print(f"Time-lagged feature {lagged_feature} not found in expanded_feature_cols.")
                    else:
                        print(f"Feature {feature} not found in feature_to_lagged_mapping.")
                        
                    return results  
                return results
        else:
                print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
                return

def CNNModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int, plots:bool,features:bool):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy()


    if ident == 1:
        #----#        
        demandDs["hour"] = demandDs["TimeStamp"].dt.hour   
        demandDs["day"] = demandDs["TimeStamp"].dt.day        
        demandDs["month"] = demandDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "PRECTOTCORR_D", "ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D", "RH2M_D", "PS_D", "PSC_D","WS10M_D","WD10M_D", "hour", "day", "month"
        ]
        demandx = demandDs[feature_cols].values
        demandy = demandDs['MW'].values.reshape(-1, 1)
        
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(demandx)
        y_scaled = scaler_y.fit_transform(demandy)
        

        seq_length = 1
        X_seq, y_seq = create_sequences_3d(X_scaled, y_scaled, seq_length)

        X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, shuffle=False)

        

        def build_cnn_model(hp):
            model = Sequential()
            model.add(Conv1D(
                filters=hp.Int('filters1', 32, 128, step=32),
                kernel_size=1,
                activation='relu',
                input_shape=(X_train.shape[1], X_train.shape[2])
            ))
            model.add(MaxPooling1D(pool_size=1))
            model.add(Dropout(hp.Float('dropout1', 0.1, 0.5, step=0.1)))
            model.add(Conv1D(
                filters=hp.Int('filters2', 32, 128, step=32),
                kernel_size=1,
                activation='relu'
            ))
            model.add(Dropout(hp.Float('dropout2', 0.1, 0.5, step=0.1)))
            model.add(Flatten())
            model.add(Dense(hp.Int('dense_units', 32, 128, step=32), activation='relu'))
            model.add(Dense(1))
            model.compile(
                optimizer=Adam(learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                loss='mse'
            )
            return model

        tuner = kt.RandomSearch(
            build_cnn_model,
            objective='val_loss',
            max_trials=5,
            executions_per_trial=1,
            directory='cnn_tuning',
            project_name='cnn'
        )
        tuner.search(X_train, y_train, epochs=20, validation_data=(X_test, y_test), verbose=0)
        best_model = tuner.get_best_models(num_models=1)[0]
        y_pred = best_model.predict(X_test)

        y_demand_pred = scaler_y.inverse_transform(y_pred)
        y_demand_true = scaler_y.inverse_transform(y_test)

        mse = mean_squared_error(y_demand_true, y_demand_pred)
        mae = mean_absolute_error(y_demand_true, y_demand_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_demand_true, y_demand_pred)
        modelName = "CNN"
        results = [mse, mae, rmse, r2, modelName]
        
        #---#
        print('\n', "Demand CNN Model Results:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
        if plots == True:
            plt.figure(figsize=(10, 5))
            plt.plot(y_demand_true, label='Actual')
            plt.title(f'CNN Model: Actual Demand')
            plt.xlabel('Time')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()


            def plot_series(y_true, y_pred, label):
                plt.figure(figsize=(10, 5))
                plt.plot(y_true, label='Actual')
                plt.plot(y_pred, label='Predicted')
                plt.title(f'CNN Model: Actual vs Predicted {label}')
                plt.xlabel('Time')
                plt.ylabel('MW')
                plt.legend()
                plt.tight_layout()
                plt.show()

            plot_series(y_demand_true, y_demand_pred, 'Demand')
        
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
           #shap_values_supply = shap_values_flat[:, :, 0] 
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
            #assert shap_values_supply.shape[1] == X_test_flat.shape[1], "Mismatch between shap_values and X_test!"
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
            #shap.summary_plot(shap_values_supply, X_test_flat, plot_type="bar", feature_names=expanded_feature_cols)

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

    elif ident == 2:
        supplyDs["hour"] = supplyDs["TimeStamp"].dt.hour   
        supplyDs["day"] = supplyDs["TimeStamp"].dt.day        
        supplyDs["month"] = supplyDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_S", "ALLSKY_SFC_UV_INDEX_S", "T2M_S", "PRECTOTCORR_S", "ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S", "RH2M_S", "PS_S", "PSC_S","WS10M_S","WD10M_S","hour","day","month"
        ]
        supplyx = supplyDs[feature_cols].values
        supplyy = supplyDs['MW'].values.reshape(-1, 1)
        
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(supplyx)
        y_scaled = scaler_y.fit_transform(supplyy)
        

        seq_length = 1
        X_seq, y_seq = create_sequences_3d(X_scaled, y_scaled, seq_length)

        X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, shuffle=False)

        def build_cnn_model(hp):
            model = Sequential()
            model.add(Conv1D(
                filters=hp.Int('filters1', 32, 128, step=32),
                kernel_size=1,
                activation='relu',
                input_shape=(X_train.shape[1], X_train.shape[2])
            ))
            model.add(MaxPooling1D(pool_size=1))
            model.add(Dropout(hp.Float('dropout1', 0.1, 0.5, step=0.1)))
            model.add(Conv1D(
                filters=hp.Int('filters2', 32, 128, step=32),
                kernel_size=1,
                activation='relu'
            ))
            model.add(Dropout(hp.Float('dropout2', 0.1, 0.5, step=0.1)))
            model.add(Flatten())
            model.add(Dense(hp.Int('dense_units', 32, 128, step=32), activation='relu'))
            model.add(Dense(1))
            model.compile(
                optimizer=Adam(learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                loss='mse'
            )
            return model

        tuner = kt.RandomSearch(
            build_cnn_model,
            objective='val_loss',
            max_trials=5,
            executions_per_trial=1,
            directory='cnn_tuning',
            project_name='cnn'
        )
        tuner.search(X_train, y_train, epochs=20, validation_data=(X_test, y_test), verbose=0)
        best_model = tuner.get_best_models(num_models=1)[0]
        y_pred = best_model.predict(X_test)

        y_supply_pred = scaler_y.inverse_transform(y_pred)
        y_supply_true = scaler_y.inverse_transform(y_test)

        mse = mean_squared_error(y_supply_true, y_supply_pred)
        mae = mean_absolute_error(y_supply_true, y_supply_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_supply_true, y_supply_pred)
        modelName = "CNN"
        results = [mse, mae, rmse, r2, modelName]
        if plots == True:
            plt.figure(figsize=(10, 5))
            plt.plot(y_supply_true, label='Actual')
            plt.title(f'CNN Model: Actual Supply')
            plt.xlabel('Time')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()


            def plot_series(y_true, y_pred, label):
                plt.figure(figsize=(10, 5))
                plt.plot(y_true, label='Actual')
                plt.plot(y_pred, label='Predicted')
                plt.title(f'CNN Model: Actual vs Predicted {label}')
                plt.xlabel('Time')
                plt.ylabel('MW')
                plt.legend()
                plt.tight_layout()
                plt.show()

            plot_series(y_supply_true, y_supply_pred, 'Supply')
        
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
            #shap_values_demand = shap_values_flat[:, :, 0]  # Shape: (2619, 600)
            ##Select the SHAP values for the second output (e.g., Supply)
            shap_values_supply = shap_values_flat[:, :, 0] 
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
            #assert shap_values_demand.shape[1] == X_test_flat.shape[1], "Mismatch between shap_values and X_test!"
            assert len(expanded_feature_cols) == X_test_flat.shape[1], "Mismatch between expanded_feature_cols and X_test!"

            mean_shap_values = np.array(mean_shap_values).flatten()
            # Sort features by importance in descending order
            sorted_features = sorted(zip(feature_cols, mean_shap_values), key=lambda x: x[1], reverse=True)

            # Print each feature and its importance
            for feature, importance in sorted_features:
                print("{:<30} {:>12.4f}".format(feature, importance))
            # Plot SHAP summary for the first output
            #shap.summary_plot(shap_values_demand, X_test_flat, plot_type="bar", feature_names=expanded_feature_cols)
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
                        lagged_feature, shap_values_supply, X_test_flat, feature_names=expanded_feature_cols
                    )
                else:
                    print(f"Time-lagged feature {lagged_feature} not found in expanded_feature_cols.")
            else:
                print(f"Feature {feature} not found in feature_to_lagged_mapping.")       
                
            return results        
        return results

    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return
    

def GBDTModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int, plots:bool,features:bool):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy()
  
    """
    Demand GradientBoosting Results -

    -----
    Supply GradientBoosting Results - 

    """
    if (ident == 1):
    #----#        
        demandDs["hour"] = demandDs["TimeStamp"].dt.hour   
        demandDs["day"] = demandDs["TimeStamp"].dt.day        
        demandDs["month"] = demandDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "PRECTOTCORR_D", "ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D", "RH2M_D", "PS_D", "PSC_D","WS10M_D","WD10M_D", "hour", "day", "month"
        ]
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        target_cols = ['MW']
        X = scaler_X.fit_transform(demandDs[feature_cols])
        y = scaler_y.fit_transform(demandDs[target_cols])

        seq_length = 1
        X_seq, y_seq = create_sequences_2d(X, y, seq_length)


        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=0.2, shuffle=False
        )
        #X_seq, y_seq = create_seasonal_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
        #X_seq_flat = X_seq.reshape(X_seq.shape[0], -1)

        #X_seq, y_seq = create_seasonal_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
        #X_seq_flat = X_seq.reshape(X_seq.shape[0], -1)

        """
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        target_cols = ['MW','MW']

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
        print('\n', "GB Model for Demand Data:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
        if plots == True:
            plt.figure(figsize=(10, 5))
            plt.plot(y_demand_true, label='Actual')
            plt.title(f'Gradient Boosting Model: Actual Demand')
            plt.xlabel('Time')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()


            def plot_series(y_true, y_pred, label):
                plt.figure(figsize=(10, 5))
                plt.plot(y_true, label='Actual')
                plt.plot(y_pred, label='Predicted')
                plt.title(f'Gradient Boosting Model: Actual vs Predicted {label}')
                plt.xlabel('Time')
                plt.ylabel('MW')
                plt.legend()
                plt.tight_layout()
                plt.show()

            plot_series(y_demand_true, y_demand_pred, 'Demand')
        if features == True:
            shap_importance = compute_shap_values_Trees_demand(
                model_type="GBDT",
                best_tree_model=grid_search.best_estimator_,
                X_test=X_test,
                feature_cols=feature_cols,
                seq_length=seq_length
            )
            return results  
        return results
    elif (ident == 2):

        supplyDs.replace(-999,np.nan, inplace=True)
        supplyDs.dropna(inplace=True)
        supplyDs["hour"] = supplyDs["TimeStamp"].dt.hour   
        supplyDs["day"] = supplyDs["TimeStamp"].dt.day        
        supplyDs["month"] = supplyDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_S", "ALLSKY_SFC_UV_INDEX_S", "T2M_S", "PRECTOTCORR_S", "ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S", "RH2M_S", "PS_S", "PSC_S","WS10M_S","WD10M_S","hour","day","month"
        ]
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        target_cols = ['MW']
        X = scaler_X.fit_transform(supplyDs[feature_cols])
        y = scaler_y.fit_transform(supplyDs[target_cols])

        seq_length = 1
        X_seq, y_seq = create_sequences_2d(X, y, seq_length)


        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=0.2, shuffle=False
        )
        #X_seq, y_seq = create_seasonal_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
        #X_seq_flat = X_seq.reshape(X_seq.shape[0], -1)

        #X_seq, y_seq = create_seasonal_sequences(mergedDs, feature_cols, target_cols, pad_to_max=True)
        #X_seq_flat = X_seq.reshape(X_seq.shape[0], -1)

        """
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        target_cols = ['MW','MW']

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
        print('\n', "GB Model for Supply Data:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
        if plots == True:
            plt.figure(figsize=(10, 5))
            plt.plot(y_demand_true, label='Actual')
            plt.title(f'Gradient Boosting Model: Actual Supply')
            plt.xlabel('Time')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()


            def plot_series(y_true, y_pred, label):
                plt.figure(figsize=(10, 5))
                plt.plot(y_true, label='Actual')
                plt.plot(y_pred, label='Predicted')
                plt.title(f'Gradient Boosting Model: Actual vs Predicted {label}')
                plt.xlabel('Time')
                plt.ylabel('MW')
                plt.legend()
                plt.tight_layout()
                plt.show()

            plot_series(y_demand_true, y_demand_pred, 'Supply')
        if features == True:
            shap_importance = compute_shap_values_Trees_supply(
                model_type="GBDT",
                best_tree_model=grid_search.best_estimator_,
                X_test=X_test,
                feature_cols=feature_cols,
                seq_length=seq_length
            )
            return results
        return results
    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return
 
###RMSE Lower the value the better.
##R2 the closer to 1 the better the better fitr of the model to the data.
##MSE Lower the value the better the model performance.
##RMSE lower value is better, it is the square root of MSE and provides error in the same units as the target variable.
##MAE indicates the average absolute error between predicted and actual values. The smaller the MAE,
## the better the model's predictions align with the actual data. A MAE of 0 would mean a perfect prediction

 ##[mse, mae, rmse, r2]
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

   
##------Demand Functions------

#dem_TREE = decisionTreeModelDS(sp_WeatherxDem,sp_WeatherxSup,1,False,True)
#dem_RF = randomForestModelDS(sp_WeatherxDem,sp_WeatherxSup,1,False,True)
#dem_XGB = xgbModelDS(sp_WeatherxDem,sp_WeatherxSup,1,False,True)
#dem_BLSTM = biDirectionalLSTMDS(sp_WeatherxDem,sp_WeatherxSup,1,False,True)
#dem_LSTM = LSTMModelDS(sp_WeatherxDem,sp_WeatherxSup,1,False,True)
#dem_GRU = GRUModelDS(sp_WeatherxDem,sp_WeatherxSup,1,False,True)
#dem_SVR = SVRModelDS(sp_WeatherxDem,sp_WeatherxSup,1,False,True)
#dem_MLP = MLPModelDS(sp_WeatherxDem,sp_WeatherxSup,1,False,True)
#dem_CNN = CNNModelDS(sp_WeatherxDem,sp_WeatherxSup,1,False,True)
#dem_GBDT = GBDTModelDS(sp_WeatherxDem,sp_WeatherxSup,1,False,True)

##dem_NSGA2_CNN = NSGA2_CNN_ModelDS(sp_WeatherxDem,sp_WeatherxSup,1) 
##dem_NSGA3_CNN = NSGA3_CNN_ModelDS(sp_WeatherxDem,sp_WeatherxSup,1) 
##demandModelresults = [dem_TREE, dem_RF, dem_XGB, dem_BLSTM, dem_LSTM, dem_GRU, dem_SVR, dem_MLP, dem_CNN, dem_GBDT, dem_NSGA2_CNN,dem_NSGA3_CNN]
"""
demandModelresults = [dem_TREE, dem_RF, dem_XGB, dem_GB, dem_BLSTM, dem_LSTM, dem_GRU, dem_SVR, dem_MLP, dem_CNN, dem_GBDT]
demandBestResultsOrdered = BetterModelSelectionMethod(demandModelresults)

print("\n DemandXWeather Model Ranking (Best to Worst) - Hourly:")
print("{:<20} {:>12} {:>12} {:>12} {:>10}".format("Model", "MSE", "MAE", "RMSE", "R2"))
print("-" * 70)
for res in demandBestResultsOrdered:
    print("{:<20} {:>12.4f} {:>12.4f} {:>12.4f} {:>10.4f}".format(
        res[4], res[0], res[1], res[2], res[3]
    ))
"""
#sup_TREE = decisionTreeModelDS(sp_WeatherxDem,sp_WeatherxSup,2,False,True)
#sup_RF = randomForestModelDS(sp_WeatherxDem,sp_WeatherxSup,2,False,True)
#sup_XGB = xgbModelDS(sp_WeatherxDem,sp_WeatherxSup,2,False,True)
#sup_BLSTM = biDirectionalLSTMDS(sp_WeatherxDem,sp_WeatherxSup,2,False,True)
#sup_LSTM = LSTMModelDS(sp_WeatherxDem,sp_WeatherxSup,2,False,True)
#sup_GRU = GRUModelDS(sp_WeatherxDem,sp_WeatherxSup,2,False,True)
#sup_SVR = SVRModelDS(sp_WeatherxDem,sp_WeatherxSup,2,False,True)
#sup_MLP = MLPModelDS(sp_WeatherxDem,sp_WeatherxSup,2,False,True)
sup_CNN = CNNModelDS(sp_WeatherxDem,sp_WeatherxSup,2,False,True)
sup_GBDT = GBDTModelDS(sp_WeatherxDem,sp_WeatherxSup,2,False,True)
"""
#sup_NSGA2_CNN = NSGA2_CNN_ModelDS(sp_WeatherxDem,sp_WeatherxSup,2) 
#sup_NSGA3_CNN = NSGA3_CNN_ModelDS(sp_WeatherxDem,sp_WeatherxSup,2) 
##supplyModelresults = [sup_TREE, sup_RF, sup_XGB, sup_GB, sup_BLSTM,sup_LSTM, sup_GRU, sup_SVR, sup_MLP, sup_CNN,sup_GBDT, sup_NSGA2_CNN,sup_NSGA3_CNN]
 
supplyModelresults = [sup_TREE, sup_RF, sup_XGB, sup_BLSTM,sup_LSTM, sup_GRU, sup_SVR, sup_MLP, sup_CNN,sup_GBDT]
supplyBestResultsOrdered = BetterModelSelectionMethod(supplyModelresults)



print("\n SupplyXWeather Model Ranking (Best to Worst) - Hourly:")
print("{:<20} {:>12} {:>12} {:>12} {:>10}".format("Model", "MSE", "MAE", "RMSE", "R2"))
print("-" * 70)
for res in supplyBestResultsOrdered:
    print("{:<20} {:>12.4f} {:>12.4f} {:>12.4f} {:>10.4f}".format(
        res[4], res[0], res[1], res[2], res[3]
    ))
"""
