#----# Libraries
import array
from ast import Str
from calendar import c
from ctypes import Array
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
import keras_tuner as kt
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
##Questions to ask Tomorrow
"""
tuner_subdirs = [
    'tuner_dir/blstm_tuning',
    'tuner_dir/lstm_tuning',
    'tuner_dir/gru_tuning',
    'tuner_dir/mlp_tuning',
    'tuner_dir/cnn_tuning',
    'mlp_tuning',
    'cnn_tuning'
]

for subdir in tuner_subdirs:
    if os.path.exists(subdir):
        shutil.rmtree(subdir)

# Optionally, delete the parent tuner_dir if you want to remove everything
if os.path.exists('tuner_dir'):
    shutil.rmtree('tuner_dir')
"""
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None) 
print("Tuner directory:", os.path.abspath('mlp_tuning_new'))
"""
1. Should I look into adding lag into the supply dataset aspects of the method as its results for MSE specifically are wildly different
  to that of Demand dataset results.
2. Why when comparing the creation process for the merged dataset in olivers code does he
duplicate the weather data
with ALLSKY_SFC_SW_DWN ending up as ALLSKY_SFC_SW_DWN_x/ALLSKY_SFC_SW_DWN_y as an example.
"""
# For clearing cache for when the shape is changed in the tuner directory
    


sp_DemandDef = pd.read_excel(r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\Sakakah 2021 Demand dataset.xlsx")
sp_SupplyDef = pd.read_excel(r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\Sakakah 2021 PV supply dataset.xlsx")
sp_WeatherDe = pd.read_excel(r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\Weather for demand 2018.xlsx")
sp_WeatherSu = pd.read_excel(r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\weather for solar 2021.xlsx")

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
#sp_WeatherxDem = pd.merge(sp_DemandDef,sp_WeatherSu, on="TimeStamp", how="inner")
sp_WeatherDe = pd.concat([sp_WeatherSu, sp_WeatherWind.reset_index(drop=True)], axis=1)



sp_WeatherxDem = pd.merge(sp_DemandDef,sp_WeatherSu, on="TimeStamp", how="inner")
sp_WeatherxSup = pd.merge(sp_SupplyDef,sp_WeatherSu, on="TimeStamp", how="inner")
sp_WeatherxSup = pd.concat([sp_WeatherxSup, sp_WeatherWind.reset_index(drop=True)], axis=1)
sp_WeatherxDem = pd.concat([sp_WeatherxDem, sp_WeatherWind.reset_index(drop=True)], axis=1)

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

def create_monthly_sequences_3d(df, feature_cols, target_cols, pad_to_max=True):
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
        y.append(next_month[target_cols].values[0])

    X = np.array(X)
    y = np.array(y)
    return X, y

def create_monthly_sequences_2d(df, feature_cols, target_cols, pad_to_max=True):
    X, y = create_monthly_sequences_3d(df, feature_cols, target_cols, pad_to_max)
    X_flat = X.reshape(X.shape[0], -1)
    return X_flat, y

#----#

##print(sp_SupplyDef.head())
##print(sp_DemandDef.head())
##Note to self can't remember if saeed wanted me to do confusion martrixs aswell as the MSE,MAE,RMSE & R2
def decisionTreeModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int):
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
        """
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_D","ALLSKY_SFC_UV_INDEX_D","T2M_D","PRECTOTCORR_D","ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D","RH2M_D","PS_D","PSC_D","hour","day","month"
    ]
        """
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","WS10M","WD10M","hour","day","month"
        ]
        target_cols = ['MW']

        X_seq, y_seq = create_monthly_sequences_2d(demandDs, feature_cols, target_cols, pad_to_max=True)


        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_seq_scaled = scaler_X.fit_transform(X_seq)
        y_seq_scaled = scaler_y.fit_transform(y_seq)
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
        print('\n', "Decision Tree Model for Demand Data:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
        
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true, label='Actual')
        plt.title(f'Decision Tree Model: Actual Demand')
        plt.xlabel('Month Sequence')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()

        def plot_series(y_true, y_pred, label):
            plt.figure(figsize=(10, 5))
            plt.plot(y_true, label='Actual')
            plt.plot(y_pred, label='Predicted')
            plt.title(f'Decision Tree Model: Actual vs Predicted {label}')
            plt.xlabel('Month Sequence')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()

        plot_series(y_demand_true, y_demand_pred, 'Demand')
        
        return results
    elif (ident == 2):

        supplyDs.replace(-999,np.nan, inplace=True)
        supplyDs.dropna(inplace=True)
        supplyDs["hour"] = supplyDs["TimeStamp"].dt.hour   
        supplyDs["day"] = supplyDs["TimeStamp"].dt.day        
        supplyDs["month"] = supplyDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","WS10M","WD10M","hour","day","month"
        ]
        target_cols = ['MW']

        X_seq, y_seq = create_monthly_sequences_2d(supplyDs, feature_cols, target_cols, pad_to_max=True)


        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_seq_scaled = scaler_X.fit_transform(X_seq)
        y_seq_scaled = scaler_y.fit_transform(y_seq)
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=0.3, shuffle=False
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
       
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true, label='Actual')
        plt.title(f'Decision Tree Model: Actual Supply')
        plt.xlabel('Month Sequence')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()


        def plot_series(y_true, y_pred, label):
            plt.figure(figsize=(10, 5))
            plt.plot(y_true, label='Actual')
            plt.plot(y_pred, label='Predicted')
            plt.title(f'Decision Tree Model: Actual vs Predicted {label}')
            plt.xlabel('Month Sequence')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()

        plot_series(y_demand_true, y_demand_pred, 'Supply')
        
        return results
    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return
  

def randomForestModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int):
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
        "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","WS10M","WD10M","hour","day","month"
        ]
        target_cols = ['MW']

        X_seq, y_seq = create_monthly_sequences_2d(demandDs, feature_cols, target_cols, pad_to_max=True)


        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_seq_scaled = scaler_X.fit_transform(X_seq)
        y_seq_scaled = scaler_y.fit_transform(y_seq)
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=0.3, shuffle=False
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
        
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true, label='Actual')
        plt.title(f'Random Forest Model: Actual Demand')
        plt.xlabel('Month Sequence')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()


        def plot_series(y_true, y_pred, label):
             plt.figure(figsize=(10, 5))
             plt.plot(y_true, label='Actual')
             plt.plot(y_pred, label='Predicted')
             plt.title(f'Random Forest Model: Actual vs Predicted {label}')
             plt.xlabel('Month Sequence')
             plt.ylabel('MW')
             plt.legend()
             plt.tight_layout()
             plt.show()

        plot_series(y_demand_true, y_demand_pred, 'Demand')
        
        return results
    elif (ident == 2):
        supplyDs.replace(-999,np.nan, inplace=True)
        supplyDs.dropna(inplace=True)
        supplyDs["hour"] = supplyDs["TimeStamp"].dt.hour   
        supplyDs["day"] = supplyDs["TimeStamp"].dt.day        
        supplyDs["month"] = supplyDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","WS10M","WD10M","hour","day","month"
        ]
        target_cols = ['MW']

        X_seq, y_seq = create_monthly_sequences_2d(supplyDs, feature_cols, target_cols, pad_to_max=True)


        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_seq_scaled = scaler_X.fit_transform(X_seq)
        y_seq_scaled = scaler_y.fit_transform(y_seq)
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=0.3, shuffle=False
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
        
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true, label='Actual')
        plt.title(f'Random Forest Model: Actual Supply')
        plt.xlabel('Month Sequence')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()


        def plot_series(y_true, y_pred, label):
            plt.figure(figsize=(10, 5))
            plt.plot(y_true, label='Actual')
            plt.plot(y_pred, label='Predicted')
            plt.title(f'Random Forest Model: Actual vs Predicted {label}')
            plt.xlabel('Month Sequence')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()

        plot_series(y_demand_true, y_demand_pred, 'Supply')
        
        return results
    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return


def xgbModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int):
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
        "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","WS10M","WD10M","hour","day","month"
        ]
        target_cols = ['MW']

        X_seq, y_seq = create_monthly_sequences_2d(demandDs, feature_cols, target_cols, pad_to_max=True)


        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_seq_scaled = scaler_X.fit_transform(X_seq)
        y_seq_scaled = scaler_y.fit_transform(y_seq)
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=0.3, shuffle=False
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
        
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true, label='Actual')
        plt.title(f'XGB Model: Actual Demand')
        plt.xlabel('Month Sequence')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()


        def plot_series(y_true, y_pred, label):
            plt.figure(figsize=(10, 5))
            plt.plot(y_true, label='Actual')
            plt.plot(y_pred, label='Predicted')
            plt.title(f'XGB Model: Actual vs Predicted {label}')
            plt.xlabel('Month Sequence')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()

        plot_series(y_demand_true, y_demand_pred, 'Demand')
        

        return results
    elif (ident == 2):
       
        supplyDs.replace(-999,np.nan, inplace=True)
        supplyDs.dropna(inplace=True)
        supplyDs["hour"] = supplyDs["TimeStamp"].dt.hour   
        supplyDs["day"] = supplyDs["TimeStamp"].dt.day        
        supplyDs["month"] = supplyDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","WS10M","WD10M","hour","day","month"
        ]
        target_cols = ['MW']

        X_seq, y_seq = create_monthly_sequences_2d(supplyDs, feature_cols, target_cols, pad_to_max=True)


        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_seq_scaled = scaler_X.fit_transform(X_seq)
        y_seq_scaled = scaler_y.fit_transform(y_seq)
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=0.3, shuffle=False
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
        
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true, label='Actual')
        plt.title(f'XGB Model: Actual Supply')
        plt.xlabel('Month Sequence')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()


        def plot_series(y_true, y_pred, label):
            plt.figure(figsize=(10, 5))
            plt.plot(y_true, label='Actual')
            plt.plot(y_pred, label='Predicted')
            plt.title(f'XGB Model: Actual vs Predicted {label}')
            plt.xlabel('Month Sequence')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()

        plot_series(y_demand_true, y_demand_pred, 'Supply')
        
        return results
    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return

def gbModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy()
  
    if (ident == 1):
    #----#        
        demandDs["hour"] = demandDs["TimeStamp"].dt.hour   
        demandDs["day"] = demandDs["TimeStamp"].dt.day        
        demandDs["month"] = demandDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","WS10M","WD10M","hour","day","month"
        ]
        target_cols = ['MW']

        X_seq, y_seq = create_monthly_sequences_2d(demandDs, feature_cols, target_cols, pad_to_max=True)


        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_seq_scaled = scaler_X.fit_transform(X_seq)
        y_seq_scaled = scaler_y.fit_transform(y_seq)
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=0.3, shuffle=False
        )
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
        
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true, label='Actual')
        plt.title(f'Gradient Boosting Model: Actual Demand')
        plt.xlabel('Month Sequence')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()


        def plot_series(y_true, y_pred, label):
            plt.figure(figsize=(10, 5))
            plt.plot(y_true, label='Actual')
            plt.plot(y_pred, label='Predicted')
            plt.title(f'Gradient Boosting Model: Actual vs Predicted {label}')
            plt.xlabel('Month Sequence')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()

        plot_series(y_demand_true, y_demand_pred, 'Demand')
        
        return results
    elif (ident == 2):

        supplyDs.replace(-999,np.nan, inplace=True)
        supplyDs.dropna(inplace=True)
        supplyDs["hour"] = supplyDs["TimeStamp"].dt.hour   
        supplyDs["day"] = supplyDs["TimeStamp"].dt.day        
        supplyDs["month"] = supplyDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","WS10M","WD10M","hour","day","month"
        ]
        target_cols = ['MW']

        X_seq, y_seq = create_monthly_sequences_2d(supplyDs, feature_cols, target_cols, pad_to_max=True)


        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_seq_scaled = scaler_X.fit_transform(X_seq)
        y_seq_scaled = scaler_y.fit_transform(y_seq)
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
        
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true, label='Actual')
        plt.title(f'Gradient Boosting Model: Actual Supply')
        plt.xlabel('Month Sequence')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()


        def plot_series(y_true, y_pred, label):
            plt.figure(figsize=(10, 5))
            plt.plot(y_true, label='Actual')
            plt.plot(y_pred, label='Predicted')
            plt.title(f'Gradient Boosting Model: Actual vs Predicted {label}')
            plt.xlabel('Month Sequence')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()

        plot_series(y_demand_true, y_demand_pred, 'Supply')
        
        return results
    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return

def biDirectionalLSTMDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int):
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
        "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","WS10M","WD10M","hour","day","month"
        ]
        demandx = demandDs[feature_cols].values
        demandy = demandDs['MW'].values.reshape(-1, 1)
        target_cols = ['MW']
        X_seq, y_seq = create_monthly_sequences_3d(demandDs, feature_cols, target_cols, pad_to_max=True)

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        n_samples, n_timesteps, n_features = X_seq.shape
        X_seq_2d = X_seq.reshape(-1, n_features)
        X_seq_scaled_2d = scaler_X.fit_transform(X_seq_2d)
        X_seq_scaled = X_seq_scaled_2d.reshape(n_samples, n_timesteps, n_features)

        y_seq_scaled = scaler_y.fit_transform(y_seq)

        X_train, X_test, y_train, y_test = train_test_split(
            X_seq_scaled, y_seq_scaled, test_size=0.3, shuffle=False
        )

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
        
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true, label='Actual')
        plt.title(f'BLSTM Model: Actual Demand')
        plt.xlabel('Month Sequence')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()


        def plot_series(y_true, y_pred, label):
            plt.figure(figsize=(10, 5))
            plt.plot(y_true, label='Actual')
            plt.plot(y_pred, label='Predicted')
            plt.title(f'BLSTM Model: Actual vs Predicted {label}')
            plt.xlabel('Month Sequence')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()

        plot_series(y_demand_true, y_demand_pred, 'Demand')
        
        return results
    elif (ident == 2):

        supplyDs["hour"] = supplyDs["TimeStamp"].dt.hour   
        supplyDs["day"] = supplyDs["TimeStamp"].dt.day        
        supplyDs["month"] = supplyDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","WS10M","WD10M","hour","day","month"
        ]
        target_cols = ['MW']
        X_seq, y_seq = create_monthly_sequences_3d(supplyDs, feature_cols, target_cols, pad_to_max=True)

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        n_samples, n_timesteps, n_features = X_seq.shape
        X_seq_2d = X_seq.reshape(-1, n_features)
        X_seq_scaled_2d = scaler_X.fit_transform(X_seq_2d)
        X_seq_scaled = X_seq_scaled_2d.reshape(n_samples, n_timesteps, n_features)

        y_seq_scaled = scaler_y.fit_transform(y_seq)

        X_train, X_test, y_train, y_test = train_test_split(
            X_seq_scaled, y_seq_scaled, test_size=0.3, shuffle=False
        )

        
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
        
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true, label='Actual')
        plt.title(f'BLSTM Model: Actual Supply')
        plt.xlabel('Month Sequence')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()


        def plot_series(y_true, y_pred, label):
            plt.figure(figsize=(10, 5))
            plt.plot(y_true, label='Actual')
            plt.plot(y_pred, label='Predicted')
            plt.title(f'BLSTM Model: Actual vs Predicted {label}')
            plt.xlabel('Month Sequence')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()
            
        plot_series(y_demand_true, y_demand_pred, 'Supply')
        
        return results
    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return

def LSTMModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int):
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
        "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","WS10M","WD10M","hour","day","month"
        ]
        demandx = demandDs[feature_cols].values
        demandy = demandDs['MW'].values.reshape(-1, 1)
        target_cols = ['MW']
        X_seq, y_seq = create_monthly_sequences_3d(demandDs, feature_cols, target_cols, pad_to_max=True)

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        n_samples, n_timesteps, n_features = X_seq.shape
        X_seq_2d = X_seq.reshape(-1, n_features)
        X_seq_scaled_2d = scaler_X.fit_transform(X_seq_2d)
        X_seq_scaled = X_seq_scaled_2d.reshape(n_samples, n_timesteps, n_features)

        y_seq_scaled = scaler_y.fit_transform(y_seq)

        X_train, X_test, y_train, y_test = train_test_split(
            X_seq_scaled, y_seq_scaled, test_size=0.3, shuffle=False
        )

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
        
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true, label='Actual')
        plt.title(f'LSTM Model: Actual Demand')
        plt.xlabel('Month Sequence')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()


        def plot_series(y_true, y_pred, label):
            plt.figure(figsize=(10, 5))
            plt.plot(y_true, label='Actual')
            plt.plot(y_pred, label='Predicted')
            plt.title(f'LSTM Model: Actual vs Predicted {label}')
            plt.xlabel('Month Sequence')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()

        plot_series(y_demand_true, y_demand_pred, 'Demand')
        
        return results
    elif (ident == 2):
        supplyDs["hour"] = supplyDs["TimeStamp"].dt.hour   
        supplyDs["day"] = supplyDs["TimeStamp"].dt.day        
        supplyDs["month"] = supplyDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","WS10M","WD10M","hour","day","month"
        ]
        target_cols = ['MW']
        X_seq, y_seq = create_monthly_sequences_3d(supplyDs, feature_cols, target_cols, pad_to_max=True)

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        n_samples, n_timesteps, n_features = X_seq.shape
        X_seq_2d = X_seq.reshape(-1, n_features)
        X_seq_scaled_2d = scaler_X.fit_transform(X_seq_2d)
        X_seq_scaled = X_seq_scaled_2d.reshape(n_samples, n_timesteps, n_features)

        y_seq_scaled = scaler_y.fit_transform(y_seq)

        X_train, X_test, y_train, y_test = train_test_split(
            X_seq_scaled, y_seq_scaled, test_size=0.3, shuffle=False
        )
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
        
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true, label='Actual')
        plt.title(f'LSTM Model: Actual Supply')
        plt.xlabel('Month Sequence')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()


        def plot_series(y_true, y_pred, label):
            plt.figure(figsize=(10, 5))
            plt.plot(y_true, label='Actual')
            plt.plot(y_pred, label='Predicted')
            plt.title(f'LSTM Model: Actual vs Predicted {label}')
            plt.xlabel('Month Sequence')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()

        plot_series(y_demand_true, y_demand_pred, 'Supply')
        
        return results 
    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return

def GRUModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy()

   
    """
    Demand GRU Results -

    -----
    Supply GRU Results - 

    """
    if (ident == 1):
    #----#        
    ##Basic transformation for the base dataset
        demandDs["hour"] = demandDs["TimeStamp"].dt.hour   
        demandDs["day"] = demandDs["TimeStamp"].dt.day        
        demandDs["month"] = demandDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","WS10M","WD10M","hour","day","month"
        ]
        target_cols = ['MW']
        X_seq, y_seq = create_monthly_sequences_3d(demandDs, feature_cols, target_cols, pad_to_max=True)

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        n_samples, n_timesteps, n_features = X_seq.shape
        X_seq_2d = X_seq.reshape(-1, n_features)
        X_seq_scaled_2d = scaler_X.fit_transform(X_seq_2d)
        X_seq_scaled = X_seq_scaled_2d.reshape(n_samples, n_timesteps, n_features)

        y_seq_scaled = scaler_y.fit_transform(y_seq)

        X_train, X_test, y_train, y_test = train_test_split(
            X_seq_scaled, y_seq_scaled, test_size=0.3, shuffle=False
        )
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
        
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true, label='Actual')
        plt.title(f'GRU Model: Actual Demand')
        plt.xlabel('Month Sequence')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()


        def plot_series(y_true, y_pred, label):
            plt.figure(figsize=(10, 5))
            plt.plot(y_true, label='Actual')
            plt.plot(y_pred, label='Predicted')
            plt.title(f'GRU Model: Actual vs Predicted {label}')
            plt.xlabel('Month Sequence')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()

        plot_series(y_demand_true, y_demand_pred, 'Demand')
        
        return results 
    elif (ident == 2):
        supplyDs["hour"] = supplyDs["TimeStamp"].dt.hour   
        supplyDs["day"] = supplyDs["TimeStamp"].dt.day        
        supplyDs["month"] = supplyDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","WS10M","WD10M","hour","day","month"
        ]
        target_cols = ['MW']
        X_seq, y_seq = create_monthly_sequences_3d(supplyDs, feature_cols, target_cols, pad_to_max=True)

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        n_samples, n_timesteps, n_features = X_seq.shape
        X_seq_2d = X_seq.reshape(-1, n_features)
        X_seq_scaled_2d = scaler_X.fit_transform(X_seq_2d)
        X_seq_scaled = X_seq_scaled_2d.reshape(n_samples, n_timesteps, n_features)

        y_seq_scaled = scaler_y.fit_transform(y_seq)

        X_train, X_test, y_train, y_test = train_test_split(
            X_seq_scaled, y_seq_scaled, test_size=0.3, shuffle=False
        )
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
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true, label='Actual')
        plt.title(f'GRU Model: Actual Supply')
        plt.xlabel('Month Sequence')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()


        def plot_series(y_true, y_pred, label):
            plt.figure(figsize=(10, 5))
            plt.plot(y_true, label='Actual')
            plt.plot(y_pred, label='Predicted')
            plt.title(f'GRU Model: Actual vs Predicted {label}')
            plt.xlabel('Month Sequence')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()

        plot_series(y_demand_true, y_demand_pred, 'Supply')
        return results
    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return
##
def SVRModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int):
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
        "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","WS10M","WD10M","hour","day","month"
        ]
        target_cols = ['MW']

        X_seq, y_seq = create_monthly_sequences_2d(demandDs, feature_cols, target_cols, pad_to_max=True)


        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_seq_scaled = scaler_X.fit_transform(X_seq)
        y_seq_scaled = scaler_y.fit_transform(y_seq)
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=0.3, shuffle=False
        )
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
        
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true, label='Actual')
        plt.title(f'SVR Model: Actual Demand')
        plt.xlabel('Month Sequence')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()


        def plot_series(y_true, y_pred, label):
            plt.figure(figsize=(10, 5))
            plt.plot(y_true, label='Actual')
            plt.plot(y_pred, label='Predicted')
            plt.title(f'SVR Model: Actual vs Predicted {label}')
            plt.xlabel('Month Sequence')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()

        plot_series(y_demand_true, y_demand_pred, 'Demand')
        
        return results 
    elif (ident == 2):
        supplyDs["hour"] = supplyDs["TimeStamp"].dt.hour   
        supplyDs["day"] = supplyDs["TimeStamp"].dt.day        
        supplyDs["month"] = supplyDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","WS10M","WD10M","hour","day","month"
        ]
        target_cols = ['MW']

        X_seq, y_seq = create_monthly_sequences_2d(supplyDs, feature_cols, target_cols, pad_to_max=True)


        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_seq_scaled = scaler_X.fit_transform(X_seq)
        y_seq_scaled = scaler_y.fit_transform(y_seq)
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=0.3, shuffle=False
        )

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
        
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true, label='Actual')
        plt.title(f'SVR Model: Actual Supply')
        plt.xlabel('Month Sequence')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()


        def plot_series(y_true, y_pred, label):
            plt.figure(figsize=(10, 5))
            plt.plot(y_true, label='Actual')
            plt.plot(y_pred, label='Predicted')
            plt.title(f'SVR Model: Actual vs Predicted {label}')
            plt.xlabel('Month Sequence')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()

        plot_series(y_demand_true, y_demand_pred, 'Supply')
        
        return results
    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return

def MLPModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int):

   
        """
        Demand MLP Results -

        -----
        Supply MLP Results - 

        """
        demandDs = demandDs.copy()
        supplyDs = supplyDs.copy()
        if (ident == 1):
            #----#        
            ##Basic transformation for the base dataset
                demandDs["hour"] = demandDs["TimeStamp"].dt.hour   
                demandDs["day"] = demandDs["TimeStamp"].dt.day        
                demandDs["month"] = demandDs["TimeStamp"].dt.month        
                feature_cols = [
                "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
                "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","WS10M","WD10M","hour","day","month"
                ]
                target_cols = ['MW']

                X_seq, y_seq = create_monthly_sequences_2d(demandDs, feature_cols, target_cols, pad_to_max=True)


                scaler_X = MinMaxScaler()
                scaler_y = MinMaxScaler()
                X_seq_scaled = scaler_X.fit_transform(X_seq)
                y_seq_scaled = scaler_y.fit_transform(y_seq)
                X_train, X_test, y_train, y_test = train_test_split(
                    X_seq, y_seq, test_size=0.3, shuffle=False
                )
                print("X_train shape:", X_train.shape)
                print("y_train shape:", y_train.shape)
        
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
                
                plt.figure(figsize=(10, 5))
                plt.plot(y_demand_true, label='Actual')
                plt.title(f'MLP Model: Actual Demand')
                plt.xlabel('Month Sequence')
                plt.ylabel('MW')
                plt.legend()
                plt.tight_layout()
                plt.show()


                def plot_series(y_true, y_pred, label):
                    plt.figure(figsize=(10, 5))
                    plt.plot(y_true, label='Actual')
                    plt.plot(y_pred, label='Predicted')
                    plt.title(f'MLP Model: Actual vs Predicted {label}')
                    plt.xlabel('Month Sequence')
                    plt.ylabel('MW')
                    plt.legend()
                    plt.tight_layout()
                    plt.show()

                plot_series(y_demand_true, y_demand_pred, 'Demand')
                
                return results 
        elif (ident == 2):
                supplyDs["hour"] = supplyDs["TimeStamp"].dt.hour   
                supplyDs["day"] = supplyDs["TimeStamp"].dt.day        
                supplyDs["month"] = supplyDs["TimeStamp"].dt.month        
        
                feature_cols = [
                "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
                "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","WS10M","WD10M","hour","day","month"
                ]
                target_cols = ['MW']

                X_seq, y_seq = create_monthly_sequences_2d(supplyDs, feature_cols, target_cols, pad_to_max=True)


                scaler_X = MinMaxScaler()
                scaler_y = MinMaxScaler()
                X_seq_scaled = scaler_X.fit_transform(X_seq)
                y_seq_scaled = scaler_y.fit_transform(y_seq)
                X_train, X_test, y_train, y_test = train_test_split(
                    X_seq, y_seq, test_size=0.3, shuffle=False
                )
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
                
                plt.figure(figsize=(10, 5))
                plt.plot(y_demand_true, label='Actual')
                plt.title(f'MLP Model: Actual Supply')
                plt.xlabel('Month Sequence')
                plt.ylabel('MW')
                plt.legend()
                plt.tight_layout()
                plt.show()


                def plot_series(y_true, y_pred, label):
                    plt.figure(figsize=(10, 5))
                    plt.plot(y_true, label='Actual')
                    plt.plot(y_pred, label='Predicted')
                    plt.title(f'MLP Model: Actual vs Predicted {label}')
                    plt.xlabel('Month Sequence')
                    plt.ylabel('MW')
                    plt.legend()
                    plt.tight_layout()
                    plt.show()

                plot_series(y_demand_true, y_demand_pred, 'Supply')
                
                return results
        else:
                print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
                return

def CNNModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy()
    """
    Demand CNN Results -

    -----
    Supply CNN Results - 

    """

    if ident == 1:
        #----#        
        demandDs["hour"] = demandDs["TimeStamp"].dt.hour   
        demandDs["day"] = demandDs["TimeStamp"].dt.day        
        demandDs["month"] = demandDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","WS10M","WD10M","hour","day","month"
        ]
        target_cols = ['MW']
        X_seq, y_seq = create_monthly_sequences_3d(demandDs, feature_cols, target_cols, pad_to_max=True)

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        n_samples, n_timesteps, n_features = X_seq.shape
        X_seq_2d = X_seq.reshape(-1, n_features)
        X_seq_scaled_2d = scaler_X.fit_transform(X_seq_2d)
        X_seq_scaled = X_seq_scaled_2d.reshape(n_samples, n_timesteps, n_features)

        y_seq_scaled = scaler_y.fit_transform(y_seq)

        X_train, X_test, y_train, y_test = train_test_split(
            X_seq_scaled, y_seq_scaled, test_size=0.3, shuffle=False
        )

        

        def build_cnn_model(hp):
            model = Sequential()
            model.add(Conv1D(
                filters=hp.Int('filters1', 32, 128, step=32),
                kernel_size=hp.Choice('kernel_size1', [2, 3, 5]),
                activation='relu',
                input_shape=(X_train.shape[1], X_train.shape[2])
            ))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(hp.Float('dropout1', 0.1, 0.5, step=0.1)))
            model.add(Conv1D(
                filters=hp.Int('filters2', 32, 128, step=32),
                kernel_size=hp.Choice('kernel_size2', [2, 3, 5]),
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
        
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true, label='Actual')
        plt.title(f'CNN Model: Actual Demand')
        plt.xlabel('Month Sequence')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()


        def plot_series(y_true, y_pred, label):
            plt.figure(figsize=(10, 5))
            plt.plot(y_true, label='Actual')
            plt.plot(y_pred, label='Predicted')
            plt.title(f'CNN Model: Actual vs Predicted {label}')
            plt.xlabel('Month Sequence')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()

        plot_series(y_demand_true, y_demand_pred, 'Demand')
        
        return results

    elif ident == 2:
        supplyDs["hour"] = supplyDs["TimeStamp"].dt.hour   
        supplyDs["day"] = supplyDs["TimeStamp"].dt.day        
        supplyDs["month"] = supplyDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","WS10M","WD10M","hour","day","month"
        ]
        target_cols = ['MW']
        X_seq, y_seq = create_monthly_sequences_3d(supplyDs, feature_cols, target_cols, pad_to_max=True)

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        n_samples, n_timesteps, n_features = X_seq.shape
        X_seq_2d = X_seq.reshape(-1, n_features)
        X_seq_scaled_2d = scaler_X.fit_transform(X_seq_2d)
        X_seq_scaled = X_seq_scaled_2d.reshape(n_samples, n_timesteps, n_features)

        y_seq_scaled = scaler_y.fit_transform(y_seq)

        X_train, X_test, y_train, y_test = train_test_split(
            X_seq_scaled, y_seq_scaled, test_size=0.3, shuffle=False
        )

        def build_cnn_model(hp):
            model = Sequential()
            model.add(Conv1D(
                filters=hp.Int('filters1', 32, 128, step=32),
                kernel_size=hp.Choice('kernel_size1', [2, 3, 5]),
                activation='relu',
                input_shape=(X_train.shape[1], X_train.shape[2])
            ))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(hp.Float('dropout1', 0.1, 0.5, step=0.1)))
            model.add(Conv1D(
                filters=hp.Int('filters2', 32, 128, step=32),
                kernel_size=hp.Choice('kernel_size2', [2, 3, 5]),
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
        
        #---#
        print('\n', "CNN Model for Supply Data:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
        
        plt.figure(figsize=(10, 5))
        plt.plot(y_supply_true, label='Actual')
        plt.title(f'CNN Model: Actual Supply')
        plt.xlabel('Month Sequence')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()


        def plot_series(y_true, y_pred, label):
            plt.figure(figsize=(10, 5))
            plt.plot(y_true, label='Actual')
            plt.plot(y_pred, label='Predicted')
            plt.title(f'CNN Model: Actual vs Predicted {label}')
            plt.xlabel('Month Sequence')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()

        plot_series(y_supply_true, y_supply_pred, 'Supply')
        
        return results

    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return
    

def GBDTModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy()
   
    """
    Demand GBDT Results -

    -----
    Supply GBDT Results - 

        -----

    """


    if (ident == 1):
    #----#        
    ##Basic transformation for the base dataset
        demandDs["hour"] = demandDs["TimeStamp"].dt.hour   
        demandDs["day"] = demandDs["TimeStamp"].dt.day        
        demandDs["month"] = demandDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","WS10M","WD10M","hour","day","month"
        ]
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","WS10M","WD10M","hour","day","month"
        ]
        target_cols = ['MW']

        X_seq, y_seq = create_monthly_sequences_2d(demandDs, feature_cols, target_cols, pad_to_max=True)


        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_seq_scaled = scaler_X.fit_transform(X_seq)
        y_seq_scaled = scaler_y.fit_transform(y_seq)
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=0.3, shuffle=False
        )

        param_grid = {
        'estimator__C': [0.1, 1, 10, 100],           # Regularization, similar to tree depth/complexity
        'estimator__gamma': ['scale', 'auto', 0.01, 0.1, 1],  # Kernel coefficient, like learning rate
        'estimator__epsilon': [0.01, 0.1, 0.5, 1.0], # Insensitivity, similar to min_samples_split
        'estimator__kernel': ['rbf']                 # Keep kernel consistent
        }
        ##model = MultiOutputRegressor(SVR(kernel='rbf'))

        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Ensure 2D shape for inverse_transform
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)

        y_demand_pred = scaler_y.inverse_transform(y_pred)
        y_demand_true = scaler_y.inverse_transform(y_test)

        mse = mean_squared_error(y_demand_true, y_demand_pred)
        mae = mean_absolute_error(y_demand_true, y_demand_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_demand_true, y_demand_pred)
        modelName = "GBDT"
        results = [mse, mae, rmse, r2, modelName]
        
        #---#
        print('\n', "Demand GBDT Model Results:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))

        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true, label='Actual')
        plt.title(f'GBDT Model: Actual Demand')
        plt.xlabel('Month Sequence')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()


        def plot_series(y_true, y_pred, label):
            plt.figure(figsize=(10, 5))
            plt.plot(y_true, label='Actual')
            plt.plot(y_pred, label='Predicted')
            plt.title(f'GBDT Model: Actual vs Predicted {label}')
            plt.xlabel('Month Sequence')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()

        plot_series(y_demand_true, y_demand_pred, 'Demand')

        return results
    elif (ident == 2):
        supplyDs["hour"] = supplyDs["TimeStamp"].dt.hour   
        supplyDs["day"] = supplyDs["TimeStamp"].dt.day        
        supplyDs["month"] = supplyDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","WS10M","WD10M","hour","day","month"
        ]
        target_cols = ['MW']

        X_seq, y_seq = create_monthly_sequences_2d(supplyDs, feature_cols, target_cols, pad_to_max=True)


        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_seq_scaled = scaler_X.fit_transform(X_seq)
        y_seq_scaled = scaler_y.fit_transform(y_seq)
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=0.3, shuffle=False
        )
        param_grid = {
        'estimator__C': [0.1, 1, 10, 100],           # Regularization, similar to tree depth/complexity
        'estimator__gamma': ['scale', 'auto', 0.01, 0.1, 1],  # Kernel coefficient, like learning rate
        'estimator__epsilon': [0.01, 0.1, 0.5, 1.0], # Insensitivity, similar to min_samples_split
        'estimator__kernel': ['rbf']                 # Keep kernel consistent
        }
        ##model = MultiOutputRegressor(SVR(kernel='rbf'))

        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Ensure 2D shape for inverse_transform
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)

        y_demand_pred = scaler_y.inverse_transform(y_pred)
        y_demand_true = scaler_y.inverse_transform(y_test)

        mse = mean_squared_error(y_demand_true, y_demand_pred)
        mae = mean_absolute_error(y_demand_true, y_demand_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_demand_true, y_demand_pred)
        modelName = "GBDT"
        results = [mse, mae, rmse, r2, modelName]

        
        #----#
        print('\n', "GBDT Model for Supply Data:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
        
        plt.figure(figsize=(10, 5))
        plt.plot(y_demand_true, label='Actual')
        plt.title(f'GBDT Model: Actual Supply')
        plt.xlabel('Month Sequence')
        plt.ylabel('MW')
        plt.legend()
        plt.tight_layout()
        plt.show()


        def plot_series(y_true, y_pred, label):
            plt.figure(figsize=(10, 5))
            plt.plot(y_true, label='Actual')
            plt.plot(y_pred, label='Predicted')
            plt.title(f'GBDT Model: Actual vs Predicted {label}')
            plt.xlabel('Month Sequence')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()

        plot_series(y_demand_true, y_demand_pred, 'Supply')
        
        return results 
    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return
    """

def custom_mutate(individual, indpb, valid_kernel_sizes):
    if random.random() < indpb:
        individual[0] = random.choice([16, 32, 64])
    if random.random() < indpb:
        individual[1] = random.choice(valid_kernel_sizes)
    if random.random() < indpb:
        individual[2] = float(random.uniform(0.1, 0.3))
    if random.random() < indpb:
        individual[3] = random.choice([16, 32, 64])
    if random.random() < indpb:
        individual[4] = float(random.choice([1e-3, 1e-4]))
    return (individual,)
def ensure_real_result(results):
    # Only convert the first 4 metrics, leave the model name as is
    return [float(np.real(x)) if isinstance(x, (complex, np.complexfloating, np.floating, float)) else x for x in results]
def NSGA2_CNN_ModelDS(
    demandDs: pd.DataFrame,
    supplyDs: pd.DataFrame,
    ident: int,
    pop_size=5,
    ngen=3,
    cxpb=0.7,
    mutpb=0.3,
    seq_length=24,
    epochs=20,
    batch_size=16
):
    # Select dataset and target
    if ident == 1:
        df = demandDs.copy()
        target_col = 'MW' if 'MW' in df.columns else 'MW'
        model_label = 'Demand'
    elif ident == 2:
        df = supplyDs.copy()
        target_col = 'MW' if 'MW' in df.columns else 'MW'
        model_label = 'Supply'
    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return

    df.replace(-999, np.nan, inplace=True)
    df.dropna(inplace=True)
    df["hour"] = df["TimeStamp"].dt.hour
    df["day"] = df["TimeStamp"].dt.day
    df["month"] = df["TimeStamp"].dt.month

    feature_cols = [
        "ALLSKY_SFC_SW_DWN_S", "ALLSKY_SFC_UV_INDEX_S", "T2M_S", "PRECTOTCORR_S", "ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S", "RH2M_S", "PS_S", "PSC_S", "WS10M", "WD10M", "hour", "day", "month"
    ]
    target_cols = [target_col]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X = scaler_X.fit_transform(df[feature_cols])
    y = scaler_y.fit_transform(df[target_cols])

    X_seq, y_seq = create_sequences_3d(X, y, seq_length)
    num_features = len(feature_cols)
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.3, shuffle=False
    )

    if not hasattr(NSGA2_CNN_ModelDS, "creator_built"):
        creator.create("FitnessSingleCNN", base.Fitness, weights=(-1.0,))
        creator.create("IndividualSingleCNN", list, fitness=creator.FitnessSingleCNN)
        NSGA2_CNN_ModelDS.creator_built = True

    filters_choices = [16, 32, 64]
    dense_choices = [16, 32, 64]
    dropout_range = (0.1, 0.3)
    learning_rates = [1e-3, 1e-4]
    valid_kernel_sizes = [2, 3]

    toolbox = base.Toolbox()
    toolbox.register("filters1", random.choice, filters_choices)
    toolbox.register("kernel_size1", random.choice, valid_kernel_sizes)
    toolbox.register("dropout1", random.uniform, *dropout_range)
    toolbox.register("dense_units", random.choice, dense_choices)
    toolbox.register("learning_rate", random.choice, learning_rates)
    toolbox.register("individual", tools.initCycle, creator.IndividualSingleCNN,
                     (toolbox.filters1, toolbox.kernel_size1, toolbox.dropout1, toolbox.dense_units, toolbox.learning_rate), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def compute_metrics(y_true, y_pred):
        y_pred_inv = scaler_y.inverse_transform(y_pred)
        y_true_inv = scaler_y.inverse_transform(y_true)
        mse = mean_squared_error(y_true_inv, y_pred_inv)
        return float(np.real(mse))

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
                Dense(1)
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
            mse = compute_metrics(y_test, y_pred)
            if np.isnan(mse) or np.isinf(mse):
                return (float('inf'),)
            if np.isnan(y_pred).any() or np.isinf(y_pred).any():
                return (float('inf'),)
            return (mse,)
        except Exception as e:
            print(f"Error evaluating individual {individual}: {e}")
            traceback.print_exc()
            return (float('inf'),)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", custom_mutate, indpb=0.2, valid_kernel_sizes=valid_kernel_sizes)
    toolbox.register("select", tools.selNSGA2)

    # Ensure at least one valid individual in the population
    max_attempts = 10
    for attempt in range(max_attempts):
        pop = toolbox.population(n=pop_size)
        NGEN = ngen

        for gen in range(NGEN):
            print(f"===== Generation {gen + 1} =====")
            offspring = algorithms.varAnd(pop, toolbox, cxpb=cxpb, mutpb=mutpb)
            fits = list(map(toolbox.evaluate, offspring))
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
            pop = toolbox.select(pop + offspring, k=len(pop))

        valid_inds = [ind for ind in pop if ind.fitness.valid]
        if valid_inds:
            break
        else:
            print(f"Attempt {attempt+1}: No valid individuals found, retrying with new population...")

    if not valid_inds:
        print("No valid individuals found in the final population after several attempts.")
        return None

    best = tools.selBest(valid_inds, 1)[0]
    filters1, kernel_size1, dropout1, dense_units, learning_rate = best

    tf.keras.backend.clear_session()
    final_model = Sequential([
        Conv1D(filters=int(filters1), kernel_size=int(kernel_size1), activation='relu', input_shape=(seq_length, num_features)),
        MaxPooling1D(pool_size=2),
        Dropout(float(dropout1)),
        Flatten(),
        Dense(int(dense_units), activation='relu'),
        Dense(1)
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
    y_pred_inv = scaler_y.inverse_transform(final_pred)
    y_true_inv = scaler_y.inverse_transform(y_test)

    mse = mean_squared_error(y_true_inv, y_pred_inv)
    mae = mean_absolute_error(y_true_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_inv, y_pred_inv)
    modelName = f"NSGA-II CNN ({model_label})"
    results = [mse, mae, rmse, r2, modelName]

    print('\n', f"NSGA-II CNN Model Results for {model_label}:")
    print(f"Best Params: filters1={int(filters1)}, kernel_size1={int(kernel_size1)}, dropout1={float(dropout1):.2f}, dense_units={int(dense_units)}, learning_rate={float(learning_rate)}")
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))

    plt.figure(figsize=(10, 5))
    plt.plot(y_true_inv, label=f'Actual {model_label}')
    plt.plot(y_pred_inv, label=f'Predicted {model_label}')
    plt.title(f'NSGA-II CNN: Actual vs Predicted {model_label}')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return results
    """
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

dem_TREE = decisionTreeModelDS(sp_WeatherxDem,sp_WeatherxSup,1)
dem_RF = randomForestModelDS(sp_WeatherxDem,sp_WeatherxSup,1)
dem_XGB = xgbModelDS(sp_WeatherxDem,sp_WeatherxSup,1)
dem_GB =gbModelDS(sp_WeatherxDem,sp_WeatherxSup,1)
dem_BLSTM = biDirectionalLSTMDS(sp_WeatherxDem,sp_WeatherxSup,1)

dem_LSTM = LSTMModelDS(sp_WeatherxDem,sp_WeatherxSup,1)
dem_GRU = GRUModelDS(sp_WeatherxDem,sp_WeatherxSup,1)
dem_SVR = SVRModelDS(sp_WeatherxDem,sp_WeatherxSup,1)

dem_MLP = MLPModelDS(sp_WeatherxDem,sp_WeatherxSup,1)

dem_CNN = CNNModelDS(sp_WeatherxDem,sp_WeatherxSup,1)
dem_GBDT = GBDTModelDS(sp_WeatherxDem,sp_WeatherxSup,1)
##dem_NSGA2_CNN = NSGA2_CNN_ModelDS(sp_WeatherxDem,sp_WeatherxSup,1) 
##dem_NSGA3_CNN = NSGA3_CNN_ModelDS(sp_WeatherxDem,sp_WeatherxSup,1) 
##demandModelresults = [dem_TREE, dem_RF, dem_XGB, dem_GB, dem_BLSTM, dem_LSTM, dem_GRU, dem_SVR, dem_MLP, dem_CNN, dem_GBDT, dem_NSGA2_CNN,dem_NSGA3_CNN]

demandModelresults = [dem_TREE, dem_RF, dem_XGB, dem_GB, dem_BLSTM, dem_LSTM, dem_GRU, dem_SVR, dem_MLP, dem_CNN, dem_GBDT]
demandBestResultsOrdered = BetterModelSelectionMethod(demandModelresults)



sup_TREE = decisionTreeModelDS(sp_WeatherxDem,sp_WeatherxSup,2)
sup_RF = randomForestModelDS(sp_WeatherxDem,sp_WeatherxSup,2)
sup_XGB = xgbModelDS(sp_WeatherxDem,sp_WeatherxSup,2)
sup_GB =gbModelDS(sp_WeatherxDem,sp_WeatherxSup,2)
sup_BLSTM = biDirectionalLSTMDS(sp_WeatherxDem,sp_WeatherxSup,2)
sup_LSTM = LSTMModelDS(sp_WeatherxDem,sp_WeatherxSup,2)
sup_GRU = GRUModelDS(sp_WeatherxDem,sp_WeatherxSup,2)
sup_SVR = SVRModelDS(sp_WeatherxDem,sp_WeatherxSup,2)
sup_MLP = MLPModelDS(sp_WeatherxDem,sp_WeatherxSup,2)
sup_CNN = CNNModelDS(sp_WeatherxDem,sp_WeatherxSup,2)
sup_GBDT = GBDTModelDS(sp_WeatherxDem,sp_WeatherxSup,2)

#sup_NSGA2_CNN = NSGA2_CNN_ModelDS(sp_WeatherxDem,sp_WeatherxSup,2) 
#sup_NSGA3_CNN = NSGA3_CNN_ModelDS(sp_WeatherxDem,sp_WeatherxSup,2) 
##supplyModelresults = [sup_TREE, sup_RF, sup_XGB, sup_GB, sup_BLSTM,sup_LSTM, sup_GRU, sup_SVR, sup_MLP, sup_CNN,sup_GBDT, sup_NSGA2_CNN,sup_NSGA3_CNN]

supplyModelresults = [sup_TREE, sup_RF, sup_XGB, sup_GB, sup_BLSTM,sup_LSTM, sup_GRU, sup_SVR, sup_MLP, sup_CNN,sup_GBDT]
supplyBestResultsOrdered = BetterModelSelectionMethod(supplyModelresults)


print("\n Demand Model Ranking (Best to Worst) - Monthly:")
print("{:<20} {:>12} {:>12} {:>12} {:>10}".format("Model", "MSE", "MAE", "RMSE", "R2"))
print("-" * 70)
for res in demandBestResultsOrdered:
    print("{:<20} {:>12.4f} {:>12.4f} {:>12.4f} {:>10.4f}".format(
        res[4], res[0], res[1], res[2], res[3]
    ))

print("\n Supply Model Ranking (Best to Worst) - Monthly:")
print("{:<20} {:>12} {:>12} {:>12} {:>10}".format("Model", "MSE", "MAE", "RMSE", "R2"))
print("-" * 70)
for res in supplyBestResultsOrdered:
    print("{:<20} {:>12.4f} {:>12.4f} {:>12.4f} {:>10.4f}".format(
        res[4], res[0], res[1], res[2], res[3]
    ))

    





