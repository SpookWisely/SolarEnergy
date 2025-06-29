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

##Questions to ask Tomorrow 
"""
1. Should I look into adding lag into the supply dataset aspects of the method as its results for MSE specifically are wildly different
  to that of Demand dataset results.
"""
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
###RMSE Lower the value the better.
##R2 the closer to 1 the better the better fitr of the model to the data.
##MSE Lower the value the better the model performance.
##MAE indicates the average absolute error between predicted and actual values. The smaller the MAE,
## the better the model's predictions align with the actual data. A MAE of 0 would mean a perfect prediction

sp_DemandDef = pd.read_excel(r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\Sakakah 2021 Demand dataset.xlsx")
sp_SupplyDef = pd.read_excel(r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\Sakakah 2021 PV supply dataset.xlsx")
def create_sequences_with_time(data, targets, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length].flatten())
        y.append(targets[i + seq_length])
    return np.array(X), np.array(y)

def create_sequences_with_time_flatten(data, targets, seq_length):
    """
    Creates sequences and flattens each sequence (for tree/MLP models).
    Output shape: (samples, seq_length * features)
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length].flatten())
        y.append(targets[i + seq_length])
    return np.array(X), np.array(y)

def create_sequences_with_time_3d(data, targets, seq_length):
    """
    Creates sequences without flattening (for LSTM/GRU/CNN models).
    Output shape: (samples, seq_length, features)
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
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
def create_seasonal_sequences(df, feature_cols, target_cols, pad_to_max=True):
    df = df.copy()
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    df['year'] = df['TimeStamp'].dt.year
    df['month'] = df['TimeStamp'].dt.month

    # Map months to seasons
    def month_to_season(month):
        if month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Autumn'
        else:  # 12, 1, 2
            return 'Winter'

    df['season'] = df['month'].apply(month_to_season)

    # Adjust year for December, January, February to keep winters together
    df['season_year'] = df['year']
    df.loc[(df['month'] == 12), 'season_year'] += 1  # December belongs to next year's winter

    X, y = [], []
    grouped = df.groupby(['season_year', 'season'])
    seasons = sorted(grouped.groups.keys())

    max_len = max(len(grouped.get_group(s)[feature_cols]) for s in seasons[:-1]) if pad_to_max else None

    for i in range(len(seasons) - 1):
        this_season = grouped.get_group(seasons[i])
        next_season = grouped.get_group(seasons[i + 1])

        x_seq = this_season[feature_cols].values
        if pad_to_max and x_seq.shape[0] < max_len:
            pad_width = ((0, max_len - x_seq.shape[0]), (0, 0))
            x_seq = np.pad(x_seq, pad_width, mode='constant')
        X.append(x_seq)
        y.append(next_season[target_cols].values[0])  # or customize as needed

    X = np.array(X)
    y = np.array(y)
    return X, y


random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
##Note to self can't remember if saeed wanted me to do confusion martrixs aswell as the MSE,MAE,RMSE & R2
def decisionTreeModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int, plots:bool):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy()    
    """
    Decision Tree Results for Demand Dataset -
    MSE: 69.5029
    MAE: 5.7254
    RMSE: 8.3368
    R2: 0.9065
    --- 
    Decision Tree  Results for Supply Dataset -
    MSE: 2705.1184
    MAE: 24.1974
    RMSE: 52.0108
    R2: 0.7519
    """
    if (ident == 1):
    #----#        
        demandDs['DATE-TIME'] = demandDs['DATE-TIME'].astype(str)
        demandDs[['Date', 'Time']] = demandDs['DATE-TIME'].str.split(' ',expand=True)
        demandDs ['MW'] = pd.to_numeric(demandDs['MW'],errors='coerce')
        demandDs['DATE-TIME'] = pd.to_datetime(demandDs['DATE-TIME'], errors='coerce')

        demandDs.dropna(subset=['MW','DATE-TIME'], inplace=True)
    
        ###minLength = min(len(demandDs))
        ###demandDs = demandDs.iloc[minLength]

        #Extraction of Data.

        demand = demandDs['MW'].values.reshape(-1,1)
        datetime_series = demandDs['DATE-TIME']

        hour = datetime_series.dt.hour.values.reshape(-1,1)
        dayofweek = datetime_series.dt.dayofweek.values.reshape(-1,1)
        month = datetime_series.dt.month.values.reshape(-1,1)

        scaler_demand = MinMaxScaler()
        scaler_hour = MinMaxScaler()
        scaler_day = MinMaxScaler()
        scaler_month = MinMaxScaler()

        demand_scaled = scaler_demand.fit_transform(demand)
        hour_scaled = scaler_hour.fit_transform(hour)
        day_scaled = scaler_day.fit_transform(dayofweek)
        month_scaled = scaler_month.fit_transform(month)

        full_data = np.hstack((demand_scaled, hour_scaled, day_scaled, month_scaled))
        targets = (demand_scaled)

        seq_length = 24
        X, y = create_sequences_with_time_flatten(full_data, targets, seq_length)

        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        param_grid = {
            'estimator__max_depth': [3, 5, 10, None],
            'estimator__min_samples_split': [2, 5, 10]
        }
        base_model = DecisionTreeRegressor(random_state=42)
        model = MultiOutputRegressor(base_model)
        grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_

        y_pred = model.predict(X_test)

        y_demand_pred = scaler_demand.inverse_transform(y_pred[:, 0].reshape(-1, 1))
        y_demand_true = scaler_demand.inverse_transform(y_test[:, 0].reshape(-1, 1))


        mse = mean_squared_error(y_demand_true, y_demand_pred)
        mae = mean_absolute_error(y_demand_true, y_demand_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_demand_true, y_demand_pred)
        modelName = "Decision Tree"
        results = [mse, mae, rmse, r2,modelName]
        #---#
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
        
        #----#
        print('\n',"Demand Data Decision Tree Model Results:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))

        return results
    elif (ident == 2):

        supplyDs['Date & Time'] = supplyDs['Date & Time'].astype(str)
        supplyDs[['Date','Time']] = supplyDs['Date & Time'].str.split(' ',expand=True)
        supplyDs ['MW'] = pd.to_numeric(supplyDs['MW'], errors='coerce')
        supplyDs['Date & Time'] = pd.to_datetime(supplyDs['Date & Time'], errors='coerce')

        supplyDs.dropna(subset=['MW','Date & Time'], inplace=True)

        supply = supplyDs['MW'].values.reshape(-1,1)
        datetime_series = supplyDs['Date & Time']

        hour = datetime_series.dt.hour.values.reshape(-1,1)
        dayofweek = datetime_series.dt.dayofweek.values.reshape(-1,1)
        month = datetime_series.dt.month.values.reshape(-1,1)

        scaler_supply = MinMaxScaler()
        scaler_hour = MinMaxScaler()
        scaler_day = MinMaxScaler()
        scaler_month = MinMaxScaler()

        supply_scaled = scaler_supply.fit_transform(supply)
        hour_scaled = scaler_hour.fit_transform(hour)
        day_scaled = scaler_day.fit_transform(dayofweek)
        month_scaled = scaler_month.fit_transform(month)

        full_data = np.hstack((supply_scaled, hour_scaled, day_scaled, month_scaled))
        targets = (supply_scaled)

        seq_length = 24
        X, y = create_sequences_with_time_flatten(full_data, targets, seq_length)


        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        param_grid = {
            'estimator__max_depth': [3, 5, 10, None],
            'estimator__min_samples_split': [2, 5, 10]
        }
        base_model = DecisionTreeRegressor(random_state=42)
        model = MultiOutputRegressor(base_model)
        grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_supply_pred = scaler_supply.inverse_transform(y_pred[:, 0].reshape(-1, 1))
        y_supply_true = scaler_supply.inverse_transform(y_test[:, 0].reshape(-1, 1))


        mse = mean_squared_error(y_supply_true, y_supply_pred)
        mae = mean_absolute_error(y_supply_true, y_supply_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_supply_true, y_supply_pred)
        modelName = "Decision Tree"
        results = [mse, mae, rmse, r2,modelName]

        #----#
        print('\n', "Decision Tree Model for Supply Data:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
        if plots == True:
            plt.figure(figsize=(10, 5))
            plt.plot(y_supply_true, label='Actual')
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

            plot_series(y_supply_true, y_supply_pred, 'Supply')


        return results
    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return
  

def randomForestModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int, plots:bool):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy()      
    
 
    """
    Demand Random Forest Results -
    MSE: 37.2506
    MAE: 3.9268
    RMSE: 6.1033
    R2: 0.9499
    -----
    Supply Random Forest Results - 
    MSE: 1305.4629
    MAE: 17.9107
    RMSE: 36.1312
    R2: 0.8802
    """
    if (ident == 1):
    #----#        
        demandDs['DATE-TIME'] = demandDs['DATE-TIME'].astype(str)
        demandDs[['Date', 'Time']] = demandDs['DATE-TIME'].str.split(' ',expand=True)
        demandDs ['MW'] = pd.to_numeric(demandDs['MW'],errors='coerce')
        demandDs['DATE-TIME'] = pd.to_datetime(demandDs['DATE-TIME'], errors='coerce')

        demandDs.dropna(subset=['MW','DATE-TIME'], inplace=True)
   

        demand = demandDs['MW'].values.reshape(-1,1)
        datetime_series = demandDs['DATE-TIME']

        hour = datetime_series.dt.hour.values.reshape(-1,1)
        dayofweek = datetime_series.dt.dayofweek.values.reshape(-1,1)
        month = datetime_series.dt.month.values.reshape(-1,1)

        scaler_demand = MinMaxScaler()
        scaler_hour = MinMaxScaler()
        scaler_day = MinMaxScaler()
        scaler_month = MinMaxScaler()

        demand_scaled = scaler_demand.fit_transform(demand)
        hour_scaled = scaler_hour.fit_transform(hour)
        day_scaled = scaler_day.fit_transform(dayofweek)
        month_scaled = scaler_month.fit_transform(month)

        full_data = np.hstack((demand_scaled, hour_scaled, day_scaled, month_scaled))
        targets = (demand_scaled)

        seq_length = 24
        X, y = create_sequences_with_time_flatten(full_data, targets, seq_length)


        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        param_grid = {
            'estimator__n_estimators': [50, 100],
            'estimator__max_depth': [5, 10, None]
        }
        base_model = RandomForestRegressor(random_state=42)
        model = MultiOutputRegressor(base_model)
        grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_demand_pred = scaler_demand.inverse_transform(y_pred[:, 0].reshape(-1, 1))
        y_demand_true = scaler_demand.inverse_transform(y_test[:, 0].reshape(-1, 1))


        mse = mean_squared_error(y_demand_true, y_demand_pred)
        mae = mean_absolute_error(y_demand_true, y_demand_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_demand_true, y_demand_pred)
        modelName = "Random Forest"
        results = [mse, mae, rmse, r2,modelName]
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
        #----#
        print('\n',"Demand Random Forest Model Results:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))


        return results
    elif (ident == 2):

        supplyDs['Date & Time'] = supplyDs['Date & Time'].astype(str)
        supplyDs[['Date','Time']] = supplyDs['Date & Time'].str.split(' ',expand=True)
        supplyDs ['MW'] = pd.to_numeric(supplyDs['MW'], errors='coerce')
        supplyDs['Date & Time'] = pd.to_datetime(supplyDs['Date & Time'], errors='coerce')

        supplyDs.dropna(subset=['MW','Date & Time'], inplace=True)

        supply = supplyDs['MW'].values.reshape(-1,1)
        datetime_series = supplyDs['Date & Time']

        hour = datetime_series.dt.hour.values.reshape(-1,1)
        dayofweek = datetime_series.dt.dayofweek.values.reshape(-1,1)
        month = datetime_series.dt.month.values.reshape(-1,1)

        scaler_supply = MinMaxScaler()
        scaler_hour = MinMaxScaler()
        scaler_day = MinMaxScaler()
        scaler_month = MinMaxScaler()

        supply_scaled = scaler_supply.fit_transform(supply)
        hour_scaled = scaler_hour.fit_transform(hour)
        day_scaled = scaler_day.fit_transform(dayofweek)
        month_scaled = scaler_month.fit_transform(month)

        full_data = np.hstack((supply_scaled, hour_scaled, day_scaled, month_scaled))
        targets = (supply_scaled)

        seq_length = 24
        X, y = create_sequences_with_time_flatten(full_data, targets, seq_length)


        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        param_grid = {
            'estimator__n_estimators': [50, 100],
            'estimator__max_depth': [5, 10, None]
        }
        base_model = RandomForestRegressor(random_state=42)
        model = MultiOutputRegressor(base_model)
        grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_supply_pred = scaler_supply.inverse_transform(y_pred[:, 0].reshape(-1, 1))
        y_supply_true = scaler_supply.inverse_transform(y_test[:, 0].reshape(-1, 1))


        mse = mean_squared_error(y_supply_true, y_supply_pred)
        mae = mean_absolute_error(y_supply_true, y_supply_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_supply_true, y_supply_pred)
        modelName = "Random Forest"
        results = [mse, mae, rmse, r2,modelName]

        #----#
        print('\n', "Random Forest Model for Supply Data:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
        if plots == True:
            plt.figure(figsize=(10, 5))
            plt.plot(y_supply_true, label='Actual')
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

            plot_series(y_supply_true, y_supply_pred, 'Supply')
        return results
    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return


def xgbModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int, plots:bool):

    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy()   
    """
    Demand XGB Results -
    MSE: 40.8531
    MAE: 4.4390
    RMSE: 6.3916
    R2: 0.9450
    -----
    Supply XGB Results - 
    MSE: 1482.6631
    MAE: 19.7854
    RMSE: 38.5054
    R2: 0.8640
    """
    if (ident == 1):
    #----#        
        demandDs['DATE-TIME'] = demandDs['DATE-TIME'].astype(str)
        demandDs[['Date', 'Time']] = demandDs['DATE-TIME'].str.split(' ',expand=True)
        demandDs ['MW'] = pd.to_numeric(demandDs['MW'],errors='coerce')
        demandDs['DATE-TIME'] = pd.to_datetime(demandDs['DATE-TIME'], errors='coerce')

        demandDs.dropna(subset=['MW','DATE-TIME'], inplace=True)
   

        demand = demandDs['MW'].values.reshape(-1,1)
        datetime_series = demandDs['DATE-TIME']

        hour = datetime_series.dt.hour.values.reshape(-1,1)
        dayofweek = datetime_series.dt.dayofweek.values.reshape(-1,1)
        month = datetime_series.dt.month.values.reshape(-1,1)

        scaler_demand = MinMaxScaler()
        scaler_hour = MinMaxScaler()
        scaler_day = MinMaxScaler()
        scaler_month = MinMaxScaler()

        demand_scaled = scaler_demand.fit_transform(demand)
        hour_scaled = scaler_hour.fit_transform(hour)
        day_scaled = scaler_day.fit_transform(dayofweek)
        month_scaled = scaler_month.fit_transform(month)

        full_data = np.hstack((demand_scaled, hour_scaled, day_scaled, month_scaled))
        targets = (demand_scaled)

        seq_length = 24
        X, y = create_sequences_with_time_flatten(full_data, targets, seq_length)


        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        param_grid = {
            'estimator__n_estimators': [50, 100],
            'estimator__max_depth': [3, 5],
            'estimator__learning_rate': [0.05, 0.1]
        }
        base_model = xgb.XGBRegressor(random_state=42)
        model = MultiOutputRegressor(base_model)
        grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_demand_pred = scaler_demand.inverse_transform(y_pred[:, 0].reshape(-1, 1))
        y_demand_true = scaler_demand.inverse_transform(y_test[:, 0].reshape(-1, 1))


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

        return results
    elif (ident == 2):

        supplyDs['Date & Time'] = supplyDs['Date & Time'].astype(str)
        supplyDs[['Date','Time']] = supplyDs['Date & Time'].str.split(' ',expand=True)
        supplyDs ['MW'] = pd.to_numeric(supplyDs['MW'], errors='coerce')
        supplyDs['Date & Time'] = pd.to_datetime(supplyDs['Date & Time'], errors='coerce')

        supplyDs.dropna(subset=['MW','Date & Time'], inplace=True)

        supply = supplyDs['MW'].values.reshape(-1,1)
        datetime_series = supplyDs['Date & Time']

        hour = datetime_series.dt.hour.values.reshape(-1,1)
        dayofweek = datetime_series.dt.dayofweek.values.reshape(-1,1)
        month = datetime_series.dt.month.values.reshape(-1,1)

        scaler_supply = MinMaxScaler()
        scaler_hour = MinMaxScaler()
        scaler_day = MinMaxScaler()
        scaler_month = MinMaxScaler()

        supply_scaled = scaler_supply.fit_transform(supply)
        hour_scaled = scaler_hour.fit_transform(hour)
        day_scaled = scaler_day.fit_transform(dayofweek)
        month_scaled = scaler_month.fit_transform(month)

        full_data = np.hstack((supply_scaled, hour_scaled, day_scaled, month_scaled))
        targets = (supply_scaled)

        seq_length = 24
        X, y = create_sequences_with_time_flatten(full_data, targets, seq_length)


        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        param_grid = {
            'estimator__n_estimators': [50, 100],
            'estimator__max_depth': [3, 5],
            'estimator__learning_rate': [0.05, 0.1]
        }
        base_model = xgb.XGBRegressor(random_state=42)
        model = MultiOutputRegressor(base_model)
        grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_supply_pred = scaler_supply.inverse_transform(y_pred[:, 0].reshape(-1, 1))
        y_supply_true = scaler_supply.inverse_transform(y_test[:, 0].reshape(-1, 1))


        mse = mean_squared_error(y_supply_true, y_supply_pred)
        mae = mean_absolute_error(y_supply_true, y_supply_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_supply_true, y_supply_pred)
        modelName = "XGB"
        results = [mse, mae, rmse, r2,modelName]

        #----#
        print('\n', "XGB Model for Supply Data:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
        if plots == True:
            plt.figure(figsize=(10, 5))
            plt.plot(y_supply_true, label='Actual')
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

            plot_series(y_supply_true, y_supply_pred, 'Supply')

        return results
    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return

def gbModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int, plots:bool):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy() 
    """
    Demand GradientBoosting Results -
    MSE: 62.5690
    MAE: 5.4002
    RMSE: 7.9101
    R2: 0.9158
    -----
    Supply GradientBoosting Results - 
    MSE: 1056.1014
    MAE: 16.0954
    RMSE: 32.4977
    R2: 0.9031
    """
    if (ident == 1):
    #----#        
        demandDs['DATE-TIME'] = demandDs['DATE-TIME'].astype(str)
        demandDs[['Date', 'Time']] = demandDs['DATE-TIME'].str.split(' ',expand=True)
        demandDs ['MW'] = pd.to_numeric(demandDs['MW'],errors='coerce')
        demandDs['DATE-TIME'] = pd.to_datetime(demandDs['DATE-TIME'], errors='coerce')

        demandDs.dropna(subset=['MW','DATE-TIME'], inplace=True)
   

        demand = demandDs['MW'].values.reshape(-1,1)
        datetime_series = demandDs['DATE-TIME']

        hour = datetime_series.dt.hour.values.reshape(-1,1)
        dayofweek = datetime_series.dt.dayofweek.values.reshape(-1,1)
        month = datetime_series.dt.month.values.reshape(-1,1)

        scaler_demand = MinMaxScaler()
        scaler_hour = MinMaxScaler()
        scaler_day = MinMaxScaler()
        scaler_month = MinMaxScaler()

        demand_scaled = scaler_demand.fit_transform(demand)
        hour_scaled = scaler_hour.fit_transform(hour)
        day_scaled = scaler_day.fit_transform(dayofweek)
        month_scaled = scaler_month.fit_transform(month)

        full_data = np.hstack((demand_scaled, hour_scaled, day_scaled, month_scaled))
        targets = (demand_scaled)

        seq_length = 24
        X, y = create_sequences_with_time_flatten(full_data, targets, seq_length)


        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        param_grid = {
            'estimator__n_estimators': [50, 100],
            'estimator__max_depth': [3, 5],
            'estimator__learning_rate': [0.05, 0.1]
        }
        base_model = GradientBoostingRegressor(random_state=42)
        model = MultiOutputRegressor(base_model)
        grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_demand_pred = scaler_demand.inverse_transform(y_pred[:, 0].reshape(-1, 1))
        y_demand_true = scaler_demand.inverse_transform(y_test[:, 0].reshape(-1, 1))


        mse = mean_squared_error(y_demand_true, y_demand_pred)
        mae = mean_absolute_error(y_demand_true, y_demand_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_demand_true, y_demand_pred)
        modelName = "Gradient Boost"
        results = [mse, mae, rmse, r2,modelName]

        #----#
        print('\n',"Demand GB Model Results:")
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

        return results
    elif (ident == 2):

        supplyDs['Date & Time'] = supplyDs['Date & Time'].astype(str)
        supplyDs[['Date','Time']] = supplyDs['Date & Time'].str.split(' ',expand=True)
        supplyDs ['MW'] = pd.to_numeric(supplyDs['MW'], errors='coerce')
        supplyDs['Date & Time'] = pd.to_datetime(supplyDs['Date & Time'], errors='coerce')

        supplyDs.dropna(subset=['MW','Date & Time'], inplace=True)

        supply = supplyDs['MW'].values.reshape(-1,1)
        datetime_series = supplyDs['Date & Time']

        hour = datetime_series.dt.hour.values.reshape(-1,1)
        dayofweek = datetime_series.dt.dayofweek.values.reshape(-1,1)
        month = datetime_series.dt.month.values.reshape(-1,1)

        scaler_supply = MinMaxScaler()
        scaler_hour = MinMaxScaler()
        scaler_day = MinMaxScaler()
        scaler_month = MinMaxScaler()

        supply_scaled = scaler_supply.fit_transform(supply)
        hour_scaled = scaler_hour.fit_transform(hour)
        day_scaled = scaler_day.fit_transform(dayofweek)
        month_scaled = scaler_month.fit_transform(month)

        full_data = np.hstack((supply_scaled, hour_scaled, day_scaled, month_scaled))
        targets = (supply_scaled)

        seq_length = 24
        X, y = create_sequences_with_time_flatten(full_data, targets, seq_length)


        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        param_grid = {
            'estimator__n_estimators': [50, 100],
            'estimator__max_depth': [3, 5],
            'estimator__learning_rate': [0.05, 0.1]
        }
        base_model = GradientBoostingRegressor(random_state=42)
        model = MultiOutputRegressor(base_model)
        grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_supply_pred = scaler_supply.inverse_transform(y_pred[:, 0].reshape(-1, 1))
        y_supply_true = scaler_supply.inverse_transform(y_test[:, 0].reshape(-1, 1))


        mse = mean_squared_error(y_supply_true, y_supply_pred)
        mae = mean_absolute_error(y_supply_true, y_supply_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_supply_true, y_supply_pred)
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
            plt.plot(y_supply_true, label='Actual')
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

            plot_series(y_supply_true, y_supply_pred, 'Supply')
        return results
    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return

def biDirectionalLSTMDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int, plots:bool):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy()    
    """
    Demand Bidirectional LSTM Results -
    MSE: 54.4676
    MAE: 5.2916
    RMSE: 7.3802
    R2: 0.9267
    -----
    Supply Bidirectional LSTM Results - 
    MSE: 938.3314
    MAE: 16.0176
    RMSE: 30.6322
    R2: 0.9139
    """
    if (ident == 1):
    #----#        
    ##Basic transformation for the base dataset
        demandDs['DATE-TIME'] = demandDs['DATE-TIME'].astype(str)
        demandDs[['Date', 'Time']] = demandDs['DATE-TIME'].str.split(' ',expand=True)
        demandDs ['MW'] = pd.to_numeric(demandDs['MW'],errors='coerce')
        demandDs['DATE-TIME'] = pd.to_datetime(demandDs['DATE-TIME'], errors='coerce')
        demandDs.dropna(subset=['MW','DATE-TIME'], inplace=True)
   
        ##Declaration of datetime and reshaping of the MW column
        demand = demandDs['MW'].values.reshape(-1,1)
        datetime_series = demandDs['DATE-TIME']

        hour = datetime_series.dt.hour.values.reshape(-1,1)
        dayofweek = datetime_series.dt.dayofweek.values.reshape(-1,1)
        month = datetime_series.dt.month.values.reshape(-1,1)
        ##Feature Declaration for fitting later on.
        features = np.hstack([
        demandDs['MW'].values.reshape(-1, 1),
        hour.reshape(-1, 1),
        dayofweek.reshape(-1, 1),
        month.reshape(-1, 1)
        ])

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_scaled = scaler_X.fit_transform(features)
        targets = demand

      
        
        X_scaled = scaler_X.fit_transform(features)
        y_scaled = scaler_y.fit_transform(targets)

        # Suppose seq_length = 24, num_features = 4 (MW, hour, dayofweek, month)
        


        seq_length = 24
        X, y = create_sequences_with_time_3d(X_scaled, y_scaled, seq_length)


        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        best_score = float('inf')
        best_params = None
        best_model = None
        param_grid = {
            'units': [64, 100],
            'dropout': [0.2, 0.3]
        }
        for units in param_grid['units']:
            for dropout in param_grid['dropout']:
                model = Sequential([
                    Bidirectional(LSTM(units, return_sequences=True, input_shape=(seq_length, X_train.shape[2]))),
                    Dropout(dropout),
                    Bidirectional(LSTM(units, return_sequences=False)),
                    Dense(64, activation='relu'),
                    Dense(1)
                ])
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')
                early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test),
                            callbacks=[early_stopping], verbose=0)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                if mse < best_score:
                    best_score = mse
                    best_params = (units, dropout)
                    best_model = model

        y_pred = best_model.predict(X_test)
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

        return results
    elif (ident == 2):

        supplyDs['Date & Time'] = supplyDs['Date & Time'].astype(str)
        supplyDs[['Date','Time']] = supplyDs['Date & Time'].str.split(' ',expand=True)
        supplyDs ['MW'] = pd.to_numeric(supplyDs['MW'], errors='coerce')
        supplyDs['Date & Time'] = pd.to_datetime(supplyDs['Date & Time'], errors='coerce')

        supplyDs.dropna(subset=['MW','Date & Time'], inplace=True)

        supply = supplyDs['MW'].values.reshape(-1,1)
        datetime_series = supplyDs['Date & Time']

        hour = datetime_series.dt.hour.values.reshape(-1,1)
        dayofweek = datetime_series.dt.dayofweek.values.reshape(-1,1)
        month = datetime_series.dt.month.values.reshape(-1,1)

        features = np.hstack([
        supplyDs['MW'].values.reshape(-1, 1),
        hour.reshape(-1, 1),
        dayofweek.reshape(-1, 1),
        month.reshape(-1, 1)
        ])

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_scaled = scaler_X.fit_transform(features)
        targets = supply

      
        
        X_scaled = scaler_X.fit_transform(features)
        y_scaled = scaler_y.fit_transform(targets)

        # Suppose seq_length = 24, num_features = 4 (MW, hour, dayofweek, month)
        


        seq_length = 24
        X, y = create_sequences_with_time_3d(X_scaled, y_scaled, seq_length)


        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        best_score = float('inf')
        best_params = None
        best_model = None
        param_grid = {
            'units': [64, 100],
            'dropout': [0.2, 0.3]
        }
        for units in param_grid['units']:
            for dropout in param_grid['dropout']:
                model = Sequential([
                    Bidirectional(LSTM(units, return_sequences=True, input_shape=(seq_length, X_train.shape[2]))),
                    Dropout(dropout),
                    Bidirectional(LSTM(units, return_sequences=False)),
                    Dense(64, activation='relu'),
                    Dense(1)
                ])
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')
                early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test),
                            callbacks=[early_stopping], verbose=0)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                if mse < best_score:
                    best_score = mse
                    best_params = (units, dropout)
                    best_model = model

        y_pred = best_model.predict(X_test)
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
        return results
    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return

def LSTMModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int, plots:bool):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy()    
    """
    Demand LSTM Results -
    MSE: 73.2593
    MAE: 6.3673
    RMSE: 8.5592
    R2: 0.9014
    -----
    Supply LSTM Results - 
    MSE: 1048.5297
    MAE: 17.9746
    RMSE: 32.3810
    R2: 0.9038
    """
    if (ident == 1):
    #----#        
    ##Basic transformation for the base dataset
        demandDs['DATE-TIME'] = demandDs['DATE-TIME'].astype(str)
        demandDs[['Date', 'Time']] = demandDs['DATE-TIME'].str.split(' ',expand=True)
        demandDs ['MW'] = pd.to_numeric(demandDs['MW'],errors='coerce')
        demandDs['DATE-TIME'] = pd.to_datetime(demandDs['DATE-TIME'], errors='coerce')
        demandDs.dropna(subset=['MW','DATE-TIME'], inplace=True)
   
        ##Declaration of datetime and reshaping of the MW column
        demand = demandDs['MW'].values.reshape(-1,1)
        datetime_series = demandDs['DATE-TIME']

        hour = datetime_series.dt.hour.values.reshape(-1,1)
        dayofweek = datetime_series.dt.dayofweek.values.reshape(-1,1)
        month = datetime_series.dt.month.values.reshape(-1,1)
        ##Feature Declaration for fitting later on.
        features = np.hstack([
        demandDs['MW'].values.reshape(-1, 1),
        hour.reshape(-1, 1),
        dayofweek.reshape(-1, 1),
        month.reshape(-1, 1)
        ])

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_scaled = scaler_X.fit_transform(features)
        targets = demand

      
        
        X_scaled = scaler_X.fit_transform(features)
        y_scaled = scaler_y.fit_transform(targets)

        # Suppose seq_length = 24, num_features = 4 (MW, hour, dayofweek, month)
        


        seq_length = 24
        X, y = create_sequences_with_time_3d(X_scaled, y_scaled, seq_length)


        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        best_score = float('inf')
        best_params = None
        best_model = None
        param_grid = {
            'units': [64, 100],
            'dropout': [0.2, 0.3]
        }
        for units in param_grid['units']:
            for dropout in param_grid['dropout']:
                model = Sequential([
                    LSTM(units, return_sequences=True, input_shape=(seq_length, X_train.shape[2])),
                    Dropout(dropout),
                    LSTM(units, return_sequences=False),
                    Dense(64),
                    Dense(1)
                ])
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')
                early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test),
                          callbacks=[early_stopping], verbose=0)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                if mse < best_score:
                    best_score = mse
                    best_params = (units, dropout)
                    best_model = model

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
                plt.title(f'LSTM: Actual vs Predicted {label}')
                plt.xlabel('Time')
                plt.ylabel('MW')
                plt.legend()
                plt.tight_layout()
                plt.show()

            plot_series(y_demand_true, y_demand_pred, 'Demand')

        return results
    elif (ident == 2):

        supplyDs['Date & Time'] = supplyDs['Date & Time'].astype(str)
        supplyDs[['Date','Time']] = supplyDs['Date & Time'].str.split(' ',expand=True)
        supplyDs ['MW'] = pd.to_numeric(supplyDs['MW'], errors='coerce')
        supplyDs['Date & Time'] = pd.to_datetime(supplyDs['Date & Time'], errors='coerce')

        supplyDs.dropna(subset=['MW','Date & Time'], inplace=True)

        supply = supplyDs['MW'].values.reshape(-1,1)
        datetime_series = supplyDs['Date & Time']

        hour = datetime_series.dt.hour.values.reshape(-1,1)
        dayofweek = datetime_series.dt.dayofweek.values.reshape(-1,1)
        month = datetime_series.dt.month.values.reshape(-1,1)

        features = np.hstack([
        supplyDs['MW'].values.reshape(-1, 1),
        hour.reshape(-1, 1),
        dayofweek.reshape(-1, 1),
        month.reshape(-1, 1)
        ])

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_scaled = scaler_X.fit_transform(features)
        targets = supply

      
        
        X_scaled = scaler_X.fit_transform(features)
        y_scaled = scaler_y.fit_transform(targets)

        # Suppose seq_length = 24, num_features = 4 (MW, hour, dayofweek, month)
        


        seq_length = 24
        X, y = create_sequences_with_time_3d(X_scaled, y_scaled, seq_length)


        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        best_score = float('inf')
        best_params = None
        best_model = None
        param_grid = {
            'units': [64, 100],
            'dropout': [0.2, 0.3]
        }
        for units in param_grid['units']:
            for dropout in param_grid['dropout']:
                model = Sequential([
                    LSTM(units, return_sequences=True, input_shape=(seq_length, X_train.shape[2])),
                    Dropout(dropout),
                    LSTM(units, return_sequences=False),
                    Dense(64),
                    Dense(1)
                ])
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')
                early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test),
                          callbacks=[early_stopping], verbose=0)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                if mse < best_score:
                    best_score = mse
                    best_params = (units, dropout)
                    best_model = model
        
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
        return results 
    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return

def GRUModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int, plots:bool):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy()    
    """
    Demand GRU Results -
    MSE: 84.4424
    MAE: 6.0762
    RMSE: 9.1893
    R2: 0.8864
    -----
    Supply GRU Results - 
    MSE: 992.8610
    MAE: 18.6059
    RMSE: 31.5097
    R2: 0.9089
    """
    if (ident == 1):
    #----#        
    ##Basic transformation for the base dataset
        demandDs['DATE-TIME'] = demandDs['DATE-TIME'].astype(str)
        demandDs[['Date', 'Time']] = demandDs['DATE-TIME'].str.split(' ',expand=True)
        demandDs ['MW'] = pd.to_numeric(demandDs['MW'],errors='coerce')
        demandDs['DATE-TIME'] = pd.to_datetime(demandDs['DATE-TIME'], errors='coerce')
        demandDs.dropna(subset=['MW','DATE-TIME'], inplace=True)
   
        ##Declaration of datetime and reshaping of the MW column
        demand = demandDs['MW'].values.reshape(-1,1)
        datetime_series = demandDs['DATE-TIME']

        hour = datetime_series.dt.hour.values.reshape(-1,1)
        dayofweek = datetime_series.dt.dayofweek.values.reshape(-1,1)
        month = datetime_series.dt.month.values.reshape(-1,1)
        ##Feature Declaration for fitting later on.
        features = np.hstack([
        demandDs['MW'].values.reshape(-1, 1),
        hour.reshape(-1, 1),
        dayofweek.reshape(-1, 1),
        month.reshape(-1, 1)
        ])

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_scaled = scaler_X.fit_transform(features)
        targets = demand

      
        
        X_scaled = scaler_X.fit_transform(features)
        y_scaled = scaler_y.fit_transform(targets)

        # Suppose seq_length = 24, num_features = 4 (MW, hour, dayofweek, month)
        


        seq_length = 24
        X, y = create_sequences_with_time_3d(X_scaled, y_scaled, seq_length)


        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        best_score = float('inf')
        best_params = None
        best_model = None
        param_grid = {
            'units': [64, 100],
            'dropout': [0.2, 0.3]
        }
        for units in param_grid['units']:
            for dropout in param_grid['dropout']:
                model = Sequential([
                    GRU(units, return_sequences=True, input_shape=(seq_length, X_train.shape[2])),
                    Dropout(dropout),
                    GRU(units, return_sequences=False),
                    Dense(64, activation='relu'),
                    Dense(1)
                ])
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')
                early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test),
                          callbacks=[early_stopping], verbose=0)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                if mse < best_score:
                    best_score = mse
                    best_params = (units, dropout)
                    best_model = model
        # Use best_model for final prediction
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

        return results 
    elif (ident == 2):

        supplyDs['Date & Time'] = supplyDs['Date & Time'].astype(str)
        supplyDs[['Date','Time']] = supplyDs['Date & Time'].str.split(' ',expand=True)
        supplyDs ['MW'] = pd.to_numeric(supplyDs['MW'], errors='coerce')
        supplyDs['Date & Time'] = pd.to_datetime(supplyDs['Date & Time'], errors='coerce')

        supplyDs.dropna(subset=['MW','Date & Time'], inplace=True)

        supply = supplyDs['MW'].values.reshape(-1,1)
        datetime_series = supplyDs['Date & Time']

        hour = datetime_series.dt.hour.values.reshape(-1,1)
        dayofweek = datetime_series.dt.dayofweek.values.reshape(-1,1)
        month = datetime_series.dt.month.values.reshape(-1,1)

        features = np.hstack([
        supplyDs['MW'].values.reshape(-1, 1),
        hour.reshape(-1, 1),
        dayofweek.reshape(-1, 1),
        month.reshape(-1, 1)
        ])

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        X_scaled = scaler_X.fit_transform(features)
        targets = supply

      
        
        X_scaled = scaler_X.fit_transform(features)
        y_scaled = scaler_y.fit_transform(targets)

        # Suppose seq_length = 24, num_features = 4 (MW, hour, dayofweek, month)
        


        seq_length = 24
        X, y = create_sequences_with_time_3d(X_scaled, y_scaled, seq_length)


        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        ##split_index = int(len(X_reshaped) * 0.8)
        ##X_train, X_test = X_reshaped[:split_index], X_reshaped[split_index:]
        ##y_train, y_test = y_scaled[:split_index], y_scaled[split_i
        
        best_score = float('inf')
        best_params = None
        best_model = None
        param_grid = {
            'units': [64, 100],
            'dropout': [0.2, 0.3]
        }
        for units in param_grid['units']:
            for dropout in param_grid['dropout']:
                model = Sequential([
                    GRU(units, return_sequences=True, input_shape=(seq_length, X_train.shape[2])),
                    Dropout(dropout),
                    GRU(units, return_sequences=False),
                    Dense(64, activation='relu'),
                    Dense(1)
                ])
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')
                early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test),
                          callbacks=[early_stopping], verbose=0)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                if mse < best_score:
                    best_score = mse
                    best_params = (units, dropout)
                    best_model = model
        # Use best_model for final prediction
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
        return results
    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return
##
def SVRModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int, plots:bool):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy()    
    """
    ... (docstring unchanged)
    """

    if (ident == 1):
        #----#        
        demandDs['DATE-TIME'] = demandDs['DATE-TIME'].astype(str)
        demandDs[['Date', 'Time']] = demandDs['DATE-TIME'].str.split(' ',expand=True)
        demandDs ['MW'] = pd.to_numeric(demandDs['MW'],errors='coerce')
        demandDs['DATE-TIME'] = pd.to_datetime(demandDs['DATE-TIME'], errors='coerce')
        demandDs.dropna(subset=['MW','DATE-TIME'], inplace=True)

        demand = demandDs['MW'].values.reshape(-1,1)
        datetime_series = demandDs['DATE-TIME']

        hour = datetime_series.dt.hour.values.reshape(-1,1)
        dayofweek = datetime_series.dt.dayofweek.values.reshape(-1,1)
        month = datetime_series.dt.month.values.reshape(-1,1)

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        demand_scaled = scaler_X.fit_transform(np.hstack([demand, hour, dayofweek, month]))
        hour_scaled = scaler_X.fit_transform(hour)
        day_scaled = scaler_X.fit_transform(dayofweek)
        month_scaled = scaler_X.fit_transform(month)

        # Declare full_data for sequence creation
        full_data = np.hstack((demand_scaled, hour_scaled, day_scaled, month_scaled))
        targets = scaler_y.fit_transform(demand)

        seq_length = 24
        X, y = create_sequences_with_time_flatten(full_data, targets, seq_length)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3,shuffle=False
        )

        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1],
            'epsilon': [0.01, 0.1, 0.5, 1.0],
            'kernel': ['rbf']
        }
        model = SVR(kernel='rbf')
        grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train.ravel())
        model = grid.best_estimator_

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_demand_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
        y_demand_true = scaler_y.inverse_transform(y_test.reshape(-1, 1))

        mse = mean_squared_error(y_demand_true, y_demand_pred)
        mae = mean_absolute_error(y_demand_true, y_demand_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_demand_true, y_demand_pred)
        modelName = "SVR"
        results = [mse, mae, rmse, r2, modelName]

        print('\n', "Demand SVR Model Results:")
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

        return results 
    elif (ident == 2):
        supplyDs['Date & Time'] = supplyDs['Date & Time'].astype(str)
        supplyDs[['Date','Time']] = supplyDs['Date & Time'].str.split(' ',expand=True)
        supplyDs ['MW'] = pd.to_numeric(supplyDs['MW'], errors='coerce')
        supplyDs['Date & Time'] = pd.to_datetime(supplyDs['Date & Time'], errors='coerce')
        supplyDs.dropna(subset=['MW','Date & Time'], inplace=True)

        supply = supplyDs['MW'].values.reshape(-1,1)
        datetime_series = supplyDs['Date & Time']

        hour = datetime_series.dt.hour.values.reshape(-1,1)
        dayofweek = datetime_series.dt.dayofweek.values.reshape(-1,1)
        month = datetime_series.dt.month.values.reshape(-1,1)

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        supply_scaled = scaler_X.fit_transform(np.hstack([supply, hour, dayofweek, month]))
        hour_scaled = scaler_X.fit_transform(hour)
        day_scaled = scaler_X.fit_transform(dayofweek)
        month_scaled = scaler_X.fit_transform(month)

        # Declare full_data for sequence creation
        full_data = np.hstack((supply_scaled, hour_scaled, day_scaled, month_scaled))
        targets = scaler_y.fit_transform(supply)

        seq_length = 24
        X, y = create_sequences_with_time_flatten(full_data, targets, seq_length)


        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3,shuffle=False
        )

        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1],
            'epsilon': [0.01, 0.1, 0.5, 1.0],
            'kernel': ['rbf']
        }
        model = SVR(kernel='rbf')
        grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train.ravel())
        model = grid.best_estimator_

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_supply_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
        y_supply_true = scaler_y.inverse_transform(y_test.reshape(-1, 1))

        mse = mean_squared_error(y_supply_true, y_supply_pred)
        mae = mean_absolute_error(y_supply_true, y_supply_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_supply_true, y_supply_pred)
        modelName = "SVR"
        results = [mse, mae, rmse, r2, modelName]

        print('\n', "Supply SVR Model Results:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
        if plots == True:
            plt.figure(figsize=(10, 5))
            plt.plot(y_supply_true, label='Actual')
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

            plot_series(y_supply_true, y_supply_pred, 'Demand')

        return results
    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return

def MLPModelDS(demandDs: pd.DataFrame, supplyDs: pd.DataFrame, ident: int, plots:bool):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy()
    """
    Demand MLP Results -
    MSE: 148.2202
    MAE: 10.5575
    RMSE: 12.1746
    R2: 0.8006
    -----
    Supply MLP Results - 
    MSE: 1193.7714
    MAE: 19.9199
    RMSE: 34.5510
    R2: 0.8905
    """
    if ident == 1:
        #----#        
        # Basic transformation for the base dataset
        demandDs['DATE-TIME'] = demandDs['DATE-TIME'].astype(str)
        demandDs[['Date', 'Time']] = demandDs['DATE-TIME'].str.split(' ', expand=True)
        demandDs['MW'] = pd.to_numeric(demandDs['MW'], errors='coerce')
        demandDs['DATE-TIME'] = pd.to_datetime(demandDs['DATE-TIME'], errors='coerce')
        demandDs.dropna(subset=['MW', 'DATE-TIME'], inplace=True)

        # Declaration of datetime and reshaping of the MW column
        demand = demandDs['MW'].values.reshape(-1, 1)
        datetime_series = demandDs['DATE-TIME']

        hour = datetime_series.dt.hour.values.reshape(-1, 1)
        dayofweek = datetime_series.dt.dayofweek.values.reshape(-1, 1)
        month = datetime_series.dt.month.values.reshape(-1, 1)

        scaler_demand = MinMaxScaler()
        scaler_hour = MinMaxScaler()
        scaler_day = MinMaxScaler()
        scaler_month = MinMaxScaler()

        demand_scaled = scaler_demand.fit_transform(demand)
        hour_scaled = scaler_hour.fit_transform(hour)
        day_scaled = scaler_day.fit_transform(dayofweek)
        month_scaled = scaler_month.fit_transform(month)

        # Declare full_data for sequence creation
        full_data = np.hstack((demand_scaled, hour_scaled, day_scaled, month_scaled))
        targets = demand_scaled

        seq_length = 24
        X, y = create_sequences_with_time_flatten(full_data, targets, seq_length)
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        model = Sequential([
            Dense(128, activation='relu', input_dim=X_train.shape[1]),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')

        model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))

        y_pred = model.predict(X_test)

        y_demand_pred = scaler_demand.inverse_transform(y_pred)
        y_demand_true = scaler_demand.inverse_transform(y_test)

        mse = mean_squared_error(y_demand_true, y_demand_pred)
        mae = mean_absolute_error(y_demand_true, y_demand_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_demand_true, y_demand_pred)
        modelName = "MLP"
        results = [mse, mae, rmse, r2, modelName]

        print('\n', "Demand MLP Model Results:")
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

        return results

    elif ident == 2:
        supplyDs['Date & Time'] = supplyDs['Date & Time'].astype(str)
        supplyDs[['Date', 'Time']] = supplyDs['Date & Time'].str.split(' ', expand=True)
        supplyDs['MW'] = pd.to_numeric(supplyDs['MW'], errors='coerce')
        supplyDs['Date & Time'] = pd.to_datetime(supplyDs['Date & Time'], errors='coerce')
        supplyDs.dropna(subset=['MW', 'Date & Time'], inplace=True)

        supply = supplyDs['MW'].values.reshape(-1, 1)
        datetime_series = supplyDs['Date & Time']

        hour = datetime_series.dt.hour.values.reshape(-1, 1)
        dayofweek = datetime_series.dt.dayofweek.values.reshape(-1, 1)
        month = datetime_series.dt.month.values.reshape(-1, 1)

        scaler_supply = MinMaxScaler()
        scaler_hour = MinMaxScaler()
        scaler_day = MinMaxScaler()
        scaler_month = MinMaxScaler()

        supply_scaled = scaler_supply.fit_transform(supply)
        hour_scaled = scaler_hour.fit_transform(hour)
        day_scaled = scaler_day.fit_transform(dayofweek)
        month_scaled = scaler_month.fit_transform(month)

        # Declare full_data for sequence creation
        full_data = np.hstack((supply_scaled, hour_scaled, day_scaled, month_scaled))
        targets = supply_scaled

        seq_length = 24
        X, y = create_sequences_with_time_flatten(full_data, targets, seq_length)
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        model = Sequential([
            Dense(128, activation='relu', input_dim=X_train.shape[1]),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')

        model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))

        y_pred = model.predict(X_test)

        y_supply_pred = scaler_supply.inverse_transform(y_pred)
        y_supply_true = scaler_supply.inverse_transform(y_test)

        mse = mean_squared_error(y_supply_true, y_supply_pred)
        mae = mean_absolute_error(y_supply_true, y_supply_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_supply_true, y_supply_pred)
        modelName = "MLP"
        results = [mse, mae, rmse, r2, modelName]

        print('\n', "MLP Model for Supply Data:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
        if plots == True:
            plt.figure(figsize=(10, 5))
            plt.plot(y_supply_true, label='Actual')
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

            plot_series(y_supply_true, y_supply_pred, 'Supply')
        return results

    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return

def CNNModelDS(demandDs: pd.DataFrame, supplyDs: pd.DataFrame, ident: int, plots:bool):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy() 
    """
    Demand CNN Results -
    MSE: 683.9287
    MAE: 20.9247
    RMSE: 26.1520
    R2: 0.0798
    -----
    Supply CNN Results - 
    MSE: 1283.5096
    MAE: 19.5253
    RMSE: 35.8261
    R2: 0.8823
    """

    if ident == 1:
        #----#        
        demandDs['DATE-TIME'] = demandDs['DATE-TIME'].astype(str)
        demandDs[['Date', 'Time']] = demandDs['DATE-TIME'].str.split(' ', expand=True)
        demandDs['MW'] = pd.to_numeric(demandDs['MW'], errors='coerce')
        demandDs['DATE-TIME'] = pd.to_datetime(demandDs['DATE-TIME'], errors='coerce')
        demandDs.dropna(subset=['MW', 'DATE-TIME'], inplace=True)

        demand = demandDs['MW'].values.reshape(-1, 1)
        datetime_series = demandDs['DATE-TIME']

        hour = datetime_series.dt.hour.values.reshape(-1, 1)
        dayofweek = datetime_series.dt.dayofweek.values.reshape(-1, 1)
        month = datetime_series.dt.month.values.reshape(-1, 1)

        features = np.hstack([
            demandDs['MW'].values.reshape(-1, 1),
            hour,
            dayofweek,
            month
        ])

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X_scaled = scaler_X.fit_transform(features)
        y_scaled = scaler_y.fit_transform(demand)

        seq_length = 24
        X, y = create_sequences_with_time_3d(X_scaled, y_scaled, seq_length)

        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        best_score = float('inf')
        best_params = None
        best_model = None
        param_grid = {
            'filters': [32, 64],
            'dropout': [0.2, 0.3]
        }
        for filters in param_grid['filters']:
            for dropout in param_grid['dropout']:
                model = Sequential([
                    Conv1D(filters, kernel_size=3, activation='relu', input_shape=(seq_length, X_train.shape[2])),
                    MaxPooling1D(pool_size=2),
                    Dropout(dropout),
                    Conv1D(filters*2, kernel_size=3, activation='relu'),
                    Dropout(dropout),
                    Flatten(),
                    Dense(32, activation='relu'),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test), verbose=0)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                if mse < best_score:
                    best_score = mse
                    best_params = (filters, dropout)
                    best_model = model
        # Use best_model for final prediction
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

        return results

    elif ident == 2:
        supplyDs['Date & Time'] = supplyDs['Date & Time'].astype(str)
        supplyDs[['Date', 'Time']] = supplyDs['Date & Time'].str.split(' ', expand=True)
        supplyDs['MW'] = pd.to_numeric(supplyDs['MW'], errors='coerce')
        supplyDs['Date & Time'] = pd.to_datetime(supplyDs['Date & Time'], errors='coerce')
        supplyDs.dropna(subset=['MW', 'Date & Time'], inplace=True)

        supply = supplyDs['MW'].values.reshape(-1, 1)
        datetime_series = supplyDs['Date & Time']

        hour = datetime_series.dt.hour.values.reshape(-1, 1)
        dayofweek = datetime_series.dt.dayofweek.values.reshape(-1, 1)
        month = datetime_series.dt.month.values.reshape(-1, 1)

        features = np.hstack([
            supplyDs['MW'].values.reshape(-1, 1),
            hour,
            dayofweek,
            month
        ])

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X_scaled = scaler_X.fit_transform(features)
        y_scaled = scaler_y.fit_transform(supply)

        seq_length = 24
        X, y = create_sequences_with_time_3d(X_scaled, y_scaled, seq_length)

        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        best_score = float('inf')
        best_params = None
        best_model = None
        param_grid = {
            'filters': [32, 64],
            'dropout': [0.2, 0.3]
        }
        for filters in param_grid['filters']:
            for dropout in param_grid['dropout']:
                model = Sequential([
                    Conv1D(filters, kernel_size=3, activation='relu', input_shape=(seq_length, X_train.shape[2])),
                    MaxPooling1D(pool_size=2),
                    Dropout(dropout),
                    Conv1D(filters*2, kernel_size=3, activation='relu'),
                    Dropout(dropout),
                    Flatten(),
                    Dense(32, activation='relu'),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test), verbose=0)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                if mse < best_score:
                    best_score = mse
                    best_params = (filters, dropout)
                    best_model = model
        # Use best_model for final prediction
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
        print('\n', "CNN Model for Supply Data:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
        if plots == True:
            plt.figure(figsize=(10, 5))
            plt.plot(y_demand_true, label='Actual')
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

            plot_series(y_demand_true, y_demand_pred, 'Supply')


        return results

    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return
    

def GBDTModelDS(demandDs: pd.DataFrame, supplyDs: pd.DataFrame, ident: int, plots:bool):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy()
    """
    Demand GBDT Results -
    MSE: 0.2395
    MAE: 0.3767
    RMSE: 0.4894
    R2: 0.9997
    -----
    Supply GBDT Results - 
    MSE: 1.0965
    MAE: 0.1957
    RMSE: 1.0471
    R2: 0.9999
    """

    if ident == 1:
        #----#
        demandDs['DATE-TIME'] = demandDs['DATE-TIME'].astype(str)
        demandDs[['Date', 'Time']] = demandDs['DATE-TIME'].str.split(' ', expand=True)
        demandDs['MW'] = pd.to_numeric(demandDs['MW'], errors='coerce')
        demandDs['DATE-TIME'] = pd.to_datetime(demandDs['DATE-TIME'], errors='coerce')
        demandDs.dropna(subset=['MW', 'DATE-TIME'], inplace=True)

        demand = demandDs['MW'].values.reshape(-1, 1)
        datetime_series = demandDs['DATE-TIME']

        hour = datetime_series.dt.hour.values.reshape(-1, 1)
        dayofweek = datetime_series.dt.dayofweek.values.reshape(-1, 1)
        month = datetime_series.dt.month.values.reshape(-1, 1)

        scaler_demand = MinMaxScaler()
        scaler_hour = MinMaxScaler()
        scaler_day = MinMaxScaler()
        scaler_month = MinMaxScaler()

        demand_scaled = scaler_demand.fit_transform(demand)
        hour_scaled = scaler_hour.fit_transform(hour)
        day_scaled = scaler_day.fit_transform(dayofweek)
        month_scaled = scaler_month.fit_transform(month)

        # Use full_data and create_sequences_with_time_flatten
        full_data = np.hstack((demand_scaled, hour_scaled, day_scaled, month_scaled))
        targets = demand_scaled

        seq_length = 24
        X, y = create_sequences_with_time_flatten(full_data, targets, seq_length)

        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        param_grid = {
            'estimator__n_estimators': [50, 100],
            'estimator__max_depth': [3, 5],
            'estimator__learning_rate': [0.05, 0.1]
        }
        base_model = GradientBoostingRegressor(random_state=42)
        model = MultiOutputRegressor(base_model)
        grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        y_pred = model.predict(X_test)
        y_demand_pred = scaler_demand.inverse_transform(y_pred[:, 0].reshape(-1, 1))
        y_demand_true = scaler_demand.inverse_transform(y_test[:, 0].reshape(-1, 1))
        mse = mean_squared_error(y_demand_true, y_demand_pred)
        mae = mean_absolute_error(y_demand_true, y_demand_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_demand_true, y_demand_pred)
        modelName = "GBDT"
        results = [mse, mae, rmse, r2, modelName]

        print('\n', "Demand GBDT Model Results:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
        if plots == True:
            plt.figure(figsize=(10, 5))
            plt.plot(y_demand_true, label='Actual')
            plt.title(f'GBDT Model: Actual Demand')
            plt.xlabel('Time')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()

            def plot_series(y_true, y_pred, label):
                plt.figure(figsize=(10, 5))
                plt.plot(y_true, label='Actual')
                plt.plot(y_pred, label='Predicted')
                plt.title(f'GBDT Model: Actual vs Predicted {label}')
                plt.xlabel('Time')
                plt.ylabel('MW')
                plt.legend()
                plt.tight_layout()
                plt.show()

            plot_series(y_demand_true, y_demand_pred, 'Demand')

        return results

    elif ident == 2:
        supplyDs['Date & Time'] = supplyDs['Date & Time'].astype(str)
        supplyDs[['Date', 'Time']] = supplyDs['Date & Time'].str.split(' ', expand=True)
        supplyDs['MW'] = pd.to_numeric(supplyDs['MW'], errors='coerce')
        supplyDs['Date & Time'] = pd.to_datetime(supplyDs['Date & Time'], errors='coerce')
        supplyDs.dropna(subset=['MW', 'Date & Time'], inplace=True)

        supply = supplyDs['MW'].values.reshape(-1, 1)
        datetime_series = supplyDs['Date & Time']

        hour = datetime_series.dt.hour.values.reshape(-1, 1)
        dayofweek = datetime_series.dt.dayofweek.values.reshape(-1, 1)
        month = datetime_series.dt.month.values.reshape(-1, 1)

        scaler_supply = MinMaxScaler()
        scaler_hour = MinMaxScaler()
        scaler_day = MinMaxScaler()
        scaler_month = MinMaxScaler()

        supply_scaled = scaler_supply.fit_transform(supply)
        hour_scaled = scaler_hour.fit_transform(hour)
        day_scaled = scaler_day.fit_transform(dayofweek)
        month_scaled = scaler_month.fit_transform(month)

        # Use full_data and create_sequences_with_time_flatten
        full_data = np.hstack((supply_scaled, hour_scaled, day_scaled, month_scaled))
        targets = supply_scaled

        seq_length = 24
        X, y = create_sequences_with_time_flatten(full_data, targets, seq_length)

        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        param_grid = {
            'estimator__n_estimators': [50, 100],
            'estimator__max_depth': [3, 5],
            'estimator__learning_rate': [0.05, 0.1]
        }
        base_model = GradientBoostingRegressor(random_state=42)
        model = MultiOutputRegressor(base_model)
        grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        y_pred = model.predict(X_test)
        y_demand_pred = scaler_supply.inverse_transform(y_pred[:, 0].reshape(-1, 1))
        y_demand_true = scaler_supply.inverse_transform(y_test[:, 0].reshape(-1, 1))
        mse = mean_squared_error(y_demand_true, y_demand_pred)
        mae = mean_absolute_error(y_demand_true, y_demand_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_demand_true, y_demand_pred)
        modelName = "GBDT"
        results = [mse, mae, rmse, r2, modelName]

        print('\n', "GBDT Model for Supply Data:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
        if plots == True:
            plt.figure(figsize=(10, 5))
            plt.plot(y_demand_true, label='Actual')
            plt.title(f'GBDT Model: Actual Supply')
            plt.xlabel('Time')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()

            def plot_series(y_true, y_pred, label):
                plt.figure(figsize=(10, 5))
                plt.plot(y_true, label='Actual')
                plt.plot(y_pred, label='Predicted')
                plt.title(f'GBDT Model: Actual vs Predicted {label}')
                plt.xlabel('Time')
                plt.ylabel('MW')
                plt.legend()
                plt.tight_layout()
                plt.show()

            plot_series(y_demand_true, y_demand_pred, 'Supply')

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

dem_TREE = decisionTreeModelDS(sp_DemandDef, sp_SupplyDef, 1,False)
dem_RF = randomForestModelDS(sp_DemandDef, sp_SupplyDef, 1,False)
dem_XGB = xgbModelDS(sp_DemandDef, sp_SupplyDef, 1,False)
dem_GB = gbModelDS(sp_DemandDef, sp_SupplyDef, 1,False)
dem_BLSTM = biDirectionalLSTMDS(sp_DemandDef, sp_SupplyDef, 1,False)
dem_LSTM = LSTMModelDS(sp_DemandDef, sp_SupplyDef, 1,False)
dem_GRU = GRUModelDS(sp_DemandDef, sp_SupplyDef, 1,False)
dem_SVR = SVRModelDS(sp_DemandDef, sp_SupplyDef, 1,False)
dem_MLP = MLPModelDS(sp_DemandDef, sp_SupplyDef, 1,False)
dem_CNN = CNNModelDS(sp_DemandDef, sp_SupplyDef, 1,False)
dem_GBDT = GBDTModelDS(sp_DemandDef, sp_SupplyDef, 1,False)
##dem_NSGA2_CNN = NSGA2_CNN_ModelDS(sp_DemandDef, sp_SupplyDef, 1)
##dem_NSGA3_CNN = NSGA3_CNN_ModelDS(sp_DemandDef, sp_SupplyDef, 1)
##demandModelresults = [dem_TREE, dem_RF, dem_XGB, dem_GB, dem_BLSTM, dem_LSTM, dem_GRU, dem_SVR, dem_MLP, dem_CNN, dem_GBDT, dem_NSGA2_CNN, dem_NSGA3_CNN]

demandModelresults = [dem_TREE, dem_RF, dem_XGB, dem_GB, dem_BLSTM, dem_LSTM, dem_GRU, dem_SVR, dem_MLP, dem_CNN, dem_GBDT]
demandBestResultsOrdered = BetterModelSelectionMethod(demandModelresults)


sup_TREE = decisionTreeModelDS(sp_DemandDef, sp_SupplyDef, 2,False)
sup_RF = randomForestModelDS(sp_DemandDef, sp_SupplyDef, 2,False)
sup_XGB = xgbModelDS(sp_DemandDef, sp_SupplyDef, 2,False)
sup_GB = gbModelDS(sp_DemandDef, sp_SupplyDef, 2,False)
sup_BLSTM = biDirectionalLSTMDS(sp_DemandDef, sp_SupplyDef, 2,False)
sup_LSTM = LSTMModelDS(sp_DemandDef, sp_SupplyDef, 2,False)
sup_GRU = GRUModelDS(sp_DemandDef, sp_SupplyDef, 2,False)
sup_SVR = SVRModelDS(sp_DemandDef, sp_SupplyDef, 2,False)
sup_MLP = MLPModelDS(sp_DemandDef, sp_SupplyDef, 2,False)
sup_CNN = CNNModelDS(sp_DemandDef, sp_SupplyDef, 2,False)
sup_GBDT = GBDTModelDS(sp_DemandDef, sp_SupplyDef, 2,False)

#sup_NSGA2_CNN = NSGA2_CNN_ModelDS(sp_WeatherxDem,sp_WeatherxSup,2) 
#sup_NSGA3_CNN = NSGA3_CNN_ModelDS(sp_WeatherxDem,sp_WeatherxSup,2) 
##supplyModelresults = [sup_TREE, sup_RF, sup_XGB, sup_GB, sup_BLSTM,sup_LSTM, sup_GRU, sup_SVR, sup_MLP, sup_CNN,sup_GBDT, sup_NSGA2_CNN,sup_NSGA3_CNN]

supplyModelresults = [sup_TREE, sup_RF, sup_XGB, sup_GB, sup_BLSTM,sup_LSTM, sup_GRU, sup_SVR, sup_MLP, sup_CNN,sup_GBDT]
supplyBestResultsOrdered = BetterModelSelectionMethod(supplyModelresults)


print("\n Demand Solo Model Ranking (Best to Worst) - Daily:")
print("{:<20} {:>12} {:>12} {:>12} {:>10}".format("Model", "MSE", "MAE", "RMSE", "R2"))
print("-" * 70)
for res in demandBestResultsOrdered:
    print("{:<20} {:>12.4f} {:>12.4f} {:>12.4f} {:>10.4f}".format(
        res[4], res[0], res[1], res[2], res[3]
    ))

print("\n Supply Solo Model Ranking (Best to Worst) - Daily:")
print("{:<20} {:>12} {:>12} {:>12} {:>10}".format("Model", "MSE", "MAE", "RMSE", "R2"))
print("-" * 70)
for res in supplyBestResultsOrdered:
    print("{:<20} {:>12.4f} {:>12.4f} {:>12.4f} {:>10.4f}".format(
        res[4], res[0], res[1], res[2], res[3]
    ))