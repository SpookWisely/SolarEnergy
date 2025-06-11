#----# Libraries
import array
from ast import Str
from ctypes import Array
from enum import Enum
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
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional,GRU, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from keras_tuner import RandomSearch, Hyperband,BayesianOptimization
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
"""

###RMSE Lower the value the better.
##R2 the closer to 1 the better the better fitr of the model to the data.
##MSE Lower the value the better the model performance.
##MAE indicates the average absolute error between predicted and actual values. The smaller the MAE,
## the better the model's predictions align with the actual data. A MAE of 0 would mean a perfect prediction

sp_DemandDef = pd.read_excel(r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\Sakakah 2021 Demand dataset.xlsx")
sp_SupplyDef = pd.read_excel(r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\Sakakah 2021 PV supply dataset.xlsx")

##Note to self can't remember if saeed wanted me to do confusion martrixs aswell as the MSE,MAE,RMSE & R2
def decisionTreeModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int):
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


    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length].flatten())
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)
    
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
        X, y = create_sequences_with_time(full_data, targets, seq_length)


        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        base_model = DecisionTreeRegressor(random_state=42)
        model = MultiOutputRegressor(base_model)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_demand_pred = scaler_demand.inverse_transform(y_pred[:, 0].reshape(-1, 1))
        y_demand_true = scaler_demand.inverse_transform(y_test[:, 0].reshape(-1, 1))


        mse = mean_squared_error(y_demand_true, y_demand_pred)
        mae = mean_absolute_error(y_demand_true, y_demand_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_demand_true, y_demand_pred)
        results = [mse, mae, rmse, r2]
        #---#
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
        X, y = create_sequences_with_time(full_data, targets, seq_length)


        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        base_model = DecisionTreeRegressor(random_state=42)
        model = MultiOutputRegressor(base_model)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_supply_pred = scaler_supply.inverse_transform(y_pred[:, 0].reshape(-1, 1))
        y_supply_true = scaler_supply.inverse_transform(y_test[:, 0].reshape(-1, 1))


        mse = mean_squared_error(y_supply_true, y_supply_pred)
        mae = mean_absolute_error(y_supply_true, y_supply_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_supply_true, y_supply_pred)
        results = [mse, mae, rmse, r2]

        #----#
        print('\n', "Decision Tree Model for Supply Data:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
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
  

def randomForestModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int):
      
    
    
    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length].flatten())
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)
   
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
        X, y = create_sequences_with_time(full_data, targets, seq_length)


        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        base_model = RandomForestRegressor(n_estimators =100,random_state=42)
        model = MultiOutputRegressor(base_model)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_demand_pred = scaler_demand.inverse_transform(y_pred[:, 0].reshape(-1, 1))
        y_demand_true = scaler_demand.inverse_transform(y_test[:, 0].reshape(-1, 1))


        mse = mean_squared_error(y_demand_true, y_demand_pred)
        mae = mean_absolute_error(y_demand_true, y_demand_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_demand_true, y_demand_pred)
        results = [mse, mae, rmse, r2]
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
        X, y = create_sequences_with_time(full_data, targets, seq_length)


        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        base_model = RandomForestRegressor(n_estimators =100,random_state=42)
        model = MultiOutputRegressor(base_model)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_supply_pred = scaler_supply.inverse_transform(y_pred[:, 0].reshape(-1, 1))
        y_supply_true = scaler_supply.inverse_transform(y_test[:, 0].reshape(-1, 1))


        mse = mean_squared_error(y_supply_true, y_supply_pred)
        mae = mean_absolute_error(y_supply_true, y_supply_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_supply_true, y_supply_pred)
        results = [mse, mae, rmse, r2]

        #----#
        print('\n', "Random Forest Model for Supply Data:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
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


def xgbModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int):
    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length].flatten())
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)
   
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
        X, y = create_sequences_with_time(full_data, targets, seq_length)


        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        base_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        model = MultiOutputRegressor(base_model)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_demand_pred = scaler_demand.inverse_transform(y_pred[:, 0].reshape(-1, 1))
        y_demand_true = scaler_demand.inverse_transform(y_test[:, 0].reshape(-1, 1))


        mse = mean_squared_error(y_demand_true, y_demand_pred)
        mae = mean_absolute_error(y_demand_true, y_demand_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_demand_true, y_demand_pred)
        results = [mse, mae, rmse, r2]

        #----#
        print('\n',"Demand XGB Model Results:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
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
        X, y = create_sequences_with_time(full_data, targets, seq_length)


        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        base_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        model = MultiOutputRegressor(base_model)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_supply_pred = scaler_supply.inverse_transform(y_pred[:, 0].reshape(-1, 1))
        y_supply_true = scaler_supply.inverse_transform(y_test[:, 0].reshape(-1, 1))


        mse = mean_squared_error(y_supply_true, y_supply_pred)
        mae = mean_absolute_error(y_supply_true, y_supply_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_supply_true, y_supply_pred)
        results = [mse, mae, rmse, r2]

        #----#
        print('\n', "XGB Model for Supply Data:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
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

def gbModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int):
    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length].flatten())
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)
   
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
        X, y = create_sequences_with_time(full_data, targets, seq_length)


        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        base_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model = MultiOutputRegressor(base_model)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_demand_pred = scaler_demand.inverse_transform(y_pred[:, 0].reshape(-1, 1))
        y_demand_true = scaler_demand.inverse_transform(y_test[:, 0].reshape(-1, 1))


        mse = mean_squared_error(y_demand_true, y_demand_pred)
        mae = mean_absolute_error(y_demand_true, y_demand_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_demand_true, y_demand_pred)
        results = [mse, mae, rmse, r2]

        #----#
        print('\n',"Demand GB Model Results:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
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
        X, y = create_sequences_with_time(full_data, targets, seq_length)


        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        base_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model = MultiOutputRegressor(base_model)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_supply_pred = scaler_supply.inverse_transform(y_pred[:, 0].reshape(-1, 1))
        y_supply_true = scaler_supply.inverse_transform(y_test[:, 0].reshape(-1, 1))


        mse = mean_squared_error(y_supply_true, y_supply_pred)
        mae = mean_absolute_error(y_supply_true, y_supply_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_supply_true, y_supply_pred)
        results = [mse, mae, rmse, r2]

        #----#
        print('\n', "GB Model for Supply Data:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
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

def biDirectionalLSTMDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int):
    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)
   
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
        X, y = create_sequences_with_time(X_scaled,y_scaled, seq_length)


        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        ##split_index = int(len(X_reshaped) * 0.8)
        ##X_train, X_test = X_reshaped[:split_index], X_reshaped[split_index:]
        ##y_train, y_test = y_scaled[:split_index], y_scaled[split_i
        
        model = Sequential ([
            Bidirectional(LSTM(100,return_sequences=True, input_shape=(seq_length,X_train.shape[2]))),
                            Dropout(0.2),
                            Bidirectional(LSTM(100,return_sequences=False)),
                            Dense(64,activation='relu'),
                            Dense(1)
        ])
       
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')

        early_stopping = EarlyStopping(monitor='val_loss', patience=10,restore_best_weights=True)

        model.fit(X_train,y_train,epochs=100, batch_size=16, validation_data=(X_test,y_test),
             callbacks =[early_stopping])

        y_pred = model.predict(X_test)

        y_demand_pred = scaler_y.inverse_transform(y_pred)
        y_demand_true = scaler_y.inverse_transform(y_test)


        mse = mean_squared_error(y_demand_true, y_demand_pred)
        mae = mean_absolute_error(y_demand_true, y_demand_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_demand_true, y_demand_pred)
        results = [mse, mae, rmse, r2]

        #----#
        print('\n',"Demand BLSTM Model Results:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
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
        X, y = create_sequences_with_time(X_scaled,y_scaled, seq_length)


        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        ##split_index = int(len(X_reshaped) * 0.8)
        ##X_train, X_test = X_reshaped[:split_index], X_reshaped[split_index:]
        ##y_train, y_test = y_scaled[:split_index], y_scaled[split_i
        
        model = Sequential ([
            Bidirectional(LSTM(100,return_sequences=True, input_shape=(seq_length,X_train.shape[2]))),
                            Dropout(0.2),
                            Bidirectional(LSTM(100,return_sequences=False)),
                            Dense(64,activation='relu'),
                            Dense(1)
        ])
       
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')

        early_stopping = EarlyStopping(monitor='val_loss', patience=10,restore_best_weights=True)

        model.fit(X_train,y_train,epochs=100, batch_size=16, validation_data=(X_test,y_test),
             callbacks =[early_stopping])

        y_pred = model.predict(X_test)

        y_supply_pred = scaler_y.inverse_transform(y_pred)
        y_supply_true = scaler_y.inverse_transform(y_test)


        mse = mean_squared_error(y_supply_true, y_supply_pred)
        mae = mean_absolute_error(y_supply_true, y_supply_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_supply_true, y_supply_pred)
        results = [mse, mae, rmse, r2]

        #----#
        print('\n', "BLSTM Model for Supply Data:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
        plt.figure(figsize=(10, 5))
        plt.plot(y_supply_true, label='Actual')
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

        plot_series(y_supply_true, y_supply_pred, 'Supply')
        return results
    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return

def LSTMModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int):
    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)
   
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
        X, y = create_sequences_with_time(X_scaled,y_scaled, seq_length)


        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        ##split_index = int(len(X_reshaped) * 0.8)
        ##X_train, X_test = X_reshaped[:split_index], X_reshaped[split_index:]
        ##y_train, y_test = y_scaled[:split_index], y_scaled[split_i
        
        model = Sequential ([
            LSTM(64,return_sequences=True, input_shape=(seq_length,X_train.shape[2])),
                            Dropout(0.2),
                            LSTM(100,return_sequences=False),
                            Dense(64),
                            Dense(1)
        ])
       
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')

        early_stopping = EarlyStopping(monitor='val_loss', patience=10,restore_best_weights=True)

        model.fit(X_train,y_train,epochs=100, batch_size=16, validation_data=(X_test,y_test),
             callbacks =[early_stopping])

        y_pred = model.predict(X_test)

        y_demand_pred = scaler_y.inverse_transform(y_pred)
        y_demand_true = scaler_y.inverse_transform(y_test)


        mse = mean_squared_error(y_demand_true, y_demand_pred)
        mae = mean_absolute_error(y_demand_true, y_demand_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_demand_true, y_demand_pred)
        results = [mse, mae, rmse, r2]
        
        #----#
        print('\n',"Demand LSTM Model Results:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
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
        X, y = create_sequences_with_time(X_scaled,y_scaled, seq_length)


        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        ##split_index = int(len(X_reshaped) * 0.8)
        ##X_train, X_test = X_reshaped[:split_index], X_reshaped[split_index:]
        ##y_train, y_test = y_scaled[:split_index], y_scaled[split_i
        
        model = Sequential ([
            LSTM(64,return_sequences=True, input_shape=(seq_length,X_train.shape[2])),
                            Dropout(0.2),
                            LSTM(100,return_sequences=False),
                            Dense(64),
                            Dense(1)
        ])
       
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')

        early_stopping = EarlyStopping(monitor='val_loss', patience=10,restore_best_weights=True)

        model.fit(X_train,y_train,epochs=100, batch_size=16, validation_data=(X_test,y_test),
             callbacks =[early_stopping])

        y_pred = model.predict(X_test)

        y_supply_pred = scaler_y.inverse_transform(y_pred)
        y_supply_true = scaler_y.inverse_transform(y_test)


        mse = mean_squared_error(y_supply_true, y_supply_pred)
        mae = mean_absolute_error(y_supply_true, y_supply_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_supply_true, y_supply_pred)
        results = [mse, mae, rmse, r2]

        #----#
        print('\n', "LSTM Model for Supply Data:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
        plt.figure(figsize=(10, 5))
        plt.plot(y_supply_true, label='Actual')
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

        plot_series(y_supply_true, y_supply_pred, 'Supply')
        return results 
    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return

def GRUModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int):
    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)
   
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
        X, y = create_sequences_with_time(X_scaled,y_scaled, seq_length)


        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        ##split_index = int(len(X_reshaped) * 0.8)
        ##X_train, X_test = X_reshaped[:split_index], X_reshaped[split_index:]
        ##y_train, y_test = y_scaled[:split_index], y_scaled[split_i
        
        model = Sequential ([
            GRU(64,return_sequences=True, input_shape=(seq_length,X_train.shape[2])),
                            Dropout(0.2),
                            GRU(100,return_sequences=False),
                            Dense(64,activation='relu'),
                            Dense(1)
        ])
       
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')

        early_stopping = EarlyStopping(monitor='val_loss', patience=10,restore_best_weights=True)

        model.fit(X_train,y_train,epochs=100, batch_size=16, validation_data=(X_test,y_test),
             callbacks =[early_stopping])

        y_pred = model.predict(X_test)

        y_demand_pred = scaler_y.inverse_transform(y_pred)
        y_demand_true = scaler_y.inverse_transform(y_test)


        mse = mean_squared_error(y_demand_true, y_demand_pred)
        mae = mean_absolute_error(y_demand_true, y_demand_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_demand_true, y_demand_pred)
        results = [mse, mae, rmse, r2]
        
        #----#
        print('\n',"Demand GRU Model Results:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
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
        X, y = create_sequences_with_time(X_scaled,y_scaled, seq_length)


        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        ##split_index = int(len(X_reshaped) * 0.8)
        ##X_train, X_test = X_reshaped[:split_index], X_reshaped[split_index:]
        ##y_train, y_test = y_scaled[:split_index], y_scaled[split_i
        
        model = Sequential ([
            GRU(64,return_sequences=True, input_shape=(seq_length,X_train.shape[2])),
                            Dropout(0.2),
                            GRU(100,return_sequences=False),
                            Dense(64,activation='relu'),
                            Dense(1)
        ])
       
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')

        early_stopping = EarlyStopping(monitor='val_loss', patience=10,restore_best_weights=True)

        model.fit(X_train,y_train,epochs=100, batch_size=16, validation_data=(X_test,y_test),
             callbacks =[early_stopping])

        y_pred = model.predict(X_test)

        y_supply_pred = scaler_y.inverse_transform(y_pred)
        y_supply_true = scaler_y.inverse_transform(y_test)


        mse = mean_squared_error(y_supply_true, y_supply_pred)
        mae = mean_absolute_error(y_supply_true, y_supply_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_supply_true, y_supply_pred)
        results = [mse, mae, rmse, r2]

        #----#
        print('\n', "GRU Model for Supply Data:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
        plt.figure(figsize=(10, 5))
        plt.plot(y_supply_true, label='Actual')
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

        plot_series(y_supply_true, y_supply_pred, 'Supply')
        return results
    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return
##
def SVRModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int):
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
    
    WITH HYPERPARAMS RESULTS -
    Demand SVR Results -
    MSE: 9.2057
    MAE: 2.7258
    RMSE: 3.0341
    R2: 0.9876
    -----
    Supply GRU Results - 
    MSE: 6.3049
    MAE: 2.0740
    RMSE: 2.5110
    R2: 0.9994
        -----
    WITHOUT HYPER PARAMS
    Demand ---
    MSE: 1527.5193
    MAE: 35.8387
    RMSE: 39.0835
    R2: -1.0585
    ----
    Supply ---
    MSE: 1184.9307
    MAE: 31.1441
    RMSE: 34.4228
    R2: 0.8912
    
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
        ##X, y = create_sequences_with_time(X_scaled,y_scaled, seq_length)


        split_index = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
        y_train, y_test = y_scaled[:split_index], y_scaled[split_index:]
        
        ##split_index = int(len(X_reshaped) * 0.8)
        ##X_train, X_test = X_reshaped[:split_index], X_reshaped[split_index:]
        ##y_train, y_test = y_scaled[:split_index], y_scaled[split_i
        
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
        results = [mse, mae, rmse, r2]

        #----#
        print('\n',"Demand SVR Model Results:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
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
        ##X, y = create_sequences_with_time(X_scaled,y_scaled, seq_length)


        split_index = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
        y_train, y_test = y_scaled[:split_index], y_scaled[split_index:]
        
        ##split_index = int(len(X_reshaped) * 0.8)
        ##X_train, X_test = X_reshaped[:split_index], X_reshaped[split_index:]
        ##y_train, y_test = y_scaled[:split_index], y_scaled[split_i
        
        param_grid = {
        'estimator__C': [0.1, 1, 10, 100],           # Regularization, similar to tree depth/complexity
        'estimator__gamma': ['scale', 'auto', 0.01, 0.1, 1],  # Kernel coefficient, like learning rate
        'estimator__epsilon': [0.01, 0.1, 0.5, 1.0], # Insensitivity, similar to min_samples_split
        'estimator__kernel': ['rbf']                 # Keep kernel consistent
        }
        ##model = MultiOutputRegressor(SVR(kernel='rbf'))
        model = MultiOutputRegressor(SVR())
        
        grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid.fit(X_train, y_train)
        print('\n',"Best parameters:", grid.best_params_)

        grid.fit(X_train,y_train)

        y_pred = grid.predict(X_test)

        y_pred = grid.predict(X_test)

        y_supply_pred = scaler_y.inverse_transform(y_pred)
        y_supply_true = scaler_y.inverse_transform(y_test)


        mse = mean_squared_error(y_supply_true, y_supply_pred)
        mae = mean_absolute_error(y_supply_true, y_supply_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_supply_true, y_supply_pred)
        results = [mse, mae, rmse, r2]

        #----#
        print('\n', "SVR Model for Supply Data:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))

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

        plot_series(y_supply_true, y_supply_pred, 'Supply')

        return results
    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return

def MLPModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int):
 def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length].flatten())
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)
   
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
        X, y = create_sequences_with_time(X_scaled,y_scaled, seq_length)

        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        ##split_index = int(len(X_reshaped) * 0.8)
        ##X_train, X_test = X_reshaped[:split_index], X_reshaped[split_index:]
        ##y_train, y_test = y_scaled[:split_index], y_scaled[split_i
        
        model = Sequential ([
            Dense(128, activation='relu', input_dim=X_train.shape[1]),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1) 
        ])
       
        model.compile(optimizer='adam', loss='mse')

        model.fit(X_train,y_train,epochs=100, batch_size=16, validation_data=(X_test,y_test))

        y_pred = model.predict(X_test)

        y_demand_pred = scaler_y.inverse_transform(y_pred)
        y_demand_true = scaler_y.inverse_transform(y_test)


        mse = mean_squared_error(y_demand_true, y_demand_pred)
        mae = mean_absolute_error(y_demand_true, y_demand_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_demand_true, y_demand_pred)
        results = [mse, mae, rmse, r2]

        #----#
        print('\n',"Demand MLP Model Results:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
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

        


        seq_length = 24
        X, y = create_sequences_with_time(X_scaled,y_scaled, seq_length)


        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        ##split_index = int(len(X_reshaped) * 0.8)
        ##X_train, X_test = X_reshaped[:split_index], X_reshaped[split_index:]
        ##y_train, y_test = y_scaled[:split_index], y_scaled[split_i
        
        model = Sequential ([
            Dense(128, activation='relu', input_dim=X_train.shape[1]),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1) 
        ])
       
        model.compile(optimizer='adam', loss='mse')

        model.fit(X_train,y_train,epochs=100, batch_size=16, validation_data=(X_test,y_test))

        y_pred = model.predict(X_test)

        y_supply_pred = scaler_y.inverse_transform(y_pred)
        y_supply_true = scaler_y.inverse_transform(y_test)


        mse = mean_squared_error(y_supply_true, y_supply_pred)
        mae = mean_absolute_error(y_supply_true, y_supply_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_supply_true, y_supply_pred)
        results = [mse, mae, rmse, r2]

        #----#
        print('\n', "MLP Model for Supply Data:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
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

def CNNModelDS(demandDs: pd.DataFrame, supplyDs: pd.DataFrame, ident: int):
    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)

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
        X, y = create_sequences_with_time(X_scaled, y_scaled, seq_length)

        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        model = Sequential([
            Conv1D(64, kernel_size=3, activation='relu', input_shape=(seq_length, X_train.shape[2])),           
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            Conv1D(128, kernel_size=3, activation='relu'),
            Dropout(0.2),
            Flatten(),
            Dense(32, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')

        model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))

        y_pred = model.predict(X_test)

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
        X, y = create_sequences_with_time(X_scaled, y_scaled, seq_length)

        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        model = Sequential([
            Conv1D(64, kernel_size=3, activation='relu', input_shape=(seq_length, X_train.shape[2])),           
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            Conv1D(128, kernel_size=3, activation='relu'),
            Dropout(0.2),
            Flatten(),
            Dense(32, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')

        model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))
        y_pred = model.predict(X_test)

        y_supply_pred = scaler_y.inverse_transform(y_pred)
        y_supply_true = scaler_y.inverse_transform(y_test)

        mse = mean_squared_error(y_supply_true, y_supply_pred)
        mae = mean_absolute_error(y_supply_true, y_supply_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_supply_true, y_supply_pred)
        results = [mse, mae, rmse, r2]
        
        #---#
        print('\n', "CNN Model for Supply Data:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
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


        return results

    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return
    

def GBDTModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int):
    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length].flatten())
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)
   
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
        -----

    """


    if (ident == 1):
    #----#        
    ##Basic transformation for the base dataset
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
        X, y = create_sequences_with_time(X_scaled, y_scaled, seq_length)

        split_index = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
        y_train, y_test = y_scaled[:split_index], y_scaled[split_index:]

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
        results = [mse, mae, rmse, r2]
        
        #---#
        print('\n', "Demand GBDT Model Results:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
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
        ##X, y = create_sequences_with_time(X_scaled,y_scaled, seq_length)


        split_index = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
        y_train, y_test = y_scaled[:split_index], y_scaled[split_index:]
        
        ##split_index = int(len(X_reshaped) * 0.8)
        ##X_train, X_test = X_reshaped[:split_index], X_reshaped[split_index:]
        ##y_train, y_test = y_scaled[:split_index], y_scaled[split_i
        
        param_grid = {
        'estimator__C': [0.1, 1, 10, 100],           # Regularization, similar to tree depth/complexity
        'estimator__gamma': ['scale', 'auto', 0.01, 0.1, 1],  # Kernel coefficient, like learning rate
        'estimator__epsilon': [0.01, 0.1, 0.5, 1.0], # Insensitivity, similar to min_samples_split
        'estimator__kernel': ['rbf']                 # Keep kernel consistent
        }
        ##model = MultiOutputRegressor(SVR(kernel='rbf'))
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X_scaled = scaler_X.fit_transform(features)
        y_scaled = scaler_y.fit_transform(supply)

        seq_length = 24
        X, y = create_sequences_with_time(X_scaled, y_scaled, seq_length)

        split_index = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
        y_train, y_test = y_scaled[:split_index], y_scaled[split_index:]

        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # Ensure 2D shape for inverse_transform
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)

        y_supply_pred = scaler_y.inverse_transform(y_pred)
        y_supply_true = scaler_y.inverse_transform(y_test)

        mse = mean_squared_error(y_supply_true, y_supply_pred)
        mae = mean_absolute_error(y_supply_true, y_supply_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_supply_true, y_supply_pred)
        results = [mse, mae, rmse, r2]
        
        #----#
        print('\n', "GBDT Model for Supply Data:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))

        plt.figure(figsize=(10, 5))
        plt.plot(y_supply_true, label='Actual')
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

        plot_series(y_supply_true, y_supply_pred, 'Supply')

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
def BestModelChoice(setOne:Array,setTwo:Array,modelNameOne:str,modelNameTwo:str):
        scoreOne = 0
        scoreTwo = 0
        bestresult = []
        bestModelName = ''  
        if (setOne[0] < setTwo[0]): #MSE
            scoreOne += 1
        else:
            scoreTwo += 1      
        ##
        if (setOne[1] < setTwo[1]): #MAE
            scoreOne += 1
        else:
            scoreTwo += 1      
        ##
        if (setOne[2] < setTwo[2]): #RMSE
            scoreOne += 1
        else:
            scoreTwo += 1
        ##
        r2OneScore = 1 - setOne[3]
        r2TwoScore = 1 - setTwo[3] 
        if (r2OneScore < r2TwoScore): #R2
            scoreOne += 1
        else:
            scoreTwo += 1
       ##
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
   
##------Demand Functions------
##decisionTreeModelDS(sp_DemandDef,sp_SupplyDef,1)
##randomForestModelDS(sp_DemandDef,sp_SupplyDef,1)
"""
CurrentTopResult, CurrentTopMLName = BestModelChoice(
  decisionTreeModelDS(sp_DemandDef,sp_SupplyDef,1),
  randomForestModelDS(sp_DemandDef,sp_SupplyDef,1),
 'DecisionTree','RandomForest'
 )
"""
##xgbModelDS(sp_DemandDef,sp_SupplyDef,1)
"""
CurrentTopResult, CurrentTopMLName = BestModelChoice(
 CurrentTopResult,
 xgbModelDS(sp_DemandDef,sp_SupplyDef,1),
 CurrentTopMLName,'XGB'
 )
"""
##gbModelDS(sp_DemandDef,sp_SupplyDef,1)
"""
CurrentTopResult, CurrentTopMLName = BestModelChoice(
 CurrentTopResult,
 gbModelDS(sp_DemandDef,sp_SupplyDef,1),
 CurrentTopMLName,'GB'
 )
"""
##biDirectionalLSTMDS(sp_DemandDef,sp_SupplyDef,1)
"""
CurrentTopResult, CurrentTopMLName = BestModelChoice(
CurrentTopResult,
biDirectionalLSTMDS(sp_DemandDef,sp_SupplyDef,1),
CurrentTopMLName,'BSTMD'
)
"""
##LSTMModelDS(sp_DemandDef,sp_SupplyDef,1)
"""
CurrentTopResult, CurrentTopMLName = BestModelChoice(
CurrentTopResult,
LSTMModelDS(sp_DemandDef,sp_SupplyDef,1),
CurrentTopMLName,'LSTM'
 )
"""
##GRUModelDS(sp_DemandDef,sp_SupplyDef,1)
"""
CurrentTopResult, CurrentTopMLName = BestModelChoice(
CurrentTopResult,
GRUModelDS(sp_DemandDef,sp_SupplyDef,1),
CurrentTopMLName,'GRU'
 )
"""
##SVRModelDS(sp_DemandDef,sp_SupplyDef,1)
"""
CurrentTopResult, CurrentTopMLName = BestModelChoice(
CurrentTopResult,
SVRModelDS(sp_DemandDef,sp_SupplyDef,1),
CurrentTopMLName,'SVR'
)
"""
##MLPModelDS(sp_DemandDef,sp_SupplyDef,1)
"""
CurrentTopResult, CurrentTopMLName = BestModelChoice(
CurrentTopResult,
MLPModelDS(sp_DemandDef,sp_SupplyDef,1),
CurrentTopMLName,'MLP'
 )
"""
##CNNModelDS(sp_DemandDef,sp_SupplyDef,1)
"""
CurrentTopResult, CurrentTopMLName = BestModelChoice(CurrentTopResult,
CNNModelDS(sp_DemandDef,sp_SupplyDef,1),
CurrentTopMLName,'CNN'
)
"""
##GBDTModelDS(sp_DemandDef,sp_SupplyDef,1) 
"""
CurrentTopResult, CurrentTopMLName = BestModelChoice(
CurrentTopResult,
GBDTModelDS(sp_DemandDef,sp_SupplyDef,1),
CurrentTopMLName,'GBDT'
 )
"""
"""
print('\n', "Best Model for solo demand DataSet is : " ,CurrentTopMLName)
print("MSE: {:.4f}".format(CurrentTopResult[0]))
print("MAE: {:.4f}".format(CurrentTopResult[1]))
print("RMSE: {:.4f}".format(CurrentTopResult[2]))
print("R2: {:.4f}".format(CurrentTopResult[3]))

"""

#------Supply Functions------
##decisionTreeModelDS(sp_DemandDef,sp_SupplyDef,2)
##randomForestModelDS(sp_DemandDef,sp_SupplyDef,2)
"""
CurrentTopResult, CurrentTopMLName = BestModelChoice(
  decisionTreeModelDS(sp_DemandDef,sp_SupplyDef,2),
  randomForestModelDS(sp_DemandDef,sp_SupplyDef,2),
 'DecisionTree','RandomForest'
 )
"""
##xgbModelDS(sp_DemandDef,sp_SupplyDef,2)
"""
CurrentTopResult, CurrentTopMLName = BestModelChoice(
 CurrentTopResult,
 xgbModelDS(sp_DemandDef,sp_SupplyDef,2),
 CurrentTopMLName,'XGB'
 )
"""
##gbModelDS(sp_DemandDef,sp_SupplyDef,2)
"""
CurrentTopResult, CurrentTopMLName = BestModelChoice(
 CurrentTopResult,
 gbModelDS(sp_DemandDef,sp_SupplyDef,2),
 CurrentTopMLName,'GB'
 )
"""
##biDirectionalLSTMDS(sp_DemandDef,sp_SupplyDef,2)
"""
CurrentTopResult, CurrentTopMLName = BestModelChoice(
CurrentTopResult,
biDirectionalLSTMDS(sp_DemandDef,sp_SupplyDef,2),
CurrentTopMLName,'BSTMD'
)
"""
##LSTMModelDS(sp_DemandDef,sp_SupplyDef,2)
"""
CurrentTopResult, CurrentTopMLName = BestModelChoice(
CurrentTopResult,
LSTMModelDS(sp_DemandDef,sp_SupplyDef,2),
CurrentTopMLName,'LSTM'
 )
"""
##GRUModelDS(sp_DemandDef,sp_SupplyDef,2)
"""
CurrentTopResult, CurrentTopMLName = BestModelChoice(
CurrentTopResult,
GRUModelDS(sp_DemandDef,sp_SupplyDef,2),
CurrentTopMLName,'GRU'
 )
"""
##SVRModelDS(sp_DemandDef,sp_SupplyDef,2)
"""
CurrentTopResult, CurrentTopMLName = BestModelChoice(
CurrentTopResult,
SVRModelDS(sp_DemandDef,sp_SupplyDef,2),
CurrentTopMLName,'SVR'
)
"""
##MLPModelDS(sp_DemandDef,sp_SupplyDef,2)
"""
CurrentTopResult, CurrentTopMLName = BestModelChoice(
CurrentTopResult,
MLPModelDS(sp_DemandDef,sp_SupplyDef,2),
CurrentTopMLName,'MLP'
 )
"""
##CNNModelDS(sp_DemandDef,sp_SupplyDef,2)
"""
CurrentTopResult, CurrentTopMLName = BestModelChoice(CurrentTopResult,
CNNModelDS(sp_DemandDef,sp_SupplyDef,2),
CurrentTopMLName,'CNN'
)
"""
GBDTModelDS(sp_DemandDef,sp_SupplyDef,2) 
"""
CurrentTopResult, CurrentTopMLName = BestModelChoice(
CurrentTopResult,
GBDTModelDS(sp_DemandDef,sp_SupplyDef,2),
CurrentTopMLName,'GBDT'
 )
"""
"""
print('\n', "Best Model for solo demand DataSet is : " ,CurrentTopMLName)
print("MSE: {:.4f}".format(CurrentTopResult[0]))
print("MAE: {:.4f}".format(CurrentTopResult[1]))
print("RMSE: {:.4f}".format(CurrentTopResult[2]))
print("R2: {:.4f}".format(CurrentTopResult[3]))

"""
"""
For the smaller datasets that are used
it seems that the GBDT model is the best performing model for both demand and supply datasets.
This is likely due to the small size of the datasets and the nature of the data.
Though could be that for the other models to shine they could need more cleaning of the data
to help them perform better.
But not sure how that could be achived.
"""


