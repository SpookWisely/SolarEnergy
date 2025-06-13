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
2. Why when comparing the creation process for the merged dataset in olivers code does he
duplicate the weather data
with ALLSKY_SFC_SW_DWN ending up as ALLSKY_SFC_SW_DWN_x/ALLSKY_SFC_SW_DWN_y as an example.
"""


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

sp_DemandDef.drop(columns=["DATE-TIME"], inplace=True)
sp_SupplyDef.drop(columns=["Date & Time"], inplace=True)
sp_WeatherDe.drop(columns=["YEAR", "MO", "DY", "HR"], inplace=True)
#Thought maybe that the demand dataset was specifically to be tied with demand
#but I'm still getting bad results so can uncomment
# this to turn dataset back to just using 2021 weather DS
#sp_WeatherxDem = pd.merge(sp_DemandDef,sp_WeatherSu, on="TimeStamp", how="inner")

sp_WeatherxDem = pd.merge(sp_DemandDef,sp_WeatherDe, on="TimeStamp", how="inner")
sp_WeatherxSup = pd.merge(sp_SupplyDef,sp_WeatherSu, on="TimeStamp", how="inner")
##linear interpolation to smooth out the datasets 
##So far has worked out great for supply but demand is getting bad results.
sp_WeatherxDem.interpolate(method='linear', inplace=True)
sp_WeatherxSup.interpolate(method='linear', inplace=True)
print(sp_WeatherxDem.head())
print(sp_WeatherxSup.head())
 

#desave_loc = r"E:\AI Lecture Notes\Datasets\Merged\DemandxWeather.csv"
##susave_loc = r"E:\AI Lecture Notes\Datasets\Merged\SupplyxWeather.csv"
#sp_WeatherxDem.to_csv(desave_loc, index=False)
##sp_WeatherxSup.to_csv(susave_loc, index=False)


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


    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length].flatten())
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)
    
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
        "ALLSKY_SFC_SW_DWN_D","ALLSKY_SFC_UV_INDEX_D","T2M_D","PRECTOTCORR_D","ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D","RH2M_D","PS_D","PSC_D","hour","day","month"
    ]
        X = demandDs[feature_cols].values
        y = demandDs['MW'].values.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False  # shuffle=False for time series
    )

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_test_scaled = scaler_y.transform(y_test)
        base_model = DecisionTreeRegressor(random_state=42)
        base_model.fit(X_train_scaled, y_train_scaled)

        y_pred = base_model.predict(X_test_scaled)

        y_demand_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
        y_demand_true = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1))


        mse = mean_squared_error(y_demand_true, y_demand_pred)
        mae = mean_absolute_error(y_demand_true, y_demand_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_demand_true, y_demand_pred)
        results = [mse, mae, rmse, r2]
        #----#
        print('\n',"Demand Data Decision Tree Model Results:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
        """
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
        """
        return results
    elif (ident == 2):

        supplyDs.replace(-999,np.nan, inplace=True)
        supplyDs.dropna(inplace=True)
        supplyDs["hour"] = supplyDs["TimeStamp"].dt.hour   
        supplyDs["day"] = supplyDs["TimeStamp"].dt.day        
        supplyDs["month"] = supplyDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","hour","day","month"
    ]
        X = supplyDs[feature_cols].values
        y = supplyDs['MW'].values.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # shuffle=False for time series
    )

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_test_scaled = scaler_y.transform(y_test)
        base_model = DecisionTreeRegressor(random_state=42)
        base_model.fit(X_train_scaled, y_train_scaled)

        y_pred = base_model.predict(X_test_scaled)

        y_supply_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
        y_supply_true = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1))


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
        """
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
        """
        return results
    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return
  

def randomForestModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy()     
    
    
    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length].flatten())
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)
   
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
        "ALLSKY_SFC_SW_DWN_D","ALLSKY_SFC_UV_INDEX_D","T2M_D","PRECTOTCORR_D","ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D","RH2M_D","PS_D","PSC_D","hour","day","month"
    ]
        X = demandDs[feature_cols].values
        y = demandDs['MW'].values.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False  # shuffle=False for time series
    )

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_test_scaled = scaler_y.transform(y_test)
        base_model = RandomForestRegressor(n_estimators =100,random_state=42)
        
        base_model.fit(X_train_scaled, y_train_scaled)

        y_pred = base_model.predict(X_test_scaled)

        y_demand_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
        y_demand_true = scaler_y.inverse_transform(y_test.reshape(-1, 1))


        mse = mean_squared_error(y_demand_true, y_demand_pred)
        mae = mean_absolute_error(y_demand_true, y_demand_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_demand_true, y_demand_pred)
        results = [mse, mae, rmse, r2]
        #----#
        print('\n',"Demand Random Forest Model Results:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
        """
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
        """
        return results
    elif (ident == 2):
        supplyDs.replace(-999,np.nan, inplace=True)
        supplyDs.dropna(inplace=True)
        supplyDs["hour"] = supplyDs["TimeStamp"].dt.hour   
        supplyDs["day"] = supplyDs["TimeStamp"].dt.day        
        supplyDs["month"] = supplyDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","hour","day","month"
    ]
        X = supplyDs[feature_cols].values
        y = supplyDs['MW'].values.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False  # shuffle=False for time series
    )

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_test_scaled = scaler_y.transform(y_test)
        base_model = RandomForestRegressor(n_estimators =100,random_state=42)
        base_model.fit(X_train_scaled, y_train_scaled)

        y_pred = base_model.predict(X_test_scaled)

        y_supply_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
        y_supply_true = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1))




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
        """
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
        """
        return results
    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return


def xgbModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy()
    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length].flatten())
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)
   
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
        "ALLSKY_SFC_SW_DWN_D","ALLSKY_SFC_UV_INDEX_D","T2M_D","PRECTOTCORR_D","ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D","RH2M_D","PS_D","PSC_D","hour","day","month"
    ]
        X = demandDs[feature_cols].values
        y = demandDs['MW'].values.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False  # shuffle=False for time series
    )

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_test_scaled = scaler_y.transform(y_test)

        base_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        ##model = MultiOutputRegressor(base_model)
        base_model.fit(X_train_scaled, y_train_scaled)

        y_pred = base_model.predict(X_test_scaled)

        y_demand_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
        y_demand_true = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1))

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
        """
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
        """

        return results
    elif (ident == 2):
       
        supplyDs.replace(-999,np.nan, inplace=True)
        supplyDs.dropna(inplace=True)
        supplyDs["hour"] = supplyDs["TimeStamp"].dt.hour   
        supplyDs["day"] = supplyDs["TimeStamp"].dt.day        
        supplyDs["month"] = supplyDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","hour","day","month"
    ]
        X = supplyDs[feature_cols].values
        y = supplyDs['MW'].values.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False  # shuffle=False for time series
    )

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_test_scaled = scaler_y.transform(y_test)
        base_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
       ## model = MultiOutputRegressor(base_model)
        base_model.fit(X_train_scaled, y_train_scaled)

        y_pred = base_model.predict(X_test_scaled)

        y_supply_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
        y_supply_true = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1))

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
        """
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
        """
        return results
    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return

def gbModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy()
    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length].flatten())
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)
   
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
        "ALLSKY_SFC_SW_DWN_D","ALLSKY_SFC_UV_INDEX_D","T2M_D","PRECTOTCORR_D","ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D","RH2M_D","PS_D","PSC_D","hour","day","month"
    ]
        X = demandDs[feature_cols].values
        y = demandDs['MW'].values.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False  # shuffle=False for time series
    )

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_test_scaled = scaler_y.transform(y_test)

        base_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        ##model = MultiOutputRegressor(base_model)
        base_model.fit(X_train_scaled, y_train_scaled)

        y_pred = base_model.predict(X_test_scaled)

        y_demand_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
        y_demand_true = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1))



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
        """
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
        """
        return results
    elif (ident == 2):

        supplyDs.replace(-999,np.nan, inplace=True)
        supplyDs.dropna(inplace=True)
        supplyDs["hour"] = supplyDs["TimeStamp"].dt.hour   
        supplyDs["day"] = supplyDs["TimeStamp"].dt.day        
        supplyDs["month"] = supplyDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","hour","day","month"
    ]
        X = supplyDs[feature_cols].values
        y = supplyDs['MW'].values.reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False  # shuffle=False for time series
    )

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train)
        y_test_scaled = scaler_y.transform(y_test)
        base_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        ##model = MultiOutputRegressor(base_model)
        base_model.fit(X_train_scaled, y_train_scaled)

        y_pred = base_model.predict(X_test_scaled)

        y_supply_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
        y_supply_true = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1))


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
        """
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
        """
        return results
    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return

def biDirectionalLSTMDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy()
    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)
   
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
        "ALLSKY_SFC_SW_DWN_D","ALLSKY_SFC_UV_INDEX_D","T2M_D","PRECTOTCORR_D","ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D","RH2M_D","PS_D","PSC_D","hour","day","month"
    ]
        demandx = demandDs[feature_cols].values
        demandy = demandDs['MW'].values.reshape(-1, 1)
        
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(demandx)
        y_scaled = scaler_y.fit_transform(demandy)
        

        seq_length = 24
        X_seq,y_seq = create_sequences_with_time(X_scaled, y_scaled, seq_length)
        X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.3, shuffle=False)

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
        """
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
        """
        return results
    elif (ident == 2):

        supplyDs["hour"] = supplyDs["TimeStamp"].dt.hour   
        supplyDs["day"] = supplyDs["TimeStamp"].dt.day        
        supplyDs["month"] = supplyDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","hour","day","month"
    ]
        supplyx = supplyDs[feature_cols].values
        supplyy = supplyDs['MW'].values.reshape(-1, 1)
        
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(supplyx)
        y_scaled = scaler_y.fit_transform(supplyy)
        

        seq_length = 24
        X_seq,y_seq = create_sequences_with_time(X_scaled, y_scaled, seq_length)
        X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.3, shuffle=False)

        
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
        print('\n', "BLSTM Model for Supply Data:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
        """
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
            """
        plot_series(y_demand_true, y_demand_pred, 'Supply')

        return results
    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return

def LSTMModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy()
    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)
   
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
        "ALLSKY_SFC_SW_DWN_D","ALLSKY_SFC_UV_INDEX_D","T2M_D","PRECTOTCORR_D","ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D","RH2M_D","PS_D","PSC_D","hour","day","month"
    ]
        demandx = demandDs[feature_cols].values
        demandy = demandDs['MW'].values.reshape(-1, 1)
        
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(demandx)
        y_scaled = scaler_y.fit_transform(demandy)
        

        seq_length = 24
        X_seq,y_seq = create_sequences_with_time(X_scaled, y_scaled, seq_length)
        X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.3, shuffle=False)

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
        """
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
        """
        return results
    elif (ident == 2):
        supplyDs["hour"] = supplyDs["TimeStamp"].dt.hour   
        supplyDs["day"] = supplyDs["TimeStamp"].dt.day        
        supplyDs["month"] = supplyDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","hour","day","month"
    ]
        supplyx = supplyDs[feature_cols].values
        supplyy = supplyDs['MW'].values.reshape(-1, 1)
        
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(supplyx)
        y_scaled = scaler_y.fit_transform(supplyy)
        

        seq_length = 24
        X_seq,y_seq = create_sequences_with_time(X_scaled, y_scaled, seq_length)
        X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.3, shuffle=False)
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
        print('\n', "LSTM Model for Supply Data:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
        """
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
        """
        return results 
    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return

def GRUModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy()
    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)
   
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
        "ALLSKY_SFC_SW_DWN_D","ALLSKY_SFC_UV_INDEX_D","T2M_D","PRECTOTCORR_D","ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D","RH2M_D","PS_D","PSC_D","hour","day","month"
    ]
        demandx = demandDs[feature_cols].values
        demandy = demandDs['MW'].values.reshape(-1, 1)
        
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(demandx)
        y_scaled = scaler_y.fit_transform(demandy)
        

        seq_length = 24
        X_seq,y_seq = create_sequences_with_time(X_scaled, y_scaled, seq_length)
        X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.3, shuffle=False)

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
        """
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
        """
        return results 
    elif (ident == 2):
        supplyDs["hour"] = supplyDs["TimeStamp"].dt.hour   
        supplyDs["day"] = supplyDs["TimeStamp"].dt.day        
        supplyDs["month"] = supplyDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","hour","day","month"
    ]
        supplyx = supplyDs[feature_cols].values
        supplyy = supplyDs['MW'].values.reshape(-1, 1)
        
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(supplyx)
        y_scaled = scaler_y.fit_transform(supplyy)
        

        seq_length = 24
        X_seq,y_seq = create_sequences_with_time(X_scaled, y_scaled, seq_length)
        X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.3, shuffle=False)
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
        print('\n', "GRU Model for Supply Data:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
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
def SVRModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy()
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
        "ALLSKY_SFC_SW_DWN_D","ALLSKY_SFC_UV_INDEX_D","T2M_D","PRECTOTCORR_D","ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D","RH2M_D","PS_D","PSC_D","hour","day","month"
    ]
        demandx = demandDs[feature_cols].values
        demandy = demandDs['MW'].values.reshape(-1, 1)
        
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(demandx)
        y_scaled = scaler_y.fit_transform(demandy)
        
        """
        seq_length = 24
        X_seq,y_seq = create_sequences_with_time(X_scaled, y_scaled, seq_length)
        X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.3, shuffle=False)
        """
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
        results = [mse, mae, rmse, r2]

        #----#
        print('\n',"Demand SVR Model Results:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
        """
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
        """
        return results 
    elif (ident == 2):
        supplyDs["hour"] = supplyDs["TimeStamp"].dt.hour   
        supplyDs["day"] = supplyDs["TimeStamp"].dt.day        
        supplyDs["month"] = supplyDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","hour","day","month"
    ]
        supplyx = supplyDs[feature_cols].values
        supplyy = supplyDs['MW'].values.reshape(-1, 1)
        
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(supplyx)
        y_scaled = scaler_y.fit_transform(supplyy)
        
        """
        seq_length = 24
        X_seq,y_seq = create_sequences_with_time(X_scaled, y_scaled, seq_length)
        X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.3, shuffle=False)
        """
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
        results = [mse, mae, rmse, r2]

        #----#
        print('\n', "SVR Model for Supply Data:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
        """
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
        """
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
        "ALLSKY_SFC_SW_DWN_D","ALLSKY_SFC_UV_INDEX_D","T2M_D","PRECTOTCORR_D","ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D","RH2M_D","PS_D","PSC_D","hour","day","month"
    ]
        demandx = demandDs[feature_cols].values
        demandy = demandDs['MW'].values.reshape(-1, 1)
        
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(demandx)
        y_scaled = scaler_y.fit_transform(demandy)
        

        seq_length = 24
        X_seq,y_seq = create_sequences_with_time(X_scaled, y_scaled, seq_length)
        X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.3, shuffle=False)

        
        
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
        """
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
        """
        return results 
 elif (ident == 2):
        supplyDs["hour"] = supplyDs["TimeStamp"].dt.hour   
        supplyDs["day"] = supplyDs["TimeStamp"].dt.day        
        supplyDs["month"] = supplyDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","hour","day","month"
    ]
        supplyx = supplyDs[feature_cols].values
        supplyy = supplyDs['MW'].values.reshape(-1, 1)
        
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(supplyx)
        y_scaled = scaler_y.fit_transform(supplyy)
        

        seq_length = 24
        X_seq,y_seq = create_sequences_with_time(X_scaled, y_scaled, seq_length)
        X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.3, shuffle=False)
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
        print('\n', "MLP Model for Supply Data:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
        """
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
        """
        return results
 else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return

def CNNModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy()
    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)

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
        "ALLSKY_SFC_SW_DWN_D","ALLSKY_SFC_UV_INDEX_D","T2M_D","PRECTOTCORR_D","ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D","RH2M_D","PS_D","PSC_D","hour","day","month"
    ]
        demandx = demandDs[feature_cols].values
        demandy = demandDs['MW'].values.reshape(-1, 1)
        
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(demandx)
        y_scaled = scaler_y.fit_transform(demandy)
        

        seq_length = 24
        X_seq,y_seq = create_sequences_with_time(X_scaled, y_scaled, seq_length)
        X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.3, shuffle=False)

        

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
        """
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
        """
        return results

    elif ident == 2:
        supplyDs["hour"] = supplyDs["TimeStamp"].dt.hour   
        supplyDs["day"] = supplyDs["TimeStamp"].dt.day        
        supplyDs["month"] = supplyDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","hour","day","month"
    ]
        supplyx = supplyDs[feature_cols].values
        supplyy = supplyDs['MW'].values.reshape(-1, 1)
        
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(supplyx)
        y_scaled = scaler_y.fit_transform(supplyy)
        

        seq_length = 24
        X_seq,y_seq = create_sequences_with_time(X_scaled, y_scaled, seq_length)
        X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.3, shuffle=False)

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
        """
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
        """
        return results

    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return
    

def GBDTModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame,ident:int):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy()
    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length].flatten())
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)
   
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
        "ALLSKY_SFC_SW_DWN_D","ALLSKY_SFC_UV_INDEX_D","T2M_D","PRECTOTCORR_D","ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D","RH2M_D","PS_D","PSC_D","hour","day","month"
    ]
        demandx = demandDs[feature_cols].values
        demandy = demandDs['MW'].values.reshape(-1, 1)
        
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(demandx)
        y_scaled = scaler_y.fit_transform(demandy)
        

        seq_length = 24
        X_seq,y_seq = create_sequences_with_time(X_scaled, y_scaled, seq_length)
        X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.3, shuffle=False)

        

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
        supplyDs["hour"] = supplyDs["TimeStamp"].dt.hour   
        supplyDs["day"] = supplyDs["TimeStamp"].dt.day        
        supplyDs["month"] = supplyDs["TimeStamp"].dt.month        
        
        feature_cols = [
        "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","hour","day","month"
    ]
        supplyx = supplyDs[feature_cols].values
        supplyy = supplyDs['MW'].values.reshape(-1, 1)
        
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(supplyx)
        y_scaled = scaler_y.fit_transform(supplyy)
        

        seq_length = 24
        X_seq,y_seq = create_sequences_with_time(X_scaled, y_scaled, seq_length)
        X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.3, shuffle=False)
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
        results = [mse, mae, rmse, r2]
        
        #----#
        print('\n', "GBDT Model for Supply Data:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))
        """
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
        """
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
   
##------Demand Functions------
##decisionTreeModelDS(sp_WeatherxDem,sp_WeatherxSup,1)
###randomForestModelDS(sp_WeatherxDem,sp_WeatherxSup,1)
###randomForestModelDS(sp_WeatherxDem,sp_WeatherxSup,2)
#xgbModelDS(sp_WeatherxDem,sp_WeatherxSup,1)
#gbModelDS(sp_WeatherxDem,sp_WeatherxSup,1)
#xgbModelDS(sp_WeatherxDem,sp_WeatherxSup,2)
#gbModelDS(sp_WeatherxDem,sp_WeatherxSup,2)
###biDirectionalLSTMDS(sp_WeatherxDem,sp_WeatherxSup,1)
###LSTMModelDS(sp_WeatherxDem,sp_WeatherxSup,1)
###GRUModelDS(sp_WeatherxDem,sp_WeatherxSup,1)
##SVRModelDS(sp_WeatherxDem,sp_WeatherxSup,1)
##MLPModelDS(sp_WeatherxDem,sp_WeatherxSup,1)
##CNNModelDS(sp_WeatherxDem,sp_WeatherxSup,1)
##GBDTModelDS(sp_WeatherxDem,sp_WeatherxSup,1)
print('\n', '-----SupplyxData Break')
"""
biDirectionalLSTMDS(sp_WeatherxDem,sp_WeatherxSup,2)
LSTMModelDS(sp_WeatherxDem,sp_WeatherxSup,2)
GRUModelDS(sp_WeatherxDem,sp_WeatherxSup,2)
SVRModelDS(sp_WeatherxDem,sp_WeatherxSup,2)
MLPModelDS(sp_WeatherxDem,sp_WeatherxSup,2)
CNNModelDS(sp_WeatherxDem,sp_WeatherxSup,2)
GBDTModelDS(sp_WeatherxDem,sp_WeatherxSup,2)
"""

CurrentTopResult, CurrentTopMLName = BestModelChoice(
  decisionTreeModelDS(sp_WeatherxDem,sp_WeatherxSup,1),
  randomForestModelDS(sp_WeatherxDem,sp_WeatherxSup,1),
 'DecisionTree','RandomForest'
 )

##xgbModelDS(sp_DemandDef,sp_SupplyDef,1)

CurrentTopResult, CurrentTopMLName = BestModelChoice(
 CurrentTopResult,
 xgbModelDS(sp_WeatherxDem,sp_WeatherxSup,1),
 CurrentTopMLName,'XGB'
 )

##gbModelDS(sp_DemandDef,sp_SupplyDef,1)

CurrentTopResult, CurrentTopMLName = BestModelChoice(
 CurrentTopResult,
 gbModelDS(sp_WeatherxDem,sp_WeatherxSup,1),
 CurrentTopMLName,'GB'
 )

##biDirectionalLSTMDS(sp_DemandDef,sp_SupplyDef,1)

CurrentTopResult, CurrentTopMLName = BestModelChoice(
CurrentTopResult,
biDirectionalLSTMDS(sp_WeatherxDem,sp_WeatherxSup,1),
CurrentTopMLName,'BSTMD'
)

##LSTMModelDS(sp_DemandDef,sp_SupplyDef,1)

CurrentTopResult, CurrentTopMLName = BestModelChoice(
CurrentTopResult,
LSTMModelDS(sp_WeatherxDem,sp_WeatherxSup,1),
CurrentTopMLName,'LSTM'
 )

##GRUModelDS(sp_DemandDef,sp_SupplyDef,1)

CurrentTopResult, CurrentTopMLName = BestModelChoice(
CurrentTopResult,
GRUModelDS(sp_WeatherxDem,sp_WeatherxSup,1),
CurrentTopMLName,'GRU'
 )

##SVRModelDS(sp_DemandDef,sp_SupplyDef,1)

CurrentTopResult, CurrentTopMLName = BestModelChoice(
CurrentTopResult,
SVRModelDS(sp_WeatherxDem,sp_WeatherxSup,1),
CurrentTopMLName,'SVR'
)

##MLPModelDS(sp_DemandDef,sp_SupplyDef,1)

CurrentTopResult, CurrentTopMLName = BestModelChoice(
CurrentTopResult,
MLPModelDS(sp_WeatherxDem,sp_WeatherxSup,1),
CurrentTopMLName,'MLP'
 )

##CNNModelDS(sp_DemandDef,sp_SupplyDef,1)

CurrentTopResult, CurrentTopMLName = BestModelChoice(CurrentTopResult,
CNNModelDS(sp_WeatherxDem,sp_WeatherxSup,1),
CurrentTopMLName,'CNN'
)

##GBDTModelDS(sp_DemandDef,sp_SupplyDef,1) 

CurrentTopResult, CurrentTopMLName = BestModelChoice(
CurrentTopResult,
GBDTModelDS(sp_WeatherxDem,sp_WeatherxSup,1),
CurrentTopMLName,'GBDT'
 )


print('\n', "Best Model for Demand&Weather DataSet is : " ,CurrentTopMLName)
print("MSE: {:.4f}".format(CurrentTopResult[0]))
print("MAE: {:.4f}".format(CurrentTopResult[1]))
print("RMSE: {:.4f}".format(CurrentTopResult[2]))
print("R2: {:.4f}".format(CurrentTopResult[3]))



#------Supply Functions------
##decisionTreeModelDS(sp_DemandDef,sp_SupplyDef,2)
##randomForestModelDS(sp_DemandDef,sp_SupplyDef,2)

CurrentTopResult, CurrentTopMLName = BestModelChoice(
  decisionTreeModelDS(sp_WeatherxDem,sp_WeatherxSup,2),
  randomForestModelDS(sp_WeatherxDem,sp_WeatherxSup,2),
 'DecisionTree','RandomForest'
 )
##xgbModelDS(sp_DemandDef,sp_SupplyDef,2)

CurrentTopResult, CurrentTopMLName = BestModelChoice(
 CurrentTopResult,
 xgbModelDS(sp_WeatherxDem,sp_WeatherxSup,2),
 CurrentTopMLName,'XGB'
 )

##gbModelDS(sp_DemandDef,sp_SupplyDef,2)

CurrentTopResult, CurrentTopMLName = BestModelChoice(
 CurrentTopResult,
 gbModelDS(sp_WeatherxDem,sp_WeatherxSup,2),
 CurrentTopMLName,'GB'
 )

##biDirectionalLSTMDS(sp_DemandDef,sp_SupplyDef,2)

CurrentTopResult, CurrentTopMLName = BestModelChoice(
CurrentTopResult,
biDirectionalLSTMDS(sp_WeatherxDem,sp_WeatherxSup,2),
CurrentTopMLName,'BSTMD'
)

##LSTMModelDS(sp_DemandDef,sp_SupplyDef,2)

CurrentTopResult, CurrentTopMLName = BestModelChoice(
CurrentTopResult,
LSTMModelDS(sp_WeatherxDem,sp_WeatherxSup,2),
CurrentTopMLName,'LSTM'
 )

##GRUModelDS(sp_DemandDef,sp_SupplyDef,2)

CurrentTopResult, CurrentTopMLName = BestModelChoice(
CurrentTopResult,
GRUModelDS(sp_WeatherxDem,sp_WeatherxSup,2),
CurrentTopMLName,'GRU'
 )

##SVRModelDS(sp_DemandDef,sp_SupplyDef,2)

CurrentTopResult, CurrentTopMLName = BestModelChoice(
CurrentTopResult,
SVRModelDS(sp_WeatherxDem,sp_WeatherxSup,2),
CurrentTopMLName,'SVR'
)

##MLPModelDS(sp_DemandDef,sp_SupplyDef,2)

CurrentTopResult, CurrentTopMLName = BestModelChoice(
CurrentTopResult,
MLPModelDS(sp_WeatherxDem,sp_WeatherxSup,2),
CurrentTopMLName,'MLP'
 )

##CNNModelDS(sp_DemandDef,sp_SupplyDef,2)

CurrentTopResult, CurrentTopMLName = BestModelChoice(CurrentTopResult,
CNNModelDS(sp_WeatherxDem,sp_WeatherxSup,2),
CurrentTopMLName,'CNN'
)

##GBDTModelDS(sp_DemandDef,sp_SupplyDef,2) 

CurrentTopResult, CurrentTopMLName = BestModelChoice(
CurrentTopResult,
GBDTModelDS(sp_WeatherxDem,sp_WeatherxSup,2),
CurrentTopMLName,'GBDT'
 )
"""
Best Model for solo demandxWeather DataSet is : 

--
Best Model for supplyxWeather DataSet is :  -

--
"""
print('\n', "Best Model for Supply&Weather DataSet is : " ,CurrentTopMLName)
print("MSE: {:.4f}".format(CurrentTopResult[0]))
print("MAE: {:.4f}".format(CurrentTopResult[1]))
print("RMSE: {:.4f}".format(CurrentTopResult[2]))
print("R2: {:.4f}".format(CurrentTopResult[3]))


