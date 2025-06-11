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

sp_FullMerg = pd.merge(sp_DemandDef,sp_WeatherDef, on="TimeStamp", how="inner")
sp_FullMerg = pd.merge(sp_SupplyDef,sp_FullMerg, on="TimeStamp", how="inner")
sp_FullMerg = pd.merge(sp_WeatherDe,sp_FullMerg, on="TimeStamp", how="inner")

##linear interpolation to smooth out the datasets 
##So far has worked out great for supply but demand is getting bad results.
sp_FullMerg.interpolate(method='linear', inplace=True)
print(sp_FullMerg.head())
 

#desave_loc = r"E:\AI Lecture Notes\Datasets\Merged\DemandxWeather.csv"
##susave_loc = r"E:\AI Lecture Notes\Datasets\Merged\SupplyxWeather.csv"
#sp_WeatherxDem.to_csv(desave_loc, index=False)
##sp_WeatherxSup.to_csv(susave_loc, index=False)

#------#
def decisionTreeModelDS(mergedDs:pd.DataFrame):
    """
    Decision Tree Results for Merged Dataset -

    --
    """

    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length].flatten())
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)
    
    
    #----#        

    mergedDs.replace(-999,np.nan, inplace=True)
    mergedDs.dropna(inplace=True)

    ###minLength = min(len(demandDs))
    ###demandDs = demandDs.iloc[minLength]

    #Extraction of Data.
  
    mergedDs["hour"] = mergedDs["TimeStamp"].dt.hour   
    mergedDs["day"] = mergedDs["TimeStamp"].dt.day        
    mergedDs["month"] = mergedDs["TimeStamp"].dt.month        
        
    feature_cols = [
    "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
    "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","ALLSKY_SFC_SW_DWN_D","ALLSKY_SFC_UV_INDEX_D","T2M_D","PRECTOTCORR_D","ALLSKY_KT_D",
    "CLRSKY_SFC_PAR_TOT_D","RH2M_D","PS_D","PSC_D","hour","day","month"
]
    """
    feature_cols = [
    "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
    "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","hour","day","month"
]
"""
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW','Supply_MW']
    X = scaler_X.fit_transform(mergedDs[feature_cols])
    y = scaler_y.fit_transform(mergedDs[target_cols])



    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=False
)
    base_model = MultiOutputRegressor(DecisionTreeRegressor(random_state=42))
    base_model.fit(X_train, y_train)

    y_pred = base_model.predict(X_test)

    y_demand_pred = scaler_y.inverse_transform(y_pred)
    y_demand_true = scaler_y.inverse_transform(y_test)


    mse = mean_squared_error(y_demand_true, y_demand_pred)
    mae = mean_absolute_error(y_demand_true, y_demand_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_demand_true, y_demand_pred)
    results = [mse, mae, rmse, r2]
    #----#
    print('\n',"Merged Data Decision Tree Model Results:")
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))

    #-Demand Plot-#
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:,0], label='Actual Demand')
    plt.plot(y_demand_pred[:,0], label='Predicted Demand')
    plt.title(f'Decision Tree Model: Actual vs Predicted Demand')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()
    #-Supply Plot-#
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:,1], label='Actual Supply')
    plt.plot(y_demand_pred[:,1], label='Predicted Supply')
    plt.title(f'Decision Tree Model: Actual vs Predicted Supply')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return results
   
  

def randomForestModelDS(mergedDs:pd.DataFrame):
      
    """
    Random Forest Results for Merged Dataset -

    --
    """
    
    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length].flatten())
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)
     
    mergedDs.replace(-999,np.nan, inplace=True)
    mergedDs.dropna(inplace=True)

    ###minLength = min(len(demandDs))
    ###demandDs = demandDs.iloc[minLength]

    #Extraction of Data.
  
    mergedDs["hour"] = mergedDs["TimeStamp"].dt.hour   
    mergedDs["day"] = mergedDs["TimeStamp"].dt.day        
    mergedDs["month"] = mergedDs["TimeStamp"].dt.month        
        
    feature_cols = [
    "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
    "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","ALLSKY_SFC_SW_DWN_D","ALLSKY_SFC_UV_INDEX_D","T2M_D","PRECTOTCORR_D","ALLSKY_KT_D",
    "CLRSKY_SFC_PAR_TOT_D","RH2M_D","PS_D","PSC_D","hour","day","month"
]
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW','Supply_MW']
    X = scaler_X.fit_transform(mergedDs[feature_cols])
    y = scaler_y.fit_transform(mergedDs[target_cols])



    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=False
)
    base_model = MultiOutputRegressor(RandomForestRegressor(n_estimators =100,random_state=42))
        
    base_model.fit(X_train, y_train)

    y_pred = base_model.predict(X_test)

    y_demand_pred = scaler_y.inverse_transform(y_pred)
    y_demand_true = scaler_y.inverse_transform(y_test)


    mse = mean_squared_error(y_demand_true, y_demand_pred)
    mae = mean_absolute_error(y_demand_true, y_demand_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_demand_true, y_demand_pred)
    results = [mse, mae, rmse, r2]
    #----#
    print('\n',"Merged Random Forest Model Results:")
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))

        #-Demand Plot-#
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:,0], label='Actual Demand')
    plt.plot(y_demand_pred[:,0], label='Predicted Demand')
    plt.title(f'Random Forest Model: Actual vs Predicted Demand')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()
    #-Supply Plot-#
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:,1], label='Actual Supply')
    plt.plot(y_demand_pred[:,1], label='Predicted Supply')
    plt.title(f'Random Forest Model: Actual vs Predicted Supply')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return results
  


def xgbModelDS(mergedDs:pd.DataFrame):
    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length].flatten())
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)
   
    """
    XGB for Merged Dataset -

    --
    """
    #----#        
    mergedDs.replace(-999,np.nan, inplace=True)
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
    "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","ALLSKY_SFC_SW_DWN_D","ALLSKY_SFC_UV_INDEX_D","T2M_D","PRECTOTCORR_D","ALLSKY_KT_D",
    "CLRSKY_SFC_PAR_TOT_D","RH2M_D","PS_D","PSC_D","hour","day","month"
]
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW','Supply_MW']
    X = scaler_X.fit_transform(mergedDs[feature_cols])
    y = scaler_y.fit_transform(mergedDs[target_cols])



    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=False
)
    base_model = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=100, random_state=42))
    ##model = MultiOutputRegressor(base_model)
    base_model.fit(X_train, y_train)

    y_pred = base_model.predict(X_test)

    y_demand_pred = scaler_y.inverse_transform(y_pred)
    y_demand_true = scaler_y.inverse_transform(y_test)

    mse = mean_squared_error(y_demand_true, y_demand_pred)
    mae = mean_absolute_error(y_demand_true, y_demand_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_demand_true, y_demand_pred)
    results = [mse, mae, rmse, r2]

    #----#
    print('\n',"Merged XGB Model Results:")
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))
    #-Demand Plot-#
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:,0], label='Actual Demand')
    plt.plot(y_demand_pred[:,0], label='Predicted Demand')
    plt.title(f'XGB Model: Actual vs Predicted Demand')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()
    #-Supply Plot-#
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:,1], label='Actual Supply')
    plt.plot(y_demand_pred[:,1], label='Predicted Supply')
    plt.title(f'XGB Model: Actual vs Predicted Supply')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return results

def gbModelDS(mergedDs:pd.DataFrame):
    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length].flatten())
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)
   
    """
    GradientBoosting for Merged Dataset -

    --
    """

    mergedDs["hour"] = mergedDs["TimeStamp"].dt.hour   
    mergedDs["day"] = mergedDs["TimeStamp"].dt.day        
    mergedDs["month"] = mergedDs["TimeStamp"].dt.month        
        
    feature_cols = [
    "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
    "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","ALLSKY_SFC_SW_DWN_D","ALLSKY_SFC_UV_INDEX_D","T2M_D","PRECTOTCORR_D","ALLSKY_KT_D",
    "CLRSKY_SFC_PAR_TOT_D","RH2M_D","PS_D","PSC_D","hour","day","month"
]
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW','Supply_MW']
    X = scaler_X.fit_transform(mergedDs[feature_cols])
    y = scaler_y.fit_transform(mergedDs[target_cols])



    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=False
)

    base_model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, random_state=42))
    ##model = MultiOutputRegressor(base_model)
    base_model.fit(X_train, y_train)

    y_pred = base_model.predict(X_test)

    y_demand_pred = scaler_y.inverse_transform(y_pred)
    y_demand_true = scaler_y.inverse_transform(y_test)



    mse = mean_squared_error(y_demand_true, y_demand_pred)
    mae = mean_absolute_error(y_demand_true, y_demand_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_demand_true, y_demand_pred)
    results = [mse, mae, rmse, r2]

    #----#
    print('\n',"Merged GB Model Results:")
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))
    #-Demand Plot-#
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:,0], label='Actual Demand')
    plt.plot(y_demand_pred[:,0], label='Predicted Demand')
    plt.title(f'Gradient Boosting Model: Actual vs Predicted Demand')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()
    #-Supply Plot-#
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:,1], label='Actual Supply')
    plt.plot(y_demand_pred[:,1], label='Predicted Supply')
    plt.title(f'Gradient Boosting Model: Actual vs Predicted Supply')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return results
    

def biDirectionalLSTMDS(mergedDs:pd.DataFrame):
    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)
   
    """
    Bidirectional LSTM for Merged Dataset -

    --
    """

    #----#        
    ##Basic transformation for the base dataset
    mergedDs["hour"] = mergedDs["TimeStamp"].dt.hour   
    mergedDs["day"] = mergedDs["TimeStamp"].dt.day        
    mergedDs["month"] = mergedDs["TimeStamp"].dt.month        
        
    feature_cols = [
    "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
    "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","ALLSKY_SFC_SW_DWN_D","ALLSKY_SFC_UV_INDEX_D","T2M_D","PRECTOTCORR_D","ALLSKY_KT_D",
    "CLRSKY_SFC_PAR_TOT_D","RH2M_D","PS_D","PSC_D","hour","day","month"
]
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW','Supply_MW']
    X = scaler_X.fit_transform(mergedDs[feature_cols])
    y = scaler_y.fit_transform(mergedDs[target_cols])

    seq_length = 24
    X_seq,y_seq = create_sequences_with_time(X,y,seq_length)

    X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.3, shuffle=False
)




    model = Sequential ([
        Bidirectional(LSTM(100,return_sequences=True, input_shape=(seq_length,X_train.shape[2]))),
                        Dropout(0.2),
                        Bidirectional(LSTM(100,return_sequences=False)),
                        Dense(64,activation='relu'),
                        Dense(2)
    ])
       
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10,restore_best_weights=True)

    model.fit(X_train,y_train,epochs=100, batch_size=16, validation_data=(X_test,y_test),
            callbacks =[early_stopping])

    y_pred = model.predict(X_test)

    y_demand_true = scaler_y.inverse_transform(y_test)
    y_demand_pred = scaler_y.inverse_transform(y_pred)


    mse = mean_squared_error(y_demand_true, y_demand_pred)
    mae = mean_absolute_error(y_demand_true, y_demand_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_demand_true, y_demand_pred)
    results = [mse, mae, rmse, r2]

    #----#
    print('\n',"Merged BLSTM Model Results:")
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))
    #-Demand Plot-#
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:,0], label='Actual Demand')
    plt.plot(y_demand_pred[:,0], label='Predicted Demand')
    plt.title(f'BLSTM Model: Actual vs Predicted Demand')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()
    #-Supply Plot-#
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:,1], label='Actual Supply')
    plt.plot(y_demand_pred[:,1], label='Predicted Supply')
    plt.title(f'BLSTM Model: Actual vs Predicted Supply')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return results
    

def LSTMModelDS(mergedDs:pd.DataFrame):
    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)
   
    """
    LSTM for Merged Dataset -

    --
    """

    mergedDs["hour"] = mergedDs["TimeStamp"].dt.hour   
    mergedDs["day"] = mergedDs["TimeStamp"].dt.day        
    mergedDs["month"] = mergedDs["TimeStamp"].dt.month        
        
    feature_cols = [
    "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
    "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","ALLSKY_SFC_SW_DWN_D","ALLSKY_SFC_UV_INDEX_D","T2M_D","PRECTOTCORR_D","ALLSKY_KT_D",
    "CLRSKY_SFC_PAR_TOT_D","RH2M_D","PS_D","PSC_D","hour","day","month"
]
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW','Supply_MW']
    X = scaler_X.fit_transform(mergedDs[feature_cols])
    y = scaler_y.fit_transform(mergedDs[target_cols])

    seq_length = 24
    X_seq,y_seq = create_sequences_with_time(X,y,seq_length)

    X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.3, shuffle=False
)


    model = Sequential ([
        LSTM(64,return_sequences=True, input_shape=(seq_length,X_train.shape[2])),
                        Dropout(0.2),
                        LSTM(100,return_sequences=False),
                        Dense(64),
                        Dense(2)
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
    print('\n',"Merged LSTM Model Results:")
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))
    #-Demand Plot-#
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:,0], label='Actual Demand')
    plt.plot(y_demand_pred[:,0], label='Predicted Demand')
    plt.title(f'LSTM Model: Actual vs Predicted Demand')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()
    #-Supply Plot-#
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:,1], label='Actual Supply')
    plt.plot(y_demand_pred[:,1], label='Predicted Supply')
    plt.title(f'LSTM Tree Model: Actual vs Predicted Supply')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return results
    

def GRUModelDS(mergedDs:pd.DataFrame):
    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)
   
    """
    GRU for Merged Dataset -

    --
    """

    #----#        
    ##Basic transformation for the base dataset
    mergedDs["hour"] = mergedDs["TimeStamp"].dt.hour   
    mergedDs["day"] = mergedDs["TimeStamp"].dt.day        
    mergedDs["month"] = mergedDs["TimeStamp"].dt.month        
        
    feature_cols = [
    "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
    "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","ALLSKY_SFC_SW_DWN_D","ALLSKY_SFC_UV_INDEX_D","T2M_D","PRECTOTCORR_D","ALLSKY_KT_D",
    "CLRSKY_SFC_PAR_TOT_D","RH2M_D","PS_D","PSC_D","hour","day","month"
]
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW','Supply_MW']
    X = scaler_X.fit_transform(mergedDs[feature_cols])
    y = scaler_y.fit_transform(mergedDs[target_cols])

    seq_length = 24
    X_seq,y_seq = create_sequences_with_time(X,y,seq_length)

    X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.3, shuffle=False
)

    model = Sequential ([
        GRU(64,return_sequences=True, input_shape=(seq_length,X_train.shape[2])),
                        Dropout(0.2),
                        GRU(100,return_sequences=False),
                        Dense(64,activation='relu'),
                        Dense(2)
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
    print('\n',"Merged GRU Model Results:")
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))
    #-Demand Plot-#
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:,0], label='Actual Demand')
    plt.plot(y_demand_pred[:,0], label='Predicted Demand')
    plt.title(f'GRU Model: Actual vs Predicted Demand')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()
    #-Supply Plot-#
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:,1], label='Actual Supply')
    plt.plot(y_demand_pred[:,1], label='Predicted Supply')
    plt.title(f'GRU Model: Actual vs Predicted Supply')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return results 
   
def SVRModelDS(mergedDs:pd.DataFrame):
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
    
    SVRfor Merged Dataset -

    --
    """
    
    mergedDs["hour"] = mergedDs["TimeStamp"].dt.hour   
    mergedDs["day"] = mergedDs["TimeStamp"].dt.day        
    mergedDs["month"] = mergedDs["TimeStamp"].dt.month        
        
    feature_cols = [
    "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
    "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","ALLSKY_SFC_SW_DWN_D","ALLSKY_SFC_UV_INDEX_D","T2M_D","PRECTOTCORR_D","ALLSKY_KT_D",
    "CLRSKY_SFC_PAR_TOT_D","RH2M_D","PS_D","PSC_D","hour","day","month"
]
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW','Supply_MW']
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
    print('\n',"Merged SVR Model Results:")
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))
    #-Demand Plot-#
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:,0], label='Actual Demand')
    plt.plot(y_demand_pred[:,0], label='Predicted Demand')
    plt.title(f'SVR Model: Actual vs Predicted Demand')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()
    #-Supply Plot-#
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:,1], label='Actual Supply')
    plt.plot(y_demand_pred[:,1], label='Predicted Supply')
    plt.title(f'SVR Model: Actual vs Predicted Supply')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return results 
    
def MLPModelDS(mergedDs: pd.DataFrame):
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
        "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
        "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","ALLSKY_SFC_SW_DWN_D","ALLSKY_SFC_UV_INDEX_D","T2M_D","PRECTOTCORR_D","ALLSKY_KT_D",
        "CLRSKY_SFC_PAR_TOT_D","RH2M_D","PS_D","PSC_D","hour","day","month"
    ]
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW','Supply_MW']
    X = scaler_X.fit_transform(mergedDs[feature_cols])
    y = scaler_y.fit_transform(mergedDs[target_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )

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
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:,0], label='Actual Demand')
    plt.plot(y_demand_pred[:,0], label='Predicted Demand')
    plt.title(f'MLP Model: Actual vs Predicted Demand')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:,1], label='Actual Supply')
    plt.plot(y_demand_pred[:,1], label='Predicted Supply')
    plt.title(f'MLP Model: Actual vs Predicted Supply')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return results


def CNNModelDS(mergedDs:pd.DataFrame):
    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)

    """
    CNN for Merged Dataset -

    --
    """
    mergedDs["hour"] = mergedDs["TimeStamp"].dt.hour   
    mergedDs["day"] = mergedDs["TimeStamp"].dt.day        
    mergedDs["month"] = mergedDs["TimeStamp"].dt.month        
        
    feature_cols = [
    "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
    "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","ALLSKY_SFC_SW_DWN_D","ALLSKY_SFC_UV_INDEX_D","T2M_D","PRECTOTCORR_D","ALLSKY_KT_D",
    "CLRSKY_SFC_PAR_TOT_D","RH2M_D","PS_D","PSC_D","hour","day","month"
]
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW','Supply_MW']
    X = scaler_X.fit_transform(mergedDs[feature_cols])
    y = scaler_y.fit_transform(mergedDs[target_cols])

    seq_length = 24
    X_seq,y_seq = create_sequences_with_time(X,y,seq_length)

    X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.3, shuffle=False
)



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
    #-Demand Plot-#
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:,0], label='Actual Demand')
    plt.plot(y_demand_pred[:,0], label='Predicted Demand')
    plt.title(f'CNN Model: Actual vs Predicted Demand')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()
    #-Supply Plot-#
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:,1], label='Actual Supply')
    plt.plot(y_demand_pred[:,1], label='Predicted Supply')
    plt.title(f'CNN Model: Actual vs Predicted Supply')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return results


def GBDTModelDS(mergedDs:pd.DataFrame):
    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length].flatten())
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)
   
    """
    GBDT for Merged Dataset -

    --
    """

    mergedDs["hour"] = mergedDs["TimeStamp"].dt.hour   
    mergedDs["day"] = mergedDs["TimeStamp"].dt.day        
    mergedDs["month"] = mergedDs["TimeStamp"].dt.month        
        
    feature_cols = [
    "ALLSKY_SFC_SW_DWN_S","ALLSKY_SFC_UV_INDEX_S","T2M_S","PRECTOTCORR_S","ALLSKY_KT_S",
    "CLRSKY_SFC_PAR_TOT_S","RH2M_S","PS_S","PSC_S","ALLSKY_SFC_SW_DWN_D","ALLSKY_SFC_UV_INDEX_D","T2M_D","PRECTOTCORR_D","ALLSKY_KT_D",
    "CLRSKY_SFC_PAR_TOT_D","RH2M_D","PS_D","PSC_D","hour","day","month"
]
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    target_cols = ['Demand_MW','Supply_MW']
    X = scaler_X.fit_transform(mergedDs[feature_cols])
    y = scaler_y.fit_transform(mergedDs[target_cols])



    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=False
)

    model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))
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
    #-Demand Plot-#
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:,0], label='Actual Demand')
    plt.plot(y_demand_pred[:,0], label='Predicted Demand')
    plt.title(f'GBDT Model: Actual vs Predicted Demand')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()
    #-Supply Plot-#
    plt.figure(figsize=(10, 5))
    plt.plot(y_demand_true[:,1], label='Actual Supply')
    plt.plot(y_demand_pred[:,1], label='Predicted Supply')
    plt.title(f'GBDT Model: Actual vs Predicted Supply')
    plt.xlabel('Time')
    plt.ylabel('MW')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return results

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
"""
decisionTreeModelDS(sp_FullMerg)

randomForestModelDS(sp_FullMerg)
xgbModelDS(sp_FullMerg)
gbModelDS(sp_FullMerg)
xgbModelDS(sp_FullMerg)
gbModelDS(sp_FullMerg)
biDirectionalLSTMDS(sp_FullMerg)
LSTMModelDS(sp_FullMerg)
GRUModelDS(sp_FullMerg)
SVRModelDS(sp_FullMerg)
MLPModelDS(sp_FullMerg)
CNNModelDS(sp_FullMerg)
GBDTModelDS(sp_FullMerg)
"""


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
MSE: 1644.6988
MAE: 30.8627
RMSE: 40.5549
R2: 0.4143
--
Best Result - No Hyper Params -



--
Best Result - No Lag -



--
Best Result - With Both Hyper Params & Lag -



--
"""