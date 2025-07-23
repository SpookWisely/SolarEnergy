#----# Libraries
from enum import Enum
import enum
from unicodedata import bidirectional
import numpy as np
import pandas as pd
import scipy as sp
import xgboost as xgb
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional,GRU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from keras_tuner import RandomSearch, Hyperband,BayesianOptimization
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from deap import base, creator, tools, algorithms   

##Models to Do
#######XGBoost,CNN,LTSM,BLSTM,RandomForest,GradientBoosting,SVR,DecisionTreeRegressor,MultiOutputRegressor
##Season Terms for Reference (Spring = March,April,May),Summer = (June,July,August),Autumn = (September,October,November) and Winter = (December,January,Febuary)
#----# Dataset imports and minor transformations.

sp_DemandDef = pd.read_excel(r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\Sakakah 2021 Demand dataset.xlsx")
sp_DemandTest = pd.read_excel(r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\Sakakah 2021 Demand dataset.xlsx")

#print(sp_Demand.head())
sp_SupplyDef = pd.read_excel(r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\Sakakah 2021 PV supply dataset.xlsx")
sp_SupplyTest = pd.read_excel(r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\Sakakah 2021 PV supply dataset.xlsx")
#print(sp_Supply.head())
sp_Weather = pd.read_csv(r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\weather for solar 2021.csv", skiprows=17, encoding='latin1')
##print(sp_Weather.head())



#----# Demand datset transformations
##Need to look up how get the .str.split to work with csv files instead of XLSX files.
sp_DemandTest ['MW'] = pd.to_numeric(sp_DemandTest['MW'],errors='coerce')
sp_DemandTest['DATE-TIME'] = sp_DemandTest['DATE-TIME'].astype(str)

sp_DemandTest[['Date', 'Time']] = sp_DemandTest['DATE-TIME'].str.split(' ',expand=True)
sp_DemandTest.drop('DATE-TIME', axis =1, inplace=True)
#print(sp_Demand.head())

#----# Supply dataset transformations

sp_SupplyTest ['MW'] = pd.to_numeric(sp_DemandTest['MW'], errors='coerce')
sp_SupplyTest['Date & Time'] = sp_SupplyTest['Date & Time'].astype(str)

sp_SupplyTest[['Date','Time']] = sp_SupplyTest['Date & Time'].str.split(' ',expand=True)
sp_SupplyTest.drop('Date & Time', axis =1, inplace=True)
#print(sp_Supply.head())

sp_DemandTest.set_index('Date',inplace=True)
sp_SupplyTest.set_index('Date',inplace=True)

#----# Averages for Demand and Supply datasets

sp_DemandTest['Day_Avg'] = sp_DemandTest['MW'].rolling(window=24).mean()
sp_DemandTest['Week_Avg'] = sp_DemandTest['MW'].rolling(window=24*7).mean()
sp_DemandTest['Month_Avg'] = sp_DemandTest['MW'].rolling(window=24*30).mean()

sp_SupplyTest['Day_Avg'] = sp_SupplyTest['MW'].rolling(window=24).mean()
sp_SupplyTest['Week_Avg'] = sp_SupplyTest['MW'].rolling(window=24*7).mean()
sp_SupplyTest['Month_Avg'] = sp_SupplyTest['MW'].rolling(window=24*30).mean()
#----#printing out to check the averages
sp_DemandTest.reset_index(inplace=True)
sp_SupplyTest.reset_index(inplace=True)



sp_DemandTest['Day_Avg'] = sp_DemandTest['Day_Avg'].fillna(0)
sp_DemandTest['Week_Avg'] = sp_DemandTest['Week_Avg'].fillna(0)
sp_DemandTest['Month_Avg'] = sp_DemandTest['Month_Avg'].fillna(0)


sp_SupplyTest['Day_Avg'] = sp_SupplyTest['Day_Avg'].fillna(0)
sp_SupplyTest['Week_Avg'] = sp_SupplyTest['Week_Avg'].fillna(0)
sp_SupplyTest['Month_Avg'] = sp_SupplyTest['Month_Avg'].fillna(0)

#print(sp_Demand.head())
#print(sp_Supply.head())
class Season(Enum):
        SPRING = 'Spring'
        SUMMER = 'Summer'
        AUTUMN = 'Autumn'
        WINTER = 'Winter'

def randomForestModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame):

    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length].flatten())
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)
    
    #----#        
    demandDs['DATE-TIME'] = demandDs['DATE-TIME'].astype(str)
    demandDs[['Date', 'Time']] = demandDs['DATE-TIME'].str.split(' ',expand=True)
    demandDs ['MW'] = pd.to_numeric(demandDs['MW'],errors='coerce')
    demandDs['DATE-TIME'] = pd.to_datetime(demandDs['DATE-TIME'], errors='coerce')


    supplyDs['Date & Time'] = supplyDs['Date & Time'].astype(str)
    supplyDs[['Date','Time']] = supplyDs['Date & Time'].str.split(' ',expand=True)
    supplyDs ['MW'] = pd.to_numeric(supplyDs['MW'], errors='coerce')
    supplyDs['Date & Time'] = pd.to_datetime(supplyDs['Date & Time'], errors='coerce')

    demandDs.dropna(subset=['MW','DATE-TIME'], inplace=True)
    supplyDs.dropna(subset=['MW','Date & Time'], inplace=True)


    min_length = min(len(demandDs), len(supplyDs))
    demandDs = demandDs.iloc[:min_length]
    supplyDs = supplyDs.iloc[:min_length]

    #Extraction of Data.

    demand = demandDs['MW'].values.reshape(-1,1)
    supply = supplyDs['MW'].values.reshape(-1,1)
    datetime_series = demandDs['DATE-TIME']

    hour = datetime_series.dt.hour.values.reshape(-1,1)
    dayofweek = datetime_series.dt.dayofweek.values.reshape(-1,1)
    month = datetime_series.dt.month.values.reshape(-1,1)

    scaler_demand = MinMaxScaler()
    scaler_supply = MinMaxScaler()
    scaler_hour = MinMaxScaler()
    scaler_day = MinMaxScaler()
    scaler_month = MinMaxScaler()

    demand_scaled = scaler_demand.fit_transform(demand)
    supply_scaled = scaler_supply.fit_transform(supply)
    hour_scaled = scaler_hour.fit_transform(hour)
    day_scaled = scaler_day.fit_transform(dayofweek)
    month_scaled = scaler_month.fit_transform(month)

    full_data = np.hstack((demand_scaled, supply_scaled, hour_scaled, day_scaled, month_scaled))
    targets = np.hstack((demand_scaled, supply_scaled))

    seq_length = 24
    X, y = create_sequences_with_time(full_data, targets, seq_length)


    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    base_model = RandomForestRegressor(n_estimators=100, random_state=42)
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)


    y_demand_pred = scaler_demand.inverse_transform(y_pred[:, 0].reshape(-1, 1))
    y_supply_pred = scaler_supply.inverse_transform(y_pred[:, 1].reshape(-1, 1))
    y_demand_true = scaler_demand.inverse_transform(y_test[:, 0].reshape(-1, 1))
    y_supply_true = scaler_supply.inverse_transform(y_test[:, 1].reshape(-1, 1))



    combined_true = np.concatenate((y_demand_true, y_supply_true), axis=0)
    combined_pred = np.concatenate((y_demand_pred, y_supply_pred), axis=0)



    mse = mean_squared_error(combined_true, combined_pred)
    mae = mean_absolute_error(combined_true, combined_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(combined_true, combined_pred)
    #----#
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))

    return 

def decisionTreeModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame):

    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length].flatten())
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)
    
    #----#        
    demandDs['DATE-TIME'] = demandDs['DATE-TIME'].astype(str)
    demandDs[['Date', 'Time']] = demandDs['DATE-TIME'].str.split(' ',expand=True)
    demandDs ['MW'] = pd.to_numeric(demandDs['MW'],errors='coerce')
    demandDs['DATE-TIME'] = pd.to_datetime(demandDs['DATE-TIME'], errors='coerce')


    supplyDs['Date & Time'] = supplyDs['Date & Time'].astype(str)
    supplyDs[['Date','Time']] = supplyDs['Date & Time'].str.split(' ',expand=True)
    supplyDs ['MW'] = pd.to_numeric(supplyDs['MW'], errors='coerce')
    supplyDs['Date & Time'] = pd.to_datetime(supplyDs['Date & Time'], errors='coerce')

    demandDs.dropna(subset=['MW','DATE-TIME'], inplace=True)
    supplyDs.dropna(subset=['MW','Date & Time'], inplace=True)


    min_length = min(len(demandDs), len(supplyDs))
    demandDs = demandDs.iloc[:min_length]
    supplyDs = supplyDs.iloc[:min_length]

    #Extraction of Data.

    demand = demandDs['MW'].values.reshape(-1,1)
    supply = supplyDs['MW'].values.reshape(-1,1)
    datetime_series = demandDs['DATE-TIME']

    hour = datetime_series.dt.hour.values.reshape(-1,1)
    dayofweek = datetime_series.dt.dayofweek.values.reshape(-1,1)
    month = datetime_series.dt.month.values.reshape(-1,1)

    scaler_demand = MinMaxScaler()
    scaler_supply = MinMaxScaler()
    scaler_hour = MinMaxScaler()
    scaler_day = MinMaxScaler()
    scaler_month = MinMaxScaler()

    demand_scaled = scaler_demand.fit_transform(demand)
    supply_scaled = scaler_supply.fit_transform(supply)
    hour_scaled = scaler_hour.fit_transform(hour)
    day_scaled = scaler_day.fit_transform(dayofweek)
    month_scaled = scaler_month.fit_transform(month)

    full_data = np.hstack((demand_scaled, supply_scaled, hour_scaled, day_scaled, month_scaled))
    targets = np.hstack((demand_scaled, supply_scaled))

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
    y_supply_pred = scaler_supply.inverse_transform(y_pred[:, 1].reshape(-1, 1))
    y_demand_true = scaler_demand.inverse_transform(y_test[:, 0].reshape(-1, 1))
    y_supply_true = scaler_supply.inverse_transform(y_test[:, 1].reshape(-1, 1))



    combined_true = np.concatenate((y_demand_true, y_supply_true), axis=0)
    combined_pred = np.concatenate((y_demand_pred, y_supply_pred), axis=0)



    mse = mean_squared_error(combined_true, combined_pred)
    mae = mean_absolute_error(combined_true, combined_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(combined_true, combined_pred)
    #----#
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))

    return 

def xgBoostModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame):

    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length].flatten())
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)
    
    #----#        
    demandDs['DATE-TIME'] = demandDs['DATE-TIME'].astype(str)
    demandDs[['Date', 'Time']] = demandDs['DATE-TIME'].str.split(' ',expand=True)
    demandDs ['MW'] = pd.to_numeric(demandDs['MW'],errors='coerce')
    demandDs['DATE-TIME'] = pd.to_datetime(demandDs['DATE-TIME'], errors='coerce')


    supplyDs['Date & Time'] = supplyDs['Date & Time'].astype(str)
    supplyDs[['Date','Time']] = supplyDs['Date & Time'].str.split(' ',expand=True)
    supplyDs ['MW'] = pd.to_numeric(supplyDs['MW'], errors='coerce')
    supplyDs['Date & Time'] = pd.to_datetime(supplyDs['Date & Time'], errors='coerce')

    demandDs.dropna(subset=['MW','DATE-TIME'], inplace=True)
    supplyDs.dropna(subset=['MW','Date & Time'], inplace=True)


    min_length = min(len(demandDs), len(supplyDs))
    demandDs = demandDs.iloc[:min_length]
    supplyDs = supplyDs.iloc[:min_length]

    #Extraction of Data.

    demand = demandDs['MW'].values.reshape(-1,1)
    supply = supplyDs['MW'].values.reshape(-1,1)
    datetime_series = demandDs['DATE-TIME']

    hour = datetime_series.dt.hour.values.reshape(-1,1)
    dayofweek = datetime_series.dt.dayofweek.values.reshape(-1,1)
    month = datetime_series.dt.month.values.reshape(-1,1)

    scaler_demand = MinMaxScaler()
    scaler_supply = MinMaxScaler()
    scaler_hour = MinMaxScaler()
    scaler_day = MinMaxScaler()
    scaler_month = MinMaxScaler()
    

    demand_scaled = scaler_demand.fit_transform(demand)
    supply_scaled = scaler_supply.fit_transform(supply)
    hour_scaled = scaler_hour.fit_transform(hour)
    day_scaled = scaler_day.fit_transform(dayofweek)
    month_scaled = scaler_month.fit_transform(month)

    full_data = np.hstack((demand_scaled, supply_scaled,hour_scaled, day_scaled, month_scaled ))
    targets = np.hstack((demand_scaled, supply_scaled))

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
    y_supply_pred = scaler_supply.inverse_transform(y_pred[:, 1].reshape(-1, 1))
    y_demand_true = scaler_demand.inverse_transform(y_test[:, 0].reshape(-1, 1))
    y_supply_true = scaler_supply.inverse_transform(y_test[:, 1].reshape(-1, 1))



    combined_true = np.concatenate((y_demand_true, y_supply_true), axis=0)
    combined_pred = np.concatenate((y_demand_pred, y_supply_pred), axis=0)



    mse = mean_squared_error(combined_true, combined_pred)
    mae = mean_absolute_error(combined_true, combined_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(combined_true, combined_pred)
    #----#
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))

    return 

def decisionTreeModelDSYearlyTraining(demandDs:pd.DataFrame,supplyDs:pd.DataFrame):

    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length].flatten())
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)
    
    #----#        
    demandDs['DATE-TIME'] = demandDs['DATE-TIME'].astype(str)
    demandDs[['Date', 'Time']] = demandDs['DATE-TIME'].str.split(' ',expand=True)
    demandDs ['MW'] = pd.to_numeric(demandDs['MW'],errors='coerce')
    demandDs['DATE-TIME'] = pd.to_datetime(demandDs['DATE-TIME'], errors='coerce')


    supplyDs['Date & Time'] = supplyDs['Date & Time'].astype(str)
    supplyDs[['Date','Time']] = supplyDs['Date & Time'].str.split(' ',expand=True)
    supplyDs ['MW'] = pd.to_numeric(supplyDs['MW'], errors='coerce')
    supplyDs['Date & Time'] = pd.to_datetime(supplyDs['Date & Time'], errors='coerce')

    demandDs.dropna(subset=['MW','DATE-TIME'], inplace=True)
    supplyDs.dropna(subset=['MW','Date & Time'], inplace=True)


    min_length = min(len(demandDs), len(supplyDs))
    demandDs = demandDs.iloc[:min_length]
    supplyDs = supplyDs.iloc[:min_length]

    #Extraction of Data.

    demand = demandDs['MW'].values.reshape(-1,1)
    supply = supplyDs['MW'].values.reshape(-1,1)
    datetime_series = demandDs['DATE-TIME']


    hour = datetime_series.dt.hour.values.reshape(-1,1)
    dayofweek = datetime_series.dt.dayofweek.values.reshape(-1,1)
    month = datetime_series.dt.month.values.reshape(-1,1)
    year = datetime_series.dt.year.values.reshape(-1,1)

    scaler_demand = MinMaxScaler()
    scaler_supply = MinMaxScaler()
    scaler_hour = MinMaxScaler()
    scaler_day = MinMaxScaler()
    scaler_month = MinMaxScaler()
    scaler_year = MinMaxScaler()

    demand_scaled = scaler_demand.fit_transform(demand)
    supply_scaled = scaler_supply.fit_transform(supply)
    hour_scaled = scaler_hour.fit_transform(hour)
    day_scaled = scaler_day.fit_transform(dayofweek)
    month_scaled = scaler_month.fit_transform(month)
    year_scaled = scaler_year.fit_transform(year)


    full_data = np.hstack((demand_scaled, supply_scaled, year_scaled))
    targets = np.hstack((demand_scaled, supply_scaled))

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
    y_supply_pred = scaler_supply.inverse_transform(y_pred[:, 1].reshape(-1, 1))
    y_demand_true = scaler_demand.inverse_transform(y_test[:, 0].reshape(-1, 1))
    y_supply_true = scaler_supply.inverse_transform(y_test[:, 1].reshape(-1, 1))



    combined_true = np.concatenate((y_demand_true, y_supply_true), axis=0)
    combined_pred = np.concatenate((y_demand_pred, y_supply_pred), axis=0)



    mse = mean_squared_error(combined_true, combined_pred)
    mae = mean_absolute_error(combined_true, combined_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(combined_true, combined_pred)
    #----#
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))

    return 

def decisionTreeModelSeasonalTraining(demandDs:pd.DataFrame,supplyDs:pd.DataFrame):

    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length].flatten())
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)
    
    #----#        
    demandDs['DATE-TIME'] = demandDs['DATE-TIME'].astype(str)
    demandDs[['Date', 'Time']] = demandDs['DATE-TIME'].str.split(' ',expand=True)
    demandDs ['MW'] = pd.to_numeric(demandDs['MW'],errors='coerce')
    demandDs['DATE-TIME'] = pd.to_datetime(demandDs['DATE-TIME'], errors='coerce')


    supplyDs['Date & Time'] = supplyDs['Date & Time'].astype(str)
    supplyDs[['Date','Time']] = supplyDs['Date & Time'].str.split(' ',expand=True)
    supplyDs ['MW'] = pd.to_numeric(supplyDs['MW'], errors='coerce')
    supplyDs['Date & Time'] = pd.to_datetime(supplyDs['Date & Time'], errors='coerce')


    demandDs.dropna(subset=['MW','DATE-TIME'], inplace=True)
    supplyDs.dropna(subset=['MW','Date & Time'], inplace=True)

    def get_season(month):
        if month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Autumn'
        else:
            return 'Winter'

    demandDs['Season'] = demandDs['DATE-TIME'].dt.month.apply(get_season)
    supplyDs['Season'] = supplyDs['Date & Time'].dt.month.apply(get_season)

    min_length = min(len(demandDs), len(supplyDs))
    demandDs = demandDs.iloc[:min_length]
    supplyDs = supplyDs.iloc[:min_length]

    #Extraction of Data.

   
    ##Most likely a more efficent way to do this but for now this will do.

    supplySds = supplyDs[supplyDs['Season'] == 'Spring']
    supplySmds = supplyDs[supplyDs['Season'] == 'Summer']
    supplyAds = supplyDs[supplyDs['Season'] == 'Autumn']
    supplyWds = supplyDs[supplyDs['Season'] == 'Winter']

    demandSds = demandDs[demandDs['Season'] == 'Spring']
    demandSmds = demandDs[demandDs['Season'] == 'Summer']
    demandAds = demandDs[demandDs['Season'] == 'Autumn']
    demandWds = demandDs[demandDs['Season'] == 'Winter']

    datetime_seriesS = demandSds['DATE-TIME']
    datetime_seriesSm = demandSmds['DATE-TIME']
    datetime_seriesA = demandAds['DATE-TIME'] 
    datetime_seriesW = demandWds['DATE-TIME']

    demandS = demandSds['MW'].values.reshape(-1,1)
    demandSm = demandSmds['MW'].values.reshape(-1,1)
    demandA = demandAds['MW'].values.reshape(-1,1)
    demandW = demandWds['MW'].values.reshape(-1,1)

    supplyS = supplySds['MW'].values.reshape(-1,1)
    supplySm = supplySmds['MW'].values.reshape(-1,1)
    supplyA = supplyAds['MW'].values.reshape(-1,1)
    supplyW = supplyWds['MW'].values.reshape(-1,1)

    
    min_lengthS = min(len(demandS), len(supplyS))
    demandS = demandS[:min_lengthS]
    supplyS = supplyS[:min_lengthS]

    min_lengthSm = min(len(demandSm), len(supplySm))
    demandSm = demandSm[:min_lengthSm]
    supplySm = supplySm[:min_lengthSm]

    min_lengthA = min(len(demandA), len(supplyA))
    demandA = demandA[:min_lengthA]
    supplyA = supplyA[:min_lengthA]

    min_lengthW = min(len(demandW), len(supplyW))
    demandW = demandW[:min_lengthW]
    supplyW = supplyW[:min_lengthW]


    hour = datetime_seriesS.dt.hour.values.reshape(-1,1)
    dayofweek = datetime_seriesS.dt.dayofweek.values.reshape(-1,1)
    month = datetime_seriesS.dt.month.values.reshape(-1,1)
    """
    hourSm = datetime_seriesSm.dt.hour.values.reshape(-1,1)
    dayofweekSm = datetime_seriesSm.dt.dayofweek.values.reshape(-1,1)
    monthSm = datetime_seriesSm.dt.month.values.reshape(-1,1)

    hourA  = datetime_seriesA.dt.hour.values.reshape(-1,1)
    dayofweekA  = datetime_seriesA.dt.dayofweek.values.reshape(-1,1)
    monthA = datetime_seriesA.dt.month.values.reshape(-1,1)

    hourW = datetime_seriesW.dt.hour.values.reshape(-1,1)
    dayofweekW = datetime_seriesW.dt.dayofweek.values.reshape(-1,1)
    monthW = datetime_seriesW.dt.month.values.reshape(-1,1)
    """
    I = 0
    scaler_demand = MinMaxScaler()
    scaler_supply = MinMaxScaler()
    scaler_hour = MinMaxScaler()
    scaler_day = MinMaxScaler()
    scaler_month = MinMaxScaler()
    while I  < 4:

        
        if   (I == 0):
            demand = demandS
            supply = supplyS
         
        elif (I == 1):
            demand = demandSm
            supply = supplySm
           
        elif (I == 2):
            demand = demandA
            supply = supplyA
            
        elif (I == 3):
            demand = demandW
            supply = supplyW
            
        demand_scaled = scaler_demand.fit_transform(demand)
        supply_scaled = scaler_supply.fit_transform(supply)
        hour_scaled = scaler_hour.fit_transform(hour)
        day_scaled = scaler_day.fit_transform(dayofweek)
        month_scaled = scaler_month.fit_transform(month)
        #year_scaled = scaler_year.fit_transform(year)


        full_data = np.hstack((demand_scaled, supply_scaled, ))
        targets = np.hstack((demand_scaled, supply_scaled))

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
        y_supply_pred = scaler_supply.inverse_transform(y_pred[:, 1].reshape(-1, 1))
        y_demand_true = scaler_demand.inverse_transform(y_test[:, 0].reshape(-1, 1))
        y_supply_true = scaler_supply.inverse_transform(y_test[:, 1].reshape(-1, 1))



        combined_true = np.concatenate((y_demand_true, y_supply_true), axis=0)
        combined_pred = np.concatenate((y_demand_pred, y_supply_pred), axis=0)



        mse = mean_squared_error(combined_true, combined_pred)
        mae = mean_absolute_error(combined_true, combined_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(combined_true, combined_pred)
        #----#
        if   (I == 0):
            print("-Spring Model Values-", '\n')
            print("MSE: {:.4f}".format(mse))
            print("MAE: {:.4f}".format(mae))
            print("RMSE: {:.4f}".format(rmse))
            print("R2: {:.4f}".format(r2))
        elif (I == 1):
            print("-Summer Model Values-", '\n')
            print("MSE: {:.4f}".format(mse))
            print("MAE: {:.4f}".format(mae))
            print("RMSE: {:.4f}".format(rmse))
            print("R2: {:.4f}".format(r2))
        elif (I == 2):
            print("-Autumn Model Values-", '\n')
            print("MSE: {:.4f}".format(mse))
            print("MAE: {:.4f}".format(mae))
            print("RMSE: {:.4f}".format(rmse))
            print("R2: {:.4f}".format(r2))
        elif (I == 3):
            print("-Winter Model Values-", '\n')
            print("MSE: {:.4f}".format(mse))
            print("MAE: {:.4f}".format(mae))
            print("RMSE: {:.4f}".format(rmse))
            print("R2: {:.4f}".format(r2))
    
        I += 1
        
        
    return 


def BLSTMModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame):

    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)
    
    #----#        

    demandDs['Timestamp'] = pd.to_datetime(demandDs['DATE-TIME'])
    supplyDs['Timestamp'] = pd.to_datetime(supplyDs['Date & Time'])

    ##Creates new dataset merged from both supply and demand and adds the additons of _demand 
    ## supply to the column names for extraction later on.
    mergedDs = pd.merge(demandDs[['Timestamp', 'MW']], supplyDs[['Timestamp', 'MW']], 
                        on='Timestamp', suffixes=('_demand', '_supply'))

    #Extraction of Data.

    mergedDs['MW_demand'] = pd.to_numeric(mergedDs['MW_demand'], errors = 'coerce')
    mergedDs['MW_supply'] = pd.to_numeric(mergedDs['MW_supply'], errors = 'coerce')
    mergedDs.dropna(inplace=True)

    mergedDs['hour'] = mergedDs['Timestamp'].dt.hour
    mergedDs['DOW'] = mergedDs['Timestamp'].dt.dayofweek
    mergedDs['month'] = mergedDs['Timestamp'].dt.month


    features = mergedDs[['MW_demand', 'MW_supply','hour','DOW','month']]
    targets = mergedDs[['MW_demand', 'MW_supply']].shift(-1)

    features = features.iloc[:-1]
    targets = targets.iloc[:-1]

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_x.fit_transform(features)
    y_scaled = scaler_y.fit_transform(targets)

    ##Same as all prior models just slight difference in the way the data is extracted.
    seq_length = 24
    X, y = create_sequences_with_time(X_scaled,y_scaled, seq_length)


    split_indx  = int(len(X) * 0.8)
    X_test, X_train = X[:split_indx], X[split_indx:]  
    y_test, y_train = y[:split_indx], y[split_indx:]
   
    """
    def define_Bimodel(seq_length, input_dim,neurons,dropoutrate):
        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True, input_shape=(seq_length, input_dim))),
            Dropout(0.2),
            Bidirectional(LSTM(64, return_sequences=False)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(2)
        ])
        return model

    dropoutrate = [0.0,0.2,0.4]
    neurons = [32,64,128]
    batchsize = [16,18,20]
    epochs = [50,100,150]

    paramgrid = {'neurons':neurons,'dropoutrate':dropoutrate,'batchsize':batchsize,'epochs':epochs}
    model = define_Bimodel(seq_length, X_train.shape[2],)

    grid_search = GridSearchCV(estimator=model, param_grid=paramgrid, cv=3, scoring='neg_mean_squared_error', verbose=1)
    """
    """
    model = Sequential([
        Bidirectional(LSTM(100,return_sequences=True,input_shape=(seq_length, X_train.shape[2]))),
        Dropout(0.2),
        Bidirectional(LSTM(100,return_sequences=False)),
        Dropout(0.2),
        Dense(64,activation='relu'),
        Dense(2)
        ])
    """
    model = Sequential([
    Bidirectional(LSTM(100, return_sequences=True, input_shape=(seq_length, X_train.shape[2]))),
    Dropout(0.2),
    Bidirectional(LSTM(100, return_sequences=False)),
    Dense(64, activation='relu'),
    Dense(2)  # Output: demand and supply
])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(X_train,y_train,epochs=100, batch_size=16, validation_data=(X_test,y_test),
             callbacks =[early_stopping])


    y_pred = model.predict(X_test)

    y_pred_inv = scaler_y.inverse_transform(y_pred)
    y_test_inv = scaler_y.inverse_transform(y_test)

    mse = mean_squared_error(y_test_inv,y_pred_inv)
    mae = mean_absolute_error(y_test_inv,y_pred_inv)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_inv,y_pred_inv)
    #----#
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))

    return 

def LTSMModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame):

 
    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)
    
    #----#        
  
    demandDs['Timestamp'] = pd.to_datetime(demandDs['DATE-TIME'], errors='coerce')
    supplyDs['Timestamp'] = pd.to_datetime(supplyDs['Date & Time'], errors='coerce')

    ##Creates new dataset merged from both suppl and demand and adds the additons of _demand 
    ## supply to the column names for extraction later on.
    mergedDs = pd.merge(demandDs[['Timestamp', 'MW']], supplyDs[['Timestamp', 'MW']], 
                        on='Timestamp', suffixes=('_demand', '_supply'))
   ## demandDs.dropna(subset=['MW','Timestamp'], inplace=True)
   ## supplyDs.dropna(subset=['MW','Timestamp'], inplace=True)


    min_length = min(len(demandDs), len(supplyDs))
    demandDs = demandDs.iloc[:min_length]
    supplyDs = supplyDs.iloc[:min_length]

    #Extraction of Data.

    mergedDs = pd.to_numeric(mergedDs['MW_demand'], errors = 'coerce')
    mergedDS = pd.to_numeric(mergedDs['MW_supply'], errors = 'coerce')
    datetime_series = demandDs['DATE-TIME']

    mergedDs['hour'] = mergedDs['Timestamp'].dt.hour
    mergedDs['DOW'] = mergedDs['Timestamp'].dt.dayofweek
    mergedDs['month'] = mergedDs['Timestamp'].dt.month


    features = mergedDs[['MW_demand', 'MW_supply','hour','dow','month']]
    targets = mergedDs[['MW_demand', 'MW_supply']].shift(-1)


    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_x.fit_transform(features)
    y_scaled = scaler_y.fit_transform(targets)

    ##Same as all prior models just slight difference in the way the data is extracted.
    seq_length = 24
    X, y = create_sequences_with_time(X_scaled,y_scaled, seq_length)


    split_idx  = int(len(seq_length) * 0.8)
    X_test, X_train = X[:split_idx], X[split_idx:]  
    y_test, y_train = y[:split_idx], y[split_idx:]

    ##Model Creation and training making use of hyper params for easier declaration.      
    model = Sequential([
        LSTM(64,return_sequences=True,input_shape=(seq_length, X_train.shape[2])),
        Dropout(0.2),
        LSTM(4,return_sequences=False),
        Dropout(0.2),
        Dense(64), ##activation='relu'
        Dense(2)
        ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(X_train,y_train,epochs =100, batch_size=18, validation_data=(X_test,y_test),
             callbacks =[early_stopping])


    y_pred = model.predict(X_test)

    y_pred_inv = scaler_y.inverse_transform(y_pred)
    y_test_inv = scaler_y.inverse_transform(y_test)

    mse = mean_squared_error(y_test_inv,y_pred_inv)
    mae = mean_absolute_error(y_test_inv,y_pred_inv)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_inv,y_pred_inv)
    #----#
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))

    return 

def GRUModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame):

    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length].flatten())
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)
    
    #----#        
    demandDs['DATE-TIME'] = demandDs['DATE-TIME'].astype(str)
    demandDs[['Date', 'Time']] = demandDs['DATE-TIME'].str.split(' ',expand=True)
    demandDs ['MW'] = pd.to_numeric(demandDs['MW'],errors='coerce')
    demandDs['DATE-TIME'] = pd.to_datetime(demandDs['DATE-TIME'], errors='coerce')


    supplyDs['Date & Time'] = supplyDs['Date & Time'].astype(str)
    supplyDs[['Date','Time']] = supplyDs['Date & Time'].str.split(' ',expand=True)
    supplyDs ['MW'] = pd.to_numeric(supplyDs['MW'], errors='coerce')
    supplyDs['Date & Time'] = pd.to_datetime(supplyDs['Date & Time'], errors='coerce')

    demandDs.dropna(subset=['MW','DATE-TIME'], inplace=True)
    supplyDs.dropna(subset=['MW','Date & Time'], inplace=True)


    min_length = min(len(demandDs), len(supplyDs))
    demandDs = demandDs.iloc[:min_length]
    supplyDs = supplyDs.iloc[:min_length]

    #Extraction of Data.

    demand = demandDs['MW'].values.reshape(-1,1)
    supply = supplyDs['MW'].values.reshape(-1,1)
    datetime_series = demandDs['DATE-TIME']

    hour = datetime_series.dt.hour.values.reshape(-1,1)
    dayofweek = datetime_series.dt.dayofweek.values.reshape(-1,1)
    month = datetime_series.dt.month.values.reshape(-1,1)

    scaler_demand = MinMaxScaler()
    scaler_supply = MinMaxScaler()
    scaler_hour = MinMaxScaler()
    scaler_day = MinMaxScaler()
    scaler_month = MinMaxScaler()
    

    demand_scaled = scaler_demand.fit_transform(demand)
    supply_scaled = scaler_supply.fit_transform(supply)
    hour_scaled = scaler_hour.fit_transform(hour)
    day_scaled = scaler_day.fit_transform(dayofweek)
    month_scaled = scaler_month.fit_transform(month)

    full_data = np.hstack((demand_scaled, supply_scaled,hour_scaled, day_scaled, month_scaled ))
    targets = np.hstack((demand_scaled, supply_scaled))

    seq_length = 24
    X, y = create_sequences_with_time(full_data, targets, seq_length)


    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    base_model = xgb.XGBRegressor(N_estimators=100, random_state=42)
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)


    y_demand_pred = scaler_demand.inverse_transform(y_pred[:, 0].reshape(-1, 1))
    y_supply_pred = scaler_supply.inverse_transform(y_pred[:, 1].reshape(-1, 1))
    y_demand_true = scaler_demand.inverse_transform(y_test[:, 0].reshape(-1, 1))
    y_supply_true = scaler_supply.inverse_transform(y_test[:, 1].reshape(-1, 1))



    combined_true = np.concatenate((y_demand_true, y_supply_true), axis=0)
    combined_pred = np.concatenate((y_demand_pred, y_supply_pred), axis=0)



    mse = mean_squared_error(combined_true, combined_pred)
    mae = mean_absolute_error(combined_true, combined_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(combined_true, combined_pred)
    #----#
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))

    return 

def SVRModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame):

    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length].flatten())
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)
    
    #----#        
    demandDs['DATE-TIME'] = demandDs['DATE-TIME'].astype(str)
    demandDs[['Date', 'Time']] = demandDs['DATE-TIME'].str.split(' ',expand=True)
    demandDs ['MW'] = pd.to_numeric(demandDs['MW'],errors='coerce')
    demandDs['DATE-TIME'] = pd.to_datetime(demandDs['DATE-TIME'], errors='coerce')


    supplyDs['Date & Time'] = supplyDs['Date & Time'].astype(str)
    supplyDs[['Date','Time']] = supplyDs['Date & Time'].str.split(' ',expand=True)
    supplyDs ['MW'] = pd.to_numeric(supplyDs['MW'], errors='coerce')
    supplyDs['Date & Time'] = pd.to_datetime(supplyDs['Date & Time'], errors='coerce')

    demandDs.dropna(subset=['MW','DATE-TIME'], inplace=True)
    supplyDs.dropna(subset=['MW','Date & Time'], inplace=True)


    min_length = min(len(demandDs), len(supplyDs))
    demandDs = demandDs.iloc[:min_length]
    supplyDs = supplyDs.iloc[:min_length]

    #Extraction of Data.

    demand = demandDs['MW'].values.reshape(-1,1)
    supply = supplyDs['MW'].values.reshape(-1,1)
    datetime_series = demandDs['DATE-TIME']

    hour = datetime_series.dt.hour.values.reshape(-1,1)
    dayofweek = datetime_series.dt.dayofweek.values.reshape(-1,1)
    month = datetime_series.dt.month.values.reshape(-1,1)

    scaler_demand = MinMaxScaler()
    scaler_supply = MinMaxScaler()
    scaler_hour = MinMaxScaler()
    scaler_day = MinMaxScaler()
    scaler_month = MinMaxScaler()
    

    demand_scaled = scaler_demand.fit_transform(demand)
    supply_scaled = scaler_supply.fit_transform(supply)
    hour_scaled = scaler_hour.fit_transform(hour)
    day_scaled = scaler_day.fit_transform(dayofweek)
    month_scaled = scaler_month.fit_transform(month)

    full_data = np.hstack((demand_scaled, supply_scaled,hour_scaled, day_scaled, month_scaled ))
    targets = np.hstack((demand_scaled, supply_scaled))

    seq_length = 24
    X, y = create_sequences_with_time(full_data, targets, seq_length)


    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    base_model = xgb.XGBRegressor(N_estimators=100, random_state=42)
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)


    y_demand_pred = scaler_demand.inverse_transform(y_pred[:, 0].reshape(-1, 1))
    y_supply_pred = scaler_supply.inverse_transform(y_pred[:, 1].reshape(-1, 1))
    y_demand_true = scaler_demand.inverse_transform(y_test[:, 0].reshape(-1, 1))
    y_supply_true = scaler_supply.inverse_transform(y_test[:, 1].reshape(-1, 1))



    combined_true = np.concatenate((y_demand_true, y_supply_true), axis=0)
    combined_pred = np.concatenate((y_demand_pred, y_supply_pred), axis=0)



    mse = mean_squared_error(combined_true, combined_pred)
    mae = mean_absolute_error(combined_true, combined_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(combined_true, combined_pred)
    #----#
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))

    return 

def MLPModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame):

    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length].flatten())
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)
    
    #----#        
    demandDs['DATE-TIME'] = demandDs['DATE-TIME'].astype(str)
    demandDs[['Date', 'Time']] = demandDs['DATE-TIME'].str.split(' ',expand=True)
    demandDs ['MW'] = pd.to_numeric(demandDs['MW'],errors='coerce')
    demandDs['DATE-TIME'] = pd.to_datetime(demandDs['DATE-TIME'], errors='coerce')


    supplyDs['Date & Time'] = supplyDs['Date & Time'].astype(str)
    supplyDs[['Date','Time']] = supplyDs['Date & Time'].str.split(' ',expand=True)
    supplyDs ['MW'] = pd.to_numeric(supplyDs['MW'], errors='coerce')
    supplyDs['Date & Time'] = pd.to_datetime(supplyDs['Date & Time'], errors='coerce')

    demandDs.dropna(subset=['MW','DATE-TIME'], inplace=True)
    supplyDs.dropna(subset=['MW','Date & Time'], inplace=True)


    min_length = min(len(demandDs), len(supplyDs))
    demandDs = demandDs.iloc[:min_length]
    supplyDs = supplyDs.iloc[:min_length]

    #Extraction of Data.

    demand = demandDs['MW'].values.reshape(-1,1)
    supply = supplyDs['MW'].values.reshape(-1,1)
    datetime_series = demandDs['DATE-TIME']

    hour = datetime_series.dt.hour.values.reshape(-1,1)
    dayofweek = datetime_series.dt.dayofweek.values.reshape(-1,1)
    month = datetime_series.dt.month.values.reshape(-1,1)

    scaler_demand = MinMaxScaler()
    scaler_supply = MinMaxScaler()
    scaler_hour = MinMaxScaler()
    scaler_day = MinMaxScaler()
    scaler_month = MinMaxScaler()
    

    demand_scaled = scaler_demand.fit_transform(demand)
    supply_scaled = scaler_supply.fit_transform(supply)
    hour_scaled = scaler_hour.fit_transform(hour)
    day_scaled = scaler_day.fit_transform(dayofweek)
    month_scaled = scaler_month.fit_transform(month)

    full_data = np.hstack((demand_scaled, supply_scaled,hour_scaled, day_scaled, month_scaled ))
    targets = np.hstack((demand_scaled, supply_scaled))

    seq_length = 24
    X, y = create_sequences_with_time(full_data, targets, seq_length)


    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    base_model = xgb.XGBRegressor(N_estimators=100, random_state=42)
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)


    y_demand_pred = scaler_demand.inverse_transform(y_pred[:, 0].reshape(-1, 1))
    y_supply_pred = scaler_supply.inverse_transform(y_pred[:, 1].reshape(-1, 1))
    y_demand_true = scaler_demand.inverse_transform(y_test[:, 0].reshape(-1, 1))
    y_supply_true = scaler_supply.inverse_transform(y_test[:, 1].reshape(-1, 1))



    combined_true = np.concatenate((y_demand_true, y_supply_true), axis=0)
    combined_pred = np.concatenate((y_demand_pred, y_supply_pred), axis=0)



    mse = mean_squared_error(combined_true, combined_pred)
    mae = mean_absolute_error(combined_true, combined_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(combined_true, combined_pred)
    #----#
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))

    return 

def CNNModelDS(demandDs:pd.DataFrame,supplyDs:pd.DataFrame):

    def create_sequences_with_time(data, targets, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length].flatten())
            y.append(targets[i + seq_length])
        return np.array(X), np.array(y)
    
    #----#        
    demandDs['DATE-TIME'] = demandDs['DATE-TIME'].astype(str)
    demandDs[['Date', 'Time']] = demandDs['DATE-TIME'].str.split(' ',expand=True)
    demandDs ['MW'] = pd.to_numeric(demandDs['MW'],errors='coerce')
    demandDs['DATE-TIME'] = pd.to_datetime(demandDs['DATE-TIME'], errors='coerce')


    supplyDs['Date & Time'] = supplyDs['Date & Time'].astype(str)
    supplyDs[['Date','Time']] = supplyDs['Date & Time'].str.split(' ',expand=True)
    supplyDs ['MW'] = pd.to_numeric(supplyDs['MW'], errors='coerce')
    supplyDs['Date & Time'] = pd.to_datetime(supplyDs['Date & Time'], errors='coerce')

    demandDs.dropna(subset=['MW','DATE-TIME'], inplace=True)
    supplyDs.dropna(subset=['MW','Date & Time'], inplace=True)


    min_length = min(len(demandDs), len(supplyDs))
    demandDs = demandDs.iloc[:min_length]
    supplyDs = supplyDs.iloc[:min_length]

    #Extraction of Data.

    demand = demandDs['MW'].values.reshape(-1,1)
    supply = supplyDs['MW'].values.reshape(-1,1)
    datetime_series = demandDs['DATE-TIME']

    hour = datetime_series.dt.hour.values.reshape(-1,1)
    dayofweek = datetime_series.dt.dayofweek.values.reshape(-1,1)
    month = datetime_series.dt.month.values.reshape(-1,1)

    scaler_demand = MinMaxScaler()
    scaler_supply = MinMaxScaler()
    scaler_hour = MinMaxScaler()
    scaler_day = MinMaxScaler()
    scaler_month = MinMaxScaler()
    

    demand_scaled = scaler_demand.fit_transform(demand)
    supply_scaled = scaler_supply.fit_transform(supply)
    hour_scaled = scaler_hour.fit_transform(hour)
    day_scaled = scaler_day.fit_transform(dayofweek)
    month_scaled = scaler_month.fit_transform(month)

    full_data = np.hstack((demand_scaled, supply_scaled,hour_scaled, day_scaled, month_scaled ))
    targets = np.hstack((demand_scaled, supply_scaled))

    seq_length = 24
    X, y = create_sequences_with_time(full_data, targets, seq_length)


    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    base_model = xgb.XGBRegressor(N_estimators=100, random_state=42)
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)


    y_demand_pred = scaler_demand.inverse_transform(y_pred[:, 0].reshape(-1, 1))
    y_supply_pred = scaler_supply.inverse_transform(y_pred[:, 1].reshape(-1, 1))
    y_demand_true = scaler_demand.inverse_transform(y_test[:, 0].reshape(-1, 1))
    y_supply_true = scaler_supply.inverse_transform(y_test[:, 1].reshape(-1, 1))



    combined_true = np.concatenate((y_demand_true, y_supply_true), axis=0)
    combined_pred = np.concatenate((y_demand_pred, y_supply_pred), axis=0)



    mse = mean_squared_error(combined_true, combined_pred)
    mae = mean_absolute_error(combined_true, combined_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(combined_true, combined_pred)
    #----#
    print("MSE: {:.4f}".format(mse))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    print("R2: {:.4f}".format(r2))

    return 
##xgBoostModelDS(sp_DemandDef,sp_SupplyDef)
## decisionTreeModelDS(sp_DemandDef,sp_SupplyDef)
## randomForestModelDS(sp_DemandDef,sp_SupplyDef)
##decisionTreeModelSeasonalTraining(sp_DemandDef,sp_SupplyDef)
##decisionTreeModelDSYearlyTraining(sp_DemandDef,sp_SupplyDef)
BLSTMModelDS(sp_DemandDef,sp_SupplyDef)


