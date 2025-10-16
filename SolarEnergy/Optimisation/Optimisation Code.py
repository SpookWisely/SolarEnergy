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
from decimal import Decimal, getcontext
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

## https://www.iea.org/reports/key-world-energy-statistics-2021
## https://www.iea.org/data-and-statistics/data-product/world-energy-statistics-and-balances
## https://www.enerdata.net/estore/energy-market/saudi-arabia/
## https://www.climatiq.io/data/emission-factor/9d4f9d9a-b332-4275-9946-70643df87ac9 - Source 4

# Taken from source 4 above 

getcontext().prec = 28

EF_target = Decimal('0.1716')
EF_oil = Decimal('0.249')
EF_gas = Decimal('0.117')

f_oil = (EF_target - EF_gas) / (EF_oil - EF_gas)  # = 0.413636...
f_gas = Decimal('1') - f_oil                      # = 0.586364...

oil_contrib = f_oil * EF_oil
gas_contrib = f_gas * EF_gas
EF = oil_contrib + gas_contrib

print(f"EF_oil={EF_oil}, f_oil={float(f_oil):.6f}")
print(f"EF_gas={EF_gas}, f_gas={float(f_gas):.6f}")
print(f"Contributions -> oil={float(oil_contrib):.6f}, gas={float(gas_contrib):.6f}")
print(f"EF = {float(EF):.4f}")  # 0.1716

##Since data isn't free had to roll back the calculation for 2021 to see how I would
##arrive at their result they had of 0.1716 for the emmision factor for Saudi Arabia.

##So using the calculation from the image you gave me
##EF = (f_oil × EF_oil) + (f_gas × EF_gas)

###the result calculation could be as follows:
##EF = ((0.413636 * 0.249) + (0.586364 * 0.117))

sp_DemandDef = pd.read_excel(r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\Datasets\Sakakah 2021 Demand dataset.xlsx")
sp_SupplyDef = pd.read_excel(r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\Datasets\Sakakah 2021 PV supply dataset.xlsx")
print("Data Read OK!")

sp_DemandDef['DATE-TIME'] = sp_DemandDef['DATE-TIME'].astype(str)
sp_DemandDef[['Date', 'Time']] = sp_DemandDef['DATE-TIME'].str.split(' ',expand=True)
sp_DemandDef ['MW'] = pd.to_numeric(sp_DemandDef['MW'],errors='coerce')
sp_DemandDef['DATE-TIME'] = pd.to_datetime(sp_DemandDef['DATE-TIME'], errors='coerce')

sp_DemandDef.dropna(subset=['MW','DATE-TIME'], inplace=True)
#sp_DemandDef = sp_DemandDef['MW'].values.reshape(-1,1)

sp_SupplyDef['Date & Time'] = sp_SupplyDef['Date & Time'].astype(str)
sp_SupplyDef[['Date','Time']] = sp_SupplyDef['Date & Time'].str.split(' ',expand=True)
sp_SupplyDef ['MW'] = pd.to_numeric(sp_SupplyDef['MW'], errors='coerce')
sp_SupplyDef['Date & Time'] = pd.to_datetime(sp_SupplyDef['Date & Time'], errors='coerce')

sp_SupplyDef.dropna(subset=['MW','Date & Time'], inplace=True)
#sp_SupplyDef = sp_SupplyDef['MW'].values.reshape(-1,1)

## This section is about the calculation of baseline emissions and the cap calculation
# using the demand and supply data given using the calculations of:
# BaselineFossil_t = max(Demand_t - Solar_t, 0) in MW   
# TotalBaselineFossil_MWh = sum(BaselineFossil_t * step_hours) in MWh
# BaselineEmissions_tCO2 = TotalBaselineFossil_MWh * EF in tCO2
demand_arr = sp_DemandDef[['MW']].to_numpy()
supply_arr = sp_SupplyDef[['MW']].to_numpy()

# Align on timestamps (files already match, so inner join should preserve all rows)
df_dem = sp_DemandDef.rename(columns={'DATE-TIME': 'ts', 'MW': 'demand_MW'})[['ts', 'demand_MW']]
df_pv = sp_SupplyDef.rename(columns={'Date & Time': 'ts', 'MW': 'solar_MW'})[['ts', 'solar_MW']]
df = pd.merge(df_dem, df_pv, on='ts', how='inner').sort_values('ts').reset_index(drop=True)

# Infer timestep in hours from the most common interval
if len(df) >= 2:
    step_seconds_mode = df['ts'].diff().dt.total_seconds().dropna().mode()
    step_hours = float(step_seconds_mode.iloc[0] / 3600.0) if not step_seconds_mode.empty else 1.0
else:
    step_hours = 1.0

# Baseline fossil use (no storage): max(Demand - Solar, 0) in MW
baseline_fossil_MW = np.maximum(df['demand_MW'].to_numpy() - df['solar_MW'].to_numpy(), 0.0)

# Convert to energy per step (MWh) and aggregate
baseline_fossil_energy_MWh = baseline_fossil_MW * step_hours
total_baseline_fossil_MWh = float(baseline_fossil_energy_MWh.sum())

# Also compute total solar and demand energy for reference
total_demand_MWh = float((df['demand_MW'].to_numpy() * step_hours).sum())
total_solar_MWh = float((df['solar_MW'].to_numpy() * step_hours).sum())

# Emissions (tCO2): sum( BaselineFossil_t * EF )
EF_float = float(EF)  # = 0.1716 tCO2/MWh
baseline_emissions_tCO2 = total_baseline_fossil_MWh * EF_float

# Emission Cap for 45% reduction target
emission_cap_tCO2 = baseline_emissions_tCO2 * 0.55

print(f"Timestep (hours): {step_hours:.3f}")
print(f"Aligned points: {len(df)}")
print(f"Total demand energy (MWh): {total_demand_MWh:,.2f}")
print(f"Total solar energy  (MWh): {total_solar_MWh:,.2f}")
print(f"Total baseline fossil energy (MWh): {total_baseline_fossil_MWh:,.2f}")
print(f"Baseline emissions (tCO2): {baseline_emissions_tCO2:,.2f}")
print(f"Emission cap @45% reduction (tCO2): {emission_cap_tCO2:,.2f}")


## Next part To do is the estimation of the energy storage size.
# Equations will be Netflow = Solar (Time?) - Demand (Time?)
# SOC = Energy_Storage_(t-1) + Netflow_t
# Ereq = max(Demand_t - Solar_t, 0) - min(Demand_t - Solar_t, 0)
# Emin =  EReq * 0.85
# EMax = Ereq * 1.15

"""
For the next part total up the solar 
"""
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

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

