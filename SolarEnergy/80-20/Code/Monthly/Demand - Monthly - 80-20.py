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
print(sp_SupplyDef.head())
print("\n",sp_DemandDef.head())
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
    return X, y#----#



random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
def decisionTreeModelDS(demandDs: pd.DataFrame, supplyDs: pd.DataFrame, ident: int, plots: bool):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy()

    if ident == 1:
        # Rename DATE-TIME to TimeStamp for compatibility
        demandDs.rename(columns={'DATE-TIME': 'TimeStamp'}, inplace=True)

        # Ensure MW column is numeric and drop NaN values
        demandDs['MW'] = pd.to_numeric(demandDs['MW'], errors='coerce')
        demandDs.dropna(subset=['MW', 'TimeStamp'], inplace=True)

        # Scale the demand data
        scaler_demand = MinMaxScaler()
        demandDs['MW'] = scaler_demand.fit_transform(demandDs[['MW']])

        # Create monthly sequences
        feature_cols = ['MW']
        target_cols = ['MW']
        X, y = create_monthly_sequences(demandDs, feature_cols, target_cols, pad_to_max=True)
        X = X.reshape(X.shape[0], -1)  # Flatten for Decision Tree compatibility

        # Split into training and testing sets
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # Train Decision Tree model
        param_grid = {
            'estimator__max_depth': [3, 5, 10, None],
            'estimator__min_samples_split': [2, 5, 10]
        }
        base_model = DecisionTreeRegressor(random_state=42)
        model = MultiOutputRegressor(base_model)
        grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_

        # Make predictions
        y_pred = model.predict(X_test)

        # Inverse transform predictions and true values
        y_demand_pred = scaler_demand.inverse_transform(y_pred)
        y_demand_true = scaler_demand.inverse_transform(y_test)

        # Calculate metrics
        mse = mean_squared_error(y_demand_true, y_demand_pred)
        mae = mean_absolute_error(y_demand_true, y_demand_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_demand_true, y_demand_pred)
        modelName = "Decision Tree"
        results = [mse, mae, rmse, r2, modelName]

        # Plot results if requested
        if plots:
            plt.figure(figsize=(10, 5))
            plt.plot(y_demand_true, label='Actual')
            plt.plot(y_demand_pred, label='Predicted')
            plt.title('Decision Tree Model: Actual vs Predicted Demand')
            plt.xlabel('Time')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()

        print('\n', "Demand Data Decision Tree Model Results:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))

        return results

    elif ident == 2:
        # Similar changes for supply dataset
        supplyDs.rename(columns={'Date & Time': 'TimeStamp'}, inplace=True)
        supplyDs['MW'] = pd.to_numeric(supplyDs['MW'], errors='coerce')
        supplyDs.dropna(subset=['MW', 'TimeStamp'], inplace=True)

        scaler_supply = MinMaxScaler()
        supplyDs['MW'] = scaler_supply.fit_transform(supplyDs[['MW']])

        feature_cols = ['MW']
        target_cols = ['MW']
        X, y = create_monthly_sequences(supplyDs, feature_cols, target_cols, pad_to_max=True)
        X = X.reshape(X.shape[0], -1)

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

        y_supply_pred = scaler_supply.inverse_transform(y_pred)
        y_supply_true = scaler_supply.inverse_transform(y_test)

        mse = mean_squared_error(y_supply_true, y_supply_pred)
        mae = mean_absolute_error(y_supply_true, y_supply_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_supply_true, y_supply_pred)
        modelName = "Decision Tree"
        results = [mse, mae, rmse, r2, modelName]

        print('\n', "Supply Data Decision Tree Model Results:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))

        return results

    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return


def randomForestModelDS(demandDs: pd.DataFrame, supplyDs: pd.DataFrame, ident: int, plots: bool):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy()

    if ident == 1:
        # Rename DATE-TIME to TimeStamp for compatibility
        demandDs.rename(columns={'DATE-TIME': 'TimeStamp'}, inplace=True)

        # Ensure MW column is numeric and drop NaN values
        demandDs['MW'] = pd.to_numeric(demandDs['MW'], errors='coerce')
        demandDs.dropna(subset=['MW', 'TimeStamp'], inplace=True)

        # Scale the demand data
        scaler_demand = MinMaxScaler()
        demandDs['MW'] = scaler_demand.fit_transform(demandDs[['MW']])

        # Create monthly sequences
        feature_cols = ['MW']
        target_cols = ['MW']
        X, y = create_monthly_sequences(demandDs, feature_cols, target_cols, pad_to_max=True)
        X = X.reshape(X.shape[0], -1)  # Flatten for Random Forest compatibility

        # Split into training and testing sets
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # Train Random Forest model
        param_grid = {
            'estimator__n_estimators': [50, 100],
            'estimator__max_depth': [5, 10, None]
        }
        base_model = RandomForestRegressor(random_state=42)
        model = MultiOutputRegressor(base_model)
        grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_

        # Make predictions
        y_pred = model.predict(X_test)

        # Inverse transform predictions and true values
        y_demand_pred = scaler_demand.inverse_transform(y_pred)
        y_demand_true = scaler_demand.inverse_transform(y_test)

        # Calculate metrics
        mse = mean_squared_error(y_demand_true, y_demand_pred)
        mae = mean_absolute_error(y_demand_true, y_demand_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_demand_true, y_demand_pred)
        modelName = "Random Forest"
        results = [mse, mae, rmse, r2, modelName]

        # Plot results if requested
        if plots:
            plt.figure(figsize=(10, 5))
            plt.plot(y_demand_true, label='Actual')
            plt.plot(y_demand_pred, label='Predicted')
            plt.title('Random Forest Model: Actual vs Predicted Demand')
            plt.xlabel('Time')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()

        print('\n', "Demand Data Random Forest Model Results:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))

        return results

    elif ident == 2:
        # Similar changes for supply dataset
        supplyDs.rename(columns={'Date & Time': 'TimeStamp'}, inplace=True)
        supplyDs['MW'] = pd.to_numeric(supplyDs['MW'], errors='coerce')
        supplyDs.dropna(subset=['MW', 'TimeStamp'], inplace=True)

        scaler_supply = MinMaxScaler()
        supplyDs['MW'] = scaler_supply.fit_transform(supplyDs[['MW']])

        feature_cols = ['MW']
        target_cols = ['MW']
        X, y = create_monthly_sequences(supplyDs, feature_cols, target_cols, pad_to_max=True)
        X = X.reshape(X.shape[0], -1)

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

        y_pred = model.predict(X_test)

        y_supply_pred = scaler_supply.inverse_transform(y_pred)
        y_supply_true = scaler_supply.inverse_transform(y_test)

        mse = mean_squared_error(y_supply_true, y_supply_pred)
        mae = mean_absolute_error(y_supply_true, y_supply_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_supply_true, y_supply_pred)
        modelName = "Random Forest"
        results = [mse, mae, rmse, r2, modelName]

        print('\n', "Supply Data Random Forest Model Results:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))

        return results

    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return


def gbModelDS(demandDs: pd.DataFrame, supplyDs: pd.DataFrame, ident: int, plots: bool):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy()

    if ident == 1:
        # Rename DATE-TIME to TimeStamp for compatibility
        demandDs.rename(columns={'DATE-TIME': 'TimeStamp'}, inplace=True)

        # Ensure MW column is numeric and drop NaN values
        demandDs['MW'] = pd.to_numeric(demandDs['MW'], errors='coerce')
        demandDs.dropna(subset=['MW', 'TimeStamp'], inplace=True)

        # Scale the demand data
        scaler_demand = MinMaxScaler()
        demandDs['MW'] = scaler_demand.fit_transform(demandDs[['MW']])

        # Create monthly sequences
        feature_cols = ['MW']
        target_cols = ['MW']
        X, y = create_monthly_sequences(demandDs, feature_cols, target_cols, pad_to_max=True)
        X = X.reshape(X.shape[0], -1)  # Flatten for Gradient Boosting compatibility

        # Split into training and testing sets
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # Train Gradient Boosting model
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1]
        }
        model = GradientBoostingRegressor(random_state=42)
        grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_

        # Make predictions
        y_pred = model.predict(X_test)

        # Inverse transform predictions and true values
        y_demand_pred = scaler_demand.inverse_transform(y_pred.reshape(-1, 1))
        y_demand_true = scaler_demand.inverse_transform(y_test)

        # Calculate metrics
        mse = mean_squared_error(y_demand_true, y_demand_pred)
        mae = mean_absolute_error(y_demand_true, y_demand_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_demand_true, y_demand_pred)
        modelName = "Gradient Boosting"
        results = [mse, mae, rmse, r2, modelName]

        # Plot results if requested
        if plots:
            plt.figure(figsize=(10, 5))
            plt.plot(y_demand_true, label='Actual')
            plt.plot(y_demand_pred, label='Predicted')
            plt.title('Gradient Boosting Model: Actual vs Predicted Demand')
            plt.xlabel('Time')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()

        print('\n', "Demand Data Gradient Boosting Model Results:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))

        return results

    elif ident == 2:
        # Similar changes for supply dataset
        supplyDs.rename(columns={'Date & Time': 'TimeStamp'}, inplace=True)
        supplyDs['MW'] = pd.to_numeric(supplyDs['MW'], errors='coerce')
        supplyDs.dropna(subset=['MW', 'TimeStamp'], inplace=True)

        scaler_supply = MinMaxScaler()
        supplyDs['MW'] = scaler_supply.fit_transform(supplyDs[['MW']])

        feature_cols = ['MW']
        target_cols = ['MW']
        X, y = create_monthly_sequences(supplyDs, feature_cols, target_cols, pad_to_max=True)
        X = X.reshape(X.shape[0], -1)

        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1]
        }
        model = GradientBoostingRegressor(random_state=42)
        grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_

        y_pred = model.predict(X_test)

        y_supply_pred = scaler_supply.inverse_transform(y_pred.reshape(-1, 1))
        y_supply_true = scaler_supply.inverse_transform(y_test)

        mse = mean_squared_error(y_supply_true, y_supply_pred)
        mae = mean_absolute_error(y_supply_true, y_supply_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_supply_true, y_supply_pred)
        modelName = "Gradient Boosting"
        results = [mse, mae, rmse, r2, modelName]

        print('\n', "Supply Data Gradient Boosting Model Results:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))

        return results

    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return


def xgbModelDS(demandDs: pd.DataFrame, supplyDs: pd.DataFrame, ident: int, plots: bool):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy()

    if ident == 1:
        # Rename DATE-TIME to TimeStamp for compatibility
        demandDs.rename(columns={'DATE-TIME': 'TimeStamp'}, inplace=True)

        # Ensure MW column is numeric and drop NaN values
        demandDs['MW'] = pd.to_numeric(demandDs['MW'], errors='coerce')
        demandDs.dropna(subset=['MW', 'TimeStamp'], inplace=True)

        # Scale the demand data
        scaler_demand = MinMaxScaler()
        demandDs['MW'] = scaler_demand.fit_transform(demandDs[['MW']])

        # Create monthly sequences
        feature_cols = ['MW']
        target_cols = ['MW']
        X, y = create_monthly_sequences(demandDs, feature_cols, target_cols, pad_to_max=True)
        X = X.reshape(X.shape[0], -1)  # Flatten for XGBoost compatibility

        # Split into training and testing sets
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # Train XGBoost model
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

        # Make predictions
        y_pred = model.predict(X_test)

        # Inverse transform predictions and true values
        y_demand_pred = scaler_demand.inverse_transform(y_pred)
        y_demand_true = scaler_demand.inverse_transform(y_test)

        # Calculate metrics
        mse = mean_squared_error(y_demand_true, y_demand_pred)
        mae = mean_absolute_error(y_demand_true, y_demand_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_demand_true, y_demand_pred)
        modelName = "XGBoost"
        results = [mse, mae, rmse, r2, modelName]

        # Plot results if requested
        if plots:
            plt.figure(figsize=(10, 5))
            plt.plot(y_demand_true, label='Actual')
            plt.plot(y_demand_pred, label='Predicted')
            plt.title('XGBoost Model: Actual vs Predicted Demand')
            plt.xlabel('Time')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()

        print('\n', "Demand Data XGBoost Model Results:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))

        return results

    elif ident == 2:
        # Similar changes for supply dataset
        supplyDs.rename(columns={'Date & Time': 'TimeStamp'}, inplace=True)
        supplyDs['MW'] = pd.to_numeric(supplyDs['MW'], errors='coerce')
        supplyDs.dropna(subset=['MW', 'TimeStamp'], inplace=True)

        scaler_supply = MinMaxScaler()
        supplyDs['MW'] = scaler_supply.fit_transform(supplyDs[['MW']])

        feature_cols = ['MW']
        target_cols = ['MW']
        X, y = create_monthly_sequences(supplyDs, feature_cols, target_cols, pad_to_max=True)
        X = X.reshape(X.shape[0], -1)

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

def biDirectionalLSTMDS(demandDs: pd.DataFrame, supplyDs: pd.DataFrame, ident: int, plots: bool):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy()

    if ident == 1:
        # Rename DATE-TIME to TimeStamp for compatibility
        demandDs.rename(columns={'DATE-TIME': 'TimeStamp'}, inplace=True)

        # Ensure MW column is numeric and drop NaN values
        demandDs['MW'] = pd.to_numeric(demandDs['MW'], errors='coerce')
        demandDs.dropna(subset=['MW', 'TimeStamp'], inplace=True)

        # Scale the demand data
        scaler_demand = MinMaxScaler()
        demandDs['MW'] = scaler_demand.fit_transform(demandDs[['MW']])

        # Create monthly sequences
        feature_cols = ['MW']
        target_cols = ['MW']
        X, y = create_monthly_sequences(demandDs, feature_cols, target_cols, pad_to_max=True)

        # Reshape for BiDirectional LSTM compatibility
        X = X.reshape(X.shape[0], X.shape[1], len(feature_cols))

        # Split into training and testing sets
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # Define and train BiDirectional LSTM model
        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))),
            Dropout(0.2),
            Bidirectional(LSTM(64, return_sequences=False)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test), verbose=0)

        # Make predictions
        y_pred = model.predict(X_test)

        # Inverse transform predictions and true values
        y_demand_pred = scaler_demand.inverse_transform(y_pred)
        y_demand_true = scaler_demand.inverse_transform(y_test)

        # Calculate metrics
        mse = mean_squared_error(y_demand_true, y_demand_pred)
        mae = mean_absolute_error(y_demand_true, y_demand_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_demand_true, y_demand_pred)
        modelName = "BiDirectional LSTM"
        results = [mse, mae, rmse, r2, modelName]

        # Plot results if requested
        if plots:
            plt.figure(figsize=(10, 5))
            plt.plot(y_demand_true, label='Actual')
            plt.plot(y_demand_pred, label='Predicted')
            plt.title('BiDirectional LSTM Model: Actual vs Predicted Demand')
            plt.xlabel('Time')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()

        print('\n', "Demand Data BiDirectional LSTM Model Results:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))

        return results

    elif ident == 2:
        # Similar changes for supply dataset
        supplyDs.rename(columns={'Date & Time': 'TimeStamp'}, inplace=True)
        supplyDs['MW'] = pd.to_numeric(supplyDs['MW'], errors='coerce')
        supplyDs.dropna(subset=['MW', 'TimeStamp'], inplace=True)

        scaler_supply = MinMaxScaler()
        supplyDs['MW'] = scaler_supply.fit_transform(supplyDs[['MW']])

        feature_cols = ['MW']
        target_cols = ['MW']
        X, y = create_monthly_sequences(supplyDs, feature_cols, target_cols, pad_to_max=True)

        # Reshape for BiDirectional LSTM compatibility
        X = X.reshape(X.shape[0], X.shape[1], len(feature_cols))

        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))),
            Dropout(0.2),
            Bidirectional(LSTM(64, return_sequences=False)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test), verbose=0)

        y_pred = model.predict(X_test)

        y_supply_pred = scaler_supply.inverse_transform(y_pred)
        y_supply_true = scaler_supply.inverse_transform(y_test)

        mse = mean_squared_error(y_supply_true, y_supply_pred)
        mae = mean_absolute_error(y_supply_true, y_supply_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_supply_true, y_supply_pred)
        modelName = "BLSTM"
        results = [mse, mae, rmse, r2, modelName]

        print('\n', "Supply Data BiDirectional LSTM Model Results:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))

        return results

    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return


def LSTMModelDS(demandDs: pd.DataFrame, supplyDs: pd.DataFrame, ident: int, plots: bool):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy()

    if ident == 1:
        # Rename DATE-TIME to TimeStamp for compatibility
        demandDs.rename(columns={'DATE-TIME': 'TimeStamp'}, inplace=True)

        # Ensure MW column is numeric and drop NaN values
        demandDs['MW'] = pd.to_numeric(demandDs['MW'], errors='coerce')
        demandDs.dropna(subset=['MW', 'TimeStamp'], inplace=True)

        # Scale the demand data
        scaler_demand = MinMaxScaler()
        demandDs['MW'] = scaler_demand.fit_transform(demandDs[['MW']])

        # Create monthly sequences
        feature_cols = ['MW']
        target_cols = ['MW']
        X, y = create_monthly_sequences(demandDs, feature_cols, target_cols, pad_to_max=True)

        # Reshape for LSTM compatibility
        X = X.reshape(X.shape[0], X.shape[1], len(feature_cols))

        # Split into training and testing sets
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # Define and train LSTM model
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test), verbose=0)

        # Make predictions
        y_pred = model.predict(X_test)

        # Inverse transform predictions and true values
        y_demand_pred = scaler_demand.inverse_transform(y_pred)
        y_demand_true = scaler_demand.inverse_transform(y_test)

        # Calculate metrics
        mse = mean_squared_error(y_demand_true, y_demand_pred)
        mae = mean_absolute_error(y_demand_true, y_demand_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_demand_true, y_demand_pred)
        modelName = "LSTM"
        results = [mse, mae, rmse, r2, modelName]

        # Plot results if requested
        if plots == True:
            plt.figure(figsize=(10, 5))
            plt.plot(y_demand_true, label='Actual')
            plt.plot(y_demand_pred, label='Predicted')
            plt.title('LSTM Model: Actual vs Predicted Demand')
            plt.xlabel('Time')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()

        print('\n', "Demand Data LSTM Model Results:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))

        return results

    elif ident == 2:
        # Similar changes for supply dataset
        supplyDs.rename(columns={'Date & Time': 'TimeStamp'}, inplace=True)
        supplyDs['MW'] = pd.to_numeric(supplyDs['MW'], errors='coerce')
        supplyDs.dropna(subset=['MW', 'TimeStamp'], inplace=True)

        scaler_supply = MinMaxScaler()
        supplyDs['MW'] = scaler_supply.fit_transform(supplyDs[['MW']])

        feature_cols = ['MW']
        target_cols = ['MW']
        X, y = create_monthly_sequences(supplyDs, feature_cols, target_cols, pad_to_max=True)

        # Reshape for LSTM compatibility
        X = X.reshape(X.shape[0], X.shape[1], len(feature_cols))

        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test), verbose=0)

        y_pred = model.predict(X_test)

        y_supply_pred = scaler_supply.inverse_transform(y_pred)
        y_supply_true = scaler_supply.inverse_transform(y_test)

        mse = mean_squared_error(y_supply_true, y_supply_pred)
        mae = mean_absolute_error(y_supply_true, y_supply_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_supply_true, y_supply_pred)
        modelName = "LSTM"
        results = [mse, mae, rmse, r2, modelName]
        
        if plots == True:
            plt.figure(figsize=(10, 5))
            plt.plot(y_supply_true, label='Actual')
            plt.plot(y_supply_pred, label='Predicted')
            plt.title('LSTM Model: Actual vs Predicted Supply')
            plt.xlabel('Time')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()

        print('\n', "Supply Data LSTM Model Results:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))

        return results

    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return


def GRUModelDS(demandDs: pd.DataFrame, supplyDs: pd.DataFrame, ident: int, plots: bool):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy()

    if ident == 1:
        # Rename DATE-TIME to TimeStamp for compatibility
        demandDs.rename(columns={'DATE-TIME': 'TimeStamp'}, inplace=True)

        # Ensure MW column is numeric and drop NaN values
        demandDs['MW'] = pd.to_numeric(demandDs['MW'], errors='coerce')
        demandDs.dropna(subset=['MW', 'TimeStamp'], inplace=True)

        # Scale the demand data
        scaler_demand = MinMaxScaler()
        demandDs['MW'] = scaler_demand.fit_transform(demandDs[['MW']])

        # Create monthly sequences
        feature_cols = ['MW']
        target_cols = ['MW']
        X, y = create_monthly_sequences(demandDs, feature_cols, target_cols, pad_to_max=True)

        # Reshape for GRU compatibility
        X = X.reshape(X.shape[0], X.shape[1], len(feature_cols))

        # Split into training and testing sets
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # Define and train GRU model
        model = Sequential([
            GRU(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            GRU(64, return_sequences=False),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test), verbose=0)

        # Make predictions
        y_pred = model.predict(X_test)

        # Inverse transform predictions and true values
        y_demand_pred = scaler_demand.inverse_transform(y_pred)
        y_demand_true = scaler_demand.inverse_transform(y_test)

        # Calculate metrics
        mse = mean_squared_error(y_demand_true, y_demand_pred)
        mae = mean_absolute_error(y_demand_true, y_demand_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_demand_true, y_demand_pred)
        modelName = "GRU"
        results = [mse, mae, rmse, r2, modelName]

        # Plot results if requested
        if plots:
            plt.figure(figsize=(10, 5))
            plt.plot(y_demand_true, label='Actual')
            plt.plot(y_demand_pred, label='Predicted')
            plt.title('GRU Model: Actual vs Predicted Demand')
            plt.xlabel('Time')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()

        print('\n', "Demand Data GRU Model Results:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))

        return results

    elif ident == 2:
        # Similar changes for supply dataset
        supplyDs.rename(columns={'Date & Time': 'TimeStamp'}, inplace=True)
        supplyDs['MW'] = pd.to_numeric(supplyDs['MW'], errors='coerce')
        supplyDs.dropna(subset=['MW', 'TimeStamp'], inplace=True)

        scaler_supply = MinMaxScaler()
        supplyDs['MW'] = scaler_supply.fit_transform(supplyDs[['MW']])

        feature_cols = ['MW']
        target_cols = ['MW']
        X, y = create_monthly_sequences(supplyDs, feature_cols, target_cols, pad_to_max=True)

        # Reshape for GRU compatibility
        X = X.reshape(X.shape[0], X.shape[1], len(feature_cols))

        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        model = Sequential([
            GRU(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            GRU(64, return_sequences=False),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test), verbose=0)

        y_pred = model.predict(X_test)

        y_supply_pred = scaler_supply.inverse_transform(y_pred)
        y_supply_true = scaler_supply.inverse_transform(y_test)

        mse = mean_squared_error(y_supply_true, y_supply_pred)
        mae = mean_absolute_error(y_supply_true, y_supply_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_supply_true, y_supply_pred)
        modelName = "GRU"
        results = [mse, mae, rmse, r2, modelName]

        print('\n', "Supply Data GRU Model Results:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))

        return results

    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return


def SVRModelDS(demandDs: pd.DataFrame, supplyDs: pd.DataFrame, ident: int, plots: bool):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy()

    if ident == 1:
        # Rename DATE-TIME to TimeStamp for compatibility
        demandDs.rename(columns={'DATE-TIME': 'TimeStamp'}, inplace=True)

        # Ensure MW column is numeric and drop NaN values
        demandDs['MW'] = pd.to_numeric(demandDs['MW'], errors='coerce')
        demandDs.dropna(subset=['MW', 'TimeStamp'], inplace=True)

        # Scale the demand data
        scaler_demand = MinMaxScaler()
        demandDs['MW'] = scaler_demand.fit_transform(demandDs[['MW']])

        # Create monthly sequences
        feature_cols = ['MW']
        target_cols = ['MW']
        X, y = create_monthly_sequences(demandDs, feature_cols, target_cols, pad_to_max=True)
        X = X.reshape(X.shape[0], -1)  # Flatten for SVR compatibility

        # Split into training and testing sets
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # Train SVR model
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'epsilon': [0.1, 0.5]
        }
        model = SVR(kernel='rbf')
        grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train.ravel())
        model = grid.best_estimator_

        # Make predictions
        y_pred = model.predict(X_test)

        # Inverse transform predictions and true values
        y_demand_pred = scaler_demand.inverse_transform(y_pred.reshape(-1, 1))
        y_demand_true = scaler_demand.inverse_transform(y_test)

        # Calculate metrics
        mse = mean_squared_error(y_demand_true, y_demand_pred)
        mae = mean_absolute_error(y_demand_true, y_demand_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_demand_true, y_demand_pred)
        modelName = "SVR"
        results = [mse, mae, rmse, r2, modelName]

        # Plot results if requested
        if plots:
            plt.figure(figsize=(10, 5))
            plt.plot(y_demand_true, label='Actual')
            plt.plot(y_demand_pred, label='Predicted')
            plt.title('SVR Model: Actual vs Predicted Demand')
            plt.xlabel('Time')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()

        print('\n', "Demand Data SVR Model Results:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))

        return results

    elif ident == 2:
        # Similar changes for supply dataset
        supplyDs.rename(columns={'Date & Time': 'TimeStamp'}, inplace=True)
        supplyDs['MW'] = pd.to_numeric(supplyDs['MW'], errors='coerce')
        supplyDs.dropna(subset=['MW', 'TimeStamp'], inplace=True)

        scaler_supply = MinMaxScaler()
        supplyDs['MW'] = scaler_supply.fit_transform(supplyDs[['MW']])

        feature_cols = ['MW']
        target_cols = ['MW']
        X, y = create_monthly_sequences(supplyDs, feature_cols, target_cols, pad_to_max=True)
        X = X.reshape(X.shape[0], -1)

        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'epsilon': [0.1, 0.5]
        }
        model = SVR(kernel='rbf')
        grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train.ravel())
        model = grid.best_estimator_

        y_pred = model.predict(X_test)

        y_supply_pred = scaler_supply.inverse_transform(y_pred.reshape(-1, 1))
        y_supply_true = scaler_supply.inverse_transform(y_test)

        mse = mean_squared_error(y_supply_true, y_supply_pred)
        mae = mean_absolute_error(y_supply_true, y_supply_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_supply_true, y_supply_pred)
        modelName = "SVR"
        results = [mse, mae, rmse, r2, modelName]

        print('\n', "Supply Data SVR Model Results:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))

        return results

    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return


def MLPModelDS(demandDs: pd.DataFrame, supplyDs: pd.DataFrame, ident: int, plots: bool):
        demandDs = demandDs.copy()
        supplyDs = supplyDs.copy()

        if ident == 1:
            # Rename DATE-TIME to TimeStamp for compatibility
            demandDs.rename(columns={'DATE-TIME': 'TimeStamp'}, inplace=True)

            # Ensure MW column is numeric and drop NaN values
            demandDs['MW'] = pd.to_numeric(demandDs['MW'], errors='coerce')
            demandDs.dropna(subset=['MW', 'TimeStamp'], inplace=True)

            # Scale the demand data
            scaler_demand = MinMaxScaler()
            demandDs['MW'] = scaler_demand.fit_transform(demandDs[['MW']])

            # Create monthly sequences
            feature_cols = ['MW']
            target_cols = ['MW']
            X, y = create_monthly_sequences(demandDs, feature_cols, target_cols, pad_to_max=True)
            X = X.reshape(X.shape[0], -1)  # Flatten for MLP compatibility

            # Split into training and testing sets
            split_index = int(len(X) * 0.8)
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]

            # Define and train MLP model
            model = Sequential([
                Dense(128, activation='relu', input_dim=X_train.shape[1]),
                Dropout(0.2),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test), verbose=0)

            # Make predictions
            y_pred = model.predict(X_test)

            # Inverse transform predictions and true values
            y_demand_pred = scaler_demand.inverse_transform(y_pred)
            y_demand_true = scaler_demand.inverse_transform(y_test)

            # Calculate metrics
            mse = mean_squared_error(y_demand_true, y_demand_pred)
            mae = mean_absolute_error(y_demand_true, y_demand_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_demand_true, y_demand_pred)
            modelName = "MLP"
            results = [mse, mae, rmse, r2, modelName]

            # Plot results if requested
            if plots:
                plt.figure(figsize=(10, 5))
                plt.plot(y_demand_true, label='Actual')
                plt.plot(y_demand_pred, label='Predicted')
                plt.title('MLP Model: Actual vs Predicted Demand')
                plt.xlabel('Time')
                plt.ylabel('MW')
                plt.legend()
                plt.tight_layout()
                plt.show()

            print('\n', "Demand Data MLP Model Results:")
            print("MSE: {:.4f}".format(mse))
            print("MAE: {:.4f}".format(mae))
            print("RMSE: {:.4f}".format(rmse))
            print("R2: {:.4f}".format(r2))

            return results

        elif ident == 2:
            # Similar changes for supply dataset
            supplyDs.rename(columns={'Date & Time': 'TimeStamp'}, inplace=True)
            supplyDs['MW'] = pd.to_numeric(supplyDs['MW'], errors='coerce')
            supplyDs.dropna(subset=['MW', 'TimeStamp'], inplace=True)

            scaler_supply = MinMaxScaler()
            supplyDs['MW'] = scaler_supply.fit_transform(supplyDs[['MW']])

            feature_cols = ['MW']
            target_cols = ['MW']
            X, y = create_monthly_sequences(supplyDs, feature_cols, target_cols, pad_to_max=True)
            X = X.reshape(X.shape[0], -1)

            split_index = int(len(X) * 0.8)
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]

            model = Sequential([
                Dense(128, activation='relu', input_dim=X_train.shape[1]),
                Dropout(0.2),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test), verbose=0)

            y_pred = model.predict(X_test)

            y_supply_pred = scaler_supply.inverse_transform(y_pred)
            y_supply_true = scaler_supply.inverse_transform(y_test)

            mse = mean_squared_error(y_supply_true, y_supply_pred)
            mae = mean_absolute_error(y_supply_true, y_supply_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_supply_true, y_supply_pred)
            modelName = "MLP"
            results = [mse, mae, rmse, r2, modelName]

            print('\n', "Supply Data MLP Model Results:")
            print("MSE: {:.4f}".format(mse))
            print("MAE: {:.4f}".format(mae))
            print("RMSE: {:.4f}".format(rmse))
            print("R2: {:.4f}".format(r2))

            return results

        else:
            print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
            return

def CNNModelDS(demandDs: pd.DataFrame, supplyDs: pd.DataFrame, ident: int, plots: bool):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy()

    if ident == 1:
        # Rename DATE-TIME to TimeStamp for compatibility
        demandDs.rename(columns={'DATE-TIME': 'TimeStamp'}, inplace=True)

        # Ensure MW column is numeric and drop NaN values
        demandDs['MW'] = pd.to_numeric(demandDs['MW'], errors='coerce')
        demandDs.dropna(subset=['MW', 'TimeStamp'], inplace=True)

        # Scale the demand data
        scaler_demand = MinMaxScaler()
        demandDs['MW'] = scaler_demand.fit_transform(demandDs[['MW']])

        # Create monthly sequences
        feature_cols = ['MW']
        target_cols = ['MW']
        X, y = create_monthly_sequences(demandDs, feature_cols, target_cols, pad_to_max=True)

        # Reshape for CNN compatibility
        X = X.reshape(X.shape[0], X.shape[1], len(feature_cols))

        # Split into training and testing sets
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # Define and train CNN model
        model = Sequential([
            Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test), verbose=0)

        # Make predictions
        y_pred = model.predict(X_test)

        # Inverse transform predictions and true values
        y_demand_pred = scaler_demand.inverse_transform(y_pred)
        y_demand_true = scaler_demand.inverse_transform(y_test)

        # Calculate metrics
        mse = mean_squared_error(y_demand_true, y_demand_pred)
        mae = mean_absolute_error(y_demand_true, y_demand_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_demand_true, y_demand_pred)
        modelName = "CNN"
        results = [mse, mae, rmse, r2, modelName]

        # Plot results if requested
        if plots:
            plt.figure(figsize=(10, 5))
            plt.plot(y_demand_true, label='Actual')
            plt.plot(y_demand_pred, label='Predicted')
            plt.title('CNN Model: Actual vs Predicted Demand')
            plt.xlabel('Time')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()

        print('\n', "Demand Data CNN Model Results:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))

        return results

    elif ident == 2:
        # Similar changes for supply dataset
        supplyDs.rename(columns={'Date & Time': 'TimeStamp'}, inplace=True)
        supplyDs['MW'] = pd.to_numeric(supplyDs['MW'], errors='coerce')
        supplyDs.dropna(subset=['MW', 'TimeStamp'], inplace=True)

        scaler_supply = MinMaxScaler()
        supplyDs['MW'] = scaler_supply.fit_transform(supplyDs[['MW']])

        feature_cols = ['MW']
        target_cols = ['MW']
        X, y = create_monthly_sequences(supplyDs, feature_cols, target_cols, pad_to_max=True)

        # Reshape for CNN compatibility
        X = X.reshape(X.shape[0], X.shape[1], len(feature_cols))

        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        model = Sequential([
            Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test), verbose=0)

        y_pred = model.predict(X_test)

        y_supply_pred = scaler_supply.inverse_transform(y_pred)
        y_supply_true = scaler_supply.inverse_transform(y_test)

        mse = mean_squared_error(y_supply_true, y_supply_pred)
        mae = mean_absolute_error(y_supply_true, y_supply_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_supply_true, y_supply_pred)
        modelName = "CNN"
        results = [mse, mae, rmse, r2, modelName]

        print('\n', "Supply Data CNN Model Results:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))

        return results

    else:
        print("Invalid identifier. Please use 1 for demand data or 2 for supply data.")
        return


def GBDTModelDS(demandDs: pd.DataFrame, supplyDs: pd.DataFrame, ident: int, plots: bool):
    demandDs = demandDs.copy()
    supplyDs = supplyDs.copy()

    if ident == 1:
        # Rename DATE-TIME to TimeStamp for compatibility
        demandDs.rename(columns={'DATE-TIME': 'TimeStamp'}, inplace=True)

        # Ensure MW column is numeric and drop NaN values
        demandDs['MW'] = pd.to_numeric(demandDs['MW'], errors='coerce')
        demandDs.dropna(subset=['MW', 'TimeStamp'], inplace=True)

        # Scale the demand data
        scaler_demand = MinMaxScaler()
        demandDs['MW'] = scaler_demand.fit_transform(demandDs[['MW']])

        # Create monthly sequences
        feature_cols = ['MW']
        target_cols = ['MW']
        X, y = create_monthly_sequences(demandDs, feature_cols, target_cols, pad_to_max=True)
        X = X.reshape(X.shape[0], -1)  # Flatten for GBDT compatibility

        # Split into training and testing sets
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # Train GBDT model
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1]
        }
        model = GradientBoostingRegressor(random_state=42)
        grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_

        # Make predictions
        y_pred = model.predict(X_test)

        # Inverse transform predictions and true values
        y_demand_pred = scaler_demand.inverse_transform(y_pred.reshape(-1, 1))
        y_demand_true = scaler_demand.inverse_transform(y_test)

        # Calculate metrics
        mse = mean_squared_error(y_demand_true, y_demand_pred)
        mae = mean_absolute_error(y_demand_true, y_demand_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_demand_true, y_demand_pred)
        modelName = "GBDT"
        results = [mse, mae, rmse, r2, modelName]

        # Plot results if requested
        if plots:
            plt.figure(figsize=(10, 5))
            plt.plot(y_demand_true, label='Actual')
            plt.plot(y_demand_pred, label='Predicted')
            plt.title('GBDT Model: Actual vs Predicted Demand')
            plt.xlabel('Time')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()

        print('\n', "Demand Data GBDT Model Results:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))

        return results

    elif ident == 2:
        # Similar changes for supply dataset
        supplyDs.rename(columns={'Date & Time': 'TimeStamp'}, inplace=True)
        supplyDs['MW'] = pd.to_numeric(supplyDs['MW'], errors='coerce')
        supplyDs.dropna(subset=['MW', 'TimeStamp'], inplace=True)

        scaler_supply = MinMaxScaler()
        supplyDs['MW'] = scaler_supply.fit_transform(supplyDs[['MW']])

        feature_cols = ['MW']
        target_cols = ['MW']
        X, y = create_monthly_sequences(supplyDs, feature_cols, target_cols, pad_to_max=True)
        X = X.reshape(X.shape[0], -1)

        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1]
        }
        model = GradientBoostingRegressor(random_state=42)
        grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_

        y_pred = model.predict(X_test)

        y_supply_pred = scaler_supply.inverse_transform(y_pred.reshape(-1, 1))
        y_supply_true = scaler_supply.inverse_transform(y_test)

        mse = mean_squared_error(y_supply_true, y_supply_pred)
        mae = mean_absolute_error(y_supply_true, y_supply_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_supply_true, y_supply_pred)
        modelName = "GBDT"
        results = [mse, mae, rmse, r2, modelName]

        # Plot results if requested
        if plots:
            plt.figure(figsize=(10, 5))
            plt.plot(y_supply_true, label='Actual')
            plt.plot(y_supply_pred, label='Predicted')
            plt.title('GBDT Model: Actual vs Predicted Supply')
            plt.xlabel('Time')
            plt.ylabel('MW')
            plt.legend()
            plt.tight_layout()
            plt.show()

        print('\n', "Supply Data GBDT Model Results:")
        print("MSE: {:.4f}".format(mse))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
        print("R2: {:.4f}".format(r2))

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
"""
dem_TREE = decisionTreeModelDS(sp_DemandDef, sp_SupplyDef, 1,True)
dem_RF = randomForestModelDS(sp_DemandDef, sp_SupplyDef, 1,True)
dem_XGB = xgbModelDS(sp_DemandDef, sp_SupplyDef, 1,True)
dem_GB = gbModelDS(sp_DemandDef, sp_SupplyDef, 1,True)
dem_BLSTM = biDirectionalLSTMDS(sp_DemandDef, sp_SupplyDef, 1,True)
dem_LSTM = LSTMModelDS(sp_DemandDef, sp_SupplyDef, 1,True)
dem_GRU = GRUModelDS(sp_DemandDef, sp_SupplyDef, 1,True)
dem_SVR = SVRModelDS(sp_DemandDef, sp_SupplyDef, 1,True)
dem_MLP = MLPModelDS(sp_DemandDef, sp_SupplyDef, 1,True)
dem_CNN = CNNModelDS(sp_DemandDef, sp_SupplyDef, 1,True)
dem_GBDT = GBDTModelDS(sp_DemandDef, sp_SupplyDef, 1,True)

demandModelresults = [dem_TREE, dem_RF, dem_XGB, dem_GB, dem_BLSTM, dem_LSTM, dem_GRU, dem_SVR, dem_MLP, dem_CNN, dem_GBDT]
demandBestResultsOrdered = BetterModelSelectionMethod(demandModelresults)

"""
sup_TREE = decisionTreeModelDS(sp_DemandDef, sp_SupplyDef, 2,True)
sup_RF = randomForestModelDS(sp_DemandDef, sp_SupplyDef, 2,True)
sup_XGB = xgbModelDS(sp_DemandDef, sp_SupplyDef, 2,True)
sup_GB = gbModelDS(sp_DemandDef, sp_SupplyDef, 2,True)
sup_BLSTM = biDirectionalLSTMDS(sp_DemandDef, sp_SupplyDef, 2,True)
sup_LSTM = LSTMModelDS(sp_DemandDef, sp_SupplyDef, 2,True)
sup_GRU = GRUModelDS(sp_DemandDef, sp_SupplyDef, 2,True)
sup_SVR = SVRModelDS(sp_DemandDef, sp_SupplyDef, 2,True)
sup_MLP = MLPModelDS(sp_DemandDef, sp_SupplyDef, 2,True)
sup_CNN = CNNModelDS(sp_DemandDef, sp_SupplyDef, 2,True)
sup_GBDT = GBDTModelDS(sp_DemandDef, sp_SupplyDef, 2,True)

#sup_NSGA2_CNN = NSGA2_CNN_ModelDS(sp_WeatherxDem,sp_WeatherxSup,2) 
#sup_NSGA3_CNN = NSGA3_CNN_ModelDS(sp_WeatherxDem,sp_WeatherxSup,2) 
##supplyModelresults = [sup_TREE, sup_RF, sup_XGB, sup_GB, sup_BLSTM,sup_LSTM, sup_GRU, sup_SVR, sup_MLP, sup_CNN,sup_GBDT, sup_NSGA2_CNN,sup_NSGA3_CNN]


supplyModelresults = [sup_TREE, sup_RF, sup_XGB, sup_GB, sup_BLSTM,sup_LSTM, sup_GRU, sup_SVR, sup_MLP, sup_CNN,sup_GBDT]
supplyBestResultsOrdered = BetterModelSelectionMethod(supplyModelresults)

"""
print("\n Demand Solo Model Ranking (Best to Worst) - Monthly:")
print("{:<20} {:>12} {:>12} {:>12} {:>10}".format("Model", "MSE", "MAE", "RMSE", "R2"))
print("-" * 70)
for res in demandBestResultsOrdered:
    print("{:<20} {:>12.4f} {:>12.4f} {:>12.4f} {:>10.4f}".format(
        res[4], res[0], res[1], res[2], res[3]
    ))
"""
print("\n Supply Solo Model Ranking (Best to Worst) - Monthly:")
print("{:<20} {:>12} {:>12} {:>12} {:>10}".format("Model", "MSE", "MAE", "RMSE", "R2"))
print("-" * 70)
for res in supplyBestResultsOrdered:
    print("{:<20} {:>12.4f} {:>12.4f} {:>12.4f} {:>10.4f}".format(
        res[4], res[0], res[1], res[2], res[3]
      ))


