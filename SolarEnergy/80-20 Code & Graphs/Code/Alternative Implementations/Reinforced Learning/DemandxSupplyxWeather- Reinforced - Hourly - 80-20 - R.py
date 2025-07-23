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
from sklearn.metrics import classification_report
#----#
#Reinforced Learning Imports
from stable_baselines3 import DQN, PPO,A2C
import gym
from gym import spaces



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

sp_DemandDef = pd.read_excel(r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\Datasets\Sakakah 2021 Demand dataset.xlsx")
sp_SupplyDef = pd.read_excel(r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\Datasets\Sakakah 2021 PV supply dataset.xlsx")
sp_WeatherDef = pd.read_excel(r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\Datasets\weather for solar NEW 2021.xlsx")
sp_WeatherDe = pd.read_excel(r"C:\Users\Harry\source\repos\SolarEnergy\SolarEnergy\Datasets\Weather for demand 2018.xlsx")

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
sp_FullMerg = pd.merge(sp_WeatherDe, sp_FullMerg, on="TimeStamp", how="inner")

##linear interpolation to smooth out the datasets
sp_FullMerg.replace(-999, np.nan, inplace=True)
sp_FullMerg.interpolate(method='linear', inplace=True)
sp_FullMerg.dropna(inplace=True)
sp_FullMerg["UnixTime"] = sp_FullMerg["TimeStamp"].apply(lambda x: x.timestamp())
sp_FullMerg["Year"] = sp_FullMerg["TimeStamp"].dt.year
sp_FullMerg["Month"] = sp_FullMerg["TimeStamp"].dt.month
sp_FullMerg["Day"] = sp_FullMerg["TimeStamp"].dt.day
sp_FullMerg["Hour"] = sp_FullMerg["TimeStamp"].dt.hour
#This code is used to print the descriptive statistics of the merged dataset. Which was then used for the 
#threshold calculation for the classification report which ended up being :•	[112.21, 187.74, 237.23, 313.32, 397.67]
#print(sp_FullMerg['Demand_MW'].describe())
#print(sp_FullMerg['Supply_MW'].describe())

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
sp_FullMerg["hour"] = sp_FullMerg["TimeStamp"].dt.hour
sp_FullMerg["day"] = sp_FullMerg["TimeStamp"].dt.day
sp_FullMerg["month"] = sp_FullMerg["TimeStamp"].dt.month
feature_cols = [
    "ALLSKY_SFC_SW_DWN_S", "ALLSKY_SFC_UV_INDEX_S", "T2M_S", "PRECTOTCORR_S", "ALLSKY_KT_S",
    "CLRSKY_SFC_PAR_TOT_S", "RH2M_S", "PS_S", "PSC_S", "WS10M_S", "WD10M_S", 
    "ALLSKY_SFC_SW_DWN_D", "ALLSKY_SFC_UV_INDEX_D", "T2M_D", "PRECTOTCORR_D", "ALLSKY_KT_D",
    "CLRSKY_SFC_PAR_TOT_D", "RH2M_D", "PS_D", "PSC_D", "WS10M_D", "WD10M_D", "hour", "day", "month"
]

hyperparameters = {
    "PPO": {
        "learning_rate": 0.0003,
        "n_steps": 2048,
        "batch_size": 64,
        "gae_lambda": 0.95,
        "gamma": 0.99,
        "clip_range": 0.2,
        "ent_coef": 0.01
    },
    "DQN": {
        "learning_rate": 0.0005,
        "buffer_size": 1000000,
        "exploration_fraction": 0.1,
        "exploration_final_eps": 0.01,
        "batch_size": 32,
        "train_freq": 4,
        "gamma": 0.99,
        "target_update_interval": 1000
    },
    "A2C": {
        "learning_rate": 0.0007,
        "n_steps": 5,
        "gamma": 0.99,
        "vf_coef": 0.25,
        "ent_coef": 0.01,
        "max_grad_norm": 0.5
    }
}
train_data, test_data = train_test_split(sp_FullMerg, test_size=0.2, random_state=42)

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
def generate_classification_report(y_true, y_pred, thresholds, model_name, data_type):
    """
    Generate a structured classification report for regression models by binning predictions into classes.
    :param y_true: Actual values (continuous).
    :param y_pred: Predicted values (continuous).
    :param thresholds: List of thresholds to bin the values into classes.
    :param model_name: Name of the model for identification in the output.
    :param data_type: Type of data being evaluated ("Demand" or "Supply").
    """
    # Dynamically adjust thresholds to cover the range of values
    thresholds = sorted(set([y_true.min(), y_pred.min()] + thresholds + [y_true.max(), y_pred.max()]))

    # Convert continuous values to discrete classes
    y_true_classes = pd.cut(y_true.flatten(), bins=thresholds, labels=False)
    y_pred_classes = pd.cut(y_pred.flatten(), bins=thresholds, labels=False)
    valid_indices = ~np.isnan(y_true_classes) & ~np.isnan(y_pred_classes)
    y_true_classes = y_true_classes[valid_indices]
    y_pred_classes = y_pred_classes[valid_indices]

    # Debugging output
    print("Thresholds:", thresholds)
    print("NaN in y_true_classes:", pd.isna(y_true_classes).sum())
    print("NaN in y_pred_classes:", pd.isna(y_pred_classes).sum())

    # Generate classification report
    report_dict = classification_report(y_true_classes, y_pred_classes, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    print(f"\nStructured Classification Report for {model_name} ({data_type}):")
    print(report_df.round(2))

def evaluate_model(env, model, seq_length, identifier="Model"):
    """
    Evaluate the model for both Demand_MW and Supply_MW and combine metrics.
    
    Args:
        env: The environment used for evaluation.
        model: The trained RL model.
        seq_length: Sequence length for time-based features.
        identifier: String identifier for the model (e.g., "DQN", "PPO").
    """
    # Reset the environment
    state = env.reset()
    actual_demand_values = []
    actual_supply_values = []
    predicted_demand_values = []
    predicted_supply_values = []

    # Generate predictions
    while True:
        action, _ = model.predict(state)

        # Handle discrete or continuous actions
        if isinstance(env.action_space, gym.spaces.Discrete):
            # Discrete action space: map action index to continuous values
            demand_scale, supply_scale = env.action_mapping[action]
        else:
            # Continuous action space: use action array directly
            demand_scale, supply_scale = action

        # Predicted values based on actions
        actual_demand = env.data.iloc[env.current_step]["Demand_MW"]
        actual_supply = env.data.iloc[env.current_step]["Supply_MW"]
        predicted_demand = demand_scale * actual_demand
        predicted_supply = supply_scale * actual_supply

        # Store values for evaluation
        predicted_demand_values.append(predicted_demand)
        predicted_supply_values.append(predicted_supply)
        actual_demand_values.append(actual_demand)
        actual_supply_values.append(actual_supply)

        # Step the environment
        state, _, done, _ = env.step(action)
        if done:
            break

    # Convert to numpy arrays
    actual_demand_values = np.array(actual_demand_values)
    actual_supply_values = np.array(actual_supply_values)
    predicted_demand_values = np.array(predicted_demand_values)
    predicted_supply_values = np.array(predicted_supply_values)

    # Calculate metrics for Demand
    mse_demand = mean_squared_error(actual_demand_values, predicted_demand_values)
    mae_demand = mean_absolute_error(actual_demand_values, predicted_demand_values)
    rmse_demand = np.sqrt(mse_demand)
    r2_demand = r2_score(actual_demand_values, predicted_demand_values)

    # Calculate metrics for Supply
    mse_supply = mean_squared_error(actual_supply_values, predicted_supply_values)
    mae_supply = mean_absolute_error(actual_supply_values, predicted_supply_values)
    rmse_supply = np.sqrt(mse_supply)
    r2_supply = r2_score(actual_supply_values, predicted_supply_values)

    # Calculate combined metrics
    mse_combined = mean_squared_error(
        np.concatenate([actual_demand_values, actual_supply_values]),
        np.concatenate([predicted_demand_values, predicted_supply_values])
    )
    mae_combined = mean_absolute_error(
        np.concatenate([actual_demand_values, actual_supply_values]),
        np.concatenate([predicted_demand_values, predicted_supply_values])
    )
    rmse_combined = np.sqrt(mse_combined)
    r2_combined = r2_score(
        np.concatenate([actual_demand_values, actual_supply_values]),
        np.concatenate([predicted_demand_values, predicted_supply_values])
    )

    # Print metrics
    print(f"\nEvaluation Results for {identifier}:")
    print("Demand Metrics:")
    print(f"MSE: {mse_demand:.4f}, MAE: {mae_demand:.4f}, RMSE: {rmse_demand:.4f}, R2: {r2_demand:.4f}")
    print("Supply Metrics:")
    print(f"MSE: {mse_supply:.4f}, MAE: {mae_supply:.4f}, RMSE: {rmse_supply:.4f}, R2: {r2_supply:.4f}")
    print("Combined Metrics:")
    print(f"MSE: {mse_combined:.4f}, MAE: {mae_combined:.4f}, RMSE: {rmse_combined:.4f}, R2: {r2_combined:.4f}")

    # Combine results into a dictionary for clarity
    results = {
        "Demand": {
            "MSE": mse_demand,
            "MAE": mae_demand,
            "RMSE": rmse_demand,
            "R2": r2_demand,
            "Model": identifier
        },
        "Supply": {
            "MSE": mse_supply,
            "MAE": mae_supply,
            "RMSE": rmse_supply,
            "R2": r2_supply,
            "Model": identifier

        },
        "Combined": {
            "MSE": mse_combined,
            "MAE": mae_combined,
            "RMSE": rmse_combined,
            "R2": r2_combined,
            "Model": identifier

        }
    }

    return results
class EnergyEnvironment(gym.Env):
    def __init__(self, data, feature_columns, seq_length=1, seed=777, algorithm="PPO"):
        """
        Initialize the EnergyEnvironment class.

        Args:
            data (pd.DataFrame): The dataset containing all columns.
            feature_columns (list): List of column names to be used as features.
            seq_length (int): Sequence length for time-based features.
            seed (int): Random seed for reproducibility.
            algorithm (str): The RL algorithm being used ("PPO", "DQN", "A2C").
        """
        super(EnergyEnvironment, self).__init__()
        self.data = data
        self.feature_columns = feature_columns
        self.seq_length = seq_length
        self.current_step = seq_length  # Start after lagged data
        self.algorithm = algorithm

        # Set the random seed
        self.seed(seed)

        # Create the initial state using the specified feature columns
        self.state = self.data.iloc[self.current_step - seq_length:self.current_step][
            feature_columns
        ].values.astype(np.float32).flatten()

        # Define action space based on the algorithm
        if self.algorithm == "DQN":
            self.num_actions_demand = 5
            self.num_actions_supply = 5
            self.action_space = spaces.Discrete(self.num_actions_demand * self.num_actions_supply)
            self.action_mapping = self._create_action_mapping()
        elif self.algorithm in ["PPO", "A2C"]:
            self.action_space = spaces.Box(low=0.5, high=1.5, shape=(2,), dtype=np.float32)

        # Define observation space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state.shape[0],), dtype=np.float32)

    def _create_action_mapping(self):
        """
        Create a mapping from discrete actions to continuous values for Demand_MW and Supply_MW.
        """
        demand_values = np.linspace(0.5, 1.5, self.num_actions_demand)  # Scale for Demand_MW
        supply_values = np.linspace(0.5, 1.5, self.num_actions_supply)  # Scale for Supply_MW
        action_mapping = []
        for demand in demand_values:
            for supply in supply_values:
                action_mapping.append((demand, supply))
        return action_mapping

    def seed(self, seed=None):
        """
        Set the random seed for reproducibility.

        Args:
            seed (int): Random seed.
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)  # Set seed for numpy
        random.seed(seed)  # Set seed for Python's random module
        return [seed]

    def step(self, action):
        """
        Apply actions and update the environment state.

        Args:
            action (int or np.ndarray): Discrete action index (DQN) or continuous action array (PPO/A2C).

        Returns:
            tuple: Updated state, reward, done flag, and additional info.
        """
        if self.algorithm == "DQN":
            # Map the discrete action to continuous values
            demand_scale, supply_scale = self.action_mapping[action]
        else:
            # Use continuous actions directly
            demand_scale, supply_scale = action

        # Extract the actual values for Demand_MW and Supply_MW
        actual_demand = self.data.iloc[self.current_step]["Demand_MW"]
        actual_supply = self.data.iloc[self.current_step]["Supply_MW"]

        # Predicted values based on the actions
        predicted_demand = demand_scale * actual_demand  # Scale prediction for demand
        predicted_supply = supply_scale * actual_supply  # Scale prediction for supply

        # Calculate reward
        reward = self.calculate_reward(predicted_demand, predicted_supply)

        # Update the current step
        self.current_step += 1
        done = self.current_step >= len(self.data)

        if not done:
            # Update the state using the specified feature columns
            self.state = self.data.iloc[self.current_step - self.seq_length:self.current_step][
                self.feature_columns
            ].values.astype(np.float32).flatten()
        else:
            self.state = None

        # Return the next state, reward, done flag, and an empty info dictionary
        return self.state, reward, done, {}

    def reset(self):
        """
        Reset the environment to the initial state.

        Returns:
            np.ndarray: Initial state.
        """
        self.current_step = self.seq_length
        # Reset the state using the specified feature columns
        self.state = self.data.iloc[self.current_step - self.seq_length:self.current_step][
            self.feature_columns
        ].values.astype(np.float32).flatten()
        return self.state

    def calculate_reward(self, predicted_demand, predicted_supply):
        """
        Calculate the reward based on the difference between actual and predicted values.

        Args:
            predicted_demand (float): Predicted value for Demand_MW.
            predicted_supply (float): Predicted value for Supply_MW.

        Returns:
            float: Total reward combining Demand_MW and Supply_MW rewards.
        """
        # Extract the actual values for Demand_MW and Supply_MW
        actual_demand = self.data.iloc[self.current_step]["Demand_MW"]
        actual_supply = self.data.iloc[self.current_step]["Supply_MW"]

        # Calculate absolute errors
        error_demand = abs(actual_demand - predicted_demand)
        error_supply = abs(actual_supply - predicted_supply)

        # Reward is inversely proportional to the error
        reward_demand = 1 / (1 + error_demand)  # Higher reward for smaller error
        reward_supply = 1 / (1 + error_supply)  # Higher reward for smaller error

        # Combine rewards into a single value
        total_reward = reward_demand + reward_supply

        return total_reward




def PPO_RL_ModelDS(train_data: pd.DataFrame, test_data: pd.DataFrame, seq_length: int = 1, timesteps: int = 10000):
    env = EnergyEnvironment(train_data, feature_columns=feature_cols, seq_length=seq_length)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        **hyperparameters["PPO"]  # Pass hyperparameters
    )
    model.learn(total_timesteps=timesteps)

    # Evaluate the model on the test data
    test_env = EnergyEnvironment(test_data, feature_columns=feature_cols, seq_length=seq_length)
    results = evaluate_model(test_env, model, seq_length, "PPO")
    return results


def DQN_RL_ModelDS(train_data: pd.DataFrame, test_data: pd.DataFrame, seq_length: int = 1, timesteps: int = 200000):
    env = EnergyEnvironment(train_data, feature_columns=feature_cols, seq_length=seq_length, algorithm="DQN")
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        **hyperparameters["DQN"]  # Pass hyperparameters
    )
    model.learn(total_timesteps=timesteps)

    # Evaluate the model on the test data
    test_env = EnergyEnvironment(test_data, feature_columns=feature_cols, seq_length=seq_length, algorithm="DQN")
    results = evaluate_model(test_env, model, seq_length, "DQN")
    return results

def A2C_RL_ModelDS(train_data: pd.DataFrame, test_data: pd.DataFrame, seq_length: int = 1, timesteps: int = 50000):
    env = EnergyEnvironment(train_data, feature_columns=feature_cols, seq_length=seq_length, algorithm="A2C")
    model = A2C(
        "MlpPolicy",
        env,
        verbose=1,
        **hyperparameters["A2C"]  # Pass hyperparameters
    )
    model.learn(total_timesteps=timesteps)

    # Evaluate the model on the test data
    test_env = EnergyEnvironment(test_data, feature_columns=feature_cols, seq_length=seq_length, algorithm="A2C")
    results = evaluate_model(test_env, model, seq_length, "A2C")
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

def BetterModelSelectionMethod(model_results_list):
    """
    Orders the models from best to worst based on MSE, MAE, RMSE, and R2.
    Lower MSE, MAE, RMSE are better; higher R2 is better.

    Args:
        model_results_list (list): List of model results dictionaries to be sorted.

    Returns:
        list: Sorted list of model results dictionaries.
    """
    def compare_results(res1, res2):
        score1 = 0
        score2 = 0

        # MSE
        if res1["MSE"] < res2["MSE"]:
            score1 += 1
        else:
            score2 += 1

        # MAE
        if res1["MAE"] < res2["MAE"]:
            score1 += 1
        else:
            score2 += 1

        # RMSE
        if res1["RMSE"] < res2["RMSE"]:
            score1 += 1
        else:
            score2 += 1

        # R2
        if res1["R2"] > res2["R2"]:  # Higher R2 is better
            score1 += 1
        else:
            score2 += 1

        return score1 > score2

    # Sort using a nested loop (selection sort style)
    n = len(model_results_list)
    ordered = model_results_list.copy()
    for i in range(n):
        best_idx = i
        for j in range(i + 1, n):
            if compare_results(ordered[j], ordered[best_idx]):
                best_idx = j
        if best_idx != i:
            ordered[i], ordered[best_idx] = ordered[best_idx], ordered[i]
    return ordered

PPO_results = PPO_RL_ModelDS(train_data, test_data)
PPO_resultsDem = PPO_results["Demand"]
PPO_resultsSup = PPO_results["Supply"]
PPO_resultsConcat = PPO_results["Combined"]

DNQ_results = DQN_RL_ModelDS(train_data, test_data)
DNQ_resultsDem = DNQ_results["Demand"]
DNQ_resultsSup = DNQ_results["Supply"]
DNQ_resultsConcat = DNQ_results["Combined"]

A2C_results = A2C_RL_ModelDS(train_data, test_data)
A2C_resultsDem = A2C_results["Demand"]
A2C_resultsSup = A2C_results["Supply"]
A2C_resultsConcat = A2C_results["Combined"]

modelresultsDemand = [PPO_resultsDem,DNQ_resultsDem ,A2C_resultsDem ]
modelresultsSupply = [PPO_resultsSup,DNQ_resultsSup , A2C_resultsSup]
modelresultsCombined = [PPO_resultsConcat,DNQ_resultsConcat ,A2C_resultsConcat ]
print(modelresultsDemand)
print(modelresultsSupply)
print(modelresultsCombined)
#for feature, importance in Shap_DecTree["Sorted Feature Importance"]:
#    print(f"{feature:<30} {importance:.4f}"



BestResultsOrderedDemand = BetterModelSelectionMethod(modelresultsDemand)
BestResultsOrderedSupply = BetterModelSelectionMethod(modelresultsSupply)
BestResultsOrderedCombined = BetterModelSelectionMethod(modelresultsCombined)

print("\nReinforced Model Ranking (Best to Worst) Hourly (Demand):")
print("{:<20} {:>12} {:>12} {:>12} {:>10}".format("Model", "MSE", "MAE", "RMSE", "R2"))
print("-" * 70)
for res in BestResultsOrderedDemand:
    print("{:<20} {:>12.4f} {:>12.4f} {:>12.4f} {:>10.4f}".format(
        res.get("Model", "Unknown"),  # Use .get() to handle missing keys gracefully
        res["MSE"],
        res["MAE"],
        res["RMSE"],
        res["R2"]
    ))

print("\nReinforced Model Ranking (Best to Worst) Hourly (Supply):")
print("{:<20} {:>12} {:>12} {:>12} {:>10}".format("Model", "MSE", "MAE", "RMSE", "R2"))
print("-" * 70)
for res in BestResultsOrderedSupply:
    print("{:<20} {:>12.4f} {:>12.4f} {:>12.4f} {:>10.4f}".format(
        res.get("Model", "Unknown"),  # Use .get() to handle missing keys gracefully
        res["MSE"],
        res["MAE"],
        res["RMSE"],
        res["R2"]
    ))

print("\nReinforced Model Ranking (Best to Worst) Hourly (Combined):")
print("{:<20} {:>12} {:>12} {:>12} {:>10}".format("Model", "MSE", "MAE", "RMSE", "R2"))
print("-" * 70)
for res in BestResultsOrderedCombined:
    print("{:<20} {:>12.4f} {:>12.4f} {:>12.4f} {:>10.4f}".format(
        res.get("Model", "Unknown"),  # Use .get() to handle missing keys gracefully
        res["MSE"],
        res["MAE"],
        res["RMSE"],
        res["R2"]
    ))
