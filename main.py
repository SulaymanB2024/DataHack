#!/usr/bin/env python3
"""
GG.py - MLDS Datahack 2025: Wind Speed Forecasting and Insurance Pricing

How to Use:
-----------
1) Make sure you have a folder named 'data' in the same directory as this script.
   Inside 'data', you should have:
      - training_data.csv
      - event_1.csv, event_2.csv, ..., event_10.csv
2) Run: python GG.py
3) A 'submission.csv' file will be created under 'submission/'.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# ============================
# 1. Data Loading and Preprocessing
# ============================
def load_data():
    """
    Loads training data and event files from the 'data' folder.

    Returns:
        df_train (pd.DataFrame): Training data.
        events (dict): Dict mapping event file name -> DataFrame filtered for GANopolis.
    """
    training_path = os.path.join("data", "training_data.csv")
    if not os.path.exists(training_path):
        raise FileNotFoundError(f"[ERROR] Cannot find {training_path}")

    # Load training data
    df_train = pd.read_csv(training_path)
    print(f"[INFO] Training data loaded: {df_train.shape[0]} records.")

    # Gather all event files matching 'event_*.csv' in the 'data' folder
    event_files = sorted(glob.glob(os.path.join("data", "event_*.csv")))
    if not event_files:
        raise FileNotFoundError("[ERROR] No event files found in the 'data' folder.")

    # Load each event file and filter for GANopolis
    events = {}
    for file in event_files:
        df_event = pd.read_csv(file)
        df_gano = df_event[df_event['city'] == 'GANopolis'].copy()
        event_name = os.path.basename(file)
        events[event_name] = df_gano
        print(f"[INFO] {event_name} loaded: {df_gano.shape[0]} records for GANopolis.")
    return df_train, events

def add_features(df):
    """
    Adds derived features to the dataset:
      - Computes wind speed if missing.
      - Creates time-based features: hour_of_day, day (if day is not numeric).

    Args:
        df (pd.DataFrame): Input data.

    Returns:
        pd.DataFrame: Data with additional features.
    """
    # Compute wind speed if not present or if all null
    if 'wind speed' not in df.columns or df['wind speed'].isnull().all():
        df['wind speed'] = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2)

    # Create hour_of_day if 'hour' exists
    if 'hour' in df.columns:
        df['hour_of_day'] = df['hour'] % 24

    # Ensure 'day' is numeric; if missing, derive from hour
    if 'day' not in df.columns:
        # If 'day' is missing, create from hour // 24
        df['day'] = df['hour'] // 24
    else:
        if not np.issubdtype(df['day'].dtype, np.number):
            df['day'] = pd.to_numeric(df['day'], errors='coerce')

    return df

# ============================
# 2. Exploratory Data Analysis (EDA)
# ============================
def perform_eda(df):
    """
    Performs basic exploratory data analysis.

    Args:
        df (pd.DataFrame): Training data.
    """
    print("\n[EDA] Head of the training data:")
    print(df.head())
    print("\n[EDA] Summary statistics:")
    print(df.describe())

    # Plot wind speed over time if columns exist
    if 'wind speed' in df.columns and 'hour' in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(df['hour'], df['wind speed'], label='Wind Speed')
        plt.xlabel('Hour')
        plt.ylabel('Wind Speed')
        plt.title('Wind Speed over Simulation Hours (Training Data)')
        plt.legend()
        plt.show()
    else:
        print("[EDA] 'hour' or 'wind speed' not found; skipping plot.")

# ============================
# 3. Model Training: Forecasting Wind Speed
# ============================
def train_wind_model(df, feature_cols, target_col):
    """
    Trains a RandomForestRegressor using time-series cross-validation.

    Args:
        df (pd.DataFrame): Training data with necessary features.
        feature_cols (list): List of feature names.
        target_col (str): Target variable name.

    Returns:
        best_model (RandomForestRegressor): Trained model with best parameters.
        cv_mse (float): Cross-validated MSE of the best model.
    """
    # Filter down to the desired features and target
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Define a time series cross-validator
    tscv = TimeSeriesSplit(n_splits=5)

    # Instantiate model and hyperparameter grid
    rf = RandomForestRegressor(random_state=42)
    param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10, None]}

    # Use GridSearchCV with neg_mean_squared_error for scoring
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_
    cv_mse = -grid_search.best_score_

    print(f"[TRAINING] Best RandomForest parameters: {grid_search.best_params_}")
    print(f"[TRAINING] Cross-validated MSE: {cv_mse:.4f}")

    return best_model, cv_mse

# ============================
# 4. Forecasting for Event Data
# ============================
def forecast_event(model, df_event, feature_cols):
    """
    Forecasts wind speed for an event using the provided model.

    Args:
        model (RandomForestRegressor): Trained forecasting model.
        df_event (pd.DataFrame): Event data for GANopolis.
        feature_cols (list): List of feature names used for prediction.

    Returns:
        pd.DataFrame: Event data with 'predicted_wind_speed' column.
    """
    df_event = add_features(df_event)
    X_event = df_event[feature_cols].copy()

    # Generate predictions
    df_event['predicted_wind_speed'] = model.predict(X_event)

    # Quick visualization of predictions
    if 'hour' in df_event.columns:
        plt.figure(figsize=(10, 4))
        plt.plot(df_event['hour'], df_event['predicted_wind_speed'], label='Predicted Wind Speed')
        plt.xlabel('Hour')
        plt.ylabel('Predicted Wind Speed')
        plt.title(f"Wind Speed Forecast for {len(df_event)} records (GANopolis)")
        plt.legend()
        plt.show()
    else:
        print("[FORECAST] 'hour' column not found; skipping plot.")

    return df_event

# ============================
# 5. Insurance Pricing Optimization
# ============================
def optimize_price(average_damage):
    """
    Optimizes the insurance price given average damage per customer.

    Pricing formulas:
      demand = 10000 - (10000 / 500) * price
      unit_contribution_margin = price - damage
      profit = demand * unit_contribution_margin

    Args:
        average_damage (float): Estimated average damage per customer over 5 days.

    Returns:
        (float, float): (best_price, best_profit)
    """
    prices = np.linspace(0, 1000, 100)  # Search range: 0 to 1000, in 100 steps
    best_profit = -np.inf
    best_price = 0.0

    for price in prices:
        demand = 10000 - (10000 / 500) * price
        unit_margin = price - average_damage
        profit = demand * unit_margin
        if profit > best_profit:
            best_profit = profit
            best_price = price

    return best_price, best_profit

# ============================
# 6. Generate Submission
# ============================
def generate_submission(events, prices_results, forecast_period=120):
    """
    Creates the submission DataFrame containing event forecasts and optimal pricing.

    Args:
        events (dict): Dictionary of event DataFrames.
        prices_results (dict): Dictionary mapping event_name -> {'optimal_price': ..., 'expected_profit': ...}
        forecast_period (int): Number of hours for the forecast period.

    Returns:
        pd.DataFrame: DataFrame ready to be saved to CSV.
    """
    submission_rows = []
    for event_name, df_event in events.items():
        # Use last 'forecast_period' hours as the forecast period
        if df_event.shape[0] >= forecast_period:
            forecast_df = df_event.tail(forecast_period)
        else:
            forecast_df = df_event

        avg_wind_speed = forecast_df['predicted_wind_speed'].mean()
        price_info = prices_results.get(event_name, {})
        price = price_info.get('optimal_price', np.nan)

        submission_rows.append({
            'event': event_name,
            'predicted_wind_speed_mean': avg_wind_speed,
            'optimal_price': price
        })

    submission_df = pd.DataFrame(submission_rows)
    return submission_df

# ============================
# 7. Main Execution Pipeline
# ============================
def main():
    print("D:")
    """
    Main pipeline:
      1) Load data from 'data' folder
      2) Perform EDA (optional)
      3) Train wind speed model
      4) Forecast wind speeds for events
      5) Compute insurance pricing
      6) Generate submission CSV
    """
    # 1. Load data from the 'data' folder
    df_train, events = load_data()

    # 2. Add features & Perform EDA (optional)
    df_train = add_features(df_train)
    perform_eda(df_train)

    # 3. Train model
    feature_cols = ['pressure', 'air_temp', 'ground_temp', 'hour_of_day', 'day']
    target_col = 'wind speed'
    model, mse = train_wind_model(df_train, feature_cols, target_col)

    # 4. Forecast wind speed & optimize pricing
    prices_results = {}
    for event_name, df_event in events.items():
        df_event = forecast_event(model, df_event, feature_cols)
        events[event_name] = df_event  # Store updated event data

        # Calculate average damage (if available) or default to 100
        if 'damage' in df_event.columns and not df_event['damage'].isnull().all():
            avg_damage = df_event['damage'].mean()
        else:
            avg_damage = 100.0

        # Optimize price
        optimal_price, expected_profit = optimize_price(avg_damage)
        prices_results[event_name] = {
            'optimal_price': optimal_price,
            'expected_profit': expected_profit
        }
        print(f"[PRICING] {event_name}: Optimal Price = {optimal_price:.2f}, "
              f"Expected Profit = {expected_profit:.2f}")

    # 5. Generate submission
    submission_df = generate_submission(events, prices_results, forecast_period=120)
    submission_path = os.path.join("submission", "submission.csv")
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission_df.to_csv(submission_path, index=False)
    print(f"[INFO] Submission file saved at: {submission_path}")

if __name__ == "__main__":
    main()


