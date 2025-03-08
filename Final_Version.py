import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler
import os
from scipy.optimize import minimize
import joblib
import json


# Function to read input data
def read_input_data(input_file):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File {input_file} not found.")
    return pd.read_csv(input_file)


# Enhanced preprocessing with more features
def preprocess_data(data):
    data = data.copy()

    # Basic interpolation for missing values
    data = data.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

    # Cyclical features for hour of the day
    data['hour_sin'] = np.sin(2 * np.pi * data['hour_of_day'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour_of_day'] / 24)

    # Cyclical features for day (seasonal patterns)
    if 'day' in data.columns:
        data['day_sin'] = np.sin(2 * np.pi * data['day'] / 365)  # Assuming yearly cycle
        data['day_cos'] = np.cos(2 * np.pi * data['day'] / 365)

    # Lagged features for wind speed - more comprehensive
    for lag in [1, 2, 3, 6, 12, 24, 48]:
        data[f'windspeed_lag_{lag}'] = data['windspeed'].shift(lag)

    # Weather-related features
    data['pressure_gradient'] = data['pressure'].diff()
    data['temp_gradient'] = data['air_temp'] - data['ground_temp']
    data['temp_change'] = data['air_temp'].diff()

    # Wind components and directional features
    data['wind_direction'] = np.arctan2(data['velocity_y'], data['velocity_x'])
    data['wind_direction_sin'] = np.sin(data['wind_direction'])
    data['wind_direction_cos'] = np.cos(data['wind_direction'])

    # Interaction features
    data['pressure_temp_interaction'] = data['pressure'] * data['air_temp']

    # Rolling statistics (multiple windows)
    for window in [6, 12, 24, 48]:
        data[f'windspeed_roll_mean_{window}'] = data['windspeed'].rolling(window=window, min_periods=1).mean()
        data[f'windspeed_roll_std_{window}'] = data['windspeed'].rolling(window=window, min_periods=1).std()
        data[f'windspeed_roll_max_{window}'] = data['windspeed'].rolling(window=window, min_periods=1).max()
        data[f'pressure_roll_mean_{window}'] = data['pressure'].rolling(window=window, min_periods=1).mean()
        data[f'temp_roll_mean_{window}'] = data['air_temp'].rolling(window=window, min_periods=1).mean()

    # Differences between rolling statistics
    data['windspeed_roll_diff'] = data['windspeed_roll_mean_24'] - data['windspeed_roll_mean_48']
    data['pressure_roll_diff'] = data['pressure_roll_mean_24'] - data['pressure_roll_mean_48']

    # Cap outliers using more robust methods - use IQR method
    for col in ['windspeed', 'pressure', 'air_temp', 'ground_temp']:
        if col in data.columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data[col] = data[col].clip(lower_bound, upper_bound)

    return data


# Function to analyze wind patterns with more visualizations
def analyze_wind_speed(data, city_name='GANopolis', save_path=None):
    plt.figure(figsize=(15, 8))

    # Plot 1: Wind Speed Time Series
    plt.subplot(2, 2, 1)
    plt.plot(data['hour'], data['windspeed'], label='Wind Speed')
    plt.title(f'Wind Speed Time Series - {city_name}')
    plt.xlabel('Hour')
    plt.ylabel('Wind Speed')
    plt.grid(True)

    # Plot 2: Autocorrelation
    plt.subplot(2, 2, 2)
    plot_acf(data['windspeed'].dropna(), lags=48, ax=plt.gca())
    plt.title('Autocorrelation of Wind Speed')

    # Plot 3: Partial Autocorrelation
    plt.subplot(2, 2, 3)
    plot_pacf(data['windspeed'].dropna(), lags=48, ax=plt.gca())
    plt.title('Partial Autocorrelation of Wind Speed')

    # Plot 4: Wind Speed by Hour of Day
    plt.subplot(2, 2, 4)
    hourly_avg = data.groupby('hour_of_day')['windspeed'].mean()
    plt.bar(hourly_avg.index, hourly_avg.values)
    plt.title('Average Wind Speed by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Wind Speed')
    plt.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

    # Additional plots for relationships between variables
    plt.figure(figsize=(15, 10))

    # Plot 5: Wind Speed vs Pressure
    plt.subplot(2, 2, 1)
    plt.scatter(data['pressure'], data['windspeed'], alpha=0.5)
    plt.title('Wind Speed vs Pressure')
    plt.xlabel('Pressure')
    plt.ylabel('Wind Speed')
    plt.grid(True)

    # Plot 6: Wind Speed vs Temperature
    plt.subplot(2, 2, 2)
    plt.scatter(data['air_temp'], data['windspeed'], alpha=0.5)
    plt.title('Wind Speed vs Air Temperature')
    plt.xlabel('Air Temperature')
    plt.ylabel('Wind Speed')
    plt.grid(True)

    # Plot 7: Wind Speed vs Damage
    if 'damage' in data.columns:
        plt.subplot(2, 2, 3)
        plt.scatter(data['windspeed'], data['damage'], alpha=0.5)
        plt.title('Wind Speed vs Damage')
        plt.xlabel('Wind Speed')
        plt.ylabel('Damage')
        plt.grid(True)

    # Plot 8: Wind Direction visualization
    if 'velocity_x' in data.columns and 'velocity_y' in data.columns:
        plt.subplot(2, 2, 4)
        plt.quiver(data['hour'][::24], data['windspeed'][::24],
                   data['velocity_x'][::24], data['velocity_y'][::24],
                   scale=50, width=0.002)
        plt.title('Wind Direction Samples (every 24 hours)')
        plt.xlabel('Hour')
        plt.ylabel('Wind Speed')
        plt.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path.replace('.png', '_relationships.png'))
    plt.show()


# Enhanced SARIMA Model with auto-optimization
class EnhancedSARIMAModel:
    def __init__(self, auto_optimize=True):
        self.model = None
        self.order = None
        self.seasonal_order = None
        self.auto_optimize = auto_optimize
        self.scaler = StandardScaler()


```python


def optimize_parameters(self, data):
    best_aic = float("inf")
    best_order = None
    best_seasonal_order = None

    # Define parameter grid to search
    p_values = range(0, 3)
    d_values = range(0, 2)
    q_values = range(0, 3)
    P_values = range(0, 2)
    D_values = range(0, 2)
    Q_values = range(0, 2)
    s_values = [24]  # Seasonal period (24 hours)

    # Use a subset of data for optimization to save time
    train_data = data.iloc[-336:].copy()  # Last 2 weeks

    print("Optimizing SARIMA parameters...")
    best_model = None

    # Simplified grid search - test fewer combinations
    orders = [(1, 1, 1), (2, 1, 0), (0, 1, 1), (1, 0, 1)]
    seasonal_orders = [(1, 1, 1, 24), (0, 1, 1, 24)]

    for order in orders:
        for seasonal_order in seasonal_orders:
            try:
                model = SARIMAX(
                    train_data['windspeed'],
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                model_fit = model.fit(disp=False, maxiter=50)
                aic = model_fit.aic

                if aic < best_aic:
                    best_aic = aic
                    best_order = order
                    best_seasonal_order = seasonal_order
                    best_model = model_fit
                    print(f"New best: Order={order}, Seasonal={seasonal_order}, AIC={aic}")
            except:
                continue

    self.order = best_order
    self.seasonal_order = best_seasonal_order
    print(f"Best SARIMA parameters: order={best_order}, seasonal_order={best_seasonal_order}, AIC={best_aic}")
    return best_model


def fit(self, data):
    train_data = data.copy()

    if self.auto_optimize:
        best_model = self.optimize_parameters(train_data)
        if best_model is not None:
            self.model = best_model
            return

    # If optimization fails or is disabled, use default parameters
    if not self.order:
        self.order = (1, 1, 1)
    if not self.seasonal_order:
        self.seasonal_order = (1, 1, 1, 24)

    try:
        self.model = SARIMAX(
            train_data['windspeed'],
            order=self.order,
            seasonal_order=self.seasonal_order
        ).fit(disp=False)
    except:
        # Fallback to simpler model if complex one fails
        self.order = (1, 0, 0)
        self.seasonal_order = (0, 1, 0, 24)
        self.model = SARIMAX(
            train_data['windspeed'],
            order=self.order,
            seasonal_order=self.seasonal_order
        ).fit(disp=False)


def predict(self, steps):
    return self.model.forecast(steps=steps)


# Enhanced XGBoost Model
class EnhancedXGBoostModel:
    def __init__(self, lag=48, optimize=True):
        self.model = None
        self.lag = lag
        self.optimize = optimize
        self.feature_importance = None
        self.scaler = StandardScaler()

        # Default parameters
        self.params = {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }

    def get_feature_columns(self):
        lag_features = [f'windspeed_lag_{i}' for i in range(1, self.lag + 1)]
        time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        weather_features = ['pressure', 'air_temp', 'ground_temp', 'pressure_gradient', 'temp_gradient']
        wind_features = ['wind_direction_sin', 'wind_direction_cos']
        rolling_features = [
            'windspeed_roll_mean_24', 'windspeed_roll_std_24', 'windspeed_roll_max_24',
            'pressure_roll_mean_24', 'temp_roll_mean_24', 'windspeed_roll_diff', 'pressure_roll_diff'
        ]

        # Combine all features
        all_features = lag_features + time_features + weather_features + wind_features + rolling_features
        return all_features

    def create_supervised_data(self, data):
        df = data.copy()

        # Create lagged features if not already in the dataframe
        for i in range(1, self.lag + 1):
            if f'windspeed_lag_{i}' not in df.columns:
                df[f'windspeed_lag_{i}'] = df['windspeed'].shift(i)

        # Target is next hour's wind speed
        df['target'] = df['windspeed'].shift(-1)

        # Drop rows with NaN values
        return df.dropna()

    def optimize_hyperparameters(self, X, y):
        print("Optimizing XGBoost hyperparameters...")

        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [4, 6, 8],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }

        # Use time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)

        # Create base model
        base_model = XGBRegressor(random_state=42)

        # Perform grid search with time series cross-validation
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )

        # Fit the grid search to the data
        grid_search.fit(X, y)

        # Get the best parameters
        best_params = grid_search.best_params_
        print(f"Best XGBoost parameters: {best_params}")

        return best_params

    def fit(self, data):
        # Prepare the supervised dataset
        supervised = self.create_supervised_data(data)

        # Get feature columns
        all_features = self.get_feature_columns()

        # Filter to only include columns that exist in the dataset
        features = [col for col in all_features if col in supervised.columns]

        # Select features and target
        X = supervised[features]
        y = supervised['target']

        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=features)

        # Optimize hyperparameters if enabled
        if self.optimize:
            try:
                best_params = self.optimize_hyperparameters(X_scaled, y)
                self.params.update(best_params)
            except Exception as e:
                print(f"Hyperparameter optimization failed: {e}. Using default parameters.")

        # Create and fit the model with the best parameters
        self.model = XGBRegressor(**self.params)
        self.model.fit(X_scaled, y)

        # Calculate feature importance
        self.feature_importance = dict(zip(features, self.model.feature_importances_))

        # Print top 10 important features
        sorted
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        print("Top 10 important features:")
        for feature, importance in sorted_features[:10]:
            print(f"{feature}: {importance}")

    def predict(self, data, steps):
        # Prepare the supervised dataset
        supervised = self.create_supervised_data(data)

        # Get feature columns
        all_features = self.get_feature_columns()

        # Filter to only include columns that exist in the dataset
        features = [col for col in all_features if col in supervised.columns]

        # Select features
        X = supervised[features]

        # Scale features
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=features)

        # Make predictions
        predictions = self.model.predict(X_scaled)

        # Extend predictions for the desired number of steps
        extended_predictions = list(predictions)
        last_known_windspeed = supervised['windspeed'].iloc[-1]

        for _ in range(steps - len(predictions)):
            # Create a dummy dataframe for prediction
            dummy_data = {f'windspeed_lag_{i + 1}': [
                extended_predictions[-i - 1] if i < len(extended_predictions) else last_known_windspeed] for i in
                          range(self.lag)}
            dummy_df = pd.DataFrame(dummy_data)

            # Add other necessary features with realistic values
            for col in features:
                if col not in dummy_df.columns and col != 'target':
                    if 'sin' in col or 'cos' in col:
                        dummy_df[col] = [0.5]  # Placeholder for cyclic features
                    elif 'pressure' in col:
                        dummy_df[col] = [1000]  # Placeholder
                    elif 'temp' in col:
                        dummy_df[col] = [20]  # Placeholder
                    elif 'roll' in col:
                        dummy_df[col] = [supervised[col].iloc[-1] if col in supervised.columns else 0]
                    else:
                        dummy_df[col] = [0]

            # Scale the dummy dataframe
            dummy_scaled = self.scaler.transform(dummy_df[features])
            dummy_scaled = pd.DataFrame(dummy_scaled, columns=features)

            # Predict next step
            next_prediction = self.model.predict(dummy_scaled)[0]
            extended_predictions.append(next_prediction)

        return np.array(extended_predictions[:steps])

    # Random Forest Model
    class RandomForestModel:
        def __init__(self, lag=48):
            self.model = None
            self.lag = lag
            self.scaler = StandardScaler()
            self.params = {
                'n_estimators': 200,
                'max_depth': 8,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            }

        def get_feature_columns(self):
            lag_features = [f'windspeed_lag_{i}' for i in range(1, self.lag + 1)]
            time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']
            weather_features = ['pressure', 'air_temp', 'ground_temp', 'pressure_gradient', 'temp_gradient']
            wind_features = ['wind_direction_sin', 'wind_direction_cos']
            rolling_features = [
                'windspeed_roll_mean_24', 'windspeed_roll_std_24', 'windspeed_roll_max_24',
                'pressure_roll_mean_24', 'temp_roll_mean_24', 'windspeed_roll_diff', 'pressure_roll_diff'
            ]

            # Combine all features
            all_features = lag_features + time_features + weather_features + wind_features + rolling_features
            return all_features

        def create_supervised_data(self, data):
            df = data.copy()

            # Create lagged features if not already in the dataframe
            for i in range(1, self.lag + 1):
                if f'windspeed_lag_{i}' not in df.columns:
                    df[f'windspeed_lag_{i}'] = df['windspeed'].shift(i)

            # Target is next hour's wind speed
            df['target'] = df['windspeed'].shift(-1)

            # Drop rows with NaN values
            return df.dropna()

        def fit(self, data):
            # Prepare the supervised dataset
            supervised = self.create_supervised_data(data)

            # Get feature columns
            all_features = self.get_feature_columns()

            # Filter to only include columns that exist in the dataset
            features = [col for col in all_features if col in supervised.columns]

            # Select features and target
            X = supervised[features]
            y = supervised['target']

            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=features)

            # Create and fit the model
            self.model = RandomForestRegressor(**self.params)
            self.model.fit(X_scaled, y)

        def predict(self, data, steps):
            # Prepare the supervised dataset
            supervised = self.create_supervised_data(data)

            # Get feature columns
            all_features = self.get_feature_columns()

            # Filter to only include columns that exist in the dataset
            features = [col for col in all_features if col in supervised.columns]

            # Select features
            X = supervised[features]

            # Scale features
            X_scaled = self.scaler.transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=features)

            # Make predictions
            predictions = self.model.predict(X_scaled)

            # Extend predictions for the desired number of steps
            extended_predictions = list(predictions)
            last_known_windspeed = supervised['windspeed'].iloc[-1]

            for _ in range(steps - len(predictions)):
                # Create a dummy dataframe for prediction
                dummy_data = {f'windspeed_lag_{i + 1}': [
                    extended_predictions[-i - 1] if i < len(extended_predictions) else last_known_windspeed] for i in
                              range(self.lag)}
                dummy_df = pd.DataFrame(dummy_data)

                # Add other necessary features with realistic values
                for col in features:
                    if col not in dummy_df.columns and col != 'target':
                        if 'sin' in col or 'cos' in col:
                            dummy_df[col] = [0.5]  # Placeholder for cyclic features
                        elif 'pressure' in col:
                            dummy_df[col] = [1000]  # Placeholder
                        elif 'temp' in col:
                            dummy_df[col] = [20]  # Placeholder
                        elif 'roll' in col:
                            dummy_df[col] = [supervised[col].iloc[-1] if col in supervised.columns else 0]
                        else:
                            dummy_df[col] = [0]

                # Scale the dummy dataframe
                dummy_scaled = self.scaler.transform(dummy_df[features])
                dummy_scaled = pd.DataFrame(dummy_scaled, columns=features)

                # Predict next step
                next_prediction = self.model.predict(dummy_scaled)[0]
                extended_predictions.append(next_prediction)

            return np.array(extended_predictions[:steps])

    # Ensemble Model
    class EnsembleModel:
        def __init__(self):
            self.models = []

        def add_model(self, model):
            self.models.append(model)

        def fit(self, data):
            for model in self.models:
                model.fit(data)

        def predict(self, data, steps):
            all_predictions = []
            for model in self.models:
                all_predictions.append(model.predict(data, steps))

            # Average the predictions
            ensemble_predictions = np.mean(all_predictions, axis=0)
            return ensemble_predictions

    # Enhanced Damage Model
    class EnhancedDamageModel:
    def __init__(self, use_ensemble=False):
        self.model = None
        self.use_ensemble = use_ensemble
        self.scaler = StandardScaler()

    def create_features(self, windspeed):
        df = pd.DataFrame({'windspeed': windspeed})

        # Add polynomial features
        df['windspeed_squared'] = df['windspeed'] ** 2
        df['windspeed_cubed'] = df['windspeed'] ** 3

        # Add exponential features
        df['windspeed_exp'] = np.exp(df['windspeed'])

        # Add log features, handling zero values
        df['windspeed_log'] = np.log(df['windspeed'].replace(0, 1e-10))

        # Add interaction features
        df['windspeed_squared_log'] = df['windspeed_squared'] * df['windspeed_log']

        return df

    def fit(self, X, y):
        X_enhanced = self.create_features(X['windspeed'])
        X_scaled = self.scaler.fit_transform(X_enhanced)

        if self.use_ensemble:
            # Ensemble of XGBoost and Random Forest
            xgb = XGBRegressor(n_estimators=100, max_depth=6, random_state=42)
            rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)

            xgb.fit(X_scaled, y)
            rf.fit(X_scaled, y)

            # Create a VotingRegressor
            self.model = VotingRegressor(estimators=[('xgb', xgb), ('rf', rf)])
            self.model.fit(X_scaled, y)
        else:
            # Single XGBoost model
            self.model = XGBRegressor(n_estimators=100, max_depth=6, random_state=42)
            self.model.fit(X_scaled, y)

    def predict(self, windspeed):
        X_enhanced = self.create_features(windspeed['windspeed'])
        X_scaled = self.scaler.transform(X_enhanced)
        return self.model.predict(X_scaled)

# Insurance Pricing Model based on expected damage
class InsurancePricingModel:
    def __init__(self):
        self.damage_model = None

    def fit(self, windspeed_data, damage_data):
        # Create a damage model
        self.damage_model = EnhancedDamageModel(use_ensemble=True)

        # Prepare data
        X = pd.DataFrame({'windspeed': windspeed_data})
        y = damage_data

        # Fit the damage model
        self.damage_model.fit(X, y)

    def calculate_profit(self, price, predicted_damage):
        """
        Calculate profit based on pricing, expected damage, and demand curve

        Profit = Revenue - Costs
        Revenue = Number of customers * Price
        Costs = Damage * Number of customers
        Number of customers = f(price) - demand curve
        """
        # Demand curve: Number of customers decreases as price increases
        # Using a simple demand function: customers = max(1000 - 20 * price, 0)
        customers = max(1000 - 20 * price, 0)

        # Revenue calculation
        revenue = customers * price

        # Cost calculation (damage payout)
        costs = customers * predicted_damage

        # Profit calculation
        profit = revenue - costs

        return profit, customers, revenue, costs

    def find_optimal_price(self, predicted_damage, price_range=(0, 100), step=0.5):
        """Find the price that maximizes profit given predicted damage"""
        prices = np.arange(price_range[0], price_range[1], step)
        best_profit = float('-inf')
        optimal_price = 0

        for price in prices:
            profit, _, _, _ = self.calculate_profit(price, predicted_damage)
            if profit > best_profit:
                best_profit = profit
                optimal_price = price

        return optimal_price, best_profit

# Main prediction pipeline
class WindSpeedPredictionPipeline:
    def __init__(self, use_ensemble=True, output_path=None):
        self.ensemble = None
        self.use_ensemble = use_ensemble
        self.output_path = output_path or "outputs"
        self.pricing_model = InsurancePricingModel()

        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def train(self, training_data):
        """Train models on historical data"""
        print("Training weather prediction models...")

        # Filter data for GANopolis
        city_data = training_data[training_data['city'] == 'GANopolis'].copy()

        # Preprocess data
        processed_data = preprocess_data(city_data)

        # Create ensemble model
        self.ensemble = EnsembleModel()

        # Add component models
        sarima_model = EnhancedSARIMAModel(auto_optimize=True)
        xgb_model = EnhancedXGBoostModel(lag=48, optimize=True)
        rf_model = RandomForestModel(lag=48)

        # Add models to ensemble
        self.ensemble.add_model(sarima_model)
        self.ensemble.add_model(xgb_model)
        self.ensemble.add_model(rf_model)

        # Fit ensemble
        self.ensemble.fit(processed_data)

        # Train pricing model
        print("Training insurance pricing model...")
        self.pricing_model.fit(
            city_data['windspeed'].values,
            city_data['damage'].values
        )

        # Save models
        self.save_models()

        return self

    def predict_event(self, event_data, event_id, prediction_hours=120):
        """Make predictions for a specific event"""
        print(f"Predicting for event {event_id}...")

        # Filter data for GANopolis
        city_event_data = event_data[event_data['city'] == 'GANopolis'].copy()

        # Preprocess data
        processed_data = preprocess_data(city_event_data)

        # Predict wind speed for next 120 hours (5 days)
        wind_predictions = self.ensemble.predict(processed_data, prediction_hours)

        # Create a damage model for this event
        damage_model = EnhancedDamageModel()

        # Train damage model on this event's data
        damage_model.fit(
            pd.DataFrame({'windspeed': city_event_data['windspeed']}),
            city_event_data['damage']
        )

        # Predict damage for forecasted wind speeds
        predicted_damage = damage_model.predict(pd.DataFrame({'windspeed': wind_predictions}))

        # Calculate total expected damage over the 5-day period
        total_expected_damage = np.sum(predicted_damage)

        # Find optimal price
        optimal_price, expected_profit = self.pricing_model.find_optimal_price(total_expected_damage)

        # Generate plots
        self.generate_prediction_plots(
            city_event_data,
            wind_predictions,
            predicted_damage,
            event_id
        )

        return {
            'event_id': event_id,
            'wind_predictions': wind_predictions,
            'total_expected_damage': total_expected_damage,
            'optimal_price': optimal_price,
            'expected_profit': expected_profit
        }

    ```python

    def generate_prediction_plots(self, event_data, wind_predictions, damage_predictions, event_id):
        """Generate plots for event predictions"""

    plt.figure(figsize=(15, 10))

    # Plot historical and predicted wind speed
    plt.subplot(2, 1, 1)
    plt.plot(event_data['hour'], event_data['windspeed'], label='Historical')
    predicted_hours = np.arange(
        event_data['hour'].iloc[-1] + 1,
        event_data['hour'].iloc[-1] + 1 + len(wind_predictions)
    )
    plt.plot(predicted_hours, wind_predictions, label='Predicted', linestyle='--')
    plt.title(f'Wind Speed Prediction - Event {event_id}')
    plt.xlabel('Hour')
    plt.ylabel('Wind Speed')
    plt.legend()
    plt.grid(True)

    # Plot predicted damage
    plt.subplot(2, 1, 2)
    plt.plot(predicted_hours, damage_predictions, color='red')
    plt.title(f'Predicted Damage - Event {event_id}')
    plt.xlabel('Hour')
    plt.ylabel('Damage')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{self.output_path}/event_{event_id}_predictions.png")
    plt.close()

    # Generate price-profit curve
    self.generate_price_profit_curve(damage_predictions.sum(), event_id)


def generate_price_profit_curve(self, total_damage, event_id):
    """Generate a price-profit curve for the given total damage"""
    prices = np.linspace(0, 100, 100)
    profits = []
    revenues = []
    costs = []
    customers = []

    for price in prices:
        profit, customer, revenue, cost = self.pricing_model.calculate_profit(price, total_damage)
        profits.append(profit)
        revenues.append(revenue)
        costs.append(cost)
        customers.append(customer)

    plt.figure(figsize=(12, 8))

    # Profit curve
    plt.subplot(2, 1, 1)
    plt.plot(prices, profits, label='Profit', color='green')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Find and mark optimal price
    optimal_idx = np.argmax(profits)
    optimal_price = prices[optimal_idx]
    optimal_profit = profits[optimal_idx]
    plt.scatter(optimal_price, optimal_profit, color='red', zorder=5, s=100)
    plt.annotate(f'Optimal Price: ${optimal_price:.2f}\nProfit: ${optimal_profit:.2f}',
                 (optimal_price, optimal_profit),
                 xytext=(10, 30),
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle='->'))

    plt.title(f'Profit vs Price - Event {event_id}')
    plt.xlabel('Price ($)')
    plt.ylabel('Profit ($)')
    plt.grid(True)

    # Revenue and Cost
    plt.subplot(2, 1, 2)
    plt.plot(prices, revenues, label='Revenue', color='blue')
    plt.plot(prices, costs, label='Cost', color='red')
    plt.plot(prices, customers, label='Customers', color='purple')
    plt.title('Revenue, Cost, and Customer Count')
    plt.xlabel('Price ($)')
    plt.ylabel('Amount ($) / Customer Count')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{self.output_path}/event_{event_id}_price_profit.png")
    plt.close()


def save_models(self):
    """Save trained models to disk"""
    try:
        os.makedirs(f"{self.output_path}/models", exist_ok=True)
        joblib.dump(self.ensemble, f"{self.output_path}/models/ensemble_model.pkl")
        joblib.dump(self.pricing_model, f"{self.output_path}/models/pricing_model.pkl")
        print(f"Models saved to {self.output_path}/models/")
    except Exception as e:
        print(f"Error saving models: {e}")


def load_models(self):
    """Load trained models from disk"""
    try:
        self.ensemble = joblib.load(f"{self.output_path}/models/ensemble_model.pkl")
        self.pricing_model = joblib.load(f"{self.output_path}/models/pricing_model.pkl")
        print("Models loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False


# Function to load all event data
def load_all_events(data_dir):
    events = {}
    for i in range(1, 11):
        event_file = f"{data_dir}/event_{i}.csv"
        if os.path.exists(event_file):
            events[i] = read_input_data(event_file)
    return events


# Function to prepare submission file
def prepare_submission(results, output_file="submission/submission.csv"):
    """Prepare submission file from prediction results"""
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Create DataFrame for submission
    submission = pd.DataFrame()
    submission['event_id'] = [result['event_id'] for result in results]
    submission['price'] = [result['optimal_price'] for result in results]

    # Add wind speed predictions for each hour
    for hour in range(120):  # 120 hours = 5 days
        submission[f'wind_{hour + 1}'] = [result['wind_predictions'][hour] for result in results]

    # Save to CSV
    submission.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")
    return submission


# Main execution function
def main():
    # Set paths
    data_dir = "data"
    output_dir = "outputs"
    submission_file = "submission/submission.csv"

    # Load training data
    print("Loading training data...")
    training_data = read_input_data(f"{data_dir}/training_data.csv")

    # Initialize prediction pipeline
    pipeline = WindSpeedPredictionPipeline(use_ensemble=True, output_path=output_dir)

    # Train models
    pipeline.train(training_data)

    # Load event data
    print("Loading event data...")
    events = load_all_events(data_dir)

    # Process each event
    results = []
    for event_id, event_data in events.items():
        # Make predictions for each event
        result = pipeline.predict_event(event_data, event_id)
        results.append(result)

    # Prepare submission file
    prepare_submission(results, submission_file)

    # Save team info
    team_info = {
        "team_name": "WindPredictors",
        "team_number": 1,
        "team_members": ["Member1", "Member2", "Member3"]
    }

    with open("submission/team_info.json", "w") as f:
        json.dump(team_info, f, indent=4)

    print("Done! Submission ready.")


if __name__ == "__main__":
    main()
