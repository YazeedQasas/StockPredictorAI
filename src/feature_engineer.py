""" This file creates ML-ready features from raw stock data"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class FeatureEngineer:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.feature_scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)
        self.feature_columns = []

    def create_technical_features(self, data):
        """Creates a comprehensive technical analysis features"""
        features = pd.DataFrame(index=data.index)

        # Price-based features
        features['close'] = data['Close']
        features['high'] = data['High']
        features['low'] = data['Low']
        features['open'] = data['Open']
        features['volume'] = data['Volume']

        # Price ratios and relationships
        features['high_low_ratio'] = data['High'] / data['Low']
        features['close_open_ratio'] = data['Close'] / data['Open']
        features['volume_price_ratio'] = data['Volume'] / data['Close']

        # Moving averages (multiple timeframes)
        for window in [5, 10, 20, 50]:
            features[f'ma_{window}'] = data['Close'].rolling(
                window=window).mean()
            features[f'ma_{window}_ratio'] = data['Close'] / \
                features[f'ma_{window}']
            features[f'volume_ma_{window}'] = data['Volume'].rolling(
                window=window).mean()

        # Exponential moving averages
        features['ema_12'] = data['Close'].ewm(span=12).mean()
        features['ema_26'] = data['Close'].ewm(span=26).mean()
        features['macd'] = features['ema_12'] - features['ema_26']
        features['macd_signal'] = features['macd'].ewm(span=9).mean()

        # Volatility features
        features['volatility_10'] = data['Close'].rolling(window=10).std()
        features['volatility_20'] = data['Close'].rolling(window=20).std()
        features['volatility_ratio'] = features['volatility_10'] / \
            features['volatility_20']

        # Price momentum
        for period in [1, 3, 5, 10]:
            features[f'price_change_{period}d'] = data['Close'].pct_change(
                periods=period)
            features[f'volume_change_{period}d'] = data['Volume'].pct_change(
                periods=period)

        # RSI (Relative Strength Index)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        bb_window = 20
        bb_std = 2
        features['bb_middle'] = data['Close'].rolling(window=bb_window).mean()
        bb_std_dev = data['Close'].rolling(window=bb_window).std()
        features['bb_upper'] = features['bb_middle'] + (bb_std_dev * bb_std)
        features['bb_lower'] = features['bb_middle'] - (bb_std_dev * bb_std)
        features['bb_position'] = (
            data['Close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])

        # Support and Resistance levels
        features['support_level'] = data['Low'].rolling(window=20).min()
        features['resistance_level'] = data['High'].rolling(window=20).max()
        features['support_distance'] = (
            data['Close'] - features['support_level']) / data['Close']
        features['resistance_distance'] = (
            features['resistance_level'] - data['Close']) / data['Close']

        return features.dropna()

    def create_time_features(self, data):
        """Creates time-based features"""
        features = pd.DataFrame(index=data.index)

        # Days of week, for context: 0=Monday..6=Sunday
        features['day_of_week'] = data.index.dayofweek
        features['is_monday'] = (features['day_of_week'] == 0).astype(int)
        features['is_friday'] = (features['day_of_week'] == 4).astype(int)

        # Month Features
        features['month'] = data.index.month
        features['quarter'] = data.index.quarter

        # Week of year
        features['week_of_year'] = data.index.isocalendar().week

        return features

    def create_lag_features(self, data, target_col='Close', lags=[1, 2, 3, 5, 10]):
        """Creates lagged features for time series prediction"""
        features = pd.DataFrame(index=data.index)

        for lag in lags:
            features[f'{target_col.lower()}_lag_{lag}'] = data[target_col].shift(
                lag)
            features[f'volume_lag_{lag}'] = data['Volume'].shift(lag)

        return features.dropna()

    def create_target_variable(self, data, target_col='Close', prediction_days=1):
        """Creates a target variable for prediction"""
        target = data[target_col].shift(-prediction_days)
        return target.dropna()

    def prepare_ml_dataset(self, data, prediction_days=1):
        """
        Prepare ML dataset - think of this as preparing ingredients for cooking
        """
        # Step 1: Check if we have ingredients (data)
        if data is None or data.empty:
            raise ValueError(
                "No data provided - like trying to cook without ingredients!")

        print(f"📊 Starting with {len(data)} days of stock data")

        # Step 2: Make sure we have enough data (need at least 60 days)
        if len(data) < 60:
            raise ValueError(
                f"Need at least 60 days of data, but only got {len(data)} days")

        try:
            # Step 3: Create different types of features (like different cooking techniques)

            # 3a: Technical indicators (moving averages, RSI, etc.)
            print("🔧 Creating technical indicators...")
            technical_features = self.create_technical_features(data)
            print(
                f"   ✅ Created {technical_features.shape[1]} technical features")

            # 3b: Time-based features (day of week, month, etc.)
            print("📅 Creating time features...")
            time_features = self.create_time_features(data)
            print(f"   ✅ Created {time_features.shape[1]} time features")

            # 3c: Historical patterns (what happened 1 day ago, 5 days ago, etc.)
            print("🔄 Creating lag features...")
            lag_features = self.create_lag_features(data)
            print(f"   ✅ Created {lag_features.shape[1]} lag features")

            # Step 4: Combine all features (like mixing ingredients)
            print("🔀 Combining all features...")

            # Find dates that exist in all feature sets
            common_dates = technical_features.index
            if not time_features.empty:
                common_dates = common_dates.intersection(time_features.index)
            if not lag_features.empty:
                common_dates = common_dates.intersection(lag_features.index)

            print(
                f"   📅 Found {len(common_dates)} common dates across all features")

            # Combine features for these common dates
            all_features = pd.concat([
                technical_features.loc[common_dates],
                time_features.loc[common_dates],
                lag_features.loc[common_dates]
            ], axis=1)

            print(f"   ✅ Combined into {all_features.shape[1]} total features")

            # Step 5: Create target (what we want to predict)
            print("🎯 Creating prediction target...")
            target = data['Close'].shift(-prediction_days)  # Tomorrow's price
            target = target.loc[common_dates]  # Only for common dates

            # Step 6: Clean the data (remove rows with missing values)
            print("🧹 Cleaning data...")

            # Before cleaning
            print(f"   Before cleaning: {len(all_features)} rows")

            # Remove rows where features OR target have missing values
            feature_missing = all_features.isna().any(axis=1)
            target_missing = target.isna()
            any_missing = feature_missing | target_missing

            # Keep only good rows
            clean_features = all_features[~any_missing]
            clean_target = target[~any_missing]

            # After cleaning
            print(f"   After cleaning: {len(clean_features)} rows")
            print(f"   Removed {any_missing.sum()} rows with missing data")

            # Step 7: Final validation (make sure we still have data)
            if clean_features.empty or clean_target.empty:
                raise ValueError(
                    "❌ All data was removed during cleaning - something went wrong!")

            if len(clean_features) < 10:
                raise ValueError(
                    f"❌ Only {len(clean_features)} rows left - need at least 10 for training")

            # Step 8: Success! Return the prepared data
            print(
                f"✅ Successfully prepared {clean_features.shape[1]} features for {len(clean_features)} samples")

            # Store feature names for later use
            self.feature_columns = clean_features.columns.tolist()

            return clean_features, clean_target

        except Exception as e:
            print(f"❌ Error during data preparation: {e}")
            raise e

    def scale_features(self, features, fit=True):
        """Scale features using StandardScaler"""
        if fit:
            # Fit and transform
            features_scaled = pd.DataFrame(
                self.scaler.fit_transform(features),
                index=features.index,
                columns=features.columns
            )
            # Store the fitted scaler for later use
            self._fitted_scaler = self.scaler
        else:
            # Use the previously fitted scaler
            if hasattr(self, '_fitted_scaler') and self._fitted_scaler is not None:
                features_scaled = pd.DataFrame(
                    self._fitted_scaler.transform(features),
                    index=features.index,
                    columns=features.columns
                )
            else:
                # Fallback: fit and transform if no fitted scaler exists
                features_scaled = pd.DataFrame(
                    self.scaler.fit_transform(features),
                    index=features.index,
                    columns=features.columns
                )

        return features_scaled

    def prepare_lstm_sequences(self, features, target, sequence_length=30):
        """Prepare sequences for LSTM model"""
        X, y = [], []

        for i in range(sequence_length, len(features)):
            X.append(features.iloc[i-sequence_length:i].values)
            y.append(target.iloc[i])

        return np.array(X), np.array(y)


# Testing the previous feature engineer class, helps with debugging whenever needed
if __name__ == "__main__":
    import sys
    sys.path.append('.')
    from src.data_collector import RealTimeStockTracker

    logging.basicConfig(level=logging.INFO)

    print("Testing Feature Engineering System")
    print("="*50)

    # Get data
    tracker = RealTimeStockTracker()
    data = tracker.fetch_historical_data()

    # Create feature engineer
    engineer = FeatureEngineer()

    # Test feature creation
    features, target = engineer.prepare_ml_dataset(data)

    print(
        f"Successfully Created {features.shape[1]} features for {features.shape[0]} samples")
    print(f"Created Target variable has {len(target)} values")
    print(
        f"Created Date range: {features.index[0].date()} to {features.index[-1].date()}")

    # Show feature importance by correlation
    feature_importance = features.corrwith(
        target).abs().sort_values(ascending=False)
    print(f"\nTop 10 Most Correlated Features:")
    print(feature_importance.head(10))

    # Test LSTM sequence preparation
    X_lstm, y_lstm = engineer.prepare_lstm_sequences(features, target)
    print(f"\n✓ LSTM sequences: {X_lstm.shape} -> {y_lstm.shape}")
