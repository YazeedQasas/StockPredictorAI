from src.models.base_predictor import BasePredictor

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import config
"""
Implementing Linear Regression Model for Stock Prediction
"""


class LinearStockPredictor(BasePredictor):
    def __init__(self, model_type='ridge', alpha=1.0, polynomial_degree=1):
        super().__init__(f"Linear_{model_type}")
        self.model_type = model_type
        self.alpha = alpha
        self.polynomial_degree = polynomial_degree
        self.poly_features = None

        # Initialize the appropriate model
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'ridge':
            self.model = Ridge(alpha=alpha)
        elif model_type == 'lasso':
            self.model = Lasso(alpha=alpha)
        else:
            raise ValueError(
                "model_type must be 'linear', 'ridge', or 'lasso'")

        # Add polynomial features if requested
        if polynomial_degree > 1:
            self.poly_features = PolynomialFeatures(degree=polynomial_degree)

    def _prepare_features(self, X):
        """Prepare features with polynomial transformation if needed"""
        if self.poly_features is not None:
            return self.poly_features.transform(X)
        return X

    def train(self, X_train, y_train):
        """Training the linear model"""
        self.logger.info(f"Training {self.name} model...")

        # Preparing polynomial features if needed
        if self.poly_features is not None:
            X_train_poly = self.poly_features.fit_transform(X_train)
            print(
                f"🔍 Polynomial features: {X_train.shape} -> {X_train_poly.shape}")
        else:
            X_train_poly = X_train

        self.model.fit(X_train_poly, y_train) 
        self.is_trained = True

        self.logger.info(f"{self.name} model has been trained successfully")

    def predict(self, X):
        """Makes predictions with better error handling"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        try:
            # Log what we're receiving
            print(f"🔍 {self.name} received data shape: {X.shape}")

            # Prepare features (handle polynomial if needed)
            X_prepared = self._prepare_features(X)
            print(f"🔍 {self.name} prepared data shape: {X_prepared.shape}")

            # Make prediction
            predictions = self.model.predict(X_prepared)
            print(f"✅ {self.name} prediction successful: {predictions[0]:.2f}")

            return predictions

        except Exception as e:
            print(f"❌ {self.name} prediction failed: {e}")
            # Instead of failing silently, raise the error so we can see what's wrong
            raise e

    def predict_with_confidence(self, X):
        """Make predictions with confidence intervals (This is simplified)"""
        if not self.is_trained:
            raise ValueError(f"The {self.name} model is not trained")

        predictions = self.predict(X)

        # Simple confidence estimation based on training residuals
        if hasattr(self, 'training_residuals'):
            std_residual = np.std(self.training_residuals)
            confidence_lower = predictions - 1.96 * std_residual
            confidence_upper = predictions + 1.96 * std_residual

            return predictions, confidence_lower, confidence_upper

        return predictions, predictions, predictions

    def get_feature_importance(self):
        """Get feature coefficients as importance"""
        if not self.is_trained:
            return None

        if hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_)
        return None

    def save_model(self, filepath=None):
        """Saves the trained model"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        if filepath is None:
            filepath = os.path.join(
                config.MODELS_DIR, f"{self.name.lower()}_model.pkl")

        model_data = {
            'model': self.model,
            'poly_features': self.poly_features,
            'model_type': self.model_type,
            'alpha': self.alpha,
            'polynomial_degree': self.polynomial_degree,
            'performance_metrics': self.performance_metrics
        }

        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load a trained model"""
        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.poly_features = model_data.get('poly_features')
        self.model_type = model_data['model_type']
        self.alpha = model_data['alpha']
        self.polynomial_degree = model_data['polynomial_degree']
        self.performance_metrics = model_data.get('performance_metrics', {})
        self.is_trained = True

        self.logger.info(f"Model loaded from {filepath}")


# Testing the linear model
if __name__ == "__main__":
    import sys
    import logging
    sys.path.append('../..')
    from src.data_collector import RealTimeStockTracker
    from src.feature_engineer import FeatureEngineer
    from sklearn.model_selection import train_test_split

    logging.basicConfig(level=logging.INFO)

    print("Testing Linear Stock Predictor")
    print("="*50)

    # Get data and features
    tracker = RealTimeStockTracker()
    data = tracker.fetch_historical_data()

    engineer = FeatureEngineer()
    features, target = engineer.prepare_ml_dataset(data)

    # Scale features
    features_scaled = engineer.scale_features(features)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, target, test_size=0.2, shuffle=False
    )

    # Test different linear models
    models = [
        LinearStockPredictor('linear'),
        LinearStockPredictor('ridge', alpha=1.0),
        LinearStockPredictor('lasso', alpha=0.1)
    ]

    for model in models:
        print(f"\nTesting {model.name}...")
        model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)

        # Test prediction
        sample_prediction = model.predict(X_test.tail(1))
        print(f"Sample prediction: ${sample_prediction[0]:.2f}")
