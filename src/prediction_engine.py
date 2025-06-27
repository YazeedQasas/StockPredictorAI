from src.models.linear_model import LinearStockPredictor
from src.feature_engineer import FeatureEngineer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

"""
Main Prediction Engine
Orchestrates multiple models and provides ensemble predictions
"""


class PredictionEngine:
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.models = {}
        self.ensemble_weights = {}
        self.logger = logging.getLogger(__name__)
        self.is_trained = False
        self._initialize_models()

    def _initialize_models(self):
        """Initializes all the prediction models (Linear, Ridge and Lasso)"""
        self.models = {
            'linear': LinearStockPredictor('linear'),
            'ridge': LinearStockPredictor('ridge', alpha=1.0),
            'lasso': LinearStockPredictor('lasso', alpha=0.1),
            'ridge_poly': LinearStockPredictor('ridge', alpha=1.0, polynomial_degree=2)
        }

        n_models = len(self.models)
        self.ensemble_weights = {name: 1.0 /
                                 n_models for name in self.models.keys()}

    def prepare_data(self, historical_data, prediction_days=1):
        """Preparing data for training"""
        self.logger.info("Preparing data for model training...")

        # Here, we create a feature and target
        features, target = self.feature_engineer.prepare_ml_dataset(
            historical_data, prediction_days=prediction_days)

        # Scaling features
        features_scaled = self.feature_engineer.scale_features(
            features, fit=True)

        return features_scaled, target

    def train_models(self, historical_data, test_size=0.2):
        """Train all models with historical data"""
        self.logger.info("Starting model training process...")

        # Prepare data
        features, target = self.prepare_data(historical_data)

        # Splitting the data (Time series split - no shuffling)
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, shuffle=False
        )

        # This section handles training each of the models
        model_performances = {}

        for name, model in self.models.items():
            self.logger.info(f"Training {name} model...")

            try:
                print(f"🔧 Training {name}...")
                model.train(X_train, y_train)
                print(f"✅ {name} training completed. is_trained = {model.is_trained}")
                performance = model.evaluate(X_test, y_test)
                model_performances[name] = performance
            except Exception as e:
                self.logger.error(f"Error Training {name}: {e}")
                # Indicates Poor Performance
                model_performances[name] = {
                    'r2': -1.0, 'mse': float('inf'), 'rmse': float('inf')}

            # Update ensemble weights based on performance
        self._update_ensemble_weights(model_performances)

        self.is_trained = True
        self.logger.info("All models have been trained successfully!")

        # Debug: Check final training status
        print("🔍 Final training status:")
        for name, model in self.models.items():
            print(f"   {name}: is_trained = {model.is_trained}")
            
        return model_performances

    def _update_ensemble_weights(self, performances):
        """Update ensmble weights based on the model performance"""

        # Use R^2 to weight models (Higher R^2 gets higher weight)
        r2_scores = {name: max(perf.get('r2', 0), 0)
                     for name, perf in performances.items()}

        # Normalize weights
        total_score = sum(r2_scores.values())
        if total_score > 0:
            self.ensemble_weights = {
                name: score/total_score for name, score in r2_scores.items()}
        else:
            # Equal weights if all models perform poorly
            n_models = len(self.models)
            self.ensemble_weights = {name: 1.0 /
                                     n_models for name in self.models.keys()}

        self.logger.info("Updated ensemble weights:")
        for name, weight in self.ensemble_weights.items():
            self.logger.info(f"  {name}: {weight:.3f}")

    def predict_next_day(self, current_data):
        """Predict next day's closing price with enhanced validation"""
        if not self.is_trained:
            raise ValueError(
                "Models must be trained before making predictions")

        try:
            # Enhanced data validation
            if current_data is None or current_data.empty:
                raise ValueError("Current data is None or empty")

            self.logger.info(f"Input data shape: {current_data.shape}")
            self.logger.info(
                f"Input date range: {current_data.index[0]} to {current_data.index[-1]}")

            # Ensure we have enough data for feature engineering
            min_required = 60
            if len(current_data) < min_required:
                raise ValueError(
                    f"Insufficient data for prediction. Need at least {min_required} days, got {len(current_data)}")

            # Prepare features using the same pipeline as training
            self.logger.info("Preparing features for prediction...")
            features, _ = self.feature_engineer.prepare_ml_dataset(
                current_data)

            if features.empty:
                raise ValueError("Feature engineering produced empty dataset")

            self.logger.info(
                f"Features shape after preparation: {features.shape}")

            # Ensure feature consistency with training
            if hasattr(self.feature_engineer, 'feature_columns'):
                expected_features = self.feature_engineer.feature_columns
                if list(features.columns) != expected_features:
                    self.logger.warning(
                        "Feature columns don't match training. Attempting to align...")
                    # Try to align features
                    missing_cols = set(expected_features) - \
                        set(features.columns)
                    extra_cols = set(features.columns) - set(expected_features)

                    if missing_cols:
                        self.logger.error(f"Missing features: {missing_cols}")
                    if extra_cols:
                        self.logger.warning(
                            f"Extra features (will be dropped): {extra_cols}")

                    # Keep only common features in the same order
                    common_features = [
                        col for col in expected_features if col in features.columns]
                    if len(common_features) < len(expected_features) * 0.8:  # Less than 80% match
                        raise ValueError(
                            f"Too many missing features. Expected {len(expected_features)}, got {len(common_features)}")

                    features = features[common_features]

            # Scale features using the fitted scaler
            self.logger.info("Scaling features...")
            features_scaled = self.feature_engineer.scale_features(
                features, fit=False)

            if features_scaled.empty:
                raise ValueError("Feature scaling produced empty dataset")

            self.logger.info(f"Scaled features shape: {features_scaled.shape}")

            # Get latest features for prediction
            latest_features = features_scaled.tail(1)

            if latest_features.empty:
                raise ValueError("No latest features available for prediction")

            self.logger.info(
                f"Latest features shape for prediction: {latest_features.shape}")

            # Get predictions from all models
            predictions = {}
            print("🤖 Starting model predictions...")

            for name, model in self.models.items():
                print(f"🔍 Checking model: {name}")
                print(f"   - Model trained: {model.is_trained}")

                if model.is_trained:
                    try:
                        print(f"   - Calling predict for {name}...")
                        pred = model.predict(latest_features)
                        print(f"   - Raw prediction from {name}: {pred}")
                        print(f"   - Prediction type: {type(pred)}")

                        # Extract the actual number
                        if isinstance(pred, (list, np.ndarray)):
                            pred_value = pred[0]
                        else:
                            pred_value = pred

                        predictions[name] = pred_value
                        print(f"   ✅ {name} SUCCESS: ${pred_value:.2f}")

                    except Exception as e:
                        print(f"   ❌ {name} FAILED: {e}")
                        print(f"   ❌ Error type: {type(e)}")
                        predictions[name] = None
                else:
                    print(f"   ⚠️ {name} not trained")

            print(f"🎯 Final predictions: {predictions}")

            # Check if we have any valid predictions
            valid_predictions = {k: v for k,
                                 v in predictions.items() if v is not None}
            print(f"✅ Valid predictions: {valid_predictions}")

            if not valid_predictions:
                print("❌ NO VALID PREDICTIONS - All models failed!")
                raise ValueError("No models produced valid predictions")

            # Calculate ensemble prediction
            ensemble_pred = self._calculate_ensemble_prediction(predictions)
            confidence = self._calculate_prediction_confidence(predictions)

            return {
                'ensemble_prediction': ensemble_pred,
                'individual_predictions': predictions,
                'confidence': confidence,
                'timestamp': pd.Timestamp.now()
            }

        except Exception as e:
            self.logger.error(f"Error in predict_next_day: {e}")
            return None

    def _calculate_ensemble_prediction(self, predictions):
        """Calculate weighted ensemble prediction"""
        weighted_sum = 0
        total_weight = 0

        for name, pred in predictions.items():
            if pred is not None:
                pred_value = pred[0] if isinstance(
                    pred, (list, np.ndarray)) else pred

                weight = self.ensemble_weights.get(name, 0)
                weighted_sum += pred_value * weight
                total_weight += weight

        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            # Fallback to simple average
            valid_preds = []
            for p in predictions.values():
                if p is not None:
                    # FIX: Extract number from list if needed
                    pred_value = p[0] if isinstance(
                        p, (list, np.ndarray)) else p
                    valid_preds.append(pred_value)

            return np.mean(valid_preds) if valid_preds else None

    def _calculate_prediction_confidence(self, predictions):
        """Calculate confidence based on prediction agreement"""
        valid_preds = []

        for p in predictions.values():
            if p is not None:
                # FIX: Extract number from list if needed
                pred_value = p[0] if isinstance(p, (list, np.ndarray)) else p
                valid_preds.append(pred_value)

        if len(valid_preds) < 2:
            return 0.5  # Low confidence with few predictions

        # Calculate coefficient of variation (lower = higher confidence)
        std_pred = np.std(valid_preds)
        mean_pred = np.mean(valid_preds)

        if mean_pred != 0:
            cv = std_pred / abs(mean_pred)
            # Convert to confidence (0-1 scale)
            confidence = max(0, min(1, 1 - cv))
        else:
            confidence = 0.5

        return confidence

    def get_model_summary(self):
        """Get summary of all models and their performance"""
        summary = {
            'is_trained': self.is_trained,
            'num_models': len(self.models),
            'ensemble_weights': self.ensemble_weights,
            'model_details': {}
        }

        for name, model in self.models.items():
            summary['model_details'][name] = {
                'is_trained': model.is_trained,
                'performance': model.performance_metrics
            }
        return summary

    def save_trained_models(self):
        """Save all trained models"""
        try:
            for name, model in self.models.items():
                if model.is_trained:
                    model.save_model()

            # Save ensemble weights
            import joblib
            weights_path = os.path.join(
                config.MODELS_DIR, 'ensemble_weights.pkl')
            joblib.dump(self.ensemble_weights, weights_path)

            self.logger.info("All models saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")

    def load_trained_models(self):
        """Load pre-trained models"""
        try:
            for name, model in self.models.items():
                model_path = os.path.join(
                    config.MODELS_DIR, f"{model.name.lower()}_model.pkl")
                if os.path.exists(model_path):
                    model.load_model(model_path)

            # Load ensemble weights
            import joblib
            weights_path = os.path.join(
                config.MODELS_DIR, 'ensemble_weights.pkl')
            if os.path.exists(weights_path):
                self.ensemble_weights = joblib.load(weights_path)

            self.is_trained = True
            self.logger.info("All models loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise e


# Testing the prediction engine
if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from data_collector import RealTimeStockTracker

    logging.basicConfig(level=logging.INFO)

    print("Testing Prediction Engine")
    print("="*50)

    # Get data
    tracker = RealTimeStockTracker()
    data = tracker.fetch_historical_data()

    # Create and train prediction engine
    engine = PredictionEngine()
    performances = engine.train_models(data)

    # Show model performances
    print("\nModel Performance Summary:")
    for name, perf in performances.items():
        print(f"{name:15} - R²: {perf.get('r2', 0):.4f}, "
              f"RMSE: {perf.get('rmse', 0):.4f}, "
              f"Dir. Acc: {perf.get('directional_accuracy', 0):.1f}%")

    # Make a prediction
    prediction_result = engine.predict_next_day(data)

    print(f"\nNext Day Prediction:")
    print(f"Ensemble: ${prediction_result['ensemble_prediction']:.2f}")
    print(f"Confidence: {prediction_result['confidence']:.2f}")
    print(f"Individual predictions:")
    for name, pred in prediction_result['individual_predictions'].items():
        if pred is not None:
            print(f"  {name}: ${pred:.2f}")
