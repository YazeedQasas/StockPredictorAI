from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

class BasePredictor(ABC):
    def __init__ (self, name):
        self.name = name
        self.model = None
        self.is_trained = False
        self.logger = logging.getLogger(__name__)
        self.performance_metrics = {}

    @abstractmethod
    def train(self, X_train, y_train):
        """Train the model"""
        pass

    @abstractmethod
    def predict(self, X):
        """Make predictions"""
        pass

    def evaluate(self, X_test, y_test):
        """This evaluates the model's performance"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        predictions = self.predict(X_test)

        #Calculating the metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        # Calculate directional accuracy
        actual_direction = np.sign(np.diff(y_test))
        pred_direction = np.sign(np.diff(predictions))
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100

        self.performance_metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'directional_accuracy': directional_accuracy
        }

        self.logger.info(f"{self.name} Performance:")
        self.logger.info(f"  RMSE: {rmse:.4f}")
        self.logger.info(f"  MAE: {mae:.4f}")
        self.logger.info(f"  R²: {r2:.4f}")
        self.logger.info(f"  Directional Accuracy: {directional_accuracy:.2f}%")

        return self.performance_metrics
    
def get_feature_importance(self):
    """Get feature importance if available"""
    return None