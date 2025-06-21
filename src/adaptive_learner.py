import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class AdaptiveLearningEngine:
    def __init__(self, prediction_engine, stock_tracker):
        self.prediction_engine = prediction_engine
        self.stock_tracker = stock_tracker
        self.logger = logging.getLogger(__name__)

        # Performance tracking
        self.prediction_log = []
        self.performance_threshold = 0.05  # 5% error threshold
        self.retrain_threshold = 0.10  # Retrain if error > 10%

    def log_prediction(self, predicted_price, actual_price, timestamp=None):
        """Log a prediction vs actual result"""
        if timestamp is None:
            timestamp = datetime.now()

        error = abs(predicted_price - actual_price) / actual_price

        prediction_record = {
            'timestamp': timestamp,
            'predicted_price': predicted_price,
            'actual_price': actual_price,
            'absolute_error': abs(predicted_price - actual_price),
            'percentage_error': error,
            'within_threshold': error <= self.performance_threshold
        }

        self.prediction_log.append(prediction_record)

        # Keep only last 100 predictions
        if len(self.prediction_log) > 100:
            self.prediction_log = self.prediction_log[-100:]

        self.logger.info(f"Logged prediction: ${predicted_price:.2f} vs ${actual_price:.2f} "
                         f"(Error: {error:.2%})")

        # Check if retraining is needed
        if self._should_retrain():
            self._trigger_retrain()

    def _should_retrain(self):
        """Determine if model retraining is needed"""
        if len(self.prediction_log) < 10:  # Need minimum predictions
            return False

        # Calculate recent performance
        recent_predictions = self.prediction_log[-10:]
        recent_errors = [p['percentage_error'] for p in recent_predictions]
        avg_error = np.mean(recent_errors)

        return avg_error > self.retrain_threshold

    def _trigger_retrain(self):
        """Trigger model retraining with fresh data"""
        self.logger.info("Performance degraded - triggering model retrain...")

        try:
            # Fetch fresh historical data
            fresh_data = self.stock_tracker.fetch_historical_data(period="1y")

            if fresh_data is not None:
                # Retrain models
                self.prediction_engine.train_models(fresh_data)
                self.logger.info("✅ Models retrained successfully")
            else:
                self.logger.error(
                    "❌ Failed to fetch fresh data for retraining")

        except Exception as e:
            self.logger.error(f"Error during retraining: {e}")

    def get_performance_summary(self):
        """Get performance summary statistics"""
        if not self.prediction_log:
            return None

        errors = [p['percentage_error'] for p in self.prediction_log]
        within_threshold = [p['within_threshold'] for p in self.prediction_log]

        return {
            'total_predictions': len(self.prediction_log),
            'average_error': np.mean(errors),
            'median_error': np.median(errors),
            'accuracy_rate': np.mean(within_threshold),
            'last_prediction_time': self.prediction_log[-1]['timestamp'],
            'needs_retrain': self._should_retrain()
        }
