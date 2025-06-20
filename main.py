import logging
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_collector import RealTimeStockTracker
from src.prediction_engine import PredictionEngine
from src.adaptive_learner import AdaptiveLearningEngine
from src.dashboard import RealTimeDashboard
import config

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - $(levelnames)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.LOGS_DIR, 'app.log')),
            logging.StreamHandler()
        ]
    )

def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting Real-Time Stock Predictor AI")

    tracker = RealTimeStockTracker(config.STOCK_SYMBOL)
    predictor = PredictionEngine()
    learner = AdaptiveLearningEngine(predictor, tracker)
    dashboard = RealTimeDashboard(predictor, tracker, learner)

    dashboard.run()

if __name__ == "__main__":
    main()