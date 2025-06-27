import os
from dotenv import load_dotenv

load_dotenv()

# Stock Configuration
STOCK_SYMBOL = "AMZN"
PREDICTION_DAYS = 30
UPDATE_INTERVAL = 60  # seconds

# Model Configuration
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32
TRAIN_TEST_SPLIT = 0.8

# Dashboard Configuration
DASHBOARD_PORT = 8050
REFRESH_RATE = 30  # seconds

# Data Configuration
DATA_SOURCES = {
    'yahoo': True,
    'alpha_vantage': False,  # I should set to True if I get the API key
}

# File Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
