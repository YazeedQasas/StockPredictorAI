import config
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RealTimeStockTracker:
    def __init__(self, symbol=config.STOCK_SYMBOL):
        self.symbol = symbol
        self.ticker = yf.Ticker(symbol)
        self.current_data = None
        self.historical_data = None
        self.logger = logging.getLogger(__name__)

        # Initialize data storage
        self.real_time_prices = []
        self.timestamps = []

    def fetch_current_data(self):
        """Fetch the most recent stock data"""
        try:
            # Get current day data with 1-minute intervals
            current = self.ticker.history(period="1d", interval="1m")

            if not current.empty:
                self.current_data = current
                latest = current.tail(1)

                # Store for tracking
                self.real_time_prices.append(latest['Close'].iloc[0])
                self.timestamps.append(datetime.now())

                # Keep only last 100 data points for memory efficiency
                if len(self.real_time_prices) > 100:
                    self.real_time_prices = self.real_time_prices[-100:]
                    self.timestamps = self.timestamps[-100:]

                self.logger.info(
                    f"Fetched current data for {self.symbol}: ${latest['Close'].iloc[0]:.2f}")
                return latest
            else:
                self.logger.warning("No current data available")
                return None

        except Exception as e:
            self.logger.error(f"Error fetching current data: {e}")
            return None

    def fetch_historical_data(self, period="1y"):
        """Fetch historical stock data for training"""
        try:
            self.historical_data = self.ticker.history(period=period)

            if not self.historical_data.empty:
                self.logger.info(
                    f"Fetched {len(self.historical_data)} days of historical data")

                # Save to CSV for backup
                data_path = os.path.join(
                    config.DATA_DIR, 'raw', f'{self.symbol.lower()}_historical.csv')
                self.historical_data.to_csv(data_path)

                return self.historical_data
            else:
                self.logger.warning("No historical data available")
                return None

        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}")
            return None

    def get_current_stats(self):
        """Get current stock statistics"""
        if self.current_data is None or self.current_data.empty:
            return None

        latest = self.current_data.tail(1)
        previous_close = self.current_data['Close'].iloc[-2] if len(
            self.current_data) > 1 else latest['Close'].iloc[0]

        current_price = latest['Close'].iloc[0]
        daily_change = current_price - previous_close
        daily_change_pct = (daily_change / previous_close) * 100

        stats = {
            'symbol': self.symbol,
            'current_price': current_price,
            'daily_change': daily_change,
            'daily_change_pct': daily_change_pct,
            'volume': latest['Volume'].iloc[0],
            'high': latest['High'].iloc[0],
            'low': latest['Low'].iloc[0],
            'timestamp': latest.index[0]
        }

        return stats

    def get_technical_indicators(self, data=None):
        """Calculate basic technical indicators"""
        if data is None:
            data = self.historical_data

        if data is None or data.empty:
            return None

        indicators = pd.DataFrame(index=data.index)
        indicators['Close'] = data['Close']

        # Moving averages
        indicators['ma_5'] = data['Close'].rolling(window=5).mean()
        indicators['ma_20'] = data['Close'].rolling(window=20).mean()
        indicators['ma_50'] = data['Close'].rolling(window=50).mean()

        # Volatility
        indicators['volatility'] = data['Close'].rolling(window=20).std()

        # RSI (Relative Strength Index)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators['rsi'] = 100 - (100 / (1 + rs))

        # Volume indicators
        indicators['Volume'] = data['Volume']
        indicators['volume_ma'] = data['Volume'].rolling(window=20).mean()

        return indicators.dropna()

    def is_market_open(self):
        """Check if the market is currently open (simplified)"""
        now = datetime.now()
        # Market hours: 9:30 AM - 4:00 PM EST, Monday-Friday
        if now.weekday() >= 5:  # Weekend
            return False

        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

        return market_open <= now <= market_close


# Test the data collector
if __name__ == "__main__":
    # Set up basic logging
    logging.basicConfig(level=logging.INFO)

    # Create tracker instance
    tracker = RealTimeStockTracker()

    print(f"Testing Real-Time Stock Tracker for {tracker.symbol}")
    print("=" * 50)

    # Test historical data fetch
    print("Fetching historical data...")
    historical = tracker.fetch_historical_data()
    if historical is not None:
        print(f"✓ Historical data: {len(historical)} days")
        print(
            f"  Date range: {historical.index[0].date()} to {historical.index[-1].date()}")
        print(
            f"  Price range: ${historical['Close'].min():.2f} - ${historical['Close'].max():.2f}")

    # Test current data fetch
    print("\nFetching current data...")
    current = tracker.fetch_current_data()
    if current is not None:
        stats = tracker.get_current_stats()
        print(f"✓ Current price: ${stats['current_price']:.2f}")
        print(
            f"  Daily change: ${stats['daily_change']:.2f} ({stats['daily_change_pct']:.2f}%)")
        print(f"  Volume: {stats['volume']:,}")

    # Test technical indicators
    print("\nCalculating technical indicators...")
    indicators = tracker.get_technical_indicators()
    if indicators is not None:
        latest_indicators = indicators.tail(1)
        print(f"✓ Technical indicators calculated")
        print(f"  5-day MA: ${latest_indicators['ma_5'].iloc[0]:.2f}")
        print(f"  20-day MA: ${latest_indicators['ma_20'].iloc[0]:.2f}")
        print(f"  RSI: {latest_indicators['rsi'].iloc[0]:.2f}")

    print(
        f"\nMarket status: {'OPEN' if tracker.is_market_open() else 'CLOSED'}")
