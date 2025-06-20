import sys
import os
sys.path.append('src')
sys.path.append('utils')

from src.data_collector import RealTimeStockTracker
from utils.visualization_utils import plot_stock_data, create_current_stats_table

def main():
    print("Testing Complete Stock Data System")
    print("="*50)
    
    # Initialize tracker
    tracker = RealTimeStockTracker()
    
    # Get historical data
    historical = tracker.fetch_historical_data()
    
    # Get current data
    current = tracker.fetch_current_data()
    stats = tracker.get_current_stats()
    
    # Get technical indicators
    indicators = tracker.get_technical_indicators()
    
    # Display current stats
    if stats:
        create_current_stats_table(stats)
    
    # Create visualization
    if indicators is not None:
        plot_stock_data(indicators.tail(60), f"{tracker.symbol} - Last 60 Days")
    
    print("\n✓ System test completed successfully!")

if __name__ == "__main__":
    main()