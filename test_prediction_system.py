import sys
import logging
sys.path.append('src')

from src.data_collector import RealTimeStockTracker
from src.prediction_engine import PredictionEngine

def main():
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Complete Prediction System")
    print("="*60)
    
    # Initialize components
    tracker = RealTimeStockTracker()
    engine = PredictionEngine()
    
    # Get historical data
    print("Fetching historical data...")
    data = tracker.fetch_historical_data()
    print(f"✓ Got {len(data)} days of data")
    
    # Train models
    print("\nTraining prediction models...")
    performances = engine.train_models(data)
    
    # Display results
    print("\n" + "="*60)
    print("MODEL PERFORMANCE RESULTS")
    print("="*60)
    
    for name, perf in performances.items():
        print(f"{name.upper():15}")
        print(f"  R² Score:           {perf.get('r2', 0):.4f}")
        print(f"  RMSE:              ${perf.get('rmse', 0):.4f}")
        print(f"  MAE:               ${perf.get('mae', 0):.4f}")
        print(f"  Directional Acc:    {perf.get('directional_accuracy', 0):.1f}%")
        print()
    
    # Make prediction
    print("Making next-day prediction...")
    prediction = engine.predict_next_day(data)
    
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Ensemble Prediction:  ${prediction['ensemble_prediction']:.2f}")
    print(f"Confidence Level:     {prediction['confidence']:.1%}")
    print(f"Current Price:        ${data['Close'].iloc[-1]:.2f}")
    print(f"Predicted Change:     ${prediction['ensemble_prediction'] - data['Close'].iloc[-1]:+.2f}")
    print(f"Predicted Change %:   {((prediction['ensemble_prediction'] / data['Close'].iloc[-1]) - 1) * 100:+.2f}%")
    
    print(f"\nIndividual Model Predictions:")
    for name, pred in prediction['individual_predictions'].items():
        if pred is not None:
            change = pred - data['Close'].iloc[-1]
            print(f"  {name:15}: ${pred:.2f} ({change:+.2f})")
    
    print(f"\nPrediction generated at: {prediction['timestamp']}")
    print("="*60)

if __name__ == "__main__":
    main()
