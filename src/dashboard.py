

from adaptive_learner import AdaptiveLearningEngine
from prediction_engine import PredictionEngine
from data_collector import RealTimeStockTracker
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import logging
import sys
import os
from datetime import datetime, timedelta
import pytz

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

@st.cache_resource
def load_prediction_engine():
    """Load and train prediction engine with caching"""
    print("🔄 Creating new prediction engine...")
    engine = PredictionEngine()

    # Skip loading for now - force fresh training
    print("🔧 Force training new models...")
    tracker = RealTimeStockTracker()
    data = tracker.fetch_historical_data()
    
    if data is not None:
        print("📊 Starting model training...")
        performances = engine.train_models(data)
        print(f"✅ Training completed. Performances: {performances}")

        # Debug: Check if models are actually trained
        print("🔍 Post-training model status:")
        for name, model in engine.models.items():
            print(f"   {name}: is_trained = {model.is_trained}")

    return engine


class RealTimeDashboard:
    def __init__(self):
        self.tracker = None
        self.engine = None
        self.learner = None
        self.logger = logging.getLogger(__name__)

        # Initialize session state
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
            st.session_state.prediction_history = []
            st.session_state.actual_history = []
            st.session_state.last_update = None

    def display_market_status(self):
        """Display current market status"""
        is_open, status_msg = self.tracker.is_market_open()

        if is_open:
            st.success(f"🟢 {status_msg}")
            return True
        else:
            st.warning(f"🔴 {status_msg}")
            st.info("Displaying last available market data")
            return False

    def get_market_data_with_fallback(self):
        """Get market data with fallback for closed markets"""
        # Try longer periods first for better predictions
        periods_to_try = ["1y", "6mo", "3mo", "2mo"]

        for period in periods_to_try:
            try:
                historical = self.tracker.fetch_historical_data(period=period)
                if historical is not None and not historical.empty and len(historical) >= 60:
                    st.info(
                        f"📊 Using {period} historical data ({len(historical)} days)")
                    return historical, "historical"
            except Exception as e:
                st.warning(f"Failed to fetch {period} data: {e}")
                continue

        st.error(
            "Unable to fetch sufficient historical data for reliable predictions (need at least 60 days)")
        return None, "failed"

    def get_stats_from_historical(self, historical_data):
        """Extract current stats from historical data when live data unavailable"""
        if historical_data is None or historical_data.empty:
            return None

        latest = historical_data.tail(1)
        previous = historical_data.tail(2).head(
            1) if len(historical_data) > 1 else latest

        current_price = latest['Close'].iloc[0]
        previous_price = previous['Close'].iloc[0]
        daily_change = current_price - previous_price
        daily_change_pct = (daily_change / previous_price) * \
            100 if previous_price > 0 else 0

        return {
            'symbol': config.STOCK_SYMBOL,
            'current_price': current_price,
            'daily_change': daily_change,
            'daily_change_pct': daily_change_pct,
            'volume': latest['Volume'].iloc[0],
            'high': latest['High'].iloc[0],
            'low': latest['Low'].iloc[0],
            'timestamp': latest.index[0]
        }

    def initialize_components(self):
        """Initialize all components"""
        if not st.session_state.initialized:
            with st.spinner("Initializing AI prediction system..."):
                self.tracker = RealTimeStockTracker(config.STOCK_SYMBOL)
                self.engine = load_prediction_engine()  # Use cached version

                # Load historical data
                historical_data = self.tracker.fetch_historical_data()
                if historical_data is not None:
                    self.learner = AdaptiveLearningEngine(
                        self.engine, self.tracker)
                    st.session_state.initialized = True
                    st.success("✅ AI system initialized successfully!")
                else:
                    st.error("❌ Failed to initialize system - no data available")
                    return False
        else:
            # Restore components from session
            self.tracker = RealTimeStockTracker(config.STOCK_SYMBOL)
            self.engine = load_prediction_engine()  # Use cached version

        return True

    def create_current_data_table_with_data(self, market_data, data_type, current_stats):
        """Create the real-time current data table with provided data"""
        st.subheader("📊 Real-Time Market Data")

        # Display market status first
        market_is_open = self.display_market_status()

        if market_data is not None:
            # Display data type indicator
            if data_type == "live":
                st.success("📈 Live Market Data")
            elif data_type == "historical":
                st.info("📊 Recent Historical Data")
            else:
                st.warning("⚠️ Fallback Data")

            if current_stats:
                # Create metrics display
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        label="Current Price",
                        value=f"${current_stats['current_price']:.2f}",
                        delta=f"{current_stats['daily_change']:+.2f}"
                    )

                with col2:
                    st.metric(
                        label="Daily Change %",
                        value=f"{current_stats['daily_change_pct']:+.2f}%",
                        delta=None
                    )

                with col3:
                    st.metric(
                        label="Volume",
                        value=f"{current_stats['volume']:,}",
                        delta=None
                    )

                with col4:
                    market_status = "🟢 OPEN" if market_is_open else "🔴 CLOSED"
                    st.metric(
                        label="Market Status",
                        value=market_status,
                        delta=None
                    )

                # Detailed data table
                st.subheader("Detailed Market Information")

                data_table = pd.DataFrame({
                    'Metric': ['Symbol', 'Current Price', 'Daily Change', 'Daily Change %',
                               'Today\'s High', 'Today\'s Low', 'Volume', 'Last Updated'],
                    'Value': [
                        current_stats['symbol'],
                        f"${current_stats['current_price']:.2f}",
                        f"${current_stats['daily_change']:+.2f}",
                        f"{current_stats['daily_change_pct']:+.2f}%",
                        f"${current_stats['high']:.2f}",
                        f"${current_stats['low']:.2f}",
                        f"{current_stats['volume']:,}",
                        current_stats['timestamp'].strftime(
                            '%Y-%m-%d %H:%M:%S')
                    ]
                })

                st.dataframe(data_table, use_container_width=True)
                return current_stats
            else:
                st.error("Unable to process market data")
                return None
        else:
            st.error("Unable to fetch any market data")
            return None

    def create_prediction_table_with_data(self, historical_data, current_stats):
        """Create the AI prediction table using provided data"""
        st.subheader("🤖 AI Prediction Engine")

        if historical_data is None or historical_data.empty:
            st.error("📊 No historical data available for prediction")
            return None

        try:
            extended_data = None
            # Start with longer periods
            periods_to_try = ["1y", "6mo", "3mo", "2mo"]

            for period in periods_to_try:
                try:
                    extended_data = self.tracker.fetch_historical_data(
                        period=period)
                    if extended_data is not None and not extended_data.empty and len(extended_data) >= 60:
                        st.info(
                            f"📈 Generating prediction using {len(extended_data)} days of data ({period})")
                        break
                except Exception as e:
                    continue

            if extended_data is None or extended_data.empty or len(extended_data) < 60:
                st.error(
                    "📊 Insufficient historical data for prediction (need at least 60 days)")
                return None
            # Generate prediction using the provided data
            prediction_result = self.engine.predict_next_day(extended_data)

            if prediction_result and prediction_result.get('ensemble_prediction'):
                # Store prediction for tracking
                st.session_state.prediction_history.append({
                    'timestamp': datetime.now(),
                    'predicted_price': prediction_result['ensemble_prediction'],
                    'current_price': current_stats['current_price'] if current_stats else None,
                    'confidence': prediction_result['confidence']
                })

                # Keep only last 50 predictions
                if len(st.session_state.prediction_history) > 50:
                    st.session_state.prediction_history = st.session_state.prediction_history[-50:]

                # Display prediction metrics
                predicted_price = prediction_result['ensemble_prediction']
                current_price = current_stats['current_price'] if current_stats else 0
                predicted_change = predicted_price - current_price
                predicted_change_pct = (
                    predicted_change / current_price) * 100 if current_price > 0 else 0

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        label="Next Day Prediction",
                        value=f"${predicted_price:.2f}",
                        delta=f"{predicted_change:+.2f}"
                    )

                with col2:
                    st.metric(
                        label="Predicted Change %",
                        value=f"{predicted_change_pct:+.2f}%",
                        delta=None
                    )

                with col3:
                    confidence_color = "🟢" if prediction_result[
                        'confidence'] > 0.7 else "🟡" if prediction_result['confidence'] > 0.5 else "🔴"
                    st.metric(
                        label="Confidence Level",
                        value=f"{confidence_color} {prediction_result['confidence']:.1%}",
                        delta=None
                    )

                # Individual model predictions table
                st.subheader("Individual Model Predictions")

                model_data = []
                for model_name, pred_value in prediction_result['individual_predictions'].items():
                    if pred_value is not None:
                        change = pred_value - current_price
                        change_pct = (change / current_price) * \
                            100 if current_price > 0 else 0
                        model_data.append({
                            'Model': model_name.title(),
                            'Prediction': f"${pred_value:.2f}",
                            'Change': f"${change:+.2f}",
                            'Change %': f"{change_pct:+.2f}%"
                        })

                if model_data:
                    pred_df = pd.DataFrame(model_data)
                    st.dataframe(pred_df, use_container_width=True)

                return prediction_result
            else:
                st.error(
                    "Unable to generate prediction - model returned no results")
                return None

        except Exception as e:
            error_msg = str(e)
            if "MinMaxScaler" in error_msg or "not fitted" in error_msg:
                st.error("🔧 Model needs retraining due to scaler issues")
                st.info("Please refresh the page to retrain the models")
            else:
                st.error(f"Prediction error: {error_msg}")
                st.info("This may be due to model training issues or data problems")

        return None

    def create_charts(self, historical_data):
        """Create interactive price and prediction charts"""
        st.subheader("📈 Interactive Price Analysis")

        if historical_data is not None and not historical_data.empty:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=(
                    'Stock Price with Moving Averages', 'Trading Volume'),
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )

            # Add price data
            fig.add_trace(
                go.Scatter(
                    x=historical_data.index,
                    y=historical_data['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )

            # Add moving averages
            ma_20 = historical_data['Close'].rolling(window=20).mean()
            ma_50 = historical_data['Close'].rolling(window=50).mean()

            fig.add_trace(
                go.Scatter(
                    x=historical_data.index,
                    y=ma_20,
                    mode='lines',
                    name='20-day MA',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=historical_data.index,
                    y=ma_50,
                    mode='lines',
                    name='50-day MA',
                    line=dict(color='red', width=1)
                ),
                row=1, col=1
            )

            # Add volume
            fig.add_trace(
                go.Bar(
                    x=historical_data.index,
                    y=historical_data['Volume'],
                    name='Volume',
                    marker_color='lightblue',
                    opacity=0.6
                ),
                row=2, col=1
            )

            # Update layout
            fig.update_layout(
                title=f"{config.STOCK_SYMBOL} - Real-Time Analysis Dashboard",
                height=600,
                showlegend=True,
                template="plotly_white"
            )

            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No historical data available for charts")

    def create_learning_insights(self):
        """Create learning insights section"""
        st.subheader("🧠 AI Learning Insights")

        if hasattr(self, 'learner') and self.learner:
            try:
                # Get model performance summary
                if hasattr(self.engine, 'get_model_summary'):
                    summary = self.engine.get_model_summary()

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Models Trained",
                                  summary.get('num_models', 0))
                        st.metric("System Status", "✅ Active" if summary.get(
                            'is_trained', False) else "❌ Inactive")

                    with col2:
                        # Show ensemble weights
                        weights = summary.get('ensemble_weights', {})
                        if weights:
                            st.write("**Model Weights:**")
                            for model, weight in weights.items():
                                st.write(f"- {model}: {weight:.3f}")

                # Prediction history chart
                if len(st.session_state.prediction_history) > 1:
                    st.subheader("🎯 Prediction History")

                    pred_df = pd.DataFrame(st.session_state.prediction_history)

                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=pred_df['timestamp'],
                            y=pred_df['predicted_price'],
                            mode='lines+markers',
                            name='Predicted Prices',
                            line=dict(color='red', dash='dash')
                        )
                    )

                    fig.update_layout(
                        title="Prediction History",
                        xaxis_title="Time",
                        yaxis_title="Price ($)",
                        template="plotly_white"
                    )

                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.warning(f"Learning insights unavailable: {e}")
        else:
            st.info("Learning system not initialized")

    def run(self):
        """Main dashboard runner"""
        # Page configuration
        st.set_page_config(
            page_title="Real-Time Stock Predictor AI",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Initialize components
        if not self.initialize_components():
            return

        # Main title
        st.title("🤖 Real-Time Stock Predictor AI Dashboard")
        st.markdown(
            f"**Analyzing: {config.STOCK_SYMBOL}** | **Auto-refresh every 30 seconds**")

        # Get market data first
        market_data, data_type = self.get_market_data_with_fallback()
        current_stats = None

        if market_data is not None:
            # Process the market data to get current stats
            current_stats = self.tracker.get_current_stats(
            ) if data_type == "live" else self.get_stats_from_historical(market_data)

        # Create two columns for layout
        col1, col2 = st.columns([1, 1])

        with col1:
            # Pass the fetched market data to current data table
            self.create_current_data_table_with_data(
                market_data, data_type, current_stats)

        with col2:
            # Pass the same market data to prediction table
            self.create_prediction_table_with_data(market_data, current_stats)

        # Charts section
        self.create_charts(
            market_data if market_data is not None else self.tracker.fetch_historical_data())

        # Learning insights
        self.create_learning_insights()


# Run the dashboard
if __name__ == "__main__":
    dashboard = RealTimeDashboard()
    dashboard.run()
