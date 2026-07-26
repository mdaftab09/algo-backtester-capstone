import pandas as pd
import numpy as np

class BaseStrategy:
    """
    The blueprint for all trading strategies. 
    Every strategy must inherit from this class and implement the 'generate_signals' method.
    """
    def __init__(self, name: str):
        self.name = name

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        # This acts as a placeholder. If a child class doesn't override this, it throws an error.
        raise NotImplementedError("Each strategy must implement the 'generate_signals' method.")


class DualMovingAverageCrossover(BaseStrategy):
    """
    Strategy 1: Buys when the short-term moving average crosses above the long-term moving average.
    Sells/Shorts when the short-term crosses below the long-term.
    """
    def __init__(self, short_window: int = 50, long_window: int = 200):
        super().__init__(name="Dual Moving Average Crossover")
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"Generating signals for {self.name}...")
        data = df.copy()
        
        # Calculate the moving averages
        data['SMA_Short'] = data['Close'].rolling(window=self.short_window).mean()
        data['SMA_Long'] = data['Close'].rolling(window=self.long_window).mean()
        
        # Create a signal: 1 for Long (Buy), -1 for Short (Sell), 0 for Hold
        # We start by defaulting everything to 0
        data['Signal'] = 0.0
        
        # If the short MA is greater than the long MA, we want to be Long (1)
        data.loc[data['SMA_Short'] > data['SMA_Long'], 'Signal'] = 1.0
        
        # If the short MA is less than the long MA, we want to be Short (-1)
        data.loc[data['SMA_Short'] < data['SMA_Long'], 'Signal'] = -1.0
        
        # The actual trade happens when the signal changes (the crossover).
        # We calculate the difference between today's position and yesterday's.
        data['Position_Change'] = data['Signal'].diff()
        
        return data


class AnomalyFader(BaseStrategy):
    """
    Strategy 2: Fades the anomaly. 
    If an anomaly drops the price significantly, we assume it's an overreaction and BUY.
    If an anomaly spikes the price significantly, we SELL/SHORT.
    """
    def __init__(self):
        super().__init__(name="Anomaly Fader")

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"Generating signals for {self.name}...")
        data = df.copy()
        
        data['Signal'] = 0.0
        
        # Condition 1: Anomaly detected AND the daily return was negative -> Buy the dip (1)
        buy_condition = (data['is_anomaly'] == 1) & (data['Return'] < 0)
        data.loc[buy_condition, 'Signal'] = 1.0
        
        # Condition 2: Anomaly detected AND the daily return was positive -> Short the spike (-1)
        sell_condition = (data['is_anomaly'] == 1) & (data['Return'] > 0)
        data.loc[sell_condition, 'Signal'] = -1.0
        
        # Unlike Moving Averages where we hold for a long time, we might just hold this for 1 day.
        # So we forward-fill the signal for a holding period of 5 days, then revert to 0.
        # For simplicity in this base version, let's just hold the position until the next signal.
        # We will forward fill the 1s and -1s, replacing 0s.
        data['Signal'] = data['Signal'].replace(0.0, method='ffill').fillna(0.0)
        
        data['Position_Change'] = data['Signal'].diff()
        
        return data

# --- Quick Test Block ---
if __name__ == "__main__":
    from data_loader import DataLoader
    from anomaly_detector import AnomalyDetector

    # 1. Load Data
    loader = DataLoader(ticker="BTC-USD", start_date="2023-01-01", end_date="2024-01-01")
    df = loader.fetch_and_clean_data()
    
    # 2. Add Anomalies
    detector = AnomalyDetector(df)
    df_with_anomalies = detector.generate_anomalies()
    
    # 3. Test Strategy 1
    sma_strategy = DualMovingAverageCrossover(short_window=10, long_window=50)
    sma_signals = sma_strategy.generate_signals(df_with_anomalies)
    
    # 4. Test Strategy 2
    anomaly_strategy = AnomalyFader()
    anomaly_signals = anomaly_strategy.generate_signals(df_with_anomalies)

    print("\n--- SMA Crossover Signals (Last 5 days) ---")
    print(sma_signals[['Close', 'SMA_Short', 'SMA_Long', 'Signal', 'Position_Change']].tail())
    
    print("\n--- Anomaly Fader Signals (Last 5 days) ---")
    print(anomaly_signals[['Close', 'Return', 'is_anomaly', 'Signal', 'Position_Change']].tail())