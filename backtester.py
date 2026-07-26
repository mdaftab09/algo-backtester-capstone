import pandas as pd
import metrics

class BacktestEngine:
    """
    Simulates trading a strategy over historical data to evaluate performance.
    """
    def __init__(self, data: pd.DataFrame, initial_capital: float = 10000.0, transaction_fee: float = 0.001):
        # Transaction fee of 0.001 represents 0.1% per trade (standard exchange fee)
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.transaction_fee = transaction_fee

    def run(self) -> tuple[dict, pd.DataFrame]:
        print("Starting backtest simulation...")
        
        # 1. Calculate Strategy Return
        # We must shift the signal by 1. If the signal triggers at the end of Monday, 
        # we capture the return for Tuesday. 
        self.data['Strategy_Return'] = self.data['Signal'].shift(1) * self.data['Return']
        
        # 2. Calculate Transaction Costs
        # We only pay fees when our position changes.
        self.data['Fee_Impact'] = self.data['Position_Change'].abs() * self.transaction_fee
        
        # 3. Calculate Net Return (Strategy Return minus Fees)
        self.data['Net_Return'] = self.data['Strategy_Return'] - self.data['Fee_Impact']
        
        # Drop the first row because shift(1) creates a NaN (Not a Number) value
        self.data = self.data.dropna(subset=['Net_Return'])
        
        # 4. Calculate the Equity Curve (compounding the net returns over time)
        self.data['Equity'] = self.initial_capital * (1 + self.data['Net_Return']).cumprod()
        
        # For baseline comparison: What if we just bought and held the asset?
        self.data['Buy_Hold_Equity'] = self.initial_capital * (1 + self.data['Return']).cumprod()

        # 5. Generate Performance Metrics
        sharpe = metrics.calculate_sharpe_ratio(self.data['Net_Return'])
        max_dd = metrics.calculate_max_drawdown(self.data['Equity'])
        win_rate = metrics.calculate_win_rate(self.data['Net_Return'])
        
        # Total return is (Final Equity / Initial Capital) - 1
        total_return = (self.data['Equity'].iloc[-1] / self.initial_capital) - 1
        bh_return = (self.data['Buy_Hold_Equity'].iloc[-1] / self.initial_capital) - 1
        
        stats = {
            'Total Return': f"{total_return * 100:.2f}%",
            'Buy & Hold Return': f"{bh_return * 100:.2f}%",
            'Sharpe Ratio': f"{sharpe:.2f}",
            'Max Drawdown': f"{max_dd * 100:.2f}%",
            'Win Rate': f"{win_rate * 100:.2f}%"
        }
        
        print("Backtest complete.")
        return stats, self.data

# --- Quick Test Block ---
if __name__ == "__main__":
    from data_loader import DataLoader
    from anomaly_detector import AnomalyDetector
    from strategies import DualMovingAverageCrossover

    # Load and prep data
    loader = DataLoader(ticker="BTC-USD", start_date="2023-01-01", end_date="2024-01-01")
    df = loader.fetch_and_clean_data()
    
    detector = AnomalyDetector(df)
    df = detector.generate_anomalies()
    
    # Apply strategy
    strategy = DualMovingAverageCrossover(short_window=10, long_window=50)
    df_signals = strategy.generate_signals(df)
    
    # Run Backtest
    engine = BacktestEngine(data=df_signals, initial_capital=10000)
    performance_stats, backtest_data = engine.run()
    
    print("\n--- Strategy Performance Results ---")
    for key, value in performance_stats.items():
        print(f"{key}: {value}")