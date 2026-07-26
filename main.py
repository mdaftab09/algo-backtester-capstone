import matplotlib.pyplot as plt
import seaborn as sns

# Import our custom modules
from data_loader import DataLoader
from anomaly_detector import AnomalyDetector
from strategies import AnomalyFader, DualMovingAverageCrossover
from backtester import BacktestEngine

def run_pipeline():
    print("=== Starting Quant Trading Backtest Pipeline ===\n")
    
    # 1. Data Ingestion
    loader = DataLoader(ticker="BTC-USD", start_date="2023-01-01", end_date="2024-01-01")
    df = loader.fetch_and_clean_data()
    
    # 2. Anomaly Detection
    detector = AnomalyDetector(df)
    df_anomalies = detector.generate_anomalies()
    
    # 3. Strategy Signal Generation
    # We will use the Anomaly Fader to show off your Machine Learning integration
    strategy = AnomalyFader()
    df_signals = strategy.generate_signals(df_anomalies)
    
    # 4. Backtesting Engine
    # Starting with a $10,000 portfolio and a 0.1% transaction fee
    engine = BacktestEngine(data=df_signals, initial_capital=10000.0, transaction_fee=0.001)
    stats, backtest_data = engine.run()
    
    # --- Print Final Results ---
    print("\n=== Final Backtest Results ===")
    for metric, value in stats.items():
        print(f"{metric}: {value}")
        
    # 5. Visualization
    print("\nGenerating charts... Close the chart window to exit the program.")
    
    # Set seaborn style for professional-looking plots
    sns.set_theme(style="darkgrid")
    
    # Create a figure with 2 subplots stacked vertically, sharing the time axis
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)
    
    # --- Plot 1: Asset Price & Anomalies ---
    axes[0].plot(backtest_data.index, backtest_data['Close'], label='BTC-USD Close Price', color='#1f77b4', alpha=0.8)
    
    # Highlight where the Isolation Forest found anomalies
    anomalies = backtest_data[backtest_data['is_anomaly'] == 1]
    axes[0].scatter(anomalies.index, anomalies['Close'], color='red', label='Anomaly Detected', marker='X', s=100)
    
    axes[0].set_title('Asset Price and Detected Market Anomalies', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Price (USD)')
    axes[0].legend()
    
    # --- Plot 2: Portfolio Equity Curve ---
    axes[1].plot(backtest_data.index, backtest_data['Equity'], label='Strategy Equity', color='#2ca02c', linewidth=2)
    axes[1].plot(backtest_data.index, backtest_data['Buy_Hold_Equity'], label='Buy & Hold Equity', color='#ff7f0e', linestyle='--')
    
    axes[1].set_title('Portfolio Performance vs. Buy & Hold', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Portfolio Value ($)')
    axes[1].legend()
    
    # Adjust layout so labels don't overlap
    plt.tight_layout()
    
    # Display the plot
    plt.show()

if __name__ == "__main__":
    run_pipeline()