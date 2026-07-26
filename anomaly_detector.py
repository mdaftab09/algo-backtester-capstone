import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# Import the DataLoader we built in Phase 1
from data_loader import DataLoader

class AnomalyDetector:
    """
    Identifies market anomalies using statistical methods and Machine Learning.
    """
    def __init__(self, df: pd.DataFrame):
        # We work on a copy of the dataframe to avoid modifying the original data unintentionally
        self.df = df.copy()

    def add_z_score_baseline(self, window: int = 20):
        # Calculate rolling mean and standard deviation
        rolling_mean = self.df['Return'].rolling(window=window).mean()
        rolling_std = self.df['Return'].rolling(window=window).std()
        
        # Calculate Z-Score
        self.df['Z_Score'] = (self.df['Return'] - rolling_mean) / rolling_std
        
        # Flag as anomaly if the Z-score is greater than 3 or less than -3 (3 standard deviations)
        self.df['stat_anomaly'] = np.where(self.df['Z_Score'].abs() > 3, 1, 0)

    def add_isolation_forest_anomalies(self, contamination: float = 0.05):
        # Contamination is the expected percentage of outliers in our dataset (5%)
        features = ['Return', 'Volatility_20d']
        
        # Drop rows with NaN values in our features before passing to the ML model
        temp_df = self.df.dropna(subset=features).copy()
        
        # Initialize the Isolation Forest model
        model = IsolationForest(contamination=contamination, random_state=42)
        
        # Fit the model and predict (-1 means anomaly, 1 means normal)
        preds = model.fit_predict(temp_df[features])
        
        # Convert predictions to a standard binary format: 1 for anomaly, 0 for normal
        temp_df['ml_anomaly'] = np.where(preds == -1, 1, 0)
        
        # Merge back into our main dataframe
        self.df['ml_anomaly'] = temp_df['ml_anomaly']
        self.df['ml_anomaly'] = self.df['ml_anomaly'].fillna(0).astype(int)

    def generate_anomalies(self) -> pd.DataFrame:
        print("Running statistical baseline (Z-Score)...")
        self.add_z_score_baseline()
        
        print("Running Machine Learning detector (Isolation Forest)...")
        self.add_isolation_forest_anomalies()
        
        # We will use the ML model as our primary 'is_anomaly' signal for the trading strategy
        self.df['is_anomaly'] = self.df['ml_anomaly']
        
        print("Anomaly detection complete.")
        return self.df

# --- Quick Test Block ---
if __name__ == "__main__":
    # 1. Fetch the data using Phase 1 code
    loader = DataLoader(ticker="BTC-USD", start_date="2023-01-01", end_date="2024-01-01")
    market_data = loader.fetch_and_clean_data()
    
    # 2. Pass the data to Phase 2 code
    detector = AnomalyDetector(market_data)
    analyzed_data = detector.generate_anomalies()
    
    # 3. Filter and print only the days where an anomaly was detected
    anomalies_only = analyzed_data[analyzed_data['is_anomaly'] == 1]
    
    print(f"\nTotal days analyzed: {len(analyzed_data)}")
    print(f"Total anomalies detected: {len(anomalies_only)}")
    print("\nSample of detected anomalies:")
    print(anomalies_only[['Close', 'Return', 'Z_Score', 'is_anomaly']].head())