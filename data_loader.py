import yfinance as yf
import pandas as pd
import numpy as np

class DataLoader:
    """
    A class to handle fetching, cleaning, and preprocessing financial time-series data.
    """
    def __init__(self, ticker: str, start_date: str, end_date: str):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    def fetch_and_clean_data(self) -> pd.DataFrame:
        print(f"Fetching data for {self.ticker} from {self.start_date} to {self.end_date}...")
        
        # Download data using yfinance
        df = yf.download(self.ticker, start=self.start_date, end=self.end_date, progress=False)
        
        if df.empty:
            raise ValueError(f"No data found for {self.ticker}. Check the ticker symbol or dates.")

        # --- NEW FIX: Flatten MultiIndex columns from newer yfinance versions ---
        if isinstance(df.columns, pd.MultiIndex):
            # This extracts just the top-level name (e.g., 'Close' instead of ('Close', 'BTC-USD'))
            df.columns = df.columns.get_level_values(0)

        # 1. Handle missing values
        df = df.ffill()

        # 2. Calculate Daily Returns
        # Note: Because yfinance now sometimes returns a DataFrame for a single column, 
        # we make sure we squeeze it down to a 1D Series first.
        df['Return'] = df['Close'].squeeze().pct_change()

        # 3. Calculate Rolling Volatility
        df['Volatility_20d'] = df['Return'].rolling(window=20).std()

        # Drop the first 20 rows because they will have NaN
        df = df.dropna()

        print("Data fetching and preprocessing complete.")
        return df

if __name__ == "__main__":
    loader = DataLoader(ticker="BTC-USD", start_date="2023-01-01", end_date="2024-01-01")
    market_data = loader.fetch_and_clean_data()
    print(market_data[['Close', 'Return', 'Volatility_20d']].head())