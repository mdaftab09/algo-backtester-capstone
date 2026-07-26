import pandas as pd
import numpy as np

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Measures risk-adjusted return. A Sharpe ratio > 1.0 is generally considered good.
    Formula: (Mean of excess returns / Standard deviation of returns) * sqrt(252 trading days)
    """
    excess_returns = returns - risk_free_rate
    if excess_returns.std() == 0:
        return 0.0
    return np.sqrt(252) * (excess_returns.mean() / excess_returns.std())

def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Measures the largest peak-to-trough drop in portfolio value.
    Crucial for understanding worst-case scenario risk.
    """
    # Calculate the running maximum of the equity curve
    roll_max = equity_curve.cummax()
    
    # Calculate the percentage drop from the running maximum
    drawdown = equity_curve / roll_max - 1.0
    
    return drawdown.min()

def calculate_win_rate(returns: pd.Series) -> float:
    """
    Calculates the percentage of days the strategy generated a positive return,
    ignoring days where we were not in the market (return == 0).
    """
    active_days = returns[returns != 0]
    if len(active_days) == 0:
        return 0.0
    
    winning_days = active_days[active_days > 0]
    return len(winning_days) / len(active_days)