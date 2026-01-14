import numpy as np
import pandas as pd
import yfinance as yf


def load_price_data(tickers, start="2015-01-01", end="2024-01-01"):
    """
    Load adjusted close prices for a list of tickers.
    """
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    prices = data["Close"]

    return prices


def prices_to_log_returns(prices: pd.DataFrame):
    """
    Convert price series to log returns.
    """
    log_returns = np.log(prices / prices.shift(1))
    log_returns = log_returns.dropna()

    return log_returns

def train_test_split(returns, train_ratio=0.7):
    split = int(len(returns) * train_ratio)
    return returns[:split], returns[split:]

def normalize_returns(returns):
    mean = returns.mean(axis=0)
    std = returns.std(axis=0) + 1e-8
    return (returns - mean) / std

def prepare_data(tickers, start="2015-01-01", end="2024-01-01", train_ratio=0.7):
    # Load prices
    prices = load_price_data(tickers, start, end)

    # Handle missing values safely
    prices = prices.ffill().dropna()

    # Convert to log returns
    returns = prices_to_log_returns(prices)

    # Train / test split
    train_returns, test_returns = train_test_split(returns, train_ratio)

    # Compute normalization stats ONLY on train
    mean = train_returns.mean(axis=0)
    std = train_returns.std(axis=0) + 1e-8

    # Normalize
    train_returns = (train_returns - mean) / std
    test_returns  = (test_returns - mean) / std

    return train_returns, test_returns, mean, std