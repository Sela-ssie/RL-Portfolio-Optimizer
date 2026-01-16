import numpy as np
import pandas as pd
import requests
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

def get_sp500_tickers() -> list[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    tables = pd.read_html(resp.text)
    df = tables[0]

    tickers = df["Symbol"].astype(str).tolist()
    # Yahoo uses '-' instead of '.' for class shares
    tickers = [t.replace(".", "-") for t in tickers]
    return tickers


def load_price_data_batched(
    tickers: list[str],
    start="2015-01-01",
    end="2024-01-01",
    batch_size: int = 80,
):
    """
    Download adjusted close prices in batches to reduce Yahoo failures.
    Returns a DataFrame: index=Date, columns=tickers.
    """
    all_prices = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        data = yf.download(
            batch,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        prices = data["Close"]
        all_prices.append(prices)

    prices = pd.concat(all_prices, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]  # just in case
    return prices