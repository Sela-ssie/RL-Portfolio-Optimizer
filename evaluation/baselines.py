# evaluation/baselines.py
# Run from project root:
#   python evaluation/baselines.py
#
# Baselines included:
# - Equal-Weight True Buy-and-Hold (weights drift, no trading after start)
# - Equal-Weight Monthly Rebalance (weights drift, rebalance every ~21 trading days, pays turnover costs)
# - Best Single-Asset Buy-and-Hold
# - SPY Buy-and-Hold
#
# All portfolio value paths are computed using log-returns and:
#   V_{t+1} = V_t * exp(portfolio_log_return - cost)
#
# IMPORTANT:
# Baselines use RAW log returns (NOT normalized), because normalization is only for RL training stability.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data.data_loader import load_price_data, prices_to_log_returns, get_sp500_tickers, load_price_data_batched


# -----------------------------
# Helpers: alignment + drift
# -----------------------------
def align_on_dates(a: pd.DataFrame, b: pd.DataFrame):
    """Align two return DataFrames on the intersection of dates."""
    common = a.index.intersection(b.index)
    return a.loc[common], b.loc[common]


def drift_weights(prev_w: np.ndarray, asset_log_returns_t: np.ndarray) -> np.ndarray:
    """
    Update weights after a market move, assuming NO trading during the step.
    With log returns r, gross returns are exp(r).
    """
    gross = np.exp(asset_log_returns_t)               # (N,)
    unnorm = prev_w * gross                           # proportional to post-move asset values
    return unnorm / (unnorm.sum() + 1e-12)            # normalize to sum to 1


# -----------------------------
# Portfolio simulators
# -----------------------------
def simulate_buy_and_hold(
    returns: pd.DataFrame,
    init_weights: np.ndarray,
    initial_value: float = 1.0,
):
    """
    True buy-and-hold:
    - Start with init_weights
    - Never trade again
    - Weights drift naturally with returns
    - No transaction costs (no trading)
    """
    r = returns.values  # (T, N) log returns
    T, N = r.shape

    w = np.asarray(init_weights, dtype=np.float64).copy()
    w = w / (w.sum() + 1e-12)

    values = np.empty(T + 1, dtype=np.float64)
    values[0] = float(initial_value)

    for t in range(T):
        port_log_ret = float(np.dot(w, r[t]))
        values[t + 1] = values[t] * np.exp(port_log_ret)
        w = drift_weights(w, r[t])

    return values


def simulate_rebalance_strategy(
    returns: pd.DataFrame,
    target_weights: np.ndarray,
    rebalance_every: int = 21,
    transaction_cost: float = 0.001,
    initial_value: float = 1.0,
):
    """
    Realistic rebalance:
    - Start at target_weights
    - Each day: portfolio gets returns; weights drift
    - On rebalance dates: trade back to target and pay turnover cost
    - Wealth update uses exp(log_return - cost)
    """
    r = returns.values  # (T, N) log returns
    T, N = r.shape

    target = np.asarray(target_weights, dtype=np.float64).copy()
    target = target / (target.sum() + 1e-12)

    w = target.copy()

    values = np.empty(T + 1, dtype=np.float64)
    values[0] = float(initial_value)

    for t in range(T):
        # Rebalance BEFORE realizing return at time t (typical convention)
        if rebalance_every > 0 and (t % rebalance_every == 0) and (t != 0):
            turnover = np.sum(np.abs(target - w))
            cost = transaction_cost * turnover
            w = target.copy()
        else:
            cost = 0.0

        port_log_ret = float(np.dot(w, r[t]))
        values[t + 1] = values[t] * np.exp(port_log_ret - cost)

        # After the market move, weights drift (no trading during the day)
        w = drift_weights(w, r[t])

    return values


def simulate_single_asset_buy_and_hold(
    returns: pd.DataFrame,
    asset_index: int,
    initial_value: float = 1.0,
):
    """100% in one asset, true buy-and-hold (weights drift is trivial here)."""
    T, N = returns.shape
    w = np.zeros(N, dtype=np.float64)
    w[asset_index] = 1.0
    return simulate_buy_and_hold(returns, init_weights=w, initial_value=initial_value)


def compute_best_single_asset(returns: pd.DataFrame):
    """
    Computes buy-and-hold performance for each asset
    and returns (ticker, values) for the best performer.
    """
    best_ticker = None
    best_values = None
    best_final = -np.inf

    for i, ticker in enumerate(returns.columns):
        w = np.zeros(len(returns.columns))
        w[i] = 1.0
        vals = simulate_buy_and_hold(returns, init_weights=w, initial_value=1.0)

        if vals[-1] > best_final:
            best_final = vals[-1]
            best_values = vals
            best_ticker = ticker

    return best_ticker, best_values




def plot_baselines(curves: dict[str, np.ndarray], title: str):
    """
    curves: dict mapping label -> portfolio value array
    """
    plt.figure(figsize=(10, 6))

    for label, values in curves.items():
        plt.plot(values, label=label)

    plt.title(title)
    plt.xlabel("Time step")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# -----------------------------
# Benchmark: SPY
# -----------------------------
def compute_spy_log_returns(start: str, end: str) -> pd.DataFrame:
    prices = load_price_data(["SPY"], start=start, end=end)
    prices = prices.ffill().dropna()
    return prices_to_log_returns(prices)


# -----------------------------
# Main script
# -----------------------------
if __name__ == "__main__":
    start, end = "2015-01-01", "2024-01-01"
    tc = 0.001

    # --- Load S&P 500 universe ---
    from data.data_loader import get_sp500_tickers, load_price_data_batched

    tickers = get_sp500_tickers()
    print("Total SP500 tickers:", len(tickers))

    prices = load_price_data_batched(tickers, start=start, end=end, batch_size=80)

    # Keep assets with sufficient data
    min_frac = 0.98
    prices = prices.dropna(axis=1, thresh=int(min_frac * len(prices)))
    prices = prices.ffill().dropna()

    returns = prices_to_log_returns(prices)
    print("Assets used after cleaning:", returns.shape[1])

    # --- SPY ---
    spy_rets = compute_spy_log_returns(start=start, end=end)
    returns_aligned, spy_aligned = align_on_dates(returns, spy_rets)

    # --- Target weights ---
    _, N = returns_aligned.shape
    target = np.ones(N) / N

    # --- Baselines ---
    ew_bh_vals = simulate_buy_and_hold(
        returns_aligned, init_weights=target, initial_value=1.0
    )

    ew_m_vals = simulate_rebalance_strategy(
        returns_aligned,
        target_weights=target,
        rebalance_every=21,
        transaction_cost=tc,
        initial_value=1.0,
    )

    spy_vals = simulate_buy_and_hold(
        spy_aligned,
        init_weights=np.array([1.0]),
        initial_value=1.0,
    )

    best_ticker, best_vals = compute_best_single_asset(returns_aligned)

    # --- Print results ---
    print("\n=== Baseline Results (SP500 Universe) ===")
    print("EW Buy & Hold (true) final:", ew_bh_vals[-1])
    print("EW Monthly Rebalance final:", ew_m_vals[-1])
    print("SPY Buy & Hold final:", spy_vals[-1])
    print(f"Best Single Asset ({best_ticker}) final:", best_vals[-1])

    # --- Plot ---
    plot_baselines(
        {
            "EW Buy & Hold (true)": ew_bh_vals,
            "EW Monthly Rebalance": ew_m_vals,
            "SPY Buy & Hold": spy_vals,
            f"Best Single ({best_ticker})": best_vals,
        },
        title="Baseline Portfolio Value Curves (S&P 500 Universe)",
    )