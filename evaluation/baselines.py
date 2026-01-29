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
from evaluation.metrics import summary_metrics, print_metrics_table


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
# Long/Short helpers + simulators (market-neutral)
# -----------------------------
def _normalize_simplex(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    s = float(x.sum())
    if s <= 0:
        return np.ones_like(x) / len(x)
    return x / (s + eps)


def _ls_from_scores(scores: np.ndarray, k_frac: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """
    Build long/short books from cross-sectional scores.
    Long: top k%, Short: bottom k%
    Returns w_long, w_short (each simplex weights sum to 1).
    """
    scores = np.asarray(scores, dtype=np.float64)
    N = scores.shape[0]
    k = max(1, int(k_frac * N))

    order = np.argsort(scores)
    short_idx = order[:k]
    long_idx = order[-k:]

    w_long = np.zeros(N, dtype=np.float64)
    w_short = np.zeros(N, dtype=np.float64)

    w_long[long_idx] = 1.0
    w_short[short_idx] = 1.0

    w_long = _normalize_simplex(w_long)
    w_short = _normalize_simplex(w_short)
    return w_long, w_short


def simulate_ls_rebalance_strategy(
    returns: pd.DataFrame,
    w_long_target: np.ndarray,
    w_short_target: np.ndarray,
    rebalance_every: int = 21,
    transaction_cost: float = 0.001,
    initial_value: float = 1.0,
    return_turnover: bool = False,
):
    """
    Market-neutral long/short strategy with fixed targets (rebalanced periodically).
    Portfolio exposure: w = 0.5*w_long - 0.5*w_short
    Turnover: sum(|Δw_long|)+sum(|Δw_short|) at rebalance times.
    """
    r = returns.values  # (T, N) log returns
    T, N = r.shape

    wL_tgt = _normalize_simplex(w_long_target)
    wS_tgt = _normalize_simplex(w_short_target)

    wL = wL_tgt.copy()
    wS = wS_tgt.copy()

    values = np.empty(T + 1, dtype=np.float64)
    values[0] = float(initial_value)

    turnovers = np.zeros(T, dtype=np.float64)

    for t in range(T):
        # Rebalance BEFORE return realization
        if rebalance_every > 0 and (t % rebalance_every == 0) and (t != 0):
            turnover = float(np.sum(np.abs(wL_tgt - wL)) + np.sum(np.abs(wS_tgt - wS)))
            wL = wL_tgt.copy()
            wS = wS_tgt.copy()
        else:
            turnover = 0.0

        turnovers[t] = turnover
        cost = transaction_cost * turnover

        # Signed exposure (fixed gross=1, net≈0)
        w = 0.5 * wL - 0.5 * wS

        port_log_ret = float(np.dot(w, r[t]))
        values[t + 1] = values[t] * np.exp(port_log_ret - cost)

        # Keep books fixed between rebalances (simplest + consistent with "rebalance strategy")
        # (No drift inside each book)
        # If you *want* drift, you need to model holding shares; start simple first.

    if return_turnover:
        return values, turnovers
    return values

def simulate_ls_buy_and_hold(
    returns: pd.DataFrame,
    w_long_init: np.ndarray,
    w_short_init: np.ndarray,
    initial_value: float = 1.0,
):
    r = returns.values
    T, N = r.shape

    wL = _normalize_simplex(w_long_init)
    wS = _normalize_simplex(w_short_init)

    values = np.empty(T + 1, dtype=np.float64)
    values[0] = float(initial_value)

    for t in range(T):
        w = 0.5 * wL - 0.5 * wS
        port_log_ret = float(np.dot(w, r[t]))
        values[t + 1] = values[t] * np.exp(port_log_ret)  # no trading, no cost

    return values

def simulate_ls_momentum(
    returns: pd.DataFrame,
    lookback: int = 60,
    k_frac: float = 0.1,
    rebalance_every: int = 21,
    transaction_cost: float = 0.001,
    initial_value: float = 1.0,
    return_turnover: bool = False,
):
    """
    Cross-sectional momentum long/short:
    - score_i(t) = sum_{j=t-lookback..t-1} r_{j,i}  (log-return momentum)
    - long top k%, short bottom k%
    - rebalanced every `rebalance_every` steps
    """
    r = returns.values  # (T, N)
    T, N = r.shape

    values = np.empty(T + 1, dtype=np.float64)
    values[0] = float(initial_value)
    turnovers = np.zeros(T, dtype=np.float64)

    # init: equal books
    wL = np.ones(N, dtype=np.float64) / N
    wS = np.ones(N, dtype=np.float64) / N

    for t in range(T):
        turnover = 0.0

        if t >= lookback and (rebalance_every > 0) and (t % rebalance_every == 0) and (t != 0):
            scores = r[t - lookback : t].sum(axis=0)  # (N,)
            wL_new, wS_new = _ls_from_scores(scores, k_frac=k_frac)

            turnover = float(np.sum(np.abs(wL_new - wL)) + np.sum(np.abs(wS_new - wS)))
            wL, wS = wL_new, wS_new

        turnovers[t] = turnover
        cost = transaction_cost * turnover

        w = 0.5 * wL - 0.5 * wS
        port_log_ret = float(np.dot(w, r[t]))
        values[t + 1] = values[t] * np.exp(port_log_ret - cost)

    if return_turnover:
        return values, turnovers
    return values

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
    return_turnover: bool = False,
):
    r = returns.values  # (T, N) log returns
    T, N = r.shape

    target = np.asarray(target_weights, dtype=np.float64).copy()
    target = target / (target.sum() + 1e-12)

    w = target.copy()

    values = np.empty(T + 1, dtype=np.float64)
    values[0] = float(initial_value)

    turnovers = np.zeros(T, dtype=np.float64)  # turnover per step

    for t in range(T):
        # rebalance BEFORE return realization
        if rebalance_every > 0 and (t % rebalance_every == 0) and (t != 0):
            turnover = float(np.sum(np.abs(target - w)))
            cost = transaction_cost * turnover
            w = target.copy()
        else:
            turnover = 0.0
            cost = 0.0

        turnovers[t] = turnover

        port_log_ret = float(np.dot(w, r[t]))
        values[t + 1] = values[t] * np.exp(port_log_ret - cost)

        # drift after the market move
        w = drift_weights(w, r[t])

    if return_turnover:
        return values, turnovers
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

def ew_long_short_cash(N: int, cash_idx: int):
    a_long = np.ones(N) / N
    a_short = np.zeros(N); a_short[cash_idx] = 1.0
    return np.concatenate([a_long, a_short]).astype(np.float32)

def equal_weight_market_neutral(N: int):
    w_long = np.ones(N) / N
    w_short = np.ones(N) / N
    return np.concatenate([w_long, w_short]).astype(np.float32)

def momentum_ls_action(obs_vec: np.ndarray, N: int, lookback: int, F: int, k: int = 5):
    # obs_vec = [window_flat, weights]
    k = max(1, N // 2)
    window_flat = obs_vec[:lookback * N * F]
    if F == 1:
        window = window_flat.reshape(lookback, N)
        r = window[-1]                 # last normalized return
        mom = window.sum(axis=0)       # lookback momentum proxy
    else:
        window = window_flat.reshape(lookback, N, F)
        r = window[-1, :, 0]           # feature 0 = return
        mom = window[:, :, 1].mean(axis=0) if F > 1 else r

    rank = np.argsort(mom)
    short_idx = rank[:k]
    long_idx = rank[-k:]

    a_long = np.zeros(N); a_short = np.zeros(N)
    a_long[long_idx] = 1.0 / k
    a_short[short_idx] = 1.0 / k
    return np.concatenate([a_long, a_short]).astype(np.float32)


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
    target_long_only = np.ones(N) / N

    # market-neutral equal books
    target_long = np.ones(N) / N
    target_short = np.ones(N) / N

    # --- Long-only baselines (still useful) ---
    ew_bh_vals = simulate_buy_and_hold(
        returns_aligned, init_weights=target_long_only, initial_value=1.0
    )

    ew_m_vals, ew_m_turnover = simulate_rebalance_strategy(
        returns_aligned,
        target_weights=target_long_only,
        rebalance_every=21,
        transaction_cost=tc,
        initial_value=1.0,
        return_turnover=True,
    )

    ls_ew_bh_vals = simulate_ls_buy_and_hold(
        returns_aligned,
        w_long_init=target_long,
        w_short_init=target_short,
        initial_value=1.0,
    )

    best_ticker, best_vals = compute_best_single_asset(returns_aligned)

    spy_vals = simulate_buy_and_hold(
        spy_aligned,
        init_weights=np.array([1.0]),
        initial_value=1.0,
    )

    # --- Market-neutral long/short baselines (coherent with your L/S env) ---
    ls_ew_monthly_vals, ls_ew_monthly_to = simulate_ls_rebalance_strategy(
        returns_aligned,
        w_long_target=target_long,
        w_short_target=target_short,
        rebalance_every=21,
        transaction_cost=tc,
        initial_value=1.0,
        return_turnover=True,
    )

    ls_mom_vals, ls_mom_to = simulate_ls_momentum(
        returns_aligned,
        lookback=60,
        k_frac=0.1,
        rebalance_every=21,
        transaction_cost=tc,
        initial_value=1.0,
        return_turnover=True,
    )

    # --- Print results ---
    print("\n=== Baseline Results (SP500 Universe) ===")
    print("EW Buy & Hold (true) final:", ew_bh_vals[-1])
    print("EW Monthly Rebalance final:", ew_m_vals[-1])
    print("SPY Buy & Hold final:", spy_vals[-1])
    print(f"Best Single Asset ({best_ticker}) final:", best_vals[-1])


    # --- Plot ---
    plot_baselines(
        {
            "EW Buy & Hold (long-only)": ew_bh_vals,
            "EW Monthly Rebalance (long-only)": ew_m_vals,
            "SPY Buy & Hold": spy_vals,
            f"Best Single ({best_ticker})": best_vals,
            "LS Equal-Weight Monthly (MN)": ls_ew_monthly_vals,
            "LS Momentum 60d (MN)": ls_mom_vals,
            "LS EW Buy&Hold (MN)": ls_ew_bh_vals,
        },
        title="Baseline Portfolio Value Curves (SP500 Universe)",
    )

    # Build metric summaries
    metrics = {
        "EW Buy&Hold (LO)": summary_metrics(ew_bh_vals),
        "EW Monthly (LO)": summary_metrics(ew_m_vals, turnover=ew_m_turnover),
        "SPY Buy&Hold": summary_metrics(spy_vals),
        f"Best Single ({best_ticker})": summary_metrics(best_vals),
        "LS EW Monthly (MN)": summary_metrics(ls_ew_monthly_vals, turnover=ls_ew_monthly_to),
        "LS Mom 60d (MN)": summary_metrics(ls_mom_vals, turnover=ls_mom_to),
        "LS EW Buy&Hold (MN)": summary_metrics(ls_ew_bh_vals),
    }

    print("\n=== Metrics (annualized, 252 trading days) ===")
    print_metrics_table(metrics)