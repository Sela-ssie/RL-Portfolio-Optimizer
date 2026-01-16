import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data.data_loader import prepare_data
from env.portfolio_env import PortfolioEnv


def run_random_episode(env: PortfolioEnv, max_steps: int | None = None):
    obs, info = env.reset(seed=42)

    rewards = []
    values = []
    turnovers = []
    weights_list = []

    done = False
    steps = 0

    # Track initial value if provided; otherwise start at env.portfolio_value if it exists
    if hasattr(env, "portfolio_value"):
        values.append(float(env.portfolio_value))

    while not done:
        action = env.action_space.sample()  # works for Discrete or Box

        obs, reward, done, truncated, info = env.step(action)
        done = done or truncated

        rewards.append(float(reward))

        # Try to record portfolio value from info or env
        pv = None
        if isinstance(info, dict) and "portfolio_value" in info:
            pv = info["portfolio_value"]
        elif hasattr(env, "portfolio_value"):
            pv = env.portfolio_value

        if pv is not None:
            values.append(float(pv))

        # Turnover isn’t in info dict; compute it if weights are exposed
        w = None
        if isinstance(info, dict) and "weights" in info:
            w = np.array(info["weights"], dtype=np.float32)
            weights_list.append(w)

        if len(weights_list) >= 2:
            turnovers.append(float(np.sum(np.abs(weights_list[-1] - weights_list[-2]))))

        steps += 1
        if max_steps is not None and steps >= max_steps:
            break

        # Hard safety stop
        if steps > 2_000_000:
            raise RuntimeError("Episode too long — possible termination bug.")

    return {
        "rewards": np.array(rewards, dtype=np.float64),
        "portfolio_values": np.array(values, dtype=np.float64) if len(values) else None,
        "turnovers": np.array(turnovers, dtype=np.float64) if len(turnovers) else None,
        "steps": steps,
    }


def assert_finite(name: str, arr: np.ndarray | None):
    if arr is None:
        return
    if not np.all(np.isfinite(arr)):
        bad = np.where(~np.isfinite(arr))[0][:10]
        raise ValueError(f"{name} contains non-finite values. First bad indices: {bad}")


def main():
    from data.data_loader import load_price_data, prices_to_log_returns, prepare_data

    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]

    # --- Raw log returns ---
    prices = load_price_data(tickers)
    prices = prices.ffill().dropna()
    raw_returns = prices_to_log_returns(prices)

    print("=== Raw Log Returns ===")
    print("Mean (per asset):")
    print(raw_returns.mean())
    print("\nStd (per asset):")
    print(raw_returns.std())
    print("\nGlobal mean:", raw_returns.values.mean())
    print("Global std:", raw_returns.values.std())

    # --- Normalized returns ---
    train_returns, test_returns, mean, std = prepare_data(tickers)

    print("\n=== Normalized Log Returns (Train) ===")
    print("Mean (per asset):")
    print(train_returns.mean())
    print("\nStd (per asset):")
    print(train_returns.std())
    print("\nGlobal mean:", train_returns.values.mean())
    print("Global std:", train_returns.values.std())

    # --- Environment sanity check with random policy ---

    env = PortfolioEnv(
        returns=train_returns.values if isinstance(train_returns, pd.DataFrame) else np.asarray(train_returns),
        lookback_window_size=30,
        transaction_cost=0.001,
        initial_value=1.0,
    )

    print("=== Environment ===")
    print("action_space:", env.action_space)
    print("observation_space:", env.observation_space)
    print("num_assets:", getattr(env, "num_assets", "unknown"))
    print()

    results = run_random_episode(env)

    rewards = results["rewards"]
    values = results["portfolio_values"]
    turnovers = results["turnovers"]

    # Checks
    assert_finite("rewards", rewards)
    assert_finite("portfolio_values", values)
    assert_finite("turnovers", turnovers)

    print("=== Episode Summary (Random Policy) ===")
    print("steps:", results["steps"])
    print("reward: mean =", rewards.mean(), "std =", rewards.std(), "min =", rewards.min(), "max =", rewards.max())
    print("reward: total =", rewards.sum())

    if values is not None and len(values) > 0:
        print("portfolio_value: start =", values[0], "end =", values[-1])
        # Approx total return (since we use exp updates, PV is multiplicative)
        total_return = (values[-1] / values[0]) - 1.0 if values[0] != 0 else np.nan
        print("portfolio_value: total_return =", total_return)

    if turnovers is not None and len(turnovers) > 0:
        print("turnover: mean =", turnovers.mean(), "max =", turnovers.max())

    # Quick plot
    if values is not None and len(values) > 1:
        plt.figure()
        plt.plot(values)
        plt.title("Sanity Check: Portfolio Value (Random Policy)")
        plt.xlabel("Step")
        plt.ylabel("Portfolio Value")
        plt.show()

    print("\nSanity check passed (no NaNs/infs).")


if __name__ == "__main__":
    main()
