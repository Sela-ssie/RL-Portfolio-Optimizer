import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from env.portfolio_env import PortfolioEnv


def run_random_episode(env: PortfolioEnv, max_steps: int | None = None):
    obs, info = env.reset(seed=42)

    rewards = []
    values = []
    turnovers = []
    turnovers_long = []
    turnovers_short = []

    done = False
    steps = 0

    # initial PV
    if hasattr(env, "portfolio_value"):
        values.append(float(env.portfolio_value))

    while not done:
        action = env.action_space.sample()

        obs, reward, done, truncated, info = env.step(action)
        done = bool(done or truncated)

        rewards.append(float(reward))

        # portfolio value
        pv = None
        if isinstance(info, dict) and "portfolio_value" in info:
            pv = info["portfolio_value"]
        elif hasattr(env, "portfolio_value"):
            pv = env.portfolio_value
        if pv is not None:
            values.append(float(pv))

        # turnover: prefer env-provided gross turnover (correct for long/short)
        if isinstance(info, dict):
            if "turnover" in info:
                turnovers.append(float(info["turnover"]))
            # optional diagnostics
            if "turnover_long" in info:
                turnovers_long.append(float(info["turnover_long"]))
            if "turnover_short" in info:
                turnovers_short.append(float(info["turnover_short"]))

        steps += 1
        if max_steps is not None and steps >= max_steps:
            break
        if steps > 2_000_000:
            raise RuntimeError("Episode too long — possible termination bug.")

    out = {
        "rewards": np.array(rewards, dtype=np.float64),
        "portfolio_values": np.array(values, dtype=np.float64) if len(values) else None,
        "turnovers": np.array(turnovers, dtype=np.float64) if len(turnovers) else None,
        "steps": steps,
    }
    if len(turnovers_long):
        out["turnovers_long"] = np.array(turnovers_long, dtype=np.float64)
    if len(turnovers_short):
        out["turnovers_short"] = np.array(turnovers_short, dtype=np.float64)
    return out

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

    # raw_returns: DataFrame of true log returns
    # train_returns: DataFrame of normalized returns from prepare_data()

    common_idx = raw_returns.index.intersection(train_returns.index)
    raw_train = raw_returns.loc[common_idx]
    obs_train = train_returns.loc[common_idx]

    needed = 30 + 252 + 1
    assert len(raw_train) >= needed, f"Not enough data: have {len(raw_train)}, need >= {needed}"

    env = PortfolioEnv(
        returns_raw=raw_train.values,
        obs_features=obs_train.values,
        lookback_window_size=30,
        transaction_cost=0.001,
        initial_value=1.0,
        turnover_penalty=0.0,
        drawdown_penalty=0.0,
        random_start=False,
        episode_length=252,
    )

    assert env.action_space.shape[0] == 2 * env.num_assets, (
        f"Expected long/short action dim 2N, got {env.action_space.shape[0]} vs 2*{env.num_assets}"
    )

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
        print("turnover (gross, env): mean =", turnovers.mean(), "max =", turnovers.max())

    if "turnovers_long" in results and len(results["turnovers_long"]) > 0:
        tl = results["turnovers_long"]
        ts = results["turnovers_short"]
        print("turnover_long: mean =", tl.mean(), "max =", tl.max())
        print("turnover_short: mean =", ts.mean(), "max =", ts.max())

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
