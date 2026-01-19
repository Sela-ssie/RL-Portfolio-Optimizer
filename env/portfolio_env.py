import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PortfolioEnv(gym.Env):
    """
    A custom environment for portfolio management.
    The agent allocates funds among multiple assets to maximize returns.
    """

    def __init__(
        self,
        returns_raw: np.ndarray,
        returns_obs: np.ndarray | None = None,
        lookback_window_size: int = 30,
        transaction_cost: float = 0.001,
        initial_value: float = 1.0,
        turnover_penalty: float = 0.0,
        drawdown_penalty: float = 0.0,
    ):
        super().__init__()

        self.num_assets = returns_raw.shape[1]
        self.initial_value = initial_value

        # Raw returns drive portfolio dynamics + reward
        self.returns_raw = np.asarray(returns_raw, dtype=np.float64)

        # Observation returns drive the state seen by the agent (often normalized)
        if returns_obs is None:
            self.returns_obs = self.returns_raw
        else:
            self.returns_obs = np.asarray(returns_obs, dtype=np.float64)

        # Basic sanity check: must align in time and assets
        if self.returns_obs.shape != self.returns_raw.shape:
            raise ValueError(
                f"returns_obs shape {self.returns_obs.shape} must match returns_raw shape {self.returns_raw.shape}"
            )

        self.lookback_window_size = lookback_window_size
        self.transaction_cost = transaction_cost
        self.turnover_penalty = float(turnover_penalty)
        self.drawdown_penalty = float(drawdown_penalty)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.t = self.lookback_window_size
        self.portfolio_value = self.initial_value
        self.weights = np.ones(self.num_assets, dtype=np.float64) / self.num_assets
        self.peak_value = float(self.portfolio_value)
        self.prev_drawdown = 0.0

        return self._get_observation(), {}
    
    def step(self, action):
        assert action.shape == (self.num_assets,)
        prev_weights = self.weights.copy()

        self.weights = self._action_to_weights(action)

        # Turnover and transaction cost
        turnover = float(np.sum(np.abs(self.weights - prev_weights)))
        tx_cost = turnover * self.transaction_cost

        # Portfolio log return at time t
        asset_returns = self.returns_raw[self.t]
        portfolio_log_return = float(np.dot(self.weights, asset_returns))

        # Update portfolio value
        self.portfolio_value *= float(np.exp(portfolio_log_return - tx_cost))

        # Drawdown tracking
        self.peak_value = max(float(self.peak_value), float(self.portfolio_value))
        drawdown = float(1.0 - (self.portfolio_value / (self.peak_value + 1e-12)))
        dd_increase = float(max(0.0, drawdown - float(self.prev_drawdown)))
        self.prev_drawdown = drawdown

        # Reward: return - transaction_cost - lambda*turnover - beta*drawdown_increase
        reward = (
            portfolio_log_return
            - tx_cost
            - self.turnover_penalty * turnover
            - self.drawdown_penalty * dd_increase
        )

        self.t += 1
        done = self.t >= len(self.returns_raw) - 1

        return self._get_observation(), reward, done, False, {
            "portfolio_value": float(self.portfolio_value),
            "weights": self.weights,
            "turnover": turnover,
            "tx_cost": float(tx_cost),
            "portfolio_log_return": portfolio_log_return,
            "drawdown": drawdown,
            "dd_increase": dd_increase,
        }

    def _get_observation(self):
        returns_window = self.returns_obs[self.t - self.lookback_window_size : self.t]
        obs = np.concatenate(
            [returns_window.flatten(), self.weights],
            axis=0
        )
        return obs.astype(np.float32)
    
    def _action_to_weights(self, action):
        action = np.clip(action, 0.0, 1.0)
        weights = action / (np.sum(action) + 1e-8)
        return weights