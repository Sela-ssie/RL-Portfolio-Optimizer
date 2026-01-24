import gymnasium as gym
from gymnasium import spaces
import numpy as np


class PortfolioEnv(gym.Env):
    """
    Portfolio environment.
    - returns_raw: (T, N) raw log returns used for portfolio dynamics
    - obs_features: (T, N) or (T, N, F) observations seen by agent (typically normalized features)
    """

    def __init__(
        self,
        returns_raw: np.ndarray,
        returns_obs: np.ndarray | None = None,     # keep backward compat
        obs_features: np.ndarray | None = None,    # NEW: used by your new training script
        lookback_window_size: int = 30,
        transaction_cost: float = 0.001,
        initial_value: float = 1.0,
        turnover_penalty: float = 0.0,
        drawdown_penalty: float = 0.0,
        random_start: bool = True,
        episode_length: int | None = None,
        dsr_eta: float | None = None,              # NEW: if set, reward = Differential Sharpe Ratio
        dsr_eps: float = 1e-8,
    ):
        super().__init__()

        self.returns_raw = np.asarray(returns_raw, dtype=np.float64)
        if self.returns_raw.ndim != 2:
            raise ValueError(f"returns_raw must be (T, N). Got shape {self.returns_raw.shape}")

        self.num_assets = self.returns_raw.shape[1]
        self.initial_value = float(initial_value)
        self.random_start = bool(random_start)
        self.episode_length = episode_length

        self.lookback_window_size = int(lookback_window_size)
        self.transaction_cost = float(transaction_cost)
        self.turnover_penalty = float(turnover_penalty)
        self.drawdown_penalty = float(drawdown_penalty)

        # --- Observations (2D returns or 3D features) ---
        if obs_features is not None and returns_obs is not None:
            raise ValueError("Pass only one of obs_features or returns_obs (not both).")

        if obs_features is not None:
            self.obs = np.asarray(obs_features, dtype=np.float64)
        elif returns_obs is not None:
            self.obs = np.asarray(returns_obs, dtype=np.float64)
        else:
            self.obs = self.returns_raw

        if self.obs.shape[0] != self.returns_raw.shape[0] or self.obs.shape[1] != self.returns_raw.shape[1]:
            raise ValueError(
                f"obs must align with returns_raw in (T, N). "
                f"obs={self.obs.shape}, returns_raw={self.returns_raw.shape}"
            )

        # obs can be (T, N) or (T, N, F)
        if self.obs.ndim == 2:
            self.num_features = 1
        elif self.obs.ndim == 3:
            self.num_features = self.obs.shape[2]
        else:
            raise ValueError(f"obs must be 2D or 3D. Got {self.obs.ndim}D with shape {self.obs.shape}")

        # --- Spaces ---
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2 * self.num_assets,), dtype=np.float32)

        obs_dim = self.lookback_window_size * self.num_assets * self.num_features + self.num_assets
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # --- DSR reward ---
        self.dsr_eta = None if dsr_eta is None else float(dsr_eta)
        self.dsr_eps = float(dsr_eps)

        # Running stats for DSR (reset every episode)
        self._dsr_A = 0.0  # EMA of returns
        self._dsr_B = 0.0  # EMA of returns^2

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.portfolio_value = float(self.initial_value)
        # Start market-neutral: 0.5 long, 0.5 short, equal-weighted
        N = self.num_assets
        self.w_long = np.ones(N, dtype=np.float64) / N
        self.w_short = np.ones(N, dtype=np.float64) / N
        self.weights = 0.5 * self.w_long - 0.5 * self.w_short  # signed exposures

        # Track previous legs for turnover/tx-cost
        self.prev_w_long = self.w_long.copy()
        self.prev_w_short = self.w_short.copy()

        self.peak_value = float(self.portfolio_value)
        self.prev_drawdown = 0.0

        # Reset DSR stats each episode (common + stable)
        self._dsr_A = 0.0
        self._dsr_B = 0.0

        # Choose start index
        start_min = self.lookback_window_size

        if self.random_start:
            if self.episode_length is None:
                start_max = len(self.returns_raw) - 2
            else:
                start_max = (len(self.returns_raw) - 2) - (self.episode_length - 1)

            start_max = max(start_min, start_max)
            self.t = int(self.np_random.integers(start_min, start_max + 1))
        else:
            self.t = start_min

        self.start_t = int(self.t)
        return self._get_observation(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 2 * self.num_assets:
            raise ValueError(f"Action must have shape ({2*self.num_assets},). Got {action.shape}")

        # Convert action -> signed exposure + underlying long/short legs
        prev_w_long = getattr(self, "prev_w_long", None)
        prev_w_short = getattr(self, "prev_w_short", None)

        if prev_w_long is None or prev_w_short is None:
            N = self.num_assets
            prev_w_long = np.ones(N, dtype=np.float64) / N
            prev_w_short = np.ones(N, dtype=np.float64) / N

        self.weights, self.w_long, self.w_short = self._action_to_weights(action)

        turnover_long = float(np.sum(np.abs(self.w_long - prev_w_long)))
        turnover_short = float(np.sum(np.abs(self.w_short - prev_w_short)))
        turnover = turnover_long + turnover_short

        tx_cost = turnover * self.transaction_cost

        self.prev_w_long = self.w_long.copy()
        self.prev_w_short = self.w_short.copy()

        # Portfolio log return at time t (raw)
        asset_returns = self.returns_raw[self.t]
        portfolio_log_return = float(np.dot(self.weights, asset_returns))

        # Update PV using *net* log return after tx_cost
        net_log_return = portfolio_log_return - tx_cost
        self.portfolio_value *= float(np.exp(net_log_return))

        # Drawdown tracking
        self.peak_value = max(self.peak_value, self.portfolio_value)
        drawdown = float(1.0 - (self.portfolio_value / (self.peak_value + 1e-12)))
        dd_increase = float(max(0.0, drawdown - self.prev_drawdown))
        self.prev_drawdown = drawdown

        # Optional penalties (you can keep these even with DSR)
        penalty = self.turnover_penalty * turnover + self.drawdown_penalty * dd_increase

        # Reward
        if self.dsr_eta is None:
            # old reward: net return minus extra penalties
            reward = net_log_return - penalty
        else:
            # Differential Sharpe Ratio on net return (optionally minus penalty)
            reward = self._dsr_reward(net_log_return) - penalty

        # advance time
        self.t += 1

        # termination
        if self.episode_length is None:
            done = self.t >= (len(self.returns_raw) - 1)
        else:
            steps_taken = self.t - self.start_t
            done = (steps_taken >= self.episode_length) or (self.t >= (len(self.returns_raw) - 1))

        if done and self.t > (len(self.obs) - 1):
            self.t = len(self.obs) - 1

        info = {
            "portfolio_value": float(self.portfolio_value),
            "weights": self.weights.copy(),
            "turnover": turnover,
            "tx_cost": float(tx_cost),
            "portfolio_log_return": float(portfolio_log_return),
            "net_log_return": float(net_log_return),
            "drawdown": drawdown,
            "dd_increase": dd_increase,
            "w_long": self.w_long.copy(),
            "w_short": self.w_short.copy(),
            "turnover_long": turnover_long,
            "turnover_short": turnover_short,
        }
        return self._get_observation(), float(reward), bool(done), False, info

    def _get_observation(self):
        w = self.lookback_window_size
        if self.obs.ndim == 2:
            window = self.obs[self.t - w : self.t]              # (w, N)
            flat = window.reshape(-1)                           # (w*N,)
        else:
            window = self.obs[self.t - w : self.t, :, :]        # (w, N, F)
            flat = window.reshape(-1)                           # (w*N*F,)

        obs = np.concatenate([flat, self.weights], axis=0)
        return obs.astype(np.float32)

    def _action_to_weights(self, action: np.ndarray):
        """
        Market-neutral long-short mapping.

        action = [a_long (N), a_short (N)], all >= 0
        w_long  = a_long / sum(a_long)
        w_short = a_short / sum(a_short)

        Final signed exposure uses fixed gross:
        w = 0.5*w_long - 0.5*w_short
        So: sum(w)=0 and long gross=0.5, short gross=0.5.
        """
        N = self.num_assets
        a = np.clip(action, 0.0, 1.0)

        a_long = a[:N]
        a_short = a[N:]

        sL = float(a_long.sum())
        sS = float(a_short.sum())

        if sL <= 0.0:
            w_long = np.ones(N, dtype=np.float64) / N
        else:
            w_long = (a_long / (sL + 1e-8)).astype(np.float64)

        if sS <= 0.0:
            w_short = np.ones(N, dtype=np.float64) / N
        else:
            w_short = (a_short / (sS + 1e-8)).astype(np.float64)

        w = 0.5 * w_long - 0.5 * w_short
        return w.astype(np.float64), w_long, w_short

    def _dsr_reward(self, r_t: float) -> float:
        """
        Differential Sharpe Ratio (Moody et al. style) using EMA stats:
          A_t = (1-eta)A_{t-1} + eta*r_t
          B_t = (1-eta)B_{t-1} + eta*r_t^2

        DSR_t = (B_{t-1}*dA - 0.5*A_{t-1}*dB) / (B_{t-1} - A_{t-1}^2)^(3/2)

        where dA = A_t - A_{t-1}, dB = B_t - B_{t-1}.
        """
        eta = self.dsr_eta
        A0 = self._dsr_A
        B0 = self._dsr_B

        A1 = (1.0 - eta) * A0 + eta * r_t
        B1 = (1.0 - eta) * B0 + eta * (r_t * r_t)

        dA = A1 - A0
        dB = B1 - B0

        denom = (B0 - A0 * A0)
        denom = max(denom, self.dsr_eps)
        dsr = (B0 * dA - 0.5 * A0 * dB) / (denom ** 1.5)

        self._dsr_A = A1
        self._dsr_B = B1
        return float(dsr)