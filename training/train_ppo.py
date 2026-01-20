# training/train_ppo.py
# Run from project root:
#   python -m training.train_ppo --tickers AAPL MSFT GOOG AMZN META
#
# This script trains PPO with a Dirichlet policy (actions are valid portfolio weights).

import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Dirichlet

from data.data_loader import load_price_data, prices_to_log_returns, prepare_data
from env.portfolio_env import PortfolioEnv
from evaluation.metrics import summary_metrics, print_metrics_table


TRADING_DAYS = 252


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def align_raw_to_split(raw_returns: pd.DataFrame, split_returns: pd.DataFrame) -> pd.DataFrame:
    """Align raw returns to the same dates as the normalized split."""
    return raw_returns.loc[split_returns.index]


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    rollout_steps: int = 2048
    minibatch_size: int = 256
    update_epochs: int = 10
    lr: float = 3e-4

    total_steps: int = 200_000


class ActorCritic(nn.Module):
    """
    Dirichlet policy for portfolio weights.
    Given obs -> alpha parameters (positive), sample weights ~ Dirichlet(alpha).
    Value head outputs V(s).
    """

    def __init__(self, obs_dim: int, num_assets: int, hidden: int = 256):
        super().__init__()
        self.num_assets = num_assets

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        # Policy head outputs concentration parameters (alpha > 0)
        self.policy_head = nn.Linear(hidden, num_assets)

        # Value head
        self.value_head = nn.Linear(hidden, 1)

        self.softplus = nn.Softplus()

    def forward(self, obs: torch.Tensor):
        x = self.shared(obs)
        logits = self.policy_head(x)
        # alpha must be strictly positive; add small epsilon for stability
        alpha = self.softplus(logits) + 1e-3
        value = self.value_head(x).squeeze(-1)
        return alpha, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha, value = self.forward(obs)
        dist = Dirichlet(alpha)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp, value

    @torch.no_grad()
    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        # Mean of Dirichlet = alpha / sum(alpha)
        alpha, _ = self.forward(obs)
        return alpha / alpha.sum(dim=-1, keepdim=True)


def compute_gae(rewards, values, dones, gamma, lam):
    """
    rewards: (T,)
    values: (T+1,) bootstrap with last value
    dones: (T,) 1 if done else 0
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float64)
    gae = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
    returns = adv + values[:-1]
    return adv, returns


def rollout(env: PortfolioEnv, model: ActorCritic, device: torch.device, steps: int):
    obs, _ = env.reset()

    obs_dim = env.observation_space.shape[0]
    num_assets = env.action_space.shape[0]

    obs_buf = np.zeros((steps, obs_dim), dtype=np.float32)
    act_buf = np.zeros((steps, num_assets), dtype=np.float32)
    logp_buf = np.zeros((steps,), dtype=np.float32)
    rew_buf = np.zeros((steps,), dtype=np.float32)
    done_buf = np.zeros((steps,), dtype=np.float32)
    val_buf = np.zeros((steps + 1,), dtype=np.float32)

    for t in range(steps):
        obs_buf[t] = obs

        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action_t, logp_t, value_t = model.act(obs_t)

        action = action_t.squeeze(0).cpu().numpy().astype(np.float32)
        logp = float(logp_t.item())
        value = float(value_t.item())

        act_buf[t] = action
        logp_buf[t] = logp
        val_buf[t] = value

        obs, reward, done, truncated, info = env.step(action)
        done_flag = float(done or truncated)

        rew_buf[t] = float(reward)
        done_buf[t] = done_flag

        if done_flag:
            obs, _ = env.reset()

    # Bootstrap value for last observation
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    _, v_last = model.forward(obs_t)
    val_buf[-1] = float(v_last.item())

    adv, ret = compute_gae(
        rewards=rew_buf,
        values=val_buf,
        dones=done_buf,
        gamma=float(cfg.gamma),
        lam=float(cfg.gae_lambda),
    )

    # Normalize advantages (common PPO trick)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    batch = {
        "obs": obs_buf,
        "act": act_buf,
        "logp": logp_buf,
        "adv": adv.astype(np.float32),
        "ret": ret.astype(np.float32),
        "val": val_buf[:-1],
    }
    return batch


def ppo_update(model: ActorCritic, optimizer: torch.optim.Optimizer, batch, device: torch.device, cfg: PPOConfig):
    obs = torch.tensor(batch["obs"], dtype=torch.float32, device=device)
    act = torch.tensor(batch["act"], dtype=torch.float32, device=device)
    old_logp = torch.tensor(batch["logp"], dtype=torch.float32, device=device)
    adv = torch.tensor(batch["adv"], dtype=torch.float32, device=device)
    ret = torch.tensor(batch["ret"], dtype=torch.float32, device=device)

    n = obs.shape[0]
    idxs = np.arange(n)

    for _ in range(cfg.update_epochs):
        np.random.shuffle(idxs)
        for start in range(0, n, cfg.minibatch_size):
            mb = idxs[start : start + cfg.minibatch_size]

            alpha, value = model.forward(obs[mb])
            dist = Dirichlet(alpha)

            logp = dist.log_prob(act[mb])
            entropy = dist.entropy().mean()

            ratio = torch.exp(logp - old_logp[mb])
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv[mb]
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = ((value - ret[mb]) ** 2).mean()

            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()


@torch.no_grad()
def evaluate_policy(env: PortfolioEnv, model: ActorCritic, device: torch.device):
    obs, _ = env.reset()
    done = False

    values = [float(env.portfolio_value)]
    turnovers = []

    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action = model.act_deterministic(obs_t).squeeze(0).cpu().numpy().astype(np.float32)

        obs, reward, done, truncated, info = env.step(action)
        done = done or truncated

        if isinstance(info, dict):
            if "portfolio_value" in info:
                values.append(float(info["portfolio_value"]))
            if "turnover" in info:
                turnovers.append(float(info["turnover"]))

        # safety
        if len(values) > 5_000_000:
            raise RuntimeError("Evaluation episode too long.")

    values = np.asarray(values, dtype=np.float64)
    turnovers = np.asarray(turnovers, dtype=np.float64) if len(turnovers) else None
    return values, turnovers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "GOOG", "AMZN", "META"])
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2024-01-01")
    parser.add_argument("--train_ratio", type=float, default=0.7)

    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--transaction_cost", type=float, default=0.001)
    parser.add_argument("--turnover_penalty", type=float, default=0.0)
    parser.add_argument("--drawdown_penalty", type=float, default=0.0)

    parser.add_argument("--total_steps", type=int, default=200_000)
    parser.add_argument("--rollout_steps", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_path", type=str, default="checkpoints/ppo_dirichlet.pt")

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # --- Raw log returns for PV/reward ---
    prices = load_price_data(args.tickers, start=args.start, end=args.end)
    prices = prices.ffill().dropna()
    raw_returns = prices_to_log_returns(prices)

    # --- Normalized returns for observations (train/test split) ---
    train_obs, test_obs, mean, std = prepare_data(
        args.tickers, start=args.start, end=args.end, train_ratio=args.train_ratio
    )

    # Align raw to split dates
    train_raw = align_raw_to_split(raw_returns, train_obs)
    test_raw = align_raw_to_split(raw_returns, test_obs)

    # Make envs
    train_env = PortfolioEnv(
        returns_raw=train_raw.values,
        returns_obs=train_obs.values,
        lookback_window_size=args.lookback,
        transaction_cost=args.transaction_cost,
        initial_value=1.0,
        turnover_penalty=args.turnover_penalty,
        drawdown_penalty=args.drawdown_penalty,
    )

    test_env = PortfolioEnv(
        returns_raw=test_raw.values,
        returns_obs=test_obs.values,
        lookback_window_size=args.lookback,
        transaction_cost=args.transaction_cost,
        initial_value=1.0,
        turnover_penalty=args.turnover_penalty,
        drawdown_penalty=args.drawdown_penalty,
    )

    # PPO config
    cfg = PPOConfig(
        rollout_steps=args.rollout_steps,
        total_steps=args.total_steps,
    )

    obs_dim = train_env.observation_space.shape[0]
    num_assets = train_env.action_space.shape[0]

    model = ActorCritic(obs_dim=obs_dim, num_assets=num_assets).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Training loop
    steps_done = 0
    while steps_done < cfg.total_steps:
        batch = rollout(train_env, model, device, cfg.rollout_steps)
        ppo_update(model, optimizer, batch, device, cfg)
        steps_done += cfg.rollout_steps

        # quick progress print (train)
        train_vals, train_to = evaluate_policy(train_env, model, device)
        m = summary_metrics(train_vals, turnover=train_to)
        print(f"[steps={steps_done}] train final={m['final']:.4f} sharpe={m['sharpe']:.4f} maxDD={m['max_dd']:.4f} avgTO={m['avg_turnover']:.4f}")

    # Final evaluation (test)
    test_vals, test_to = evaluate_policy(test_env, model, device)

    results = {
        "PPO (train)": summary_metrics(*evaluate_policy(train_env, model, device)),
        "PPO (test)": summary_metrics(test_vals, turnover=test_to),
    }

    print("\n=== PPO Results ===")
    print_metrics_table(results)

    # Save
    import os
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "tickers": args.tickers,
            "start": args.start,
            "end": args.end,
            "train_ratio": args.train_ratio,
            "lookback": args.lookback,
            "transaction_cost": args.transaction_cost,
            "turnover_penalty": args.turnover_penalty,
            "drawdown_penalty": args.drawdown_penalty,
        },
        args.save_path,
    )
    print(f"\nSaved checkpoint to: {args.save_path}")