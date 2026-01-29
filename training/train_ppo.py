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

from data.data_loader import load_price_data, prices_to_log_returns, get_sp500_tickers, load_price_data_batched
from env.portfolio_env import PortfolioEnv
from evaluation.baselines import momentum_ls_action, equal_weight_market_neutral
from evaluation.metrics import summary_metrics, print_metrics_table


TRADING_DAYS = 252

def add_cash_asset(returns_df: pd.DataFrame, name: str = "CASH") -> pd.DataFrame:
    out = returns_df.copy()
    out[name] = 0.0
    return out


def build_feature_tensor(
    log_returns: pd.DataFrame,
    mom_windows=(20, 60),
    vol_windows=(20, 60),
    ) -> tuple[np.ndarray, pd.Index]:
    """
    Build features from log returns.
    Returns:
      feats: (T, N, F) float32
      index: aligned date index after trimming NaNs from rolling windows
    Feature channels:
      [r_t, mom20, mom60, vol20, vol60]  (depending on windows)
    """
    r = log_returns

    feats = [r]  # raw return feature

    for w in mom_windows:
        feats.append(r.rolling(w).sum())

    for w in vol_windows:
        feats.append(r.rolling(w).std(ddof=0))

    # stack into (T, N, F)
    F = len(feats)
    T, N = r.shape
    out = np.zeros((T, N, F), dtype=np.float64)
    for i, df in enumerate(feats):
        out[:, :, i] = df.values

    # trim initial rows with NaNs from rolling features
    good = np.isfinite(out).all(axis=(1, 2))
    if not np.any(good):
        raise ValueError("All feature rows contain NaNs/Infs (check rolling windows vs data length).")

    first = int(np.argmax(good))  # first True
    out = out[first:]
    idx = r.index[first:]
    return out.astype(np.float32), idx


def normalize_feature_tensor(train_feats: np.ndarray, test_feats: np.ndarray):
    """
    Normalize by train stats only.
    Normalize per-channel globally over (time, assets).
    """
    mean = train_feats.reshape(-1, train_feats.shape[-1]).mean(axis=0)
    std = train_feats.reshape(-1, train_feats.shape[-1]).std(axis=0) + 1e-8
    train_n = (train_feats - mean) / std
    test_n = (test_feats - mean) / std
    return train_n.astype(np.float32), test_n.astype(np.float32), mean, std


def split_time_aligned(arr: np.ndarray, train_ratio: float, lookback: int):
    if len(arr) < lookback + 5:
        raise ValueError("Not enough data after trimming features for the chosen lookback.")
    split = int(len(arr) * train_ratio)
    return arr[:split], arr[split:]

def set_seed(seed: int):
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    Market-neutral long-short via two Dirichlets:
      a_long  ~ Dirichlet(alpha_long)   (N,)
      a_short ~ Dirichlet(alpha_short)  (N,)
    Action returned to env is concat: [a_long, a_short] (2N,)

    Env will map that to weights:
      w = (L/2)*a_long - (L/2)*a_short
    """

    def __init__(self, obs_dim: int, num_assets: int, hidden: int = 256):
        super().__init__()
        self.num_assets = num_assets  # N (true assets)

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        # Output 2N concentrations: first N for long, next N for short
        self.policy_head = nn.Linear(hidden, 2 * num_assets)

        self.value_head = nn.Linear(hidden, 1)

        self.softplus = nn.Softplus()

    def forward(self, obs: torch.Tensor):
        x = self.shared(obs)
        logits = self.policy_head(x)  # (B, 2N)

        logits_long, logits_short = torch.split(logits, self.num_assets, dim=-1)

        alpha_long = self.softplus(logits_long) + 1e-3
        alpha_short = self.softplus(logits_short) + 1e-3

        value = self.value_head(x).squeeze(-1)
        return alpha_long, alpha_short, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha_long, alpha_short, value = self.forward(obs)

        dist_long = Dirichlet(alpha_long)
        dist_short = Dirichlet(alpha_short)

        a_long = dist_long.sample()
        a_short = dist_short.sample()

        # PPO needs log_prob of the actual sampled action
        logp = dist_long.log_prob(a_long) + dist_short.log_prob(a_short)

        action = torch.cat([a_long, a_short], dim=-1)  # (B, 2N)
        return action, logp, value

    @torch.no_grad()
    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        # Dirichlet mean = alpha / sum(alpha)
        alpha_long, alpha_short, _ = self.forward(obs)

        a_long = alpha_long / alpha_long.sum(dim=-1, keepdim=True)
        a_short = alpha_short / alpha_short.sum(dim=-1, keepdim=True)

        return torch.cat([a_long, a_short], dim=-1)  # (B, 2N)


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


def rollout(env: PortfolioEnv, model: ActorCritic, device: torch.device, steps: int, cfg: PPOConfig, seed: int | None = None):
    obs, _ = env.reset(seed=seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]  # = 2N for long/short

    obs_buf = np.zeros((steps, obs_dim), dtype=np.float32)
    act_buf = np.zeros((steps, act_dim), dtype=np.float32)
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
            # Do NOT reseed here; reseeding every episode can repeat the same random starts.
            obs, _ = env.reset()

    # Bootstrap value for last observation
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    _, _, v_last = model.forward(obs_t)
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
    assert act.shape[1] == 2 * model.num_assets, (
        f"Action dim mismatch: act has {act.shape[1]}, expected {2*model.num_assets}"
    )
    old_logp = torch.tensor(batch["logp"], dtype=torch.float32, device=device)
    adv = torch.tensor(batch["adv"], dtype=torch.float32, device=device)
    ret = torch.tensor(batch["ret"], dtype=torch.float32, device=device)

    n = obs.shape[0]
    idxs = np.arange(n)

    for _ in range(cfg.update_epochs):
        np.random.shuffle(idxs)
        for start in range(0, n, cfg.minibatch_size):
            mb = idxs[start : start + cfg.minibatch_size]

            alpha_long, alpha_short, value = model.forward(obs[mb])

            # split actions back into (long, short)
            N = model.num_assets
            a_long = act[mb, :N]
            a_short = act[mb, N:]

            dist_long = Dirichlet(alpha_long)
            dist_short = Dirichlet(alpha_short)

            logp = dist_long.log_prob(a_long) + dist_short.log_prob(a_short)
            entropy = (dist_long.entropy() + dist_short.entropy()).mean()

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
def evaluate_policy(env, model, device, policy="ppo"):
    obs, _ = env.reset()
    done = False
    values = [float(env.portfolio_value)]
    turnovers = []

    N = env.num_assets
    F = getattr(env, "num_features", 1)
    lookback = env.lookback_window_size

    while not done:
        if policy == "ppo":
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action = model.act_deterministic(obs_t).squeeze(0).cpu().numpy().astype(np.float32)
        elif policy == "ew":
            action = equal_weight_market_neutral(N)
        elif policy == "mom":
            action = momentum_ls_action(obs, N=N, lookback=lookback, F=F, k=min(5, N))
        else:
            raise ValueError("unknown policy")

        obs, reward, done, truncated, info = env.step(action)
        done = done or truncated

        if "portfolio_value" in info: values.append(float(info["portfolio_value"]))
        if "turnover" in info: turnovers.append(float(info["turnover"]))

    return np.asarray(values, dtype=np.float64), (np.asarray(turnovers) if turnovers else None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "GOOG", "AMZN", "META"])
    parser.add_argument("--sp500", action="store_true", help="Use the full S&P 500 universe as the asset set")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2024-01-01")
    parser.add_argument("--batch_size", type=int, default=80, help="Batch size for Yahoo downloads when using --sp500")
    parser.add_argument("--min_frac", type=float, default=0.98, help="Keep tickers with at least this fraction of non-NaN prices")
    parser.add_argument("--train_ratio", type=float, default=0.7)

    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--episode_length", type=int, default=252, help="Train episode length (steps). Use ~252 for 1 trading year.")
    parser.add_argument("--transaction_cost", type=float, default=0.001)
    parser.add_argument("--turnover_penalty", type=float, default=0.0)
    parser.add_argument("--drawdown_penalty", type=float, default=0.0)

    parser.add_argument("--total_steps", type=int, default=200_000)
    parser.add_argument("--rollout_steps", type=int, default=2048)
    parser.add_argument("--eval_every", type=int, default=5, help="Evaluate on test every N rollouts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_path", type=str, default="checkpoints/ppo_dirichlet.pt")

    parser.add_argument("--add_cash", action="store_true", help="Add a CASH asset with 0 return")
    parser.add_argument("--mom_windows", nargs="*", type=int, default=[20, 60], help="Momentum windows in days")
    parser.add_argument("--vol_windows", nargs="*", type=int, default=[20, 60], help="Volatility windows in days")
    parser.add_argument("--dsr_eta", type=float, default=0.01, help="Smoothing factor for DSR reward stats")

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # -----------------------------
    # Data loading: raw returns + feature tensor + split
    # -----------------------------
    if args.sp500:
        tickers = get_sp500_tickers()
        print("Total SP500 tickers:", len(tickers))

        prices = load_price_data_batched(
            tickers,
            start=args.start,
            end=args.end,
            batch_size=args.batch_size,
        )

        # Keep assets with sufficient data
        min_count = int(args.min_frac * len(prices))
        prices = prices.dropna(axis=1, thresh=min_count)
        prices = prices.ffill().dropna()

        raw_returns_df = prices_to_log_returns(prices)
        print("Assets used after cleaning:", raw_returns_df.shape[1])

    else:
        tickers = args.tickers
        prices = load_price_data(tickers, start=args.start, end=args.end)
        prices = prices.ffill().dropna()
        raw_returns_df = prices_to_log_returns(prices)

    # (1) Add CASH asset
    if args.add_cash:
        raw_returns_df = add_cash_asset(raw_returns_df, name="CASH")

    # (2) Build feature tensor from raw returns
    feat_tensor, feat_index = build_feature_tensor(
        raw_returns_df,
        mom_windows=tuple(args.mom_windows),
        vol_windows=tuple(args.vol_windows),
    )

    # Align raw returns to the trimmed feature index
    raw_returns_df = raw_returns_df.loc[feat_index]

    # IMPORTANT: Instead of normalized returns, we split RAW and split FEATURES
    raw_arr = raw_returns_df.values.astype(np.float64)
    train_raw, test_raw = split_time_aligned(raw_arr, args.train_ratio, args.lookback)
    train_feats, test_feats = split_time_aligned(feat_tensor, args.train_ratio, args.lookback)

    # -----------------------------
    # Sanity checks on split lengths
    # -----------------------------
    min_train = args.lookback + args.episode_length + 1
    if len(train_raw) < min_train:
        raise ValueError(
            f"Train split too short: len(train_raw)={len(train_raw)} "
            f"but need >= {min_train}. "
            f"Reduce --episode_length, increase data range, or increase --train_ratio."
        )

    min_test = args.lookback + 2
    if len(test_raw) < min_test:
        raise ValueError(
            f"Test split too short: len(test_raw)={len(test_raw)} "
            f"but need >= {min_test}."
        )

    # Normalize features using train stats only
    train_obs, test_obs, feat_mean, feat_std = normalize_feature_tensor(train_feats, test_feats)

    print(f"Final shapes: train_raw={train_raw.shape}, train_obs={train_obs.shape}, test_raw={test_raw.shape}, test_obs={test_obs.shape}")

    # -----------------------------
    # Make envs (3) short random episodes for train, full horizon for test
    # -----------------------------
    train_env = PortfolioEnv(
        returns_raw=train_raw,          # (T, N)
        obs_features=train_obs,         # (T, N, F)  <-- requires env arg name obs_features
        lookback_window_size=args.lookback,
        transaction_cost=args.transaction_cost,
        initial_value=1.0,
        random_start=True,
        episode_length=args.episode_length,  # e.g. 252
        dsr_eta=args.dsr_eta,                # <-- requires env supports DSR reward
    )

    # full test horizon
    test_ep_len = len(test_raw) - args.lookback - 1
    test_ep_len = max(1, test_ep_len)

    test_env = PortfolioEnv(
        returns_raw=test_raw,
        obs_features=test_obs,
        lookback_window_size=args.lookback,
        transaction_cost=args.transaction_cost,
        initial_value=1.0,
        random_start=False,
        episode_length=test_ep_len,
        dsr_eta=args.dsr_eta,
    )

    # PPO config
    cfg = PPOConfig(
        rollout_steps=args.rollout_steps,
        total_steps=args.total_steps,
    )

    obs_dim = train_env.observation_space.shape[0]
    N = train_env.num_assets                 # true assets
    act_dim = train_env.action_space.shape[0]  # should be 2N

    assert act_dim == 2 * N, f"Expected action dim 2N, got {act_dim} vs 2*{N}"

    model = ActorCritic(obs_dim=obs_dim, num_assets=N).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Training loop
    steps_done = 0
    best_test_sharpe = -1e9
    best_test_steps = 0
    best_path = args.save_path.replace(".pt", "_best.pt")

    while steps_done < cfg.total_steps:
        batch = rollout(train_env, model, device, cfg.rollout_steps, cfg, seed=args.seed)
        ppo_update(model, optimizer, batch, device, cfg)
        steps_done += cfg.rollout_steps

        # quick progress print (train)
        train_vals, train_to = evaluate_policy(train_env, model, device)
        m = summary_metrics(train_vals, turnover=train_to)
        print(
            f"[steps={steps_done}] train final={m['final']:.4f} sharpe={m['sharpe']:.4f} "
            f"maxDD={m['max_dd']:.4f} avgTO={m['avg_turnover']:.4f}"
        )

        # periodic test evaluation (every eval_every rollouts)
        if steps_done % (cfg.rollout_steps * args.eval_every) == 0:
            test_vals_i, test_to_i = evaluate_policy(test_env, model, device)
            mt = summary_metrics(test_vals_i, turnover=test_to_i)
            print(
                f"TEST final={mt['final']:.4f} sharpe={mt['sharpe']:.4f} "
                f"maxDD={mt['max_dd']:.4f} avgTO={mt['avg_turnover']:.4f}"
            )

            # Save best checkpoint by TEST Sharpe ratio
            if mt["sharpe"] > best_test_sharpe:
                best_test_sharpe = mt["sharpe"]
                best_test_steps = steps_done

                import os
                os.makedirs(os.path.dirname(best_path), exist_ok=True)

                torch.save(model.state_dict(), best_path)
                print(f"  (saved BEST weights @ steps={best_test_steps}, sharpe={best_test_sharpe:.4f})")

    # Final evaluation (test)
    test_vals, test_to = evaluate_policy(test_env, model, device)
    ew_vals, ew_to = evaluate_policy(test_env, model, device, policy="ew")
    mom_vals, mom_to = evaluate_policy(test_env, model, device, policy="mom")

    results = {
        "EW (test)": summary_metrics(ew_vals, turnover=ew_to),
        "MOM (test)": summary_metrics(mom_vals, turnover=mom_to),
        "PPO (train)": summary_metrics(*evaluate_policy(train_env, model, device)),
        "PPO (test)": summary_metrics(*evaluate_policy(test_env, model, device, policy="ppo")),
    }

    print("\n=== PPO Results ===")
    print_metrics_table(results)

    # Save
    import os

    if os.path.exists(best_path):
        state_dict = torch.load(best_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"\nLoaded BEST weights from {best_path} (steps={best_test_steps}, sharpe={best_test_sharpe:.4f})")
        best_test_vals, best_test_to = evaluate_policy(test_env, model, device)
        best_results = {
            "PPO (best test)": summary_metrics(best_test_vals, turnover=best_test_to),
        }
        print("\n=== BEST Checkpoint (Test) ===")
        print_metrics_table(best_results)
    else:
        print("\nNo best checkpoint found; using final model.")