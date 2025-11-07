"""PPO trainer for the meta-agent gym environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import optim

from rl.policy import PPOPolicy


@dataclass
class PPOConfig:
    total_steps: int = 10_000
    rollout_length: int = 256
    ppo_epochs: int = 4
    minibatch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    device: str = "cpu"


class RolloutBuffer:
    def __init__(self) -> None:
        self.clear()

    def add(self, obs, action: int, reward: float, done: bool, log_prob: float, value: float) -> None:
        self.observations.append(np.array(obs, dtype=np.float32))
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self) -> None:
        self.observations: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.advantages: List[float] = []
        self.returns: List[float] = []

    def __len__(self) -> int:
        return len(self.rewards)


def compute_gae(buffer: RolloutBuffer, next_value: float, gamma: float, gae_lambda: float) -> Tuple[np.ndarray, np.ndarray]:
    rewards = buffer.rewards
    values = buffer.values + [next_value]
    dones = buffer.dones
    advantages = np.zeros(len(rewards), dtype=np.float32)
    gae = 0.0
    for step in reversed(range(len(rewards))):
        mask = 1.0 - float(dones[step])
        delta = rewards[step] + gamma * values[step + 1] * mask - values[step]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[step] = gae
    returns = advantages + np.array(buffer.values, dtype=np.float32)
    return advantages, returns


def collect_rollout(
    env,
    policy: PPOPolicy,
    buffer: RolloutBuffer,
    steps: int,
    device: torch.device,
    start_obs: np.ndarray,
) -> Tuple[np.ndarray, float, List[float], List[int]]:
    obs = start_obs
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    ep_reward = 0.0
    ep_length = 0

    for _ in range(steps):
        action, log_prob, value = policy.act(obs, device)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add(obs, action, reward, done, log_prob, value)

        ep_reward += reward
        ep_length += 1

        if done:
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            obs, _ = env.reset()
            ep_reward = 0.0
            ep_length = 0
        else:
            obs = next_obs

    next_value = policy.value(obs, device)
    return obs, next_value, episode_rewards, episode_lengths


def ppo_update(policy: PPOPolicy, optimizer: optim.Optimizer, buffer: RolloutBuffer, config: PPOConfig) -> Dict[str, float]:
    device = torch.device(config.device)
    obs = torch.tensor(np.stack(buffer.observations), dtype=torch.float32, device=device)
    actions = torch.tensor(buffer.actions, dtype=torch.long, device=device)
    old_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32, device=device)
    returns = torch.tensor(buffer.returns, dtype=torch.float32, device=device)
    advantages = torch.tensor(buffer.advantages, dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    total_loss = 0.0
    total_value_loss = 0.0
    total_policy_loss = 0.0
    total_entropy = 0.0

    batch_size = len(buffer)
    indices = torch.arange(batch_size)

    for _ in range(config.ppo_epochs):
        perm = indices[torch.randperm(batch_size)]
        for start in range(0, batch_size, config.minibatch_size):
            batch_idx = perm[start : start + config.minibatch_size]
            batch_obs = obs[batch_idx]
            batch_actions = actions[batch_idx]
            batch_old_log_probs = old_log_probs[batch_idx]
            batch_returns = returns[batch_idx]
            batch_advantages = advantages[batch_idx]

            log_probs, entropy, values = policy.evaluate_actions(batch_obs, batch_actions)
            ratio = torch.exp(log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (batch_returns - values).pow(2).mean()
            entropy_loss = entropy.mean()

            loss = policy_loss + config.vf_coef * value_loss - config.ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()

            total_loss += float(loss.item())
            total_value_loss += float(value_loss.item())
            total_policy_loss += float(policy_loss.item())
            total_entropy += float(entropy_loss.item())

    updates = max(1, (batch_size // config.minibatch_size) * config.ppo_epochs)
    return {
        "loss": total_loss / updates,
        "policy_loss": total_policy_loss / updates,
        "value_loss": total_value_loss / updates,
        "entropy": total_entropy / updates,
    }


def train(env, policy: PPOPolicy, config: PPOConfig) -> Dict[str, float]:
    device = torch.device(config.device)
    policy.to(device)
    optimizer = optim.Adam(policy.parameters(), lr=config.lr)
    buffer = RolloutBuffer()

    obs, _ = env.reset()
    global_step = 0
    reward_history: List[float] = []
    length_history: List[int] = []
    last_metrics: Dict[str, float] = {}

    while global_step < config.total_steps:
        steps = min(config.rollout_length, config.total_steps - global_step)
        buffer.clear()
        obs, next_value, batch_rewards, batch_lengths = collect_rollout(env, policy, buffer, steps, device, obs)
        advantages, returns = compute_gae(buffer, next_value, config.gamma, config.gae_lambda)
        buffer.advantages = advantages.tolist()
        buffer.returns = returns.tolist()

        batch_stats = ppo_update(policy, optimizer, buffer, config)
        last_metrics = batch_stats
        global_step += steps

        reward_history.extend(batch_rewards)
        length_history.extend(batch_lengths)

    mean_reward = float(np.mean(reward_history)) if reward_history else 0.0
    mean_length = float(np.mean(length_history)) if length_history else 0.0
    summary = {
        "total_steps": global_step,
        "mean_episode_reward": mean_reward,
        "mean_episode_length": mean_length,
    }
    summary.update(last_metrics)
    return summary


__all__ = ["PPOConfig", "train"]
