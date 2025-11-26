"""Policy definitions for PPO training of the meta-agent."""

from __future__ import annotations
from typing import Iterable, Sequence
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def build_mlp(input_dim: int, hidden_sizes: Sequence[int], output_dim: int, activation=nn.Tanh) -> nn.Sequential:
    """快速搭建多层感知机，用于 actor/critic 共享。"""
    layers = []
    last_dim = input_dim
    for size in hidden_sizes:
        layers.append(nn.Linear(last_dim, size))
        layers.append(activation())
        last_dim = size
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class PPOPolicy(nn.Module):
    """Actor-critic network used by PPO."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Iterable[int] = (128, 128)) -> None:
        super().__init__()
        self.actor = build_mlp(obs_dim, hidden_sizes, act_dim)
        self.critic = build_mlp(obs_dim, hidden_sizes, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        return logits, value

    def act(self, obs, device: str | torch.device | None = None) -> tuple[int, float, float]:
        # 单步采样动作，同时返回 log_prob 与 state value
        obs_tensor = self._format_obs(obs, device)
        logits = self.actor(obs_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(obs_tensor).squeeze(-1)
        return action.item(), float(log_prob.item()), float(value.item())

    def evaluate_actions(
        self,
        obs_batch: torch.Tensor,
        action_batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 评估一批动作的 log_prob/entropy/value，供 PPO 损失计算
        logits = self.actor(obs_batch)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(action_batch)
        entropy = dist.entropy()
        values = self.critic(obs_batch).squeeze(-1)
        return log_probs, entropy, values

    def value(self, obs, device: str | torch.device | None = None) -> float:
        # 仅推理 state value，用于 GAE 末尾 bootstrap
        obs_tensor = self._format_obs(obs, device)
        value = self.critic(obs_tensor).squeeze(-1)
        return float(value.item())

    def _format_obs(self, obs, device: str | torch.device | None) -> torch.Tensor:
        # 接受 numpy 或列表输入，统一为批量 Tensor
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.from_numpy(obs).float()
        else:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        if device is not None:
            obs_tensor = obs_tensor.to(device)
        return obs_tensor


__all__ = ["PPOPolicy", "build_mlp"]
