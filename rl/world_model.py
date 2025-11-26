"""基于简单双层感知机的世界模型，用于预测下一状态和即时奖励。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def _mlp(input_dim: int, hidden_sizes: Iterable[int], output_dim: int, activation=nn.ReLU) -> nn.Sequential:
    """快速构建多层感知机，减少样板代码。"""
    layers = []
    last = input_dim
    for size in hidden_sizes:
        layers.append(nn.Linear(last, size))
        layers.append(activation())
        last = size
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


@dataclass
class WorldModelConfig:
    """训练世界模型的关键超参容器。"""

    hidden_sizes: Tuple[int, ...] = (128, 128)
    lr: float = 1e-3
    device: str = "cpu"


class WorldModel(nn.Module):
    """
    预测 s_{t+1} 与 r_t 的轻量模型。
    - 输入：obs_t (B, obs_dim)、action_t (B,)。
    - 输出：pred_next_obs (B, obs_dim)、pred_reward (B,)。
    """

    def __init__(self, obs_dim: int, act_dim: int, config: WorldModelConfig | None = None) -> None:
        super().__init__()
        cfg = config or WorldModelConfig()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.config = cfg

        # 动作嵌入将离散动作编码为连续向量，便于和状态拼接。
        self.action_embed = nn.Embedding(act_dim, cfg.hidden_sizes[0])
        self.encoder = _mlp(obs_dim, cfg.hidden_sizes, cfg.hidden_sizes[-1])
        self.transition_head = nn.Linear(cfg.hidden_sizes[-1] + cfg.hidden_sizes[0], obs_dim)
        self.reward_head = nn.Linear(cfg.hidden_sizes[-1] + cfg.hidden_sizes[0], 1)
        self.to(cfg.device)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 将 obs 与动作嵌入拼接，再分别预测下一状态与奖励。
        obs_feat = self.encoder(obs)
        act_feat = self.action_embed(actions)
        joint = torch.cat([obs_feat, act_feat], dim=-1)
        next_obs = self.transition_head(joint)
        reward = self.reward_head(joint).squeeze(-1)
        return next_obs, reward

    @torch.no_grad()
    def predict(self, obs: np.ndarray, action: int) -> tuple[np.ndarray, float]:
        """推理接口，便于在想象环境中调用。"""
        device = torch.device(self.config.device)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        act_t = torch.tensor([action], dtype=torch.long, device=device)
        next_obs, reward = self.forward(obs_t, act_t)
        return next_obs.squeeze(0).cpu().numpy(), float(reward.item())

    def loss_fn(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        target_next_obs: torch.Tensor,
        target_reward: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回总损失与分量，便于监控训练质量。"""
        pred_next, pred_r = self.forward(obs, actions)
        obs_loss = torch.mean((pred_next - target_next_obs) ** 2)
        reward_loss = torch.mean((pred_r - target_reward) ** 2)
        total = obs_loss + reward_loss
        return total, obs_loss, reward_loss

    def train_step(
        self,
        batch,
        optimizer: optim.Optimizer | None = None,
    ) -> dict[str, float]:
        """
        单步训练例程：
        batch 需包含 obs、actions、next_obs、rewards（均为 Tensor 或可转 Tensor）。
        """
        opt = optimizer or optim.Adam(self.parameters(), lr=self.config.lr)
        device = torch.device(self.config.device)
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=device)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=device)

        opt.zero_grad()
        total, obs_loss, reward_loss = self.loss_fn(obs, actions, next_obs, rewards)
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        opt.step()

        return {
            "loss": float(total.item()),
            "obs_loss": float(obs_loss.item()),
            "reward_loss": float(reward_loss.item()),
        }


__all__ = ["WorldModel", "WorldModelConfig"]
