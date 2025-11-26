"""Command-line entry point for training the meta-agent with PPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import yaml

from agents.coder_agent import CoderAgent
from agents.meta_agent import MetaAgentController, StateEncoder
from agents.reviewer_agent import ReviewerAgent
from agents.tester_agent import TesterAgent
from env.code_env import CodeEnv, CodeTask
from rl.policy import PPOPolicy
from rl.trainer import PPOConfig, train


def load_yaml(path: Path) -> Dict:
    # 读取 YAML 配置，缺失时报错
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


def build_tasks(task_entries: Iterable[Dict]) -> List[CodeTask]:
    # 将配置中的任务列表转换为 CodeTask 实例
    tasks = [CodeTask(spec=item["spec"], context=item.get("context", {})) for item in task_entries]
    if not tasks:
        raise ValueError("No tasks defined in model config; add at least one under 'tasks'.")
    return tasks


def build_controller_factory(model_cfg: Dict) -> Callable[[], MetaAgentController]:
    # 返回工厂函数，便于 env 每次 reset 构建独立控制器
    controller_cfg = model_cfg.get("controller", {})
    reward_cfg = controller_cfg.get("reward")
    error_labels = controller_cfg.get("error_labels")

    def factory() -> MetaAgentController:
        state_encoder = StateEncoder(error_labels=error_labels) if error_labels else None
        return MetaAgentController(
            coder=CoderAgent(),
            tester=TesterAgent(),
            reviewer=ReviewerAgent(),
            state_encoder=state_encoder,
            reward_cfg=reward_cfg,
        )

    return factory


def parse_args() -> argparse.Namespace:
    # 仅接收 RL 与模型配置路径
    parser = argparse.ArgumentParser(description="Train the meta-agent with PPO")
    parser.add_argument("--rl-config", type=Path, default=Path("config/rl.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("config/model.yaml"))
    return parser.parse_args()


def main() -> None:
    # 主流程：加载配置 -> 构造任务/env/policy -> 运行训练 -> 打印指标
    args = parse_args()
    rl_cfg = load_yaml(args.rl_config)
    model_cfg = load_yaml(args.model_config)

    tasks = build_tasks(model_cfg.get("tasks", []))
    controller_factory = build_controller_factory(model_cfg)
    probe = controller_factory()

    policy_cfg = model_cfg.get("policy", {})
    hidden_sizes = tuple(policy_cfg.get("hidden_sizes", [128, 128]))
    policy = PPOPolicy(obs_dim=probe.state_dim, act_dim=len(probe.action_meanings), hidden_sizes=hidden_sizes)

    env_cfg = rl_cfg.get("env", {})
    env = CodeEnv(
        controller_factory=controller_factory,
        tasks=tasks,
        max_steps=env_cfg.get("max_steps", 6),
        seed=env_cfg.get("seed"),
    )

    ppo_cfg = rl_cfg.get("ppo", {})
    config = PPOConfig(**ppo_cfg)

    stats = train(env, policy, config)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
