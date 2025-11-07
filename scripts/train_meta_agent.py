"""Command-line entry point for training the meta-agent with PPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from agents.coder_agent import CoderAgent
from agents.meta_agent import MetaAgentController
from agents.reviewer_agent import ReviewerAgent
from agents.tester_agent import TesterAgent
from env.code_env import CodeEnv, CodeTask
from rl.policy import PPOPolicy
from rl.trainer import PPOConfig, train


def load_tasks(path: Path | None) -> List[CodeTask]:
    if not path:
        return [
            CodeTask(spec="实现一个打印 Hello World 的 Python 脚本"),
            CodeTask(spec="编写一个读取命令行参数的 CLI"),
        ]
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = []
    for item in data:
        tasks.append(CodeTask(spec=item["spec"], context=item.get("context", {})))
    return tasks


def build_controller() -> MetaAgentController:
    return MetaAgentController(
        coder=CoderAgent(),
        tester=TesterAgent(),
        reviewer=ReviewerAgent(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the meta-agent with PPO")
    parser.add_argument("--tasks", type=Path, default=None, help="JSON file describing training tasks")
    parser.add_argument("--total-steps", type=int, default=10_000)
    parser.add_argument("--rollout-length", type=int, default=256)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=6, help="Max tool invocations per episode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = load_tasks(args.tasks)

    controller_factory = build_controller
    probe = controller_factory()
    policy = PPOPolicy(obs_dim=probe.state_dim, act_dim=len(probe.action_meanings))

    env = CodeEnv(controller_factory=controller_factory, tasks=tasks, max_steps=args.max_steps, seed=args.seed)

    config = PPOConfig(
        total_steps=args.total_steps,
        rollout_length=args.rollout_length,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        device=args.device,
    )

    stats = train(env, policy, config)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
