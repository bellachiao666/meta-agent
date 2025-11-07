"""OpenAI Gym environment that wraps the MetaAgentController."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import gym
import numpy as np
from gym import spaces

from agents.meta_agent import MetaAgentController
from utils.logger import get_logger


@dataclass
class CodeTask:
    spec: str
    context: Dict[str, str] = field(default_factory=dict)


class CodeEnv(gym.Env):
    """Gym-compatible environment for PPO training."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        controller_factory: Callable[[], MetaAgentController],
        tasks: List[CodeTask],
        max_steps: int = 6,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not tasks:
            raise ValueError("CodeEnv requires at least one task specification")

        self.controller_factory = controller_factory
        self.controller = controller_factory()
        self.tasks = tasks
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.log = get_logger("CodeEnv")

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.controller.state_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(self.controller.action_meanings))

        self.current_task: Optional[CodeTask] = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)

        task = self._sample_task()
        self.controller = self.controller_factory()
        obs = self.controller.reset(task.spec, task.context, max_steps=self.max_steps)
        info = {"task": task.spec}
        return obs.astype(np.float32), info

    def step(self, action: int):
        obs, reward, terminated, info = self.controller.step(int(action))
        truncated = self.controller.step_count >= self.max_steps and not terminated
        return obs.astype(np.float32), float(reward), bool(terminated), bool(truncated), info

    def render(self):
        info = {
            "step_count": self.controller.step_count,
            "retry_count": self.controller.retry_count,
        }
        if self.controller.test_report:
            info["test_passed"] = self.controller.test_report.passed
            info["error_type"] = self.controller.test_report.error_type
        self.log.info("Render info: %s", info)

    def _sample_task(self) -> CodeTask:
        self.current_task = self.rng.choice(self.tasks)
        self.log.info("Selected task: %s", self.current_task.spec[:80])
        return self.current_task


__all__ = ["CodeEnv", "CodeTask"]
