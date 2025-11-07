"""Simple environment wrapper for training/evaluating the meta-agent."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from agents.meta_agent import EpisodeSummary, MetaAgent
from utils.logger import get_logger


@dataclass
class CodeTask:
    spec: str
    context: Dict[str, str] = field(default_factory=dict)


class CodeEnv:
    def __init__(self, tasks: List[CodeTask]) -> None:
        if not tasks:
            raise ValueError("CodeEnv requires at least one task")
        self.tasks = tasks
        self.current_task: Optional[CodeTask] = None
        self.last_summary: Optional[EpisodeSummary] = None
        self.log = get_logger("CodeEnv")

    def reset(self) -> CodeTask:
        self.current_task = random.choice(self.tasks)
        self.log.info("Selected task: %s", self.current_task.spec[:60])
        self.last_summary = None
        return self.current_task

    def step(self, meta_agent: MetaAgent, max_steps: int = 6) -> EpisodeSummary:
        if not self.current_task:
            self.reset()
        summary = meta_agent.run_episode(
            task_spec=self.current_task.spec,
            max_steps=max_steps,
            context=self.current_task.context,
        )
        self.last_summary = summary
        return summary

    def render(self) -> None:
        if not self.last_summary:
            self.log.info("No episode has been run yet.")
            return
        summary = self.last_summary
        self.log.info(
            "Episode success=%s steps=%d",
            summary.success,
            summary.steps,
        )
        if summary.test_report:
            self.log.info("Test report: passed=%s error=%s", summary.test_report.passed, summary.test_report.error_type)
        if summary.review_result:
            self.log.info("Review score=%.2f summary=%s", summary.review_result.score, summary.review_result.summary)


__all__ = ["CodeEnv", "CodeTask"]
