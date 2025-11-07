"""Meta-agent controller that coordinates coder, tester, and reviewer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from agents.coder_agent import CoderAgent, CodeBundle
from agents.reviewer_agent import ReviewResult, ReviewerAgent
from agents.tester_agent import TestReport, TesterAgent
from utils.logger import get_logger


@dataclass
class MetaState:
    test_passed: int = 0
    review_score: float = 0.0
    retry_count: int = 0
    error_type: str = "none"
    code_diff: float = 0.0
    time_cost: float = 0.0


class StateEncoder:
    """Encodes structured state data into RL-friendly vectors."""

    def __init__(self, error_labels: Optional[List[str]] = None) -> None:
        self.error_labels = error_labels or [
            "none",
            "logic_error",
            "syntax_error",
            "missing_tests",
            "style_issue",
        ]

    @property
    def state_dim(self) -> int:
        return 5 + len(self.error_labels)

    def encode(self, state: MetaState) -> np.ndarray:
        vec = [
            float(state.test_passed),
            state.review_score,
            float(state.retry_count),
            state.code_diff,
            state.time_cost,
        ]
        vec.extend(1.0 if state.error_type == label else 0.0 for label in self.error_labels)
        return np.array(vec, dtype=np.float32)

    def initial_state(self) -> MetaState:
        return MetaState()


class MetaAgentController:
    """Executes multi-agent actions and produces RL observations/rewards."""

    ACTIONS = ["regenerate", "refine", "request_review", "terminate"]

    def __init__(
        self,
        coder: CoderAgent,
        tester: TesterAgent,
        reviewer: ReviewerAgent,
        state_encoder: Optional[StateEncoder] = None,
        reward_cfg: Optional[Dict[str, float]] = None,
    ) -> None:
        self.coder = coder
        self.tester = tester
        self.reviewer = reviewer
        self.encoder = state_encoder or StateEncoder()
        self.reward_cfg = reward_cfg or {"alpha": 2.0, "beta": 1.0, "gamma": 0.3, "delta": 0.2}
        self.log = get_logger("MetaAgentController")

        self.state: MetaState = self.encoder.initial_state()
        self.task_spec: str = ""
        self.context: Dict = {}
        self.current_bundle: Optional[CodeBundle] = None
        self.test_report: Optional[TestReport] = None
        self.review_result: Optional[ReviewResult] = None
        self.retry_count = 0
        self.total_time = 0.0
        self.code_diff = 0.0
        self.max_steps = 0
        self.step_count = 0

    @property
    def state_dim(self) -> int:
        return self.encoder.state_dim

    @property
    def action_meanings(self) -> List[str]:
        return list(self.ACTIONS)

    def reset(self, task_spec: str, context: Optional[Dict] = None, max_steps: int = 6) -> np.ndarray:
        self.task_spec = task_spec
        self.context = context or {}
        self.current_bundle = None
        self.test_report = None
        self.review_result = None
        self.retry_count = 0
        self.total_time = 0.0
        self.code_diff = 0.0
        self.step_count = 0
        self.max_steps = max_steps
        self.state = self.encoder.initial_state()
        self.log.info("Reset controller for task (%s...)", task_spec[:60])
        return self.encoder.encode(self.state)

    def step(self, action_id: int) -> Tuple[np.ndarray, float, bool, Dict]:
        action_name = self._action_from_id(action_id)
        self.step_count += 1
        terminated = False

        if action_name == "regenerate":
            terminated = self._handle_generation(strategy="regenerate")
        elif action_name == "refine":
            terminated = self._handle_generation(strategy="refine")
        elif action_name == "request_review":
            self._handle_review_request()
        elif action_name == "terminate":
            terminated = True

        self._update_state()
        reward = self._compute_reward()
        obs = self.encoder.encode(self.state)
        info = {
            "action": action_name,
            "test_passed": bool(self.test_report.passed) if self.test_report else False,
            "error_type": self.test_report.error_type if self.test_report else "none",
            "review_score": self.review_result.score if self.review_result else 0.0,
            "retry_count": self.retry_count,
        }
        if self.review_result:
            info["review_summary"] = self.review_result.summary

        if not terminated and self.test_report and self.test_report.passed:
            terminated = True

        return obs, reward, terminated, info

    def _action_from_id(self, action_id: int) -> str:
        if action_id < 0 or action_id >= len(self.ACTIONS):
            raise ValueError(f"Invalid action id: {action_id}")
        return self.ACTIONS[action_id]

    def _handle_generation(self, strategy: str) -> bool:
        self.retry_count += 1
        hint = self.review_result.summary if (strategy == "refine" and self.review_result) else ""
        generation = self.coder.generate_code(
            task_spec=self.task_spec,
            previous_bundle=self.current_bundle,
            review_hint=hint,
            strategy=strategy,
        )
        self.current_bundle = generation.bundle
        self.code_diff = generation.similarity
        self.test_report = self.tester.run_tests(self.current_bundle, self.context)
        self.total_time += self.test_report.duration
        self.review_result = None
        return False

    def _handle_review_request(self) -> None:
        if self.current_bundle and self.test_report:
            self.review_result = self.reviewer.review(self.current_bundle, self.test_report, self.context)
        else:
            self.review_result = ReviewResult(
                score=0.1,
                summary="尚无可供审查的完整测试记录",
                hints=["先运行测试以收集日志"],
                recommended_action="regenerate",
            )

    def _update_state(self) -> None:
        self.state = MetaState(
            test_passed=int(self.test_report.passed) if self.test_report else 0,
            review_score=self.review_result.score if self.review_result else 0.0,
            retry_count=self.retry_count,
            error_type=self.test_report.error_type if self.test_report else "none",
            code_diff=self.code_diff,
            time_cost=self.total_time,
        )

    def _compute_reward(self) -> float:
        pass_flag = 1.0 if self.test_report and self.test_report.passed else 0.0
        review_score = self.review_result.score if self.review_result else 0.0
        severity = self.test_report.severity if self.test_report else 0.0
        cfg = self.reward_cfg
        return (
            cfg["alpha"] * pass_flag
            + cfg["beta"] * review_score
            - cfg["gamma"] * float(self.retry_count)
            - cfg["delta"] * severity
        )


__all__ = ["MetaAgentController", "MetaState", "StateEncoder"]
