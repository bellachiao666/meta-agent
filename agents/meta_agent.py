"""Meta-agent that coordinates coder/tester/reviewer with an RL policy."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from agents.coder_agent import CoderAgent, CodeBundle, GenerationResult
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


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    probs: np.ndarray
    info: Dict


@dataclass
class EpisodeTrajectory:
    transitions: List[Transition] = field(default_factory=list)
    success: bool = False


@dataclass
class EpisodeSummary:
    success: bool
    steps: int
    final_bundle: Optional[CodeBundle]
    test_report: Optional[TestReport]
    review_result: Optional[ReviewResult]
    trajectory: EpisodeTrajectory


class StateEncoder:
    def __init__(self, error_labels: Optional[List[str]] = None) -> None:
        self.error_labels = error_labels or ["none", "logic_error", "syntax_error", "missing_tests", "style_issue"]

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
        for label in self.error_labels:
            vec.append(1.0 if state.error_type == label else 0.0)
        return np.array(vec, dtype=float)

    def initial_state(self) -> MetaState:
        return MetaState()


class PolicyModel:
    """Tiny softmax policy updated with a simple reward-proportional rule."""

    def __init__(self, state_dim: int, action_dim: int, lr: float = 0.05) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.weights = np.zeros((action_dim, state_dim))
        self.bias = np.zeros(action_dim)

    def act(self, state_vec: np.ndarray) -> tuple[int, np.ndarray]:
        logits = self.weights @ state_vec + self.bias
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        action = int(np.random.choice(self.action_dim, p=probs))
        return action, probs

    def update(self, trajectories: List[EpisodeTrajectory]) -> None:
        for traj in trajectories:
            for transition in traj.transitions:
                grad = transition.reward
                self.weights[transition.action] += self.lr * grad * transition.state
                self.bias[transition.action] += self.lr * grad


class MetaAgent:
    ACTIONS = {
        0: "regenerate",
        1: "refine",
        2: "request_review",
        3: "terminate",
    }

    def __init__(
        self,
        coder: CoderAgent,
        tester: TesterAgent,
        reviewer: ReviewerAgent,
        state_encoder: Optional[StateEncoder] = None,
        policy: Optional[PolicyModel] = None,
        reward_cfg: Optional[Dict[str, float]] = None,
    ) -> None:
        self.coder = coder
        self.tester = tester
        self.reviewer = reviewer
        self.encoder = state_encoder or StateEncoder()
        self.policy = policy or PolicyModel(state_dim=self.encoder.state_dim, action_dim=len(self.ACTIONS))
        self.reward_cfg = reward_cfg or {"alpha": 2.0, "beta": 1.0, "gamma": 0.3, "delta": 0.2}
        self.log = get_logger("MetaAgent")

    def plan(self, task_spec: str) -> Dict:
        return {
            "task": task_spec,
            "stages": ["generate", "test", "review", "iterate"],
            "actions": list(self.ACTIONS.values()),
        }

    def act(self, state_vec: np.ndarray) -> tuple[int, np.ndarray]:
        return self.policy.act(state_vec)

    def learn(self, trajectories: List[EpisodeTrajectory]) -> None:
        self.policy.update(trajectories)

    def run_episode(self, task_spec: str, max_steps: int = 6, context: Optional[Dict] = None) -> EpisodeSummary:
        context = context or {}
        meta_state = self.encoder.initial_state()
        state_vec = self.encoder.encode(meta_state)
        trajectory = EpisodeTrajectory()

        current_bundle: Optional[CodeBundle] = None
        review_result: Optional[ReviewResult] = None
        test_report: Optional[TestReport] = None
        retry_count = 0
        total_time = 0.0
        code_diff = 0.0

        terminated = False
        for step in range(max_steps):
            action_id, probs = self.act(state_vec)
            action_name = self.ACTIONS[action_id]
            self.log.debug("Step %d action=%s", step, action_name)

            if action_name == "regenerate":
                retry_count += 1
                gen = self.coder.generate_code(task_spec, previous_bundle=current_bundle, strategy="regenerate")
                current_bundle = gen.bundle
                code_diff = gen.similarity
                test_report = self.tester.run_tests(current_bundle, context)
                total_time += test_report.duration
                review_result = None
            elif action_name == "refine":
                retry_count += 1
                hint = review_result.summary if review_result else ""
                gen = self.coder.generate_code(
                    task_spec,
                    previous_bundle=current_bundle,
                    review_hint=hint,
                    strategy="refine",
                )
                current_bundle = gen.bundle
                code_diff = gen.similarity
                test_report = self.tester.run_tests(current_bundle, context)
                total_time += test_report.duration
            elif action_name == "request_review":
                if current_bundle and test_report:
                    review_result = self.reviewer.review(current_bundle, test_report, context)
                else:
                    review_result = ReviewResult(score=0.2, summary="缺少测试结果，无法审查", hints=[], recommended_action="regenerate")
            elif action_name == "terminate":
                terminated = True

            meta_state = MetaState(
                test_passed=int(test_report.passed) if test_report else 0,
                review_score=review_result.score if review_result else 0.0,
                retry_count=retry_count,
                error_type=test_report.error_type if test_report else "none",
                code_diff=code_diff,
                time_cost=total_time,
            )
            next_state_vec = self.encoder.encode(meta_state)
            reward = self._compute_reward(test_report, review_result, retry_count)
            trajectory.transitions.append(
                Transition(
                    state=state_vec,
                    action=action_id,
                    reward=reward,
                    next_state=next_state_vec,
                    probs=probs,
                    info={"action": action_name},
                )
            )
            state_vec = next_state_vec

            if terminated or (test_report and test_report.passed):
                break

        success = bool(test_report and test_report.passed)
        trajectory.success = success
        self.learn([trajectory])
        return EpisodeSummary(
            success=success,
            steps=len(trajectory.transitions),
            final_bundle=current_bundle,
            test_report=test_report,
            review_result=review_result,
            trajectory=trajectory,
        )

    def _compute_reward(
        self,
        test_report: Optional[TestReport],
        review_result: Optional[ReviewResult],
        retry_count: int,
    ) -> float:
        cfg = self.reward_cfg
        pass_flag = 1.0 if test_report and test_report.passed else 0.0
        review_score = review_result.score if review_result else 0.0
        severity = test_report.severity if test_report else 0.0
        return cfg["alpha"] * pass_flag + cfg["beta"] * review_score - cfg["gamma"] * retry_count - cfg["delta"] * severity


__all__ = [
    "MetaAgent",
    "MetaState",
    "StateEncoder",
    "PolicyModel",
    "EpisodeSummary",
    "EpisodeTrajectory",
    "Transition",
]
