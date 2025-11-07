"""Reviewer agent for analyzing tester feedback and code diffs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from agents.coder_agent import CodeBundle
from agents.tester_agent import TestReport
from utils.logger import get_logger
from utils.prompts import get_prompt


@dataclass
class ReviewResult:
    score: float
    summary: str
    hints: List[str]
    recommended_action: str


class ReviewerAgent:
    """Produces structured review feedback for the meta-agent."""

    def __init__(self) -> None:
        self.log = get_logger("ReviewerAgent")

    def review(self, bundle: CodeBundle, test_report: TestReport, context: Optional[Dict] = None) -> ReviewResult:
        context = context or {}
        prompt = get_prompt(
            "reviewer",
            test_log=test_report.log,
            code=bundle.as_text(),
            context=context,
        )
        self.log.debug("Review prompt prepared (%d chars)", len(prompt))

        score = max(0.0, 1.0 - test_report.severity)
        hints = self._heuristic_hints(bundle, test_report)
        summary = self._build_summary(test_report, hints)
        action = "refine" if not test_report.passed else "terminate"
        self.log.debug("Generated review (score=%.2f action=%s)", score, action)
        return ReviewResult(score=score, summary=summary, hints=hints, recommended_action=action)

    def _heuristic_hints(self, bundle: CodeBundle, report: TestReport) -> List[str]:
        hints: List[str] = []
        text = bundle.as_text()
        if "TODO" in text:
            hints.append("存在 TODO 占位符，需要补充真实实现。")
        if "print(" in text and not report.passed:
            hints.append("调试输出较多，建议替换为结构化日志。")
        if "pass\n" in text:
            hints.append("检测到 pass 语句，可能尚未实现。")
        if not hints and not report.passed:
            hints.append("根据测试日志修复失败断言。")
        return hints

    def _build_summary(self, report: TestReport, hints: List[str]) -> str:
        base = "测试通过" if report.passed else f"测试失败：{report.error_type}"
        if hints:
            base += "。建议：" + "；".join(hints)
        return base


__all__ = ["ReviewerAgent", "ReviewResult"]
