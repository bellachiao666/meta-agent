"""Tester agent responsible for validating generated code."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from agents.coder_agent import CodeBundle
from utils.logger import get_logger


ERROR_SEVERITY = {
    "syntax_error": 0.9,
    "missing_tests": 0.6,
    "logic_error": 0.7,
    "style_issue": 0.3,
    "none": 0.0,
}


@dataclass
class TestReport:
    passed: bool
    error_type: str
    severity: float
    log: str
    duration: float


class TesterAgent:
    """Runs tests either via user-provided executor or heuristic checks."""

    def __init__(self, executor: Optional[Callable[[CodeBundle, Dict], Dict]] = None) -> None:
        self.executor = executor
        self.log = get_logger("TesterAgent")

    def run_tests(self, bundle: CodeBundle, context: Optional[Dict] = None) -> TestReport:
        start = time.time()
        context = context or {}

        if self.executor:
            result = self.executor(bundle, context)
            passed = bool(result.get("passed", False))
            log = result.get("log", "")
            error_type = result.get("error_type", "none" if passed else "logic_error")
        else:
            passed, log, error_type = self._heuristic_runner(bundle)

        duration = time.time() - start
        severity = ERROR_SEVERITY.get(error_type, 0.5)
        report = TestReport(passed=passed, error_type=error_type, severity=severity, log=log, duration=duration)
        self.log.debug(
            "Test result: passed=%s error_type=%s severity=%.2f duration=%.3fs",
            passed,
            error_type,
            severity,
            duration,
        )
        return report

    def _heuristic_runner(self, bundle: CodeBundle) -> tuple[bool, str, str]:
        text = bundle.as_text().lower()
        if "todo" in text or "pass\n" in text:
            return False, "Found TODO/PASS placeholder in code.", "missing_tests"
        if "notimplemented" in text or "raise notimplementederror" in text:
            return False, "Encountered NotImplementedError placeholder.", "logic_error"
        if re.search(r"print\s*\(.*debug", text):
            return False, "Debug logging detected; treat as style issue.", "style_issue"
        return True, "Heuristic tests passed.", "none"


__all__ = ["TesterAgent", "TestReport"]
