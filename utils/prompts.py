"""Centralized prompt templates for agents."""

from __future__ import annotations

from typing import Dict


PROMPTS: Dict[str, str] = {
    "coder": (
        "You are Coder, an autonomous software engineer.\n"
        "Task: {task}\n"
        "Strategy: {strategy}\n"
        "Known issues / hints: {hint}\n"
        "Previous code (for diff awareness):\n{history}\n"
        "Produce updated code as file blocks."
    ),
    "reviewer": (
        "You are Reviewer. Analyze the failing test log and summarize issues.\n"
        "Test Log:\n{test_log}\n"
        "Code Snapshot:\n{code}\n"
        "Additional context: {context}\n"
        "Return concise summary and prioritized hints."
    ),
    "tester": "You are Tester. This prompt slot is reserved for future model-based execution.",
}


def get_prompt(agent_role: str, **kwargs) -> str:
    template = PROMPTS.get(agent_role)
    if not template:
        raise KeyError(f"Unknown agent role: {agent_role}")
    return template.format(**{k: ("" if v is None else v) for k, v in kwargs.items()})


__all__ = ["get_prompt", "PROMPTS"]
