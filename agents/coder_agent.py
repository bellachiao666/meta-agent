"""Coder agent for the RL-controlled multi-agent system."""

from __future__ import annotations

import difflib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from utils.logger import get_logger
from utils.prompts import get_prompt


@dataclass
class CodeArtifact:
    """Single generated file."""

    path: str
    content: str


@dataclass
class CodeBundle:
    """Container for multiple files."""

    artifacts: List[CodeArtifact]

    def write_to(self, root: Path) -> None:
        for art in self.artifacts:
            file_path = root / art.path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(art.content, encoding="utf-8")

    def as_text(self) -> str:
        parts = []
        for art in self.artifacts:
            parts.append(f"=== file:{art.path} ===\n{art.content}\n=== end ===")
        return "\n\n".join(parts)

    def artifact_map(self) -> Dict[str, str]:
        return {art.path: art.content for art in self.artifacts}


@dataclass
class GenerationResult:
    """Structured response from the coder."""

    bundle: CodeBundle
    prompt: str
    rationale: str
    similarity: float


class CoderAgent:
    """Lightweight coder agent with pluggable LLM backend."""

    def __init__(self, llm_call: Optional[Callable[[str], str]] = None) -> None:
        self.llm_call = llm_call
        self.log = get_logger("CoderAgent")

    def generate_code(
        self,
        task_spec: str,
        previous_bundle: Optional[CodeBundle] = None,
        review_hint: str = "",
        strategy: str = "regenerate",
    ) -> GenerationResult:
        prompt = get_prompt(
            "coder",
            task=task_spec,
            history=previous_bundle.as_text() if previous_bundle else "",
            hint=review_hint,
            strategy=strategy,
        )

        if self.llm_call:
            raw = self.llm_call(prompt)
            bundle = CodeBundle(self._parse_llm_output(raw))
            rationale = "LLM-backed generation"
        else:
            bundle, rationale = self._rule_based_generation(task_spec, review_hint)

        similarity = self._estimate_similarity(previous_bundle, bundle)
        self.log.debug(
            "Generated code via %s (similarity=%.3f)",
            "llm" if self.llm_call else "rule-based",
            similarity,
        )
        return GenerationResult(bundle=bundle, prompt=prompt, rationale=rationale, similarity=similarity)

    # ---------------- internal helpers -----------------
    def _parse_llm_output(self, text: str) -> List[CodeArtifact]:
        artifacts: List[CodeArtifact] = []
        lines = text.splitlines()
        path: Optional[str] = None
        buf: List[str] = []
        for line in lines:
            if line.startswith("=== file:") and line.endswith(" ==="):
                if path is not None:
                    artifacts.append(CodeArtifact(path=path, content="\n".join(buf)))
                    buf = []
                path = line[len("=== file:") : -len(" ===")].strip()
            elif line.strip() == "=== end ===":
                if path is not None:
                    artifacts.append(CodeArtifact(path=path, content="\n".join(buf)))
                path, buf = None, []
            else:
                buf.append(line)

        if path is not None:
            artifacts.append(CodeArtifact(path=path, content="\n".join(buf)))
        return artifacts

    def _rule_based_generation(self, task_spec: str, review_hint: str) -> Tuple[CodeBundle, str]:
        spec = task_spec.lower()
        artifacts: List[CodeArtifact]

        if "hello" in spec and "world" in spec:
            artifacts = [
                CodeArtifact(
                    path="app.py",
                    content=(
                        "def main():\n"
                        "    print('Hello, world!')\n\n"
                        "if __name__ == '__main__':\n"
                        "    main()\n"
                    ),
                )
            ]
            rationale = "Default hello-world template"
        elif "cli" in spec or "command line" in spec:
            artifacts = [
                CodeArtifact(
                    path="cli.py",
                    content=(
                        "import argparse\n\n"
                        "def run(name: str) -> None:\n"
                        "    print(f'Hi, {name}!')\n\n"
                        "def build_parser():\n"
                        "    parser = argparse.ArgumentParser(description='Sample CLI')\n"
                        "    parser.add_argument('--name', default='world')\n"
                        "    return parser\n\n"
                        "def main():\n"
                        "    args = build_parser().parse_args()\n"
                        "    run(args.name)\n\n"
                        "if __name__ == '__main__':\n"
                        "    main()\n"
                    ),
                )
            ]
            rationale = "Rule-based CLI scaffold"
        else:
            summary = review_hint or "No review hints provided."
            artifacts = [
                CodeArtifact(
                    path="GENERATED_NOTES.md",
                    content=(
                        "# Task Notes\n\n"
                        f"Task: {task_spec}\n\n"
                        f"Hints: {summary}\n\n"
                        "LLM backend not connected; provided structured notes instead.\n"
                    ),
                )
            ]
            rationale = "Fallback note generation"

        return CodeBundle(artifacts=artifacts), rationale

    def _estimate_similarity(self, previous: Optional[CodeBundle], current: CodeBundle) -> float:
        if not previous:
            return 0.0
        prev_text = previous.as_text()
        curr_text = current.as_text()
        return difflib.SequenceMatcher(a=prev_text, b=curr_text).ratio()


__all__ = ["CodeArtifact", "CodeBundle", "GenerationResult", "CoderAgent"]
