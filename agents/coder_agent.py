"""
Coder agent implementation.

This module provides a minimal, dependency-light coder agent that turns a
task specification into code artifacts. It is purposely simple so it can run
without external LLM access during development, while exposing an interface
that can be swapped with a model-backed generator later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class CodeArtifact:
    path: str
    content: str


@dataclass
class CodeBundle:
    artifacts: List[CodeArtifact]

    def write_to(self, root: Path) -> None:
        for art in self.artifacts:
            file_path = root / art.path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(art.content, encoding="utf-8")


class CoderAgent:
    """
    A simple coder agent that supports two backends:
    - rule-based templates for offline/dev usage
    - pluggable callable `llm_call(prompt) -> str` for integration
    """

    def __init__(self, llm_call: Optional[callable] = None) -> None:
        self.llm_call = llm_call

    def generate_code(self, task_spec: str, context: Optional[Dict] = None) -> CodeBundle:
        """
        Generate code artifacts from a natural-language task specification.

        If an `llm_call` is provided, it will be used to generate content from a
        prompt; otherwise, a small library of deterministic templates is used.
        """
        context = context or {}

        if self.llm_call:
            prompt = self._build_prompt(task_spec, context)
            output = self.llm_call(prompt)
            artifacts = self._postprocess_llm_output(output)
            return CodeBundle(artifacts=artifacts)

        # Fallback deterministic generation
        return self._rule_based_generation(task_spec, context)

    # ---------------- internal helpers -----------------
    def _build_prompt(self, task_spec: str, context: Dict) -> str:
        sys_hint = (
            "You are a code generation agent. Output valid, runnable code. "
            "Respond ONLY with file blocks in the format:\n\n"
            "=== file:path/to/file ===\n<content>\n=== end ===\n\n"
            "Do not include explanations."
        )
        ctx_str = "\n".join(f"{k}: {v}" for k, v in context.items()) if context else ""
        return f"{sys_hint}\n\nTask:\n{task_spec}\n\nContext:\n{ctx_str}"

    def _postprocess_llm_output(self, text: str) -> List[CodeArtifact]:
        artifacts: List[CodeArtifact] = []
        lines = text.splitlines()
        path: Optional[str] = None
        buf: List[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
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
            i += 1

        # trailing buffer without end sentinel
        if path is not None:
            artifacts.append(CodeArtifact(path=path, content="\n".join(buf)))
        return artifacts

    def _rule_based_generation(self, task_spec: str, context: Dict) -> CodeBundle:
        spec = task_spec.lower()

        # Very simple intents
        if "hello world" in spec and ("python" in spec or "py" in spec or "script" in spec):
            return CodeBundle(
                artifacts=[
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
            )

        if "cli" in spec and ("python" in spec or "py" in spec):
            return CodeBundle(
                artifacts=[
                    CodeArtifact(
                        path="cli.py",
                        content=(
                            "import argparse\n\n"
                            "def run(name: str) -> None:\n"
                            "    print(f'Hi, {name}!')\n\n"
                            "def main():\n"
                            "    p = argparse.ArgumentParser()\n"
                            "    p.add_argument('--name', default='world')\n"
                            "    args = p.parse_args()\n"
                            "    run(args.name)\n\n"
                            "if __name__ == '__main__':\n"
                            "    main()\n"
                        ),
                    )
                ]
            )

        # Default fallback file mirrors the request in a README
        return CodeBundle(
            artifacts=[
                CodeArtifact(
                    path="GENERATED_NOTES.md",
                    content=(
                        "# Task Specification\n\n"
                        f"{task_spec}\n\n"
                        "This repository does not have an online model backend configured. "
                        "The coder agent produced this fallback artifact. "
                        "Provide an LLM callable to `CoderAgent` for richer outputs.\n"
                    ),
                )
            ]
        )


__all__ = ["CodeArtifact", "CodeBundle", "CoderAgent"]
