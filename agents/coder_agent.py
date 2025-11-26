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
    """单个生成文件的路径与内容。"""

    path: str
    content: str


@dataclass
class CodeBundle:
    """多文件容器，便于一次性写入或转换文本。"""

    artifacts: List[CodeArtifact]

    def write_to(self, root: Path) -> None:
        # 将 bundle 写入指定根目录，自动创建中间目录
        for art in self.artifacts:
            file_path = root / art.path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(art.content, encoding="utf-8")

    def as_text(self) -> str:
        # 以统一格式把多个文件拼成文本，便于提示或审查
        parts = []
        for art in self.artifacts:
            parts.append(f"=== file:{art.path} ===\n{art.content}\n=== end ===")
        return "\n\n".join(parts)

    def artifact_map(self) -> Dict[str, str]:
        # 生成 path -> content 的字典映射
        return {art.path: art.content for art in self.artifacts}


@dataclass
class GenerationResult:
    """Coder 输出的结构化结果，包含生成的 bundle 与元信息。"""

    bundle: CodeBundle
    prompt: str
    rationale: str
    similarity: float


class CoderAgent:
    """可插拔 LLM 后端的轻量代码生成 Agent。"""

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
            # 使用外部 LLM 回调生成代码
            raw = self.llm_call(prompt)
            bundle = CodeBundle(self._parse_llm_output(raw))
            rationale = "LLM-backed generation"
        else:
            # 否则走内置规则生成简易模板
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
        # 解析 “=== file: path === ... === end ===” 形式的多文件输出
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
        # 简单规则：根据任务关键词返回模板或备忘笔记
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
        # 基于文本 diff 估算两次生成的相似度，供 RL 奖励使用
        if not previous:
            return 0.0
        prev_text = previous.as_text()
        curr_text = current.as_text()
        return difflib.SequenceMatcher(a=prev_text, b=curr_text).ratio()


__all__ = ["CodeArtifact", "CodeBundle", "GenerationResult", "CoderAgent"]
