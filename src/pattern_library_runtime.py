from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pattern_library_registry import load_pattern_library
from pattern_library_schema import PatternLibrary


@dataclass(frozen=True)
class PatternSetRuntime:
    source_path: str
    payload: PatternLibrary

    @property
    def pattern_count(self) -> int:
        return len(self.payload.detectors)

    @property
    def library_id(self) -> str:
        return self.payload.library_id


@dataclass
class ConditionedBenchmarkRequest:
    image_descriptions: list[dict[str, Any]]
    system_prompt: str
    user_prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


def load_pattern_set_runtime(path: str | None) -> PatternSetRuntime | None:
    if not path:
        return None

    source_path = Path(path)
    library = load_pattern_library(source_path)
    return PatternSetRuntime(source_path=str(source_path), payload=library)


def condition_benchmark_request(
    *,
    image_descriptions: list[dict[str, Any]],
    system_prompt: str,
    user_prompt: str,
    pattern_runtime: PatternSetRuntime | None,
    environment_id: str | None,
    task_context: dict[str, Any] | None,
) -> ConditionedBenchmarkRequest:
    conditioned_images = [deepcopy(item) for item in image_descriptions]
    metadata: dict[str, Any] = {}

    if pattern_runtime is not None:
        metadata = {
            "pattern_set_path": pattern_runtime.source_path,
            "pattern_library_id": pattern_runtime.library_id,
            "pattern_count": pattern_runtime.pattern_count,
            "environment_id": environment_id,
            "task_context": deepcopy(task_context or {}),
        }

    return ConditionedBenchmarkRequest(
        image_descriptions=conditioned_images,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        metadata=metadata,
    )
