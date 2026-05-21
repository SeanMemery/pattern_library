from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from pattern_library_schema import PatternEvidence, PatternLibrary

DEFAULT_PATTERN_EVIDENCE_RELPATH = "trace.evidence.json"
DEFAULT_PATTERN_ANNOTATION_LIMIT = 50
DEFAULT_PATTERN_ANNOTATION_LIMITS_BY_ENV = {
    "phyre": DEFAULT_PATTERN_ANNOTATION_LIMIT,
    "iphyre": DEFAULT_PATTERN_ANNOTATION_LIMIT,
    "kinetix": DEFAULT_PATTERN_ANNOTATION_LIMIT,
    "pooltool": DEFAULT_PATTERN_ANNOTATION_LIMIT,
}

def pattern_evidence_path(
    sample_dir: Path,
    *,
    evidence_relpath: str = DEFAULT_PATTERN_EVIDENCE_RELPATH,
) -> Path:
    return sample_dir / evidence_relpath


def build_pattern_evidence(
    library: PatternLibrary,
    *,
    patterns: list,
    metadata: dict | None = None,
    pattern_limit: int | None = None,
) -> dict:
    resolved_limit = _resolve_pattern_annotation_limit(library, pattern_limit=pattern_limit)
    sampling_strategy = "full"
    limited_patterns = _select_pattern_annotations(
        library,
        patterns=patterns,
        metadata=metadata,
        pattern_limit=resolved_limit,
    )
    evidence_metadata = dict(metadata or {})
    evidence_metadata.setdefault("total_pattern_count", len(patterns))
    if resolved_limit is not None:
        evidence_metadata.setdefault("pattern_limit", resolved_limit)
        evidence_metadata.setdefault("patterns_truncated", len(patterns) > resolved_limit)
        if len(patterns) > resolved_limit:
            weighted_payload = _supervised_importance_payload(library)
            if (
                isinstance(weighted_payload, dict)
                and str(evidence_metadata.get("source") or "").strip() != "supervised-detectors"
            ):
                sampling_strategy = "weighted_importance_sample"
            else:
                sampling_strategy = "random_sample"
            evidence_metadata.setdefault("pattern_sampling_strategy", sampling_strategy)
        else:
            evidence_metadata.setdefault("pattern_sampling_strategy", sampling_strategy)
    evidence = PatternEvidence(
        library_id=library.library_id,
        schema_version=library.schema_version,
        patterns=[_strip_pattern_details(pattern) for pattern in limited_patterns],
        metadata=evidence_metadata,
    )
    return {"pattern_annotations": evidence.model_dump(mode="json")}


def write_pattern_evidence(
    sample_dir: Path,
    library: PatternLibrary,
    *,
    patterns: list,
    metadata: dict | None = None,
    evidence_relpath: str = DEFAULT_PATTERN_EVIDENCE_RELPATH,
    pattern_limit: int | None = None,
) -> Path:
    output_path = pattern_evidence_path(sample_dir, evidence_relpath=evidence_relpath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_pattern_evidence(
        library,
        patterns=patterns,
        metadata=metadata,
        pattern_limit=pattern_limit,
    )
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def load_pattern_evidence(
    sample_dir: Path,
    *,
    evidence_relpath: str = DEFAULT_PATTERN_EVIDENCE_RELPATH,
) -> PatternEvidence:
    evidence_path = pattern_evidence_path(sample_dir, evidence_relpath=evidence_relpath)
    with evidence_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    annotations = payload.get("pattern_annotations")
    if not isinstance(annotations, dict):
        raise ValueError(f"Pattern evidence missing pattern_annotations object: {evidence_path}")
    return PatternEvidence.model_validate(annotations)


def _resolve_pattern_annotation_limit(
    library: PatternLibrary,
    *,
    pattern_limit: int | None,
) -> int | None:
    if pattern_limit is not None:
        return max(0, int(pattern_limit))
    for env_id in library.environment_ids:
        if env_id in DEFAULT_PATTERN_ANNOTATION_LIMITS_BY_ENV:
            return DEFAULT_PATTERN_ANNOTATION_LIMITS_BY_ENV[env_id]
    return DEFAULT_PATTERN_ANNOTATION_LIMIT


def _select_pattern_annotations(
    library: PatternLibrary,
    *,
    patterns: list,
    metadata: dict | None,
    pattern_limit: int | None,
) -> list:
    selected_patterns = list(patterns)
    if pattern_limit is None or len(selected_patterns) <= pattern_limit:
        return selected_patterns

    if isinstance(metadata, dict) and str(metadata.get("source") or "").strip() == "supervised-detectors":
        weighted_patterns = None
    else:
        weighted_patterns = _weighted_pattern_sample(
            library,
            patterns=selected_patterns,
            metadata=metadata,
            pattern_limit=pattern_limit,
        )
    if weighted_patterns is not None:
        return weighted_patterns

    seed_parts = [library.library_id]
    if isinstance(metadata, dict):
        for key in ("trace_id", "task_id", "sample_id"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                seed_parts.append(value.strip())
    rng = random.Random("|".join(seed_parts))
    sampled_indices = sorted(rng.sample(range(len(selected_patterns)), pattern_limit))
    return [selected_patterns[index] for index in sampled_indices]


def _strip_pattern_details(pattern: object) -> object:
    preserved_metadata_keys = {
        "consensus_label",
        "consensus_input_mode",
        "confidence",
        "supporting_detector_ids",
        "supporting_slot_indices",
        "proposal_count",
        "consensus_method",
    }
    if isinstance(pattern, dict):
        cleaned = dict(pattern)
        cleaned["provenance"] = {}
        if isinstance(cleaned.get("parameters"), dict):
            cleaned["parameters"] = _round_annotation_value(cleaned["parameters"])
        metadata = cleaned.get("metadata")
        if isinstance(metadata, dict):
            cleaned["metadata"] = _round_annotation_value({
                key: metadata[key]
                for key in preserved_metadata_keys
                if key in metadata
            })
        else:
            cleaned["metadata"] = {}
        return cleaned
    try:
        parameters = pattern.parameters if isinstance(getattr(pattern, "parameters", None), dict) else {}
        metadata = pattern.metadata if isinstance(getattr(pattern, "metadata", None), dict) else {}
        preserved_metadata = _round_annotation_value({
            key: metadata[key]
            for key in preserved_metadata_keys
            if key in metadata
        })
        return pattern.model_copy(
            update={
                "parameters": _round_annotation_value(parameters),
                "provenance": {},
                "metadata": preserved_metadata,
            }
        )
    except AttributeError:
        return pattern


def _round_annotation_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, list):
        return [_round_annotation_value(item) for item in value]
    if isinstance(value, tuple):
        return [_round_annotation_value(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _round_annotation_value(item)
            for key, item in value.items()
        }
    return value


def _weighted_pattern_sample(
    library: PatternLibrary,
    *,
    patterns: list,
    metadata: dict | None,
    pattern_limit: int,
) -> list | None:
    importance_payload = _supervised_importance_payload(library)
    if importance_payload is None:
        return None
    labels = importance_payload.get("labels")
    if not isinstance(labels, list) or not labels:
        return None
    weight_lookup: dict[str, float] = {}
    for item in labels:
        if not isinstance(item, dict):
            continue
        key = _importance_key(
            input_mode=str(item.get("input_mode") or "trace"),
            detector_id=str(item.get("detector_id") or ""),
            label=str(item.get("label") or ""),
        )
        try:
            weight_lookup[key] = max(0.0, float(item.get("sampling_weight") or 0.0))
        except (TypeError, ValueError):
            continue
    if not weight_lookup:
        return None

    seed_parts = [library.library_id, "importance-weighted"]
    if isinstance(metadata, dict):
        for key in ("trace_id", "task_id", "sample_id"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                seed_parts.append(value.strip())
    rng = random.Random("|".join(seed_parts))

    weighted_entries: list[tuple[float, int]] = []
    for index, pattern in enumerate(patterns):
        key = _pattern_importance_key(pattern)
        base_weight = 1.0
        if key is not None:
            base_weight = max(1.0, float(weight_lookup.get(key, 1.0)))
        draw = max(rng.random(), 1e-12)
        priority = math.log(draw) / base_weight
        weighted_entries.append((priority, index))
    selected_indices = sorted(index for _priority, index in sorted(weighted_entries, reverse=True)[:pattern_limit])
    return [patterns[index] for index in selected_indices]


def _supervised_importance_payload(library: PatternLibrary) -> dict[str, Any] | None:
    payload = library.metadata.get("supervised_label_importance")
    return payload if isinstance(payload, dict) else None


def _pattern_importance_key(pattern: object) -> str | None:
    if isinstance(pattern, dict):
        detector_id = str(pattern.get("detector_id") or "").strip()
        label = str(pattern.get("label") or "").strip()
        metadata = pattern.get("metadata")
    else:
        detector_id = str(getattr(pattern, "detector_id", "") or "").strip()
        label = str(getattr(pattern, "label", "") or "").strip()
        metadata = getattr(pattern, "metadata", None)
    input_mode = ""
    if isinstance(metadata, dict):
        input_mode = str(metadata.get("detector_input_mode") or "").strip()
    if input_mode != "trace" or not detector_id or not label:
        return None
    return _importance_key(input_mode=input_mode, detector_id=detector_id, label=label)


def _importance_key(*, input_mode: str, detector_id: str, label: str) -> str:
    return f"{input_mode.strip()}::{detector_id.strip()}::{label.strip()}"
