from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


PATTERN_SCHEMA_VERSION = "patterns.v2"
LEGACY_PATTERN_SCHEMA_VERSION = "patterns.v1"
SUPPORTED_PATTERN_SCHEMA_VERSIONS = {
    PATTERN_SCHEMA_VERSION,
    LEGACY_PATTERN_SCHEMA_VERSION,
}
DetectorInputMode = Literal["trace", "supervised_evidence"]


def default_created_at() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class DetectorDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detector_id: str
    label: str
    description: str | None = None
    environment_ids: list[str] = Field(default_factory=list)
    source_code: str | None = None
    entrypoint: str | None = None
    input_mode: DetectorInputMode = "trace"
    config: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DetectorSlotDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slot_index: int
    detector: DetectorDefinition
    train_score: float | None = None
    cluster_index: int | None = None
    initial_weight: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LabelDetectorEnsemble(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    input_mode: DetectorInputMode = "trace"
    members: list[DetectorSlotDefinition] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Pattern(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detector_id: str
    label: str
    start_timestep: int
    end_timestep: int
    parameters: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PatternEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    library_id: str
    schema_version: str = PATTERN_SCHEMA_VERSION
    patterns: list[Pattern] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PatternTextBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    library_id: str
    schema_version: str = PATTERN_SCHEMA_VERSION
    text: str
    pattern_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class PatternLibrary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = PATTERN_SCHEMA_VERSION
    library_id: str
    environment_ids: list[str] = Field(default_factory=list)
    primitive_source: str | None = None
    scorer_config: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=default_created_at)
    detectors: list[DetectorDefinition] = Field(default_factory=list)
    label_ensembles: list[LabelDetectorEnsemble] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _synchronize_detectors_and_ensembles(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw
        payload = dict(raw)
        raw_detectors = payload.get("detectors")
        raw_ensembles = payload.get("label_ensembles")
        has_detectors = isinstance(raw_detectors, list) and bool(raw_detectors)
        has_ensembles = isinstance(raw_ensembles, list) and bool(raw_ensembles)
        if has_ensembles:
            payload["detectors"] = _flatten_label_ensembles(raw_ensembles)
        elif has_detectors:
            payload["label_ensembles"] = _derive_label_ensembles(raw_detectors)
        else:
            payload.setdefault("detectors", [])
            payload.setdefault("label_ensembles", [])
        return payload

    def model_copy(self, *, update: dict[str, Any] | None = None, deep: bool = False):
        copied = super().model_copy(update=update, deep=deep)
        payload = copied.model_dump(mode="json")
        if isinstance(update, dict):
            if "detectors" in update and "label_ensembles" not in update:
                payload.pop("label_ensembles", None)
            if "label_ensembles" in update and "detectors" not in update:
                payload.pop("detectors", None)
        return type(self).model_validate(payload)


def _flatten_label_ensembles(raw_ensembles: list[Any]) -> list[Any]:
    flattened: list[Any] = []
    for raw_ensemble in raw_ensembles:
        ensemble_payload = _as_payload_dict(raw_ensemble)
        if not ensemble_payload:
            continue
        members = ensemble_payload.get("members")
        if not isinstance(members, list):
            continue
        for raw_member in members:
            member_payload = _as_payload_dict(raw_member)
            if not member_payload:
                continue
            detector = _as_payload_dict(member_payload.get("detector"))
            if detector:
                flattened.append(detector)
    return flattened


def _derive_label_ensembles(raw_detectors: list[Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    slot_counters: dict[tuple[str, str], int] = {}
    for raw_detector in raw_detectors:
        detector_payload = _as_payload_dict(raw_detector)
        if not detector_payload:
            continue
        label = str(detector_payload.get("label") or "").strip()
        input_mode = str(detector_payload.get("input_mode") or "trace").strip() or "trace"
        if input_mode not in {"trace", "supervised_evidence"}:
            input_mode = "trace"
        if not label:
            continue
        key = (input_mode, label)
        ensemble = grouped.setdefault(
            key,
            {
                "label": label,
                "input_mode": input_mode,
                "members": [],
                "metadata": {},
            },
        )
        metadata = dict(detector_payload.get("metadata")) if isinstance(detector_payload.get("metadata"), dict) else {}
        slot_index = int(metadata.get("slot_index") or slot_counters.get(key, 0))
        slot_counters[key] = max(slot_counters.get(key, 0), slot_index + 1)
        ensemble["members"].append(
            {
                "slot_index": slot_index,
                "detector": detector_payload,
                "train_score": _coerce_optional_float(
                    metadata.get("train_score", metadata.get("score"))
                ),
                "cluster_index": _coerce_optional_int(metadata.get("cluster_index")),
                "initial_weight": _coerce_optional_float(metadata.get("initial_weight")),
                "metadata": {},
            }
        )
    for ensemble in grouped.values():
        ensemble["members"] = sorted(
            ensemble["members"],
            key=lambda item: (
                int(item.get("slot_index") or 0),
                str(
                    (
                        dict(item.get("detector"))
                        if isinstance(item.get("detector"), dict)
                        else {}
                    ).get("detector_id")
                    or ""
                ),
            ),
        )
    return list(grouped.values())


def _as_payload_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        if isinstance(dumped, dict):
            return dumped
    return {}


def _coerce_optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
