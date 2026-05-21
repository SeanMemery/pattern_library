from __future__ import annotations

import importlib
import inspect
import math
from typing import Any, Callable, Optional

from pattern_library_evidence import build_pattern_evidence
from pattern_library_schema import (
    DetectorDefinition,
    DetectorSlotDefinition,
    LabelDetectorEnsemble,
    Pattern,
    PatternLibrary,
)
CanonicalTrace = Any


DetectorCallable = Callable[[dict[str, Any], Optional[dict[str, Any]], dict[str, Any]], list[Any]]
DEFAULT_ENSEMBLE_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_TIME_CLUSTER_THRESHOLD = 5
DEFAULT_LOCATION_CLUSTER_THRESHOLD = 0.05
DEFAULT_ENV_CONSENSUS_THRESHOLDS = {
    "phyre": {
        "start_time_threshold": 5,
        "end_time_threshold": 5,
        "start_location_threshold": 0.05,
        "end_location_threshold": 0.05,
        "confidence_threshold": 0.5,
    },
    "iphyre": {
        "start_time_threshold": 5,
        "end_time_threshold": 5,
        "start_location_threshold": 0.05,
        "end_location_threshold": 0.05,
        "confidence_threshold": 0.5,
    },
    "kinetix": {
        "start_time_threshold": 7,
        "end_time_threshold": 7,
        "start_location_threshold": 5.0,
        "end_location_threshold": 5.0,
        "confidence_threshold": 0.35,
    },
    "pooltool": {
        "start_time_threshold": 7,
        "end_time_threshold": 7,
        "start_location_threshold": 0.05,
        "end_location_threshold": 0.05,
        "confidence_threshold": 0.35,
    },
}


def apply_pattern_library(
    trace: CanonicalTrace,
    library: PatternLibrary,
    *,
    existing_evidence: dict[str, Any] | None = None,
) -> list[Pattern]:
    if trace.environment not in set(library.environment_ids):
        raise ValueError(
            f"Pattern library {library.library_id} does not support environment {trace.environment}"
        )

    if not _library_uses_consensus(library):
        return _apply_pattern_library_legacy(
            trace,
            library,
            existing_evidence=existing_evidence,
        )

    patterns: list[Pattern] = []
    supervised_ensembles = _ensembles_for_input_mode(library, "trace")
    unsupervised_ensembles = _ensembles_for_input_mode(library, "supervised_evidence")
    for ensemble in supervised_ensembles:
        patterns.extend(
            _apply_label_ensemble(
                trace,
                ensemble,
                library=library,
                existing_evidence=existing_evidence,
            )
        )
    if unsupervised_ensembles:
        chained_evidence = build_pattern_evidence(
            library,
            patterns=patterns,
            metadata={"source": "supervised-detectors"},
        )
        if existing_evidence:
            chained_evidence = _merge_evidence_payloads(existing_evidence, chained_evidence)
        for ensemble in unsupervised_ensembles:
            patterns.extend(
                _apply_label_ensemble(
                    trace,
                    ensemble,
                    library=library,
                    existing_evidence=chained_evidence,
                )
            )
    return sorted(patterns, key=_pattern_sort_key)


def _apply_pattern_library_legacy(
    trace: CanonicalTrace,
    library: PatternLibrary,
    *,
    existing_evidence: dict[str, Any] | None = None,
) -> list[Pattern]:
    patterns: list[Pattern] = []
    supervised_detectors = [
        detector for detector in library.detectors if detector.input_mode == "trace"
    ]
    unsupervised_detectors = [
        detector
        for detector in library.detectors
        if detector.input_mode == "supervised_evidence"
    ]

    for detector in supervised_detectors:
        if detector.environment_ids and trace.environment not in set(detector.environment_ids):
            continue
        try:
            detector_patterns = apply_detector(
                trace,
                detector,
                existing_evidence=existing_evidence,
            )
        except Exception:
            # Generated detectors can be malformed. Skip only the failing detector
            # so one bad primitive does not abort the whole pattern application path.
            continue
        patterns.extend(detector_patterns)
    if unsupervised_detectors:
        chained_evidence = build_pattern_evidence(
            library,
            patterns=patterns,
            metadata={"source": "supervised-detectors"},
        )
        if existing_evidence:
            chained_evidence = _merge_evidence_payloads(existing_evidence, chained_evidence)
        for detector in unsupervised_detectors:
            if detector.environment_ids and trace.environment not in set(detector.environment_ids):
                continue
            try:
                detector_patterns = apply_detector(
                    trace,
                    detector,
                    existing_evidence=chained_evidence,
                )
            except Exception:
                continue
            patterns.extend(detector_patterns)
    return sorted(patterns, key=_pattern_sort_key)


def _library_uses_consensus(library: PatternLibrary) -> bool:
    strategy = str(library.metadata.get("annotation_strategy") or "").strip().lower()
    if strategy == "ensemble_consensus":
        return True
    return any(len(ensemble.members) > 1 for ensemble in library.label_ensembles)


def _ensembles_for_input_mode(
    library: PatternLibrary,
    input_mode: str,
) -> list[LabelDetectorEnsemble]:
    return [
        ensemble
        for ensemble in library.label_ensembles
        if ensemble.input_mode == input_mode and ensemble.members
    ]


def _apply_label_ensemble(
    trace: CanonicalTrace,
    ensemble: LabelDetectorEnsemble,
    *,
    library: PatternLibrary,
    existing_evidence: dict[str, Any] | None,
) -> list[Pattern]:
    if len(ensemble.members) <= 1:
        patterns: list[Pattern] = []
        for member in ensemble.members:
            detector = member.detector
            if detector.environment_ids and trace.environment not in set(detector.environment_ids):
                continue
            try:
                patterns.extend(
                    apply_detector(
                        trace,
                        detector,
                        existing_evidence=existing_evidence,
                    )
                )
            except Exception:
                continue
        return patterns

    proposals: list[dict[str, Any]] = []
    for member in ensemble.members:
        detector = member.detector
        if detector.environment_ids and trace.environment not in set(detector.environment_ids):
            continue
        try:
            emitted = apply_detector(
                trace,
                detector,
                existing_evidence=existing_evidence,
            )
        except Exception:
            continue
        weight = _detector_slot_weight(library=library, ensemble=ensemble, member=member)
        for pattern in emitted:
            proposals.append(
                {
                    "pattern": pattern,
                    "weight": weight,
                    "slot_index": int(member.slot_index),
                    "detector_id": detector.detector_id,
                }
            )
    if not proposals:
        return []
    calibration = _ensemble_label_calibration(library=library, ensemble=ensemble)
    clusters = _cluster_pattern_proposals(proposals, calibration=calibration)
    emitted_patterns: list[Pattern] = []
    for cluster in clusters:
        consensus = _consensus_pattern_for_cluster(
            cluster,
            ensemble=ensemble,
            calibration=calibration,
        )
        if consensus is not None:
            emitted_patterns.append(consensus)
    return emitted_patterns


def apply_detector(
    trace: CanonicalTrace,
    detector: DetectorDefinition,
    *,
    existing_evidence: dict[str, Any] | None = None,
) -> list[Pattern]:
    detect = _load_detector_callable(detector)
    trace_payload: dict[str, Any]
    if detector.input_mode == "supervised_evidence":
        trace_payload = _sanitized_unsupervised_trace_view(trace)
    else:
        trace_payload = trace.model_dump(mode="json")
    raw_events = detect(
        trace_payload,
        existing_evidence,
        dict(detector.config),
    )
    if not isinstance(raw_events, list):
        raise ValueError(f"Detector {detector.detector_id} returned a non-list result")
    return [normalize_pattern(detector, item, trace=trace) for item in raw_events]


def normalize_pattern(detector: DetectorDefinition, payload: Any, *, trace: CanonicalTrace | None = None) -> Pattern:
    if isinstance(payload, Pattern):
        payload = payload.model_dump(mode="json")
    if not isinstance(payload, dict):
        raise ValueError(f"Detector {detector.detector_id} returned a non-dict pattern")

    label = payload.get("label") or detector.label
    raw_parameters = payload.get("parameters", {})
    parameters = dict(raw_parameters) if isinstance(raw_parameters, dict) else {"value": raw_parameters}
    provenance = payload.get("provenance", {})
    metadata = payload.get("metadata", {})
    if isinstance(metadata, dict):
        metadata = dict(metadata)
    else:
        metadata = {"value": metadata}
    metadata.setdefault("detector_input_mode", detector.input_mode)
    start_timestep = _normalize_timestep_index(
        payload.get("start_timestep", payload.get("start_step")),
        trace=trace,
    )
    end_timestep = _normalize_timestep_index(
        payload.get("end_timestep", payload.get("end_step", start_timestep)),
        trace=trace,
    )
    if start_timestep is None or end_timestep is None:
        raise ValueError(
            f"Detector {detector.detector_id} must emit start_timestep and end_timestep for every pattern"
        )
    if start_timestep is not None and end_timestep is not None and end_timestep < start_timestep:
        start_timestep, end_timestep = end_timestep, start_timestep
    if trace is not None:
        _augment_pattern_parameters_with_locations(
            parameters=parameters,
            trace=trace,
            start_timestep=start_timestep,
            end_timestep=end_timestep,
        )
        _augment_pooltool_ball_pocket_details(
            label=str(label or ""),
            parameters=parameters,
            trace=trace,
            start_timestep=start_timestep,
            end_timestep=end_timestep,
        )

    return Pattern.model_validate(
        {
            "detector_id": payload.get("detector_id", detector.detector_id),
            "label": label,
            "start_timestep": start_timestep,
            "end_timestep": end_timestep,
            "parameters": parameters,
            "provenance": provenance if isinstance(provenance, dict) else {"value": provenance},
            "metadata": metadata,
        }
    )


_POOLTOOL_POCKET_IDS = {"lb", "lc", "lt", "rb", "rc", "rt"}
_POOLTOOL_DEFAULT_TABLE_WIDTH = 0.9906
_POOLTOOL_DEFAULT_TABLE_LENGTH = 1.9812


def _augment_pooltool_ball_pocket_details(
    *,
    label: str,
    parameters: dict[str, Any],
    trace: CanonicalTrace,
    start_timestep: int,
    end_timestep: int,
) -> None:
    if str(trace.environment or "").strip().lower() != "pooltool":
        return
    if str(label or "").strip().lower() != "ball pocketed":
        return

    candidate_ball_ids = _extract_pooltool_ball_ids(parameters)
    if not candidate_ball_ids:
        return

    pocket_id = _normalize_pooltool_pocket_id(parameters.get("pocket_id"))
    if not pocket_id:
        event_match = _find_pooltool_ball_pocket_event(
            trace=trace,
            candidate_ball_ids=candidate_ball_ids,
            start_timestep=start_timestep,
            end_timestep=end_timestep,
        )
        if event_match is not None:
            pocket_id = event_match[1]
    if not pocket_id:
        return

    if not _pooltool_ball_is_confirmed_pocketed(
        trace=trace,
        candidate_ball_ids=candidate_ball_ids,
        start_timestep=start_timestep,
        end_timestep=end_timestep,
    ):
        return

    parameters.setdefault("pocket_id", pocket_id)
    parameters.setdefault("pocket_confirmed", True)
    pocket_position = _pooltool_pocket_position(trace=trace, pocket_id=pocket_id)
    if pocket_position is not None:
        parameters.setdefault("pocket_position", pocket_position)
    object_details = parameters.get("object_details")
    if isinstance(object_details, dict):
        for object_key, raw_detail in object_details.items():
            if str(object_key).strip() not in candidate_ball_ids or not isinstance(raw_detail, dict):
                continue
            raw_detail.setdefault("pocket_id", pocket_id)
            raw_detail.setdefault("pocket_confirmed", True)
            if pocket_position is not None:
                raw_detail.setdefault("pocket_position", dict(pocket_position))


def _extract_pooltool_ball_ids(parameters: dict[str, Any]) -> set[str]:
    ball_ids: set[str] = set()
    for key in ("ball_id", "object_id"):
        value = parameters.get(key)
        if isinstance(value, (str, int)):
            token = str(value).strip()
            if token:
                ball_ids.add(token)
    for key in ("ball_ids", "object_ids"):
        value = parameters.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, (str, int)):
                    token = str(item).strip()
                    if token:
                        ball_ids.add(token)
    object_details = parameters.get("object_details")
    if isinstance(object_details, dict):
        for object_key, raw_detail in object_details.items():
            token = str(object_key).strip()
            if token:
                ball_ids.add(token)
                if token.lower() in _POOLTOOL_POCKET_IDS:
                    ball_ids.discard(token)
            if isinstance(raw_detail, dict):
                identity = raw_detail.get("ball_identity")
                if isinstance(identity, (str, int)):
                    identity_token = str(identity).strip()
                    if identity_token:
                        ball_ids.add(identity_token)
    return ball_ids


def _normalize_pooltool_pocket_id(value: Any) -> str | None:
    token = str(value or "").strip().lower()
    if token in _POOLTOOL_POCKET_IDS:
        return token
    return None


def _pooltool_pocket_position(*, trace: CanonicalTrace, pocket_id: str) -> dict[str, float] | None:
    normalized = _normalize_pooltool_pocket_id(pocket_id)
    if not normalized:
        return None
    initial_state = trace.initial_state if isinstance(trace.initial_state, dict) else {}
    try:
        table_width = float(initial_state.get("table_width") or initial_state.get("table_w") or _POOLTOOL_DEFAULT_TABLE_WIDTH)
    except (TypeError, ValueError):
        table_width = _POOLTOOL_DEFAULT_TABLE_WIDTH
    try:
        table_length = float(initial_state.get("table_length") or initial_state.get("table_l") or _POOLTOOL_DEFAULT_TABLE_LENGTH)
    except (TypeError, ValueError):
        table_length = _POOLTOOL_DEFAULT_TABLE_LENGTH
    x_value = 0.0 if normalized.startswith("l") else table_width
    if normalized.endswith("b"):
        y_value = 0.0
    elif normalized.endswith("c"):
        y_value = table_length * 0.5
    else:
        y_value = table_length
    return {"x": float(x_value), "y": float(y_value)}


def _find_pooltool_ball_pocket_event(
    *,
    trace: CanonicalTrace,
    candidate_ball_ids: set[str],
    start_timestep: int,
    end_timestep: int,
) -> tuple[str, str] | None:
    if not candidate_ball_ids:
        return None
    target_time = 0.0
    if 0 <= start_timestep < len(trace.timesteps):
        target_time = float(trace.timesteps[start_timestep].t or 0.0)
    best_match: tuple[float, str, str] | None = None
    for event in trace.events:
        if not isinstance(event.payload, dict):
            continue
        kind = str(event.kind or "").strip().lower()
        if kind != "physics_ball_pocket":
            continue
        ids = event.payload.get("ids")
        tokens: list[str] = []
        if isinstance(ids, list):
            tokens = [str(item).strip() for item in ids if str(item).strip()]
        elif isinstance(ids, (str, int)):
            tokens = [str(ids).strip()]
        if not tokens:
            continue
        ball_id = next((token for token in tokens if token in candidate_ball_ids), None)
        pocket_id = next(
            (
                normalized
                for normalized in (_normalize_pooltool_pocket_id(token) for token in tokens)
                if normalized
            ),
            None,
        )
        if not ball_id or not pocket_id:
            continue
        event_time = float(event.t or 0.0)
        distance = abs(event_time - target_time)
        if best_match is None or distance < best_match[0]:
            best_match = (distance, ball_id, pocket_id)
    if best_match is None:
        return None
    return best_match[1], best_match[2]


def _pooltool_ball_is_confirmed_pocketed(
    *,
    trace: CanonicalTrace,
    candidate_ball_ids: set[str],
    start_timestep: int,
    end_timestep: int,
) -> bool:
    if not candidate_ball_ids:
        return False
    outcome = trace.outcome if isinstance(trace.outcome, dict) else {}
    outcome_metrics = outcome.get("metrics") if isinstance(outcome.get("metrics"), dict) else {}
    confirmed_ids = {
        str(item).strip()
        for item in (outcome_metrics.get("potted_balls") or [])
        if str(item).strip()
    }
    for event in trace.events:
        if str(event.kind or "").strip().lower() != "shot_outcome":
            continue
        payload = event.payload if isinstance(event.payload, dict) else {}
        confirmed_ids.update(str(item).strip() for item in (payload.get("potted_balls") or []) if str(item).strip())
    if any(ball_id in confirmed_ids for ball_id in candidate_ball_ids):
        return True
    max_index = min(len(trace.timesteps) - 1, max(start_timestep, end_timestep) + 3)
    min_index = max(0, min(start_timestep, end_timestep))
    for idx in range(min_index, max_index + 1):
        timestep = trace.timesteps[idx]
        for obj in timestep.objects:
            ball_id = str(getattr(obj, "ball_id", "") or "").strip()
            if ball_id not in candidate_ball_ids:
                continue
            motion_state = str(getattr(obj, "motion_state", "") or "").strip().lower()
            if motion_state == "pocketed":
                return True
    return False


def _augment_pattern_parameters_with_locations(
    *,
    parameters: dict[str, Any],
    trace: CanonicalTrace,
    start_timestep: int,
    end_timestep: int,
) -> None:
    object_indices = _extract_object_indices(parameters)
    start_map = _object_positions_at_step(trace, start_timestep)
    end_map = _object_positions_at_step(trace, end_timestep)
    start_info = _object_info_at_step(trace, start_timestep)
    end_info = _object_info_at_step(trace, end_timestep)

    if not object_indices:
        moving = _detect_moving_objects(start_map, end_map)
        if moving:
            object_indices = moving
        else:
            object_indices = sorted(start_map.keys())

    object_positions: list[dict[str, Any]] = []
    object_details: dict[str, dict[str, Any]] = {}
    roles_by_index = _extract_object_roles(parameters)
    for object_index in object_indices:
        start_pos = start_map.get(object_index)
        end_pos = end_map.get(object_index)
        if start_pos is None and end_pos is None:
            continue
        object_positions.append(
            {
                "object_index": object_index,
                "start": _format_position(start_pos),
                "end": _format_position(end_pos),
            }
        )
        info_start = start_info.get(object_index, {})
        info_end = end_info.get(object_index, {})
        object_details[str(object_index)] = {
            "roles": roles_by_index.get(object_index, ["participant"]),
            "shape": info_start.get("shape") or info_end.get("shape"),
            "color": info_start.get("color") or info_end.get("color"),
            "start": _format_position(start_pos),
            "end": _format_position(end_pos),
        }

    parameters.setdefault("object_indices", object_indices)
    parameters.setdefault("object_positions", object_positions)
    if object_details:
        parameters.setdefault("object_details", object_details)


def _extract_object_indices(parameters: dict[str, Any]) -> list[int]:
    keys = (
        "object_index",
        "object_indices",
        "object_id",
        "object_ids",
        "target_object_index",
        "source_object_index",
        "object_a",
        "object_b",
    )
    indices: set[int] = set()
    for key in keys:
        value = parameters.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            indices.add(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, int):
                    indices.add(item)
        elif isinstance(value, dict):
            for item in value.values():
                if isinstance(item, int):
                    indices.add(item)
    return sorted(indices)


def _extract_object_roles(parameters: dict[str, Any]) -> dict[int, list[str]]:
    role_map: dict[int, list[str]] = {}
    role_keys = {
        "object_index": "primary",
        "target_object_index": "target",
        "source_object_index": "source",
        "object_a": "object_a",
        "object_b": "object_b",
        "object_id": "object",
    }
    for key, role in role_keys.items():
        value = parameters.get(key)
        if isinstance(value, int):
            role_map.setdefault(value, []).append(role)
    list_keys = {
        "object_indices": "participant",
        "object_ids": "participant",
    }
    for key, role in list_keys.items():
        value = parameters.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, int):
                    role_map.setdefault(item, []).append(role)
    return role_map


def _object_positions_at_step(trace: CanonicalTrace, timestep: int) -> dict[int, dict[str, float]]:
    if timestep < 0 or timestep >= len(trace.timesteps):
        return {}
    step = trace.timesteps[timestep]
    positions: dict[int, dict[str, float]] = {}
    for obj in step.objects:
        if not isinstance(obj, dict):
            continue
        idx = obj.get("object_index")
        if not isinstance(idx, int):
            continue
        pos: dict[str, float] = {}
        for axis in ("x", "y", "z"):
            value = obj.get(axis)
            if isinstance(value, (int, float)):
                pos[axis] = float(value)
        if pos:
            positions[idx] = pos
    return positions


def _object_info_at_step(trace: CanonicalTrace, timestep: int) -> dict[int, dict[str, Any]]:
    if timestep < 0 or timestep >= len(trace.timesteps):
        return {}
    step = trace.timesteps[timestep]
    info_map: dict[int, dict[str, Any]] = {}
    for obj in step.objects:
        if not isinstance(obj, dict):
            continue
        idx = obj.get("object_index")
        if not isinstance(idx, int):
            continue
        info: dict[str, Any] = {}
        shape = obj.get("shape")
        color = obj.get("color")
        if isinstance(shape, str):
            info["shape"] = shape
        if isinstance(color, str):
            info["color"] = color
        info_map[idx] = info
    return info_map


def _detect_moving_objects(
    start_map: dict[int, dict[str, float]],
    end_map: dict[int, dict[str, float]],
    *,
    epsilon: float = 1e-4,
) -> list[int]:
    indices: list[int] = []
    for idx, start_pos in start_map.items():
        end_pos = end_map.get(idx)
        if end_pos is None:
            continue
        if _position_distance(start_pos, end_pos) > epsilon:
            indices.append(idx)
    return sorted(indices)


def _position_distance(start_pos: dict[str, float], end_pos: dict[str, float]) -> float:
    dx = float(end_pos.get("x", 0.0)) - float(start_pos.get("x", 0.0))
    dy = float(end_pos.get("y", 0.0)) - float(start_pos.get("y", 0.0))
    dz = float(end_pos.get("z", 0.0)) - float(start_pos.get("z", 0.0))
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def _format_position(pos: dict[str, float] | None) -> dict[str, float] | None:
    if not pos:
        return None
    formatted: dict[str, float] = {}
    for axis in ("x", "y", "z"):
        if axis in pos:
            formatted[axis] = round(float(pos[axis]), 3)
    return formatted or None


def _normalize_timestep_index(value: Any, *, trace: CanonicalTrace | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Detector timestep index must be an int, got {value!r}")
    if value < 0:
        raise ValueError(f"Detector timestep index must be >= 0, got {value}")
    if trace is not None and trace.timesteps and value >= len(trace.timesteps):
        raise ValueError(
            f"Detector timestep index {value} is out of range for trace {trace.trace_id} with {len(trace.timesteps)} steps"
        )
    return value


def _pattern_sort_key(item: Pattern) -> tuple[int, int, str, str]:
    return item.start_timestep, item.end_timestep, item.label, item.detector_id


def _load_detector_callable(detector: DetectorDefinition) -> DetectorCallable:
    if detector.source_code:
        namespace: dict[str, Any] = {}
        exec(detector.source_code, {}, namespace)
        detect = namespace.get("detect")
        if callable(detect):
            return detect
        find_pattern = namespace.get("find_pattern")
        if callable(find_pattern):
            if detector.input_mode == "supervised_evidence":
                return _wrap_unsupervised_find_pattern(find_pattern)
            return lambda trace, _existing_evidence=None, _config=None: find_pattern(trace)
        raise ValueError(
            f"Detector {detector.detector_id} source_code must define a callable detect() or find_pattern(trace)"
        )

    if detector.entrypoint:
        module_name, _, attr_name = detector.entrypoint.partition(":")
        if not module_name or not attr_name:
            raise ValueError(
                f"Detector {detector.detector_id} entrypoint must use module:function syntax"
            )
        module = importlib.import_module(module_name)
        detect = getattr(module, attr_name, None)
        if callable(detect):
            return detect
        raise ValueError(f"Detector {detector.detector_id} entrypoint is not callable: {detector.entrypoint}")

    raise ValueError(
        f"Detector {detector.detector_id} must define either source_code or entrypoint"
    )


def _wrap_unsupervised_find_pattern(find_pattern: Callable[..., list[Any]]) -> DetectorCallable:
    parameter_count = len(inspect.signature(find_pattern).parameters)
    if parameter_count == 1:
        return lambda _trace, existing_evidence=None, _config=None: find_pattern(
            _patterns_from_evidence(existing_evidence)
        )
    if parameter_count == 2:
        return lambda _trace, existing_evidence=None, config=None: find_pattern(
            _patterns_from_evidence(existing_evidence),
            config or {},
        )
    return lambda trace, existing_evidence=None, config=None: find_pattern(
        trace,
        existing_evidence,
        config or {},
    )


def _patterns_from_evidence(existing_evidence: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(existing_evidence, dict):
        return []
    annotations = existing_evidence.get("pattern_annotations")
    if isinstance(annotations, dict):
        patterns = annotations.get("patterns")
        if isinstance(patterns, list):
            return [item for item in patterns if isinstance(item, dict)]
    patterns = existing_evidence.get("patterns")
    if isinstance(patterns, list):
        return [item for item in patterns if isinstance(item, dict)]
    return []


def _ensemble_label_calibration(
    *,
    library: PatternLibrary,
    ensemble: LabelDetectorEnsemble,
) -> dict[str, Any]:
    env_id = str((library.environment_ids or [""])[0] or "").strip().lower()
    default_thresholds = dict(DEFAULT_ENV_CONSENSUS_THRESHOLDS.get(env_id, {}))
    if not default_thresholds:
        default_thresholds = {
            "start_time_threshold": DEFAULT_TIME_CLUSTER_THRESHOLD,
            "end_time_threshold": DEFAULT_TIME_CLUSTER_THRESHOLD,
            "start_location_threshold": DEFAULT_LOCATION_CLUSTER_THRESHOLD,
            "end_location_threshold": DEFAULT_LOCATION_CLUSTER_THRESHOLD,
            "confidence_threshold": DEFAULT_ENSEMBLE_CONFIDENCE_THRESHOLD,
        }
    calibration_payload = (
        dict(library.metadata.get("ensemble_calibration"))
        if isinstance(library.metadata.get("ensemble_calibration"), dict)
        else {}
    )
    labels = calibration_payload.get("labels")
    if isinstance(labels, list):
        for item in labels:
            if not isinstance(item, dict):
                continue
            if str(item.get("label") or "").strip() != ensemble.label:
                continue
            input_mode = str(item.get("input_mode") or ensemble.input_mode).strip() or ensemble.input_mode
            if input_mode != ensemble.input_mode:
                continue
            merged = dict(item)
            threshold_payload = dict(merged.get("thresholds")) if isinstance(merged.get("thresholds"), dict) else {}
            merged["thresholds"] = {**default_thresholds, **threshold_payload}
            return merged
    return {"thresholds": default_thresholds}


def _detector_slot_weight(
    *,
    library: PatternLibrary,
    ensemble: LabelDetectorEnsemble,
    member: DetectorSlotDefinition,
) -> float:
    reliability = 1.0
    calibration = _ensemble_label_calibration(library=library, ensemble=ensemble)
    slot_weights = calibration.get("slot_weights")
    if isinstance(slot_weights, list):
        slot_index = int(member.slot_index)
        if 0 <= slot_index < len(slot_weights):
            try:
                reliability = _bounded_weight(float(slot_weights[slot_index]))
            except (TypeError, ValueError):
                reliability = 1.0
    return _bounded_weight(reliability * _detector_base_reward(member))


def _detector_base_reward(member: DetectorSlotDefinition) -> float:
    if member.train_score is not None:
        return _bounded_weight(member.train_score)
    metadata = member.detector.metadata if isinstance(member.detector.metadata, dict) else {}
    for key in ("base_reward", "train_score", "score", "val_score", "initial_weight"):
        try:
            value = float(metadata.get(key))
        except (TypeError, ValueError):
            continue
        return _bounded_weight(value)
    if member.initial_weight is not None:
        return _bounded_weight(member.initial_weight)
    return 1.0


def _bounded_weight(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _cluster_pattern_proposals(
    proposals: list[dict[str, Any]],
    *,
    calibration: dict[str, Any],
) -> list[list[dict[str, Any]]]:
    if not proposals:
        return []
    remaining = set(range(len(proposals)))
    clusters: list[list[dict[str, Any]]] = []
    while remaining:
        seed = next(iter(remaining))
        queue = [seed]
        remaining.remove(seed)
        cluster_indices = [seed]
        while queue:
            current = queue.pop()
            current_pattern = proposals[current]["pattern"]
            newly_connected: list[int] = []
            for other in list(remaining):
                other_pattern = proposals[other]["pattern"]
                if _patterns_cluster_match(current_pattern, other_pattern, calibration=calibration):
                    newly_connected.append(other)
            for other in newly_connected:
                remaining.remove(other)
                queue.append(other)
                cluster_indices.append(other)
        clusters.append([proposals[index] for index in sorted(cluster_indices)])
    return clusters


def _patterns_cluster_match(
    left: Pattern,
    right: Pattern,
    *,
    calibration: dict[str, Any],
) -> bool:
    thresholds = calibration.get("thresholds")
    threshold_payload = thresholds if isinstance(thresholds, dict) else calibration
    start_time_threshold = _coerce_non_negative_int(
        threshold_payload.get("start_time_threshold"),
        default=DEFAULT_TIME_CLUSTER_THRESHOLD,
    )
    end_time_threshold = _coerce_non_negative_int(
        threshold_payload.get("end_time_threshold"),
        default=DEFAULT_TIME_CLUSTER_THRESHOLD,
    )
    start_location_threshold = _coerce_non_negative_float(
        threshold_payload.get("start_location_threshold"),
        default=DEFAULT_LOCATION_CLUSTER_THRESHOLD,
    )
    end_location_threshold = _coerce_non_negative_float(
        threshold_payload.get("end_location_threshold"),
        default=DEFAULT_LOCATION_CLUSTER_THRESHOLD,
    )
    require_object_identity = bool(threshold_payload.get("require_object_identity", True))

    if abs(int(left.start_timestep) - int(right.start_timestep)) > start_time_threshold:
        return False
    if abs(int(left.end_timestep) - int(right.end_timestep)) > end_time_threshold:
        return False

    left_identity = _pattern_object_identity(left)
    right_identity = _pattern_object_identity(right)
    if require_object_identity and left_identity and right_identity and left_identity != right_identity:
        return False

    left_start, left_end = _pattern_centroid_positions(left)
    right_start, right_end = _pattern_centroid_positions(right)
    if (
        left_start is not None
        and right_start is not None
        and _point_distance(left_start, right_start) > start_location_threshold
    ):
        return False
    if (
        left_end is not None
        and right_end is not None
        and _point_distance(left_end, right_end) > end_location_threshold
    ):
        return False
    return True


def _consensus_pattern_for_cluster(
    cluster: list[dict[str, Any]],
    *,
    ensemble: LabelDetectorEnsemble,
    calibration: dict[str, Any],
) -> Pattern | None:
    if not cluster:
        return None
    supporting_weights: dict[int, float] = {}
    for item in cluster:
        supporting_weights.setdefault(int(item["slot_index"]), float(item["weight"]))
    confidence = sum(supporting_weights.values())
    thresholds = calibration.get("thresholds")
    threshold_payload = thresholds if isinstance(thresholds, dict) else calibration
    threshold = _coerce_non_negative_float(
        threshold_payload.get("confidence_threshold"),
        default=DEFAULT_ENSEMBLE_CONFIDENCE_THRESHOLD,
    )
    if confidence <= threshold:
        return None

    representative = max(
        cluster,
        key=lambda item: (
            float(item["weight"]),
            -int(item["pattern"].start_timestep),
            str(item["detector_id"]),
        ),
    )
    averaged_start = int(round(_weighted_average([int(item["pattern"].start_timestep) for item in cluster], weights=[float(item["weight"]) for item in cluster])))
    averaged_end = int(round(_weighted_average([int(item["pattern"].end_timestep) for item in cluster], weights=[float(item["weight"]) for item in cluster])))
    if averaged_end < averaged_start:
        averaged_start, averaged_end = averaged_end, averaged_start
    parameters = _merge_consensus_parameters(
        representative["pattern"],
        cluster,
    )
    metadata = dict(representative["pattern"].metadata or {})
    metadata.update(
        {
            "consensus_label": ensemble.label,
            "consensus_input_mode": ensemble.input_mode,
            "confidence": round(confidence, 6),
            "supporting_detector_ids": sorted({str(item["detector_id"]) for item in cluster}),
            "supporting_slot_indices": sorted(supporting_weights),
            "proposal_count": len(cluster),
            "consensus_method": "weighted_detector_agreement",
        }
    )
    return Pattern.model_validate(
        {
            "detector_id": _consensus_detector_id(ensemble),
            "label": ensemble.label,
            "start_timestep": averaged_start,
            "end_timestep": averaged_end,
            "parameters": parameters,
            "provenance": {},
            "metadata": metadata,
        }
    )


def _consensus_detector_id(ensemble: LabelDetectorEnsemble) -> str:
    safe_label = "".join(ch.lower() if ch.isalnum() else "-" for ch in ensemble.label).strip("-") or "label"
    prefix = "trace" if ensemble.input_mode == "trace" else "evidence"
    return f"consensus-{prefix}-{safe_label}"


def _merge_consensus_parameters(
    representative: Pattern,
    cluster: list[dict[str, Any]],
) -> dict[str, Any]:
    parameters = dict(representative.parameters or {})
    object_indices = sorted(
        {
            int(object_index)
            for item in cluster
            for object_index in _pattern_object_indices(item["pattern"])
        }
    )
    if object_indices:
        parameters["object_indices"] = object_indices

    object_positions = _average_object_positions(cluster)
    if object_positions:
        parameters["object_positions"] = object_positions
        centroid_start = _average_points(
            [
                position.get("start")
                for position in object_positions
                if isinstance(position, dict)
            ]
        )
        centroid_end = _average_points(
            [
                position.get("end")
                for position in object_positions
                if isinstance(position, dict)
            ]
        )
        if centroid_start is not None:
            parameters["start_position"] = centroid_start
        if centroid_end is not None:
            parameters["end_position"] = centroid_end

    object_details = _average_object_details(cluster)
    if object_details:
        parameters["object_details"] = object_details
    return parameters


def _average_object_positions(cluster: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates: dict[int, dict[str, Any]] = {}
    for item in cluster:
        weight = float(item["weight"])
        params = item["pattern"].parameters if isinstance(item["pattern"].parameters, dict) else {}
        raw_positions = params.get("object_positions")
        if not isinstance(raw_positions, list):
            continue
        for raw_position in raw_positions:
            if not isinstance(raw_position, dict):
                continue
            object_index = raw_position.get("object_index")
            if not isinstance(object_index, int):
                continue
            aggregate = aggregates.setdefault(
                object_index,
                {
                    "object_index": object_index,
                    "start_weight": 0.0,
                    "end_weight": 0.0,
                    "start_x": 0.0,
                    "start_y": 0.0,
                    "start_z": 0.0,
                    "end_x": 0.0,
                    "end_y": 0.0,
                    "end_z": 0.0,
                    "start_has_z": False,
                    "end_has_z": False,
                },
            )
            _accumulate_point(aggregate, "start", raw_position.get("start"), weight)
            _accumulate_point(aggregate, "end", raw_position.get("end"), weight)
    rows: list[dict[str, Any]] = []
    for object_index, aggregate in sorted(aggregates.items()):
        rows.append(
            {
                "object_index": object_index,
                "start": _finalize_accumulated_point(aggregate, "start"),
                "end": _finalize_accumulated_point(aggregate, "end"),
            }
        )
    return rows


def _average_object_details(cluster: list[dict[str, Any]]) -> dict[str, Any]:
    details_by_key: dict[str, dict[str, Any]] = {}
    for item in cluster:
        weight = float(item["weight"])
        params = item["pattern"].parameters if isinstance(item["pattern"].parameters, dict) else {}
        raw_details = params.get("object_details")
        if not isinstance(raw_details, dict):
            continue
        for object_key, raw_detail in raw_details.items():
            if not isinstance(raw_detail, dict):
                continue
            aggregate = details_by_key.setdefault(
                str(object_key),
                {
                    "roles": raw_detail.get("roles", raw_detail.get("role", ["participant"])),
                    "shape": raw_detail.get("shape"),
                    "color": raw_detail.get("color"),
                    "start_weight": 0.0,
                    "end_weight": 0.0,
                    "start_x": 0.0,
                    "start_y": 0.0,
                    "start_z": 0.0,
                    "end_x": 0.0,
                    "end_y": 0.0,
                    "end_z": 0.0,
                    "start_has_z": False,
                    "end_has_z": False,
                },
            )
            _accumulate_point(
                aggregate,
                "start",
                raw_detail.get("start_position", raw_detail.get("start")),
                weight,
            )
            _accumulate_point(
                aggregate,
                "end",
                raw_detail.get("end_position", raw_detail.get("end")),
                weight,
            )
    merged: dict[str, Any] = {}
    for object_key, aggregate in sorted(details_by_key.items()):
        merged[object_key] = {
            "roles": aggregate.get("roles"),
            "shape": aggregate.get("shape"),
            "color": aggregate.get("color"),
            "start_position": _finalize_accumulated_point(aggregate, "start"),
            "end_position": _finalize_accumulated_point(aggregate, "end"),
        }
    return merged


def _accumulate_point(target: dict[str, Any], prefix: str, raw_point: Any, weight: float) -> None:
    point = _coerce_point(raw_point)
    if point is None:
        return
    target[f"{prefix}_weight"] = float(target.get(f"{prefix}_weight", 0.0)) + weight
    target[f"{prefix}_x"] = float(target.get(f"{prefix}_x", 0.0)) + float(point["x"]) * weight
    target[f"{prefix}_y"] = float(target.get(f"{prefix}_y", 0.0)) + float(point["y"]) * weight
    if "z" in point:
        target[f"{prefix}_has_z"] = True
        target[f"{prefix}_z"] = float(target.get(f"{prefix}_z", 0.0)) + float(point["z"]) * weight


def _finalize_accumulated_point(source: dict[str, Any], prefix: str) -> dict[str, float] | None:
    total_weight = float(source.get(f"{prefix}_weight", 0.0))
    if total_weight <= 0.0:
        return None
    point = {
        "x": float(source.get(f"{prefix}_x", 0.0)) / total_weight,
        "y": float(source.get(f"{prefix}_y", 0.0)) / total_weight,
    }
    if bool(source.get(f"{prefix}_has_z")):
        point["z"] = float(source.get(f"{prefix}_z", 0.0)) / total_weight
    return point


def _pattern_object_identity(pattern: Pattern) -> tuple[int, ...] | tuple[str, ...] | None:
    params = pattern.parameters if isinstance(pattern.parameters, dict) else {}
    object_indices = params.get("object_indices")
    if isinstance(object_indices, list):
        normalized = [value for value in object_indices if isinstance(value, int)]
        if normalized:
            return tuple(sorted(normalized))
    object_positions = params.get("object_positions")
    if isinstance(object_positions, list):
        normalized = [
            int(item.get("object_index"))
            for item in object_positions
            if isinstance(item, dict) and isinstance(item.get("object_index"), int)
        ]
        if normalized:
            return tuple(sorted(normalized))
    object_details = params.get("object_details")
    if isinstance(object_details, dict) and object_details:
        return tuple(sorted(str(key) for key in object_details.keys()))
    return None


def _pattern_object_indices(pattern: Pattern) -> list[int]:
    identity = _pattern_object_identity(pattern)
    if isinstance(identity, tuple) and identity and isinstance(identity[0], int):
        return [int(item) for item in identity]
    return []


def _pattern_centroid_positions(pattern: Pattern) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    params = pattern.parameters if isinstance(pattern.parameters, dict) else {}
    points_start: list[dict[str, float]] = []
    points_end: list[dict[str, float]] = []
    object_positions = params.get("object_positions")
    if isinstance(object_positions, list):
        for item in object_positions:
            if not isinstance(item, dict):
                continue
            start = _coerce_point(item.get("start"))
            end = _coerce_point(item.get("end"))
            if start is not None:
                points_start.append(start)
            if end is not None:
                points_end.append(end)
    object_details = params.get("object_details")
    if isinstance(object_details, dict):
        for item in object_details.values():
            if not isinstance(item, dict):
                continue
            start = _coerce_point(item.get("start_position", item.get("start")))
            end = _coerce_point(item.get("end_position", item.get("end")))
            if start is not None:
                points_start.append(start)
            if end is not None:
                points_end.append(end)
    direct_start = _coerce_point(params.get("start_position"))
    direct_end = _coerce_point(params.get("end_position"))
    if direct_start is not None:
        points_start.append(direct_start)
    if direct_end is not None:
        points_end.append(direct_end)
    return _average_points(points_start), _average_points(points_end)


def _average_points(points: list[Any]) -> dict[str, float] | None:
    valid_points = [_coerce_point(item) for item in points]
    valid_points = [item for item in valid_points if item is not None]
    if not valid_points:
        return None
    count = float(len(valid_points))
    point = {
        "x": sum(float(item["x"]) for item in valid_points) / count,
        "y": sum(float(item["y"]) for item in valid_points) / count,
    }
    if any("z" in item for item in valid_points):
        z_values = [float(item["z"]) for item in valid_points if "z" in item]
        if z_values:
            point["z"] = sum(z_values) / float(len(z_values))
    return point


def _coerce_point(value: Any) -> dict[str, float] | None:
    if isinstance(value, dict):
        x = value.get("x")
        y = value.get("y")
        z = value.get("z")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            point = {"x": float(x), "y": float(y)}
            if isinstance(z, (int, float)):
                point["z"] = float(z)
            return point
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        x = value[0]
        y = value[1]
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            point = {"x": float(x), "y": float(y)}
            if len(value) >= 3 and isinstance(value[2], (int, float)):
                point["z"] = float(value[2])
            return point
    return None


def _point_distance(left: dict[str, float], right: dict[str, float]) -> float:
    dx = float(left.get("x", 0.0)) - float(right.get("x", 0.0))
    dy = float(left.get("y", 0.0)) - float(right.get("y", 0.0))
    dz = float(left.get("z", 0.0)) - float(right.get("z", 0.0))
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _weighted_average(values: list[int], *, weights: list[float]) -> float:
    if not values or not weights or len(values) != len(weights):
        return float(values[0]) if values else 0.0
    total_weight = sum(float(weight) for weight in weights)
    if total_weight <= 0.0:
        return float(sum(values)) / float(len(values))
    return sum(float(value) * float(weight) for value, weight in zip(values, weights)) / total_weight


def _coerce_non_negative_int(value: Any, *, default: int) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return int(default)


def _coerce_non_negative_float(value: Any, *, default: float) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return float(default)


def _sanitized_unsupervised_trace_view(trace: CanonicalTrace) -> dict[str, Any]:
    return {
        "trace_id": trace.trace_id,
        "environment": trace.environment,
        "task_id": trace.task_id,
        "timestep_count": len(trace.timesteps),
    }


def _merge_evidence_payloads(
    base_payload: dict[str, Any],
    chained_payload: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(base_payload)
    base_annotations = (
        dict(base_payload.get("pattern_annotations"))
        if isinstance(base_payload.get("pattern_annotations"), dict)
        else {}
    )
    chained_annotations = (
        dict(chained_payload.get("pattern_annotations"))
        if isinstance(chained_payload.get("pattern_annotations"), dict)
        else {}
    )
    merged_patterns: list[Any] = []
    for payload in (base_annotations.get("patterns"), chained_annotations.get("patterns")):
        if isinstance(payload, list):
            merged_patterns.extend(payload)
    merged["pattern_annotations"] = {
        **base_annotations,
        **chained_annotations,
        "patterns": merged_patterns,
    }
    return merged
