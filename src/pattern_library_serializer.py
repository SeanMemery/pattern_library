from __future__ import annotations

from typing import Any

from pattern_library_schema import Pattern, PatternEvidence, PatternTextBundle


def serialize_patterns(patterns: list[Pattern], *, max_patterns: int | None = None) -> str:
    prepared = _prepare_patterns_for_serialization(patterns, max_patterns=max_patterns)
    if not prepared:
        return "No learned patterns were activated."
    return "\n".join(_format_pattern(pattern) for pattern in prepared)


def serialize_pattern_evidence(
    evidence: PatternEvidence,
    *,
    max_patterns: int | None = None,
) -> PatternTextBundle:
    prepared = _prepare_patterns_for_serialization(evidence.patterns, max_patterns=max_patterns)
    text = (
        "No learned patterns were activated."
        if not prepared
        else "\n".join(_format_pattern(pattern) for pattern in prepared)
    )
    pattern_count = len(prepared)
    return PatternTextBundle(
        library_id=evidence.library_id,
        schema_version=evidence.schema_version,
        text=text,
        pattern_count=pattern_count,
        metadata=dict(evidence.metadata),
    )


def _format_pattern(pattern: Pattern) -> str:
    prefix = _pattern_prefix(pattern)
    parameters = dict(pattern.parameters or {})
    parameters.pop("original_pattern_label", None)
    object_positions = parameters.pop("object_positions", None)
    object_indices = parameters.pop("object_indices", None)
    object_details = parameters.pop("object_details", None)
    suppress_placeholder_participants = _should_suppress_participant_annotations(
        object_details=object_details,
        object_indices=object_indices,
    )
    if object_positions:
        parameters["objects"] = _format_object_positions(object_positions)
    if object_details and not suppress_placeholder_participants:
        parameters["object_details"] = _format_object_details(object_details)
    if object_indices is not None and not suppress_placeholder_participants:
        parameters["object_indices"] = object_indices
    if parameters:
        params = ", ".join(
            f"{key}={_format_parameter_value(value)}"
            for key, value in sorted(parameters.items())
        )
        return f"{prefix}{pattern.label} ({params})"
    return f"{prefix}{pattern.label}"


def _prepare_patterns_for_serialization(
    patterns: list[Pattern],
    *,
    max_patterns: int | None = None,
) -> list[Pattern]:
    ordered = sorted(patterns, key=_pattern_sort_key)
    merged = _merge_adjacent_equivalent_patterns(ordered)
    if max_patterns is not None:
        merged = merged[:max_patterns]
    return merged


def _merge_adjacent_equivalent_patterns(patterns: list[Pattern]) -> list[Pattern]:
    if not patterns:
        return []
    merged: list[Pattern] = [patterns[0].model_copy(deep=True)]
    for pattern in patterns[1:]:
        current = merged[-1]
        if _patterns_match_except_timesteps(current, pattern) and pattern.start_timestep <= (current.end_timestep + 1):
            if pattern.end_timestep > current.end_timestep:
                current.end_timestep = pattern.end_timestep
            if pattern.start_timestep < current.start_timestep:
                current.start_timestep = pattern.start_timestep
            continue
        merged.append(pattern.model_copy(deep=True))
    return merged


def _patterns_match_except_timesteps(left: Pattern, right: Pattern) -> bool:
    return (
        left.detector_id == right.detector_id
        and left.label == right.label
        and left.parameters == right.parameters
        and left.provenance == right.provenance
        and left.metadata == right.metadata
    )


def _should_suppress_participant_annotations(*, object_details: Any, object_indices: Any) -> bool:
    if not _is_empty_object_indices(object_indices):
        return False
    return _object_details_are_placeholders(object_details)


def _is_empty_object_indices(value: Any) -> bool:
    if isinstance(value, (list, tuple, set)):
        return len(value) == 0
    return False


def _object_details_are_placeholders(details: Any) -> bool:
    if not isinstance(details, dict):
        return False
    if not details:
        return True
    for entry in details.values():
        if not isinstance(entry, dict):
            return False
        if not _is_placeholder_object_detail_entry(entry):
            return False
    return True


def _is_placeholder_object_detail_entry(entry: dict[str, Any]) -> bool:
    allowed_keys = {"roles", "role", "color", "shape", "start", "end"}
    if any(str(key) not in allowed_keys for key in entry.keys()):
        return False
    role_value = entry.get("roles")
    if role_value is None and "role" in entry:
        role_value = entry.get("role")
    role_tokens = _normalize_role_tokens(role_value)
    if any(token != "participant" for token in role_tokens):
        return False
    color = str(entry.get("color") or "").strip()
    shape = str(entry.get("shape") or "").strip()
    if color and color != "?":
        return False
    if shape and shape != "?":
        return False
    start = _format_xy(_coerce_xy_payload(entry.get("start")))
    end = _format_xy(_coerce_xy_payload(entry.get("end")))
    return start == "—" and end == "—"


def _normalize_role_tokens(value: Any) -> list[str]:
    if value is None:
        return ["participant"]
    if isinstance(value, (list, tuple, set)):
        tokens = [str(item).strip().lower() for item in value if str(item).strip()]
        return tokens or ["participant"]
    token = str(value).strip().lower()
    return [token] if token else ["participant"]


def _format_object_positions(object_positions: list[dict[str, Any]] | Any) -> str:
    if not isinstance(object_positions, list):
        return _format_parameter_value(object_positions)
    parts: list[str] = []
    for item in object_positions:
        if not isinstance(item, dict):
            continue
        idx = item.get("object_index")
        start = item.get("start")
        end = item.get("end")
        parts.append(
            f"{idx}: { _format_xy(start) }→{ _format_xy(end) }"
        )
    return "; ".join(part for part in parts if part)


def _format_xy(value: dict[str, Any] | None) -> str:
    if not isinstance(value, dict):
        return "—"
    x = value.get("x")
    y = value.get("y")
    z = value.get("z")
    if z is not None:
        return (
            f"({float(x):.3f},{float(y):.3f},{float(z):.3f})"
            if isinstance(x, (int, float)) and isinstance(y, (int, float)) and isinstance(z, (int, float))
            else _format_parameter_value(value)
        )
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return f"({float(x):.3f},{float(y):.3f})"
    return _format_parameter_value(value)


def _format_object_details(details: dict[str, Any] | Any) -> str:
    if not isinstance(details, dict):
        return str(details)
    parts: list[str] = []
    for key in sorted(details.keys(), key=_object_detail_sort_key):
        entry = details.get(key)
        if not isinstance(entry, dict):
            continue
        roles_value = entry.get("roles")
        if roles_value is None and "role" in entry:
            roles_value = [entry.get("role")]
        role_tokens = _normalize_role_tokens(roles_value)
        role_text = "/".join(role_tokens) if role_tokens else "participant"
        shape = entry.get("shape") or entry.get("geometry_shape") or entry.get("geometry") or "?"
        color = entry.get("color") or "?"
        start = _format_xy(_coerce_xy_payload(entry.get("start"), fallback=entry.get("start_position")))
        end = _format_xy(_coerce_xy_payload(entry.get("end"), fallback=entry.get("end_position")))
        parts.append(f"{key}:{role_text}:{color}-{shape} {start}→{end}")
    return "; ".join(parts)


def _object_detail_sort_key(item: Any) -> tuple[int, int | str]:
    text = str(item)
    if text.isdigit():
        return (0, int(text))
    return (1, text)


def _coerce_xy_payload(value: Any, *, fallback: Any = None) -> dict[str, Any] | None:
    payload = value if value is not None else fallback
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, (list, tuple)) and len(payload) >= 2:
        x = payload[0]
        y = payload[1]
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return None
        point: dict[str, Any] = {"x": float(x), "y": float(y)}
        if len(payload) >= 3 and isinstance(payload[2], (int, float)):
            point["z"] = float(payload[2])
        return point
    return None


def _format_parameter_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, list):
        return "[" + ", ".join(_format_parameter_value(item) for item in value) + "]"
    if isinstance(value, tuple):
        return "[" + ", ".join(_format_parameter_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(
            f"{key}={_format_parameter_value(item)}"
            for key, item in value.items()
        ) + "}"
    return str(value)


def _pattern_prefix(pattern: Pattern) -> str:
    if pattern.start_timestep == pattern.end_timestep:
        return f"[step={pattern.start_timestep}] "
    return f"[steps={pattern.start_timestep}-{pattern.end_timestep}] "


def _pattern_sort_key(item: Pattern) -> tuple[int, int, str, str]:
    return item.start_timestep, item.end_timestep, item.label, item.detector_id
