from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pattern_library_schema import (
    PATTERN_SCHEMA_VERSION,
    SUPPORTED_PATTERN_SCHEMA_VERSIONS,
    DetectorDefinition,
    PatternLibrary,
)
from pattern_library_paths import (
    PATTERN_LIBRARY_FILENAME,
    UNSUPERVISED_RUNS_DIRNAME,
    resolve_pattern_library_path,
    resolve_pattern_run_dir,
)


def load_pattern_library(path: str | Path) -> PatternLibrary:
    input_path = Path(path)
    source_path = resolve_pattern_library_path(input_path)
    if not source_path.is_file():
        run_dir = resolve_pattern_run_dir(input_path)
        if _can_reconstruct_library_from_run_dir(run_dir, source_path=source_path):
            return _reconstruct_pattern_library_from_run_dir(run_dir)
        raise FileNotFoundError(f"Pattern library not found: {source_path}")

    with source_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    payload = _attach_supervised_importance_payload(payload, source_path=source_path)
    payload = _attach_ensemble_calibration_payload(payload, source_path=source_path)
    library = PatternLibrary.model_validate(payload)
    if library.schema_version not in SUPPORTED_PATTERN_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported pattern schema version: {library.schema_version} "
            f"(expected one of {sorted(SUPPORTED_PATTERN_SCHEMA_VERSIONS)})"
        )
    if not library.library_id.strip():
        raise ValueError("Pattern library must define a non-empty library_id")
    if not library.environment_ids:
        raise ValueError("Pattern library must define at least one environment_id")
    return library


def save_pattern_library(library: PatternLibrary, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = PatternLibrary.model_validate(library.model_dump(mode="json"))
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(normalized.model_dump(mode="json"), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def _can_reconstruct_library_from_run_dir(run_dir: Path, *, source_path: Path) -> bool:
    if source_path.name != PATTERN_LIBRARY_FILENAME or not run_dir.is_dir():
        return False
    if (run_dir / "run.json").is_file():
        return True
    return any(path.is_dir() for path in run_dir.glob("label_*"))


def _reconstruct_pattern_library_from_run_dir(run_dir: Path) -> PatternLibrary:
    run_payload = _load_json_object(run_dir / "run.json")
    environment_ids = _environment_ids_for_run(run_dir, run_payload=run_payload)
    library_id = _coerce_str(run_payload.get("library_id")) or run_dir.name
    primitive_source = _coerce_str(run_payload.get("primitive_source"))
    scorer_config = (
        dict(run_payload.get("scorer_config"))
        if isinstance(run_payload.get("scorer_config"), dict)
        else {}
    )
    default_input_mode = _coerce_str(run_payload.get("detector_input_mode")) or "trace"
    detectors: list[DetectorDefinition] = []
    for label_dir in sorted(path for path in run_dir.glob("label_*") if path.is_dir()):
        detector = _detector_from_label_dir(
            label_dir,
            environment_ids=environment_ids,
            default_input_mode=default_input_mode,
        )
        if detector is not None:
            detectors.append(detector)
    if not detectors:
        raise FileNotFoundError(
            f"No detector artifacts found to reconstruct pattern library from {run_dir}"
        )
    metadata = {
        "reconstructed_from_run_dir": str(run_dir),
        "status": _coerce_str(run_payload.get("status")),
        "completed_label_count": len(detectors),
    }
    metadata = {key: value for key, value in metadata.items() if value is not None}
    return PatternLibrary(
        library_id=library_id,
        environment_ids=environment_ids,
        primitive_source=primitive_source,
        scorer_config=scorer_config,
        detectors=detectors,
        metadata=metadata,
    )


def _detector_from_label_dir(
    label_dir: Path,
    *,
    environment_ids: list[str],
    default_input_mode: str,
) -> DetectorDefinition | None:
    label_result_payload = _load_json_object(label_dir / "label_result.json")
    detector_payload = label_result_payload.get("detector")
    candidate_dir, candidate_payload = _best_candidate_artifact(label_dir)
    if candidate_dir is None or candidate_payload is None:
        return None
    code_path = next(
        (path for path in sorted(candidate_dir.glob("*.py")) if path.is_file()),
        None,
    )
    if code_path is None:
        raise FileNotFoundError(f"Candidate source file not found under {candidate_dir}")
    source_code = code_path.read_text(encoding="utf-8")
    payload_env_ids = []
    if isinstance(detector_payload, dict) and isinstance(detector_payload.get("environment_ids"), list):
        payload_env_ids = [
            str(item).strip()
            for item in detector_payload.get("environment_ids", [])
            if str(item).strip()
        ]
    detector_id = None
    label = None
    description = None
    entrypoint = None
    config: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    input_mode = default_input_mode
    if isinstance(detector_payload, dict):
        detector_id = _coerce_str(detector_payload.get("detector_id"))
        label = _coerce_str(detector_payload.get("label"))
        description = _coerce_str(detector_payload.get("description"))
        entrypoint = _coerce_str(detector_payload.get("entrypoint"))
        if isinstance(detector_payload.get("config"), dict):
            config = dict(detector_payload.get("config"))
        if isinstance(detector_payload.get("metadata"), dict):
            metadata = dict(detector_payload.get("metadata"))
        input_mode = _coerce_str(detector_payload.get("input_mode")) or default_input_mode
    label = label or _coerce_str(candidate_payload.get("label")) or label_dir.name
    detector_id = detector_id or _fallback_detector_id(label_dir)
    if isinstance(candidate_payload.get("metadata"), dict):
        metadata.update(dict(candidate_payload.get("metadata")))
    metadata.update(
        {
            "candidate_uid": _coerce_str(candidate_payload.get("uid")),
            "parent_uid": _coerce_str(candidate_payload.get("parent_uid")),
            "secondary_parent_uid": _coerce_str(candidate_payload.get("secondary_parent_uid")),
            "origin": _coerce_str(candidate_payload.get("origin")),
            "iteration": candidate_payload.get("iteration"),
            "score": candidate_payload.get("score"),
            "train_score": candidate_payload.get("train_score"),
            "val_score": candidate_payload.get("val_score"),
            "source_path": str(code_path),
            "candidate_dir": str(candidate_dir),
        }
    )
    metadata = {key: value for key, value in metadata.items() if value is not None}
    return DetectorDefinition(
        detector_id=detector_id,
        label=label,
        description=description,
        environment_ids=payload_env_ids or list(environment_ids),
        source_code=source_code,
        entrypoint=entrypoint,
        input_mode=input_mode if input_mode in {"trace", "supervised_evidence"} else "trace",
        config=config,
        metadata=metadata,
    )


def _best_candidate_artifact(label_dir: Path) -> tuple[Path | None, dict[str, Any] | None]:
    explicit: tuple[Path, dict[str, Any]] | None = None
    ranked: list[tuple[float, str, Path, dict[str, Any]]] = []
    for candidate_dir in sorted(path for path in label_dir.glob("candidate_*") if path.is_dir()):
        payload = _load_json_object(candidate_dir / "candidate.json")
        if not payload:
            continue
        if bool(payload.get("is_best_candidate")) and explicit is None:
            explicit = (candidate_dir, payload)
        average_score = payload.get("average_score")
        score = float(average_score) if isinstance(average_score, (int, float)) else float("-inf")
        ranked.append((score, candidate_dir.name, candidate_dir, payload))
    if explicit is not None:
        return explicit
    if not ranked:
        return None, None
    _score, _name, candidate_dir, payload = sorted(ranked, reverse=True)[0]
    return candidate_dir, payload


def _environment_ids_for_run(run_dir: Path, *, run_payload: dict[str, Any]) -> list[str]:
    raw_env_ids = run_payload.get("environment_ids")
    if isinstance(raw_env_ids, list):
        env_ids = [str(item).strip() for item in raw_env_ids if str(item).strip()]
        if env_ids:
            return env_ids
    if run_dir.parent.name == UNSUPERVISED_RUNS_DIRNAME and run_dir.parent.parent.name:
        return [run_dir.parent.parent.name]
    if run_dir.parent.name:
        return [run_dir.parent.name]
    raise ValueError(f"Unable to determine environment_ids for run {run_dir}")


def _fallback_detector_id(label_dir: Path) -> str:
    prefix = label_dir.name.split("_", 2)
    if len(prefix) >= 2 and prefix[1].isdigit():
        return f"primitive-{prefix[1]}"
    return label_dir.name


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _attach_supervised_importance_payload(payload: dict[str, Any], *, source_path: Path) -> dict[str, Any]:
    next_payload = dict(payload)
    metadata = dict(next_payload.get("metadata")) if isinstance(next_payload.get("metadata"), dict) else {}
    importance_payload = _load_supervised_importance_for_library(source_path)
    if importance_payload is not None:
        metadata["supervised_label_importance"] = importance_payload
    if metadata:
        next_payload["metadata"] = metadata
    return next_payload


def _attach_ensemble_calibration_payload(payload: dict[str, Any], *, source_path: Path) -> dict[str, Any]:
    next_payload = dict(payload)
    metadata = dict(next_payload.get("metadata")) if isinstance(next_payload.get("metadata"), dict) else {}
    calibration_payload = _load_ensemble_calibration_for_library(source_path)
    if calibration_payload is not None:
        metadata["ensemble_calibration"] = calibration_payload
    if metadata:
        next_payload["metadata"] = metadata
    return next_payload


def _load_supervised_importance_for_library(source_path: Path) -> dict[str, Any] | None:
    run_dir = source_path.parent
    direct = _load_json_object(run_dir / "supervised_label_importance.json")
    if direct:
        return direct
    run_payload = _load_json_object(run_dir / "run.json")
    source_run_dir = _coerce_str(run_payload.get("source_supervised_run_dir"))
    if source_run_dir:
        source_importance = _load_json_object(Path(source_run_dir) / "supervised_label_importance.json")
        if source_importance:
            return source_importance
    return None


def _load_ensemble_calibration_for_library(source_path: Path) -> dict[str, Any] | None:
    candidates = [
        source_path.parent / "ensemble_calibration.json",
        source_path.parent / "library_calibration.json",
    ]
    for candidate in candidates:
        payload = _load_json_object(candidate)
        if payload:
            return payload
    return None


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
