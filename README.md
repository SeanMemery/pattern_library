# pattern_library

`pattern_library` is the reusable pattern-detection core for the Pattern Learning project.

## Features

- Defines the pattern library and ensemble library schemas
- Loads and saves pattern library artifacts
- Applies pattern libraries to simulation traces
- Builds and serializes pattern evidence outputs
- Provides shared runtime helpers for pattern-conditioned evaluation flows

## Layout

- `src/pattern_library_schema.py`: schema definitions
- `src/pattern_library_registry.py`: load/save logic
- `src/pattern_library_detectors.py`: pattern application logic
- `src/pattern_library_evidence.py`: evidence generation
- `src/pattern_library_serializer.py`: text serialization
- `src/pattern_library_runtime.py`: runtime helpers

This repo is intended to be consumed as a standalone dependency and as a submodule inside the main Pattern Learning repository.
