"""The reload rule, shared by file downloads and table builds, and how its outcome is reported."""

from collections.abc import Mapping
from dataclasses import dataclass


def should_build(exists: bool, reload: bool) -> bool:
    """Build a missing target; rebuild an existing one only when reload is requested."""
    return reload or not exists


@dataclass(frozen=True)
class BuildResult:
    """Outcome of a gated table build: its size, and whether this run built it."""

    rows: int
    built: bool


def row_counts(results: Mapping[str, BuildResult]) -> dict[str, int]:
    return {name: result.rows for name, result in results.items()}


def describe_results(results: Mapping[str, BuildResult]) -> str:
    return "\n".join(
        f"{name}: {'built' if result.built else 'kept'} ({result.rows} rows)"
        for name, result in results.items()
    )
