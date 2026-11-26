"""Deterministic, I/O-free split assignment.

Assignment hashes the key with SHA-256 rather than Python's ``hash()``, which is salted per
process. A key therefore lands in the same split on every run and every machine, so held-out
sets stay held out when the data is rebuilt.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

_UINT64_RANGE = 2**64


def hash_unit(key: str, salt: str) -> float:
    """Map ``key`` to a stable float in [0, 1)."""
    digest = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / _UINT64_RANGE


def assign_split(key: str, salt: str, fractions: Mapping[str, float]) -> str:
    """Pick the split whose cumulative-fraction bucket contains the key's hash."""
    position = hash_unit(key, salt)
    cumulative = 0.0
    for name, fraction in fractions.items():
        cumulative += fraction
        if position < cumulative:
            return name
    # Rounding can leave the final cumulative sum just below 1.
    return next(reversed(fractions))


def is_heldout_annotator(worker_id: str, salt: str, fraction: float) -> bool:
    return hash_unit(worker_id, salt) < fraction
