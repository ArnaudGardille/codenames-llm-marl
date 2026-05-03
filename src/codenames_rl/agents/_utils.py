"""Shared utilities for agent implementations.

Centralises the small bits of logic that were previously duplicated across
every agent (cosine similarity, retry-seed generation). Device auto-detection
lives in ``codenames_rl.utils.config.auto_detect_device``.
"""

from __future__ import annotations

import numpy as np


def cosine_similarity(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between a single vector and each row of a matrix.

    Both inputs are L2-normalised with a small epsilon so a zero vector does
    not produce NaNs.
    """
    vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    return np.dot(matrix_norm, vec_norm)


def stable_retry_seed(base_seed: int | None, attempt: int) -> int | None:
    """Deterministic seed for the n-th retry of an LLM generation.

    ``base_seed + attempt * LARGE_PRIME`` is reproducible across runs, unlike
    the previous ``hash(str(seed) + str(time.time()))`` which depended on
    ``PYTHONHASHSEED`` and the wall clock.
    """
    if base_seed is None:
        return None
    return (int(base_seed) + attempt * 2_654_435_761) % (2**31)
