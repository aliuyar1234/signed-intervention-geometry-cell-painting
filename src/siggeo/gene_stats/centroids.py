from __future__ import annotations

import numpy as np


def coordinate_median(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("Expected a non-empty 2D matrix.")
    return np.median(matrix, axis=0)


def centroid_l2(centroid: np.ndarray) -> float:
    if centroid.size == 0:
        return 0.0
    return float(np.linalg.norm(centroid) / np.sqrt(centroid.size))
