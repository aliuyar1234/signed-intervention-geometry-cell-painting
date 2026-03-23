from __future__ import annotations

import numpy as np


def pairwise_cosine_median(matrix: np.ndarray) -> float:
    if matrix.shape[0] < 2:
        return 0.0
    norms = np.linalg.norm(matrix, axis=1)
    similarities: list[float] = []
    for left in range(matrix.shape[0] - 1):
        for right in range(left + 1, matrix.shape[0]):
            denom = norms[left] * norms[right]
            if denom <= 0:
                similarities.append(0.0)
            else:
                similarities.append(float(np.dot(matrix[left], matrix[right]) / denom))
    return float(np.median(similarities)) if similarities else 0.0
