from __future__ import annotations

import numpy as np

from jmsdl.model.sparse_coding import omp_encode


def encode_samples(
    Y: np.ndarray,
    dictionary: np.ndarray,
    sparsity: int,
    tol: float = 1.0e-6,
) -> np.ndarray:
    matrix = np.asarray(Y, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D feature-by-sample matrix.")
    return omp_encode(matrix, dictionary, int(sparsity), tol=tol)


def score_samples(
    Y: np.ndarray,
    dictionary: np.ndarray,
    sparsity: int,
    tol: float = 1.0e-6,
) -> np.ndarray:
    matrix = np.asarray(Y, dtype=float)
    atoms = np.asarray(dictionary, dtype=float)
    codes = encode_samples(matrix, atoms, int(sparsity), tol=tol)
    residual = matrix - atoms @ codes
    return np.sum(residual**2, axis=0)


def detect_fault(scores: np.ndarray, threshold: float) -> np.ndarray:
    return np.asarray(scores, dtype=float) > float(threshold)
