from __future__ import annotations

import numpy as np

from jmsdl.utils.data_loader import apply_standardizer, fit_standardizer


def normalize_columns(matrix: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    values = np.asarray(matrix, dtype=float).copy()
    norms = np.linalg.norm(values, axis=0, keepdims=True)
    values /= np.where(norms < eps, 1.0, norms)
    return values


def as_feature_by_sample(matrix: np.ndarray, n_features: int | None = None) -> np.ndarray:
    """Return a feature-by-sample matrix.

    Project CSV files are stored as sample-by-feature. The core algorithms use
    the paper convention: feature-by-sample.
    """
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("Expected a 2D matrix.")
    if n_features is not None:
        if values.shape[0] == n_features:
            return values
        if values.shape[1] == n_features:
            return values.T
        raise ValueError(f"Cannot infer feature axis for n_features={n_features}.")
    if values.shape[0] <= values.shape[1]:
        return values
    return values.T


def initialize_svd_dictionary(
    Y: np.ndarray,
    n_atoms: int,
    random_state: int | None = None,
) -> np.ndarray:
    matrix = np.asarray(Y, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D feature-by-sample matrix.")
    n_features, n_samples = matrix.shape
    if n_features == 0 or n_samples == 0:
        raise ValueError("Input matrix must not be empty.")
    atom_count = int(n_atoms)
    if atom_count <= 0:
        raise ValueError("n_atoms must be positive.")

    centered = matrix - matrix.mean(axis=1, keepdims=True)
    u, singular_values, _ = np.linalg.svd(centered, full_matrices=True)
    atoms: list[np.ndarray] = []
    for index in range(min(atom_count, u.shape[1])):
        atoms.append(u[:, index].copy())

    rng = np.random.default_rng(random_state)
    rank = max(1, int(np.sum(singular_values > 1.0e-12)))
    basis = u[:, : min(rank, u.shape[1])]
    weights = singular_values[: basis.shape[1]]
    if weights.size:
        weights = weights / max(float(weights.max()), 1.0e-12)

    while len(atoms) < atom_count:
        if basis.size:
            coefficients = rng.standard_normal(basis.shape[1]) * np.sqrt(weights)
            atom = basis @ coefficients
        else:
            atom = rng.standard_normal(n_features)
        if np.linalg.norm(atom) <= 1.0e-12:
            atom = rng.standard_normal(n_features)
        atoms.append(atom)

    return normalize_columns(np.column_stack(atoms))


def initialize_dictionary_from_data(
    Y: np.ndarray,
    n_atoms: int,
    random_state: int | None = None,
    noise_scale: float = 1.0e-3,
) -> np.ndarray:
    matrix = np.asarray(Y, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D feature-by-sample matrix.")
    n_features, n_samples = matrix.shape
    if n_samples == 0:
        raise ValueError("Input matrix must contain at least one sample.")
    rng = np.random.default_rng(random_state)
    indices = rng.choice(n_samples, size=int(n_atoms), replace=n_samples < int(n_atoms))
    dictionary = matrix[:, indices].copy()
    dictionary += float(noise_scale) * rng.standard_normal((n_features, int(n_atoms)))
    return normalize_columns(dictionary)

