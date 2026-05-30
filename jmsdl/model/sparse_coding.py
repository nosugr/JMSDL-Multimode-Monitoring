from __future__ import annotations

import numpy as np


def _omp_single(
    y: np.ndarray,
    dictionary: np.ndarray,
    sparsity: int,
    residual_tol: float,
    selection_tol: float,
    ridge: float,
) -> np.ndarray:
    n_features, n_atoms = dictionary.shape
    if y.shape[0] != n_features:
        raise ValueError("Sample dimension does not match dictionary dimension.")
    if sparsity <= 0 or n_atoms == 0:
        return np.zeros(n_atoms, dtype=float)

    atom_norms = np.linalg.norm(dictionary, axis=0)
    valid = atom_norms > 1.0e-12
    if not np.any(valid):
        return np.zeros(n_atoms, dtype=float)

    normalized_dictionary = np.zeros_like(dictionary)
    normalized_dictionary[:, valid] = dictionary[:, valid] / atom_norms[valid]

    residual = y.copy()
    active: list[int] = []
    active_coefficients = np.array([], dtype=float)

    for _ in range(min(int(sparsity), int(valid.sum()))):
        correlations = np.abs(normalized_dictionary.T @ residual)
        correlations[~valid] = -np.inf
        if active:
            correlations[np.asarray(active)] = -np.inf

        chosen = int(np.argmax(correlations))
        if not np.isfinite(correlations[chosen]) or correlations[chosen] <= selection_tol:
            break

        active.append(chosen)
        active_dictionary = normalized_dictionary[:, active]
        if ridge > 0.0:
            gram = active_dictionary.T @ active_dictionary
            rhs = active_dictionary.T @ y
            system = gram + ridge * np.eye(len(active), dtype=float)
            try:
                active_coefficients = np.linalg.solve(system, rhs)
            except np.linalg.LinAlgError:
                active_coefficients, *_ = np.linalg.lstsq(system, rhs, rcond=None)
        else:
            active_coefficients, *_ = np.linalg.lstsq(active_dictionary, y, rcond=None)

        residual = y - active_dictionary @ active_coefficients
        if np.linalg.norm(residual) <= residual_tol:
            break

    coefficients = np.zeros(n_atoms, dtype=float)
    if active:
        active_array = np.asarray(active)
        coefficients[active_array] = active_coefficients / atom_norms[active_array]
    return coefficients


def omp_encode(
    Y: np.ndarray,
    dictionary: np.ndarray,
    sparsity: int,
    tol: float = 1.0e-8,
    selection_tol: float = 1.0e-12,
    ridge: float = 0.0,
) -> np.ndarray:
    """Encode a sample or feature-by-sample matrix with OMP."""
    samples = np.asarray(Y, dtype=float)
    atoms = np.asarray(dictionary, dtype=float)
    if atoms.ndim != 2:
        raise ValueError("dictionary must be a 2D matrix.")

    residual_tol = max(float(tol), 0.0)
    effective_selection_tol = max(float(selection_tol), 0.0)
    effective_ridge = max(float(ridge), 0.0)

    if samples.ndim == 1:
        return _omp_single(
            samples,
            atoms,
            int(sparsity),
            residual_tol,
            effective_selection_tol,
            effective_ridge,
        )
    if samples.ndim != 2:
        raise ValueError("Y must be a 1D sample or 2D feature-by-sample matrix.")

    codes = np.zeros((atoms.shape[1], samples.shape[1]), dtype=float)
    for column in range(samples.shape[1]):
        codes[:, column] = _omp_single(
            samples[:, column],
            atoms,
            int(sparsity),
            residual_tol,
            effective_selection_tol,
            effective_ridge,
        )
    return codes
