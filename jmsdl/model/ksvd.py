from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tqdm.auto import tqdm

from jmsdl.utils.initializer import (
    initialize_dictionary_from_data,
    initialize_svd_dictionary,
    normalize_columns,
)
from jmsdl.model.sparse_coding import omp_encode


@dataclass
class KSVDResult:
    dictionary: np.ndarray
    codes: np.ndarray
    error_history: list[float]
    n_iter: int


def _reinitialize_atom(
    Y: np.ndarray,
    dictionary: np.ndarray,
    codes: np.ndarray,
    atom_index: int,
    rng: np.random.Generator,
) -> None:
    residual = Y - dictionary @ codes
    sample_index = int(np.argmax(np.sum(residual**2, axis=0)))
    atom = residual[:, sample_index].copy()
    if np.linalg.norm(atom) <= 1.0e-12:
        atom = rng.standard_normal(Y.shape[0])
    dictionary[:, atom_index] = atom / max(float(np.linalg.norm(atom)), 1.0e-12)
    codes[atom_index, :] = 0.0


def _update_dictionary(
    Y: np.ndarray,
    dictionary: np.ndarray,
    codes: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    updated_dictionary = np.asarray(dictionary, dtype=float).copy()
    updated_codes = np.asarray(codes, dtype=float).copy()

    for atom_index in range(updated_dictionary.shape[1]):
        active = np.flatnonzero(np.abs(updated_codes[atom_index, :]) > 1.0e-12)
        if active.size == 0:
            _reinitialize_atom(Y, updated_dictionary, updated_codes, atom_index, rng)
            continue

        residual = (
            Y[:, active]
            - updated_dictionary @ updated_codes[:, active]
            + np.outer(updated_dictionary[:, atom_index], updated_codes[atom_index, active])
        )
        if np.linalg.norm(residual) <= 1.0e-12:
            _reinitialize_atom(Y, updated_dictionary, updated_codes, atom_index, rng)
            continue

        try:
            u, singular_values, vh = np.linalg.svd(residual, full_matrices=False)
        except np.linalg.LinAlgError:
            _reinitialize_atom(Y, updated_dictionary, updated_codes, atom_index, rng)
            continue

        updated_dictionary[:, atom_index] = u[:, 0]
        updated_codes[atom_index, :] = 0.0
        updated_codes[atom_index, active] = singular_values[0] * vh[0, :]

    return normalize_columns(updated_dictionary), updated_codes


def fit_ksvd(
    Y: np.ndarray,
    n_atoms: int,
    sparsity: int,
    max_iter: int = 30,
    tol: float = 1.0e-5,
    random_state: int | None = None,
    initial_dictionary: np.ndarray | None = None,
    init: str = "svd",
    show_progress: bool = False,
    progress_desc: str = "epoch[K-SVD]",
    progress_position: int = 0,
    progress_leave: bool = False,
) -> KSVDResult:
    """Fit a K-SVD dictionary for a feature-by-sample matrix.

    设 ``show_progress=True`` 时用 tqdm 显示迭代进度（每轮刷新重构误差）。
    """
    matrix = np.asarray(Y, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D feature-by-sample matrix.")
    if initial_dictionary is not None:
        dictionary = normalize_columns(np.asarray(initial_dictionary, dtype=float))
        if dictionary.shape != (matrix.shape[0], int(n_atoms)):
            raise ValueError("initial_dictionary shape does not match data and n_atoms.")
    elif init == "data":
        dictionary = initialize_dictionary_from_data(matrix, int(n_atoms), random_state=random_state)
    else:
        dictionary = initialize_svd_dictionary(matrix, int(n_atoms), random_state=random_state)

    rng = np.random.default_rng(random_state)
    error_history: list[float] = []
    previous_error: float | None = None
    codes = omp_encode(matrix, dictionary, int(sparsity), tol=tol)
    n_iter = 0

    progress_bar = tqdm(
        total=max(0, int(max_iter)),
        desc=progress_desc,
        position=progress_position,
        leave=progress_leave,
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for iteration in range(max(0, int(max_iter))):
        codes = omp_encode(matrix, dictionary, int(sparsity), tol=tol)
        dictionary, codes = _update_dictionary(matrix, dictionary, codes, rng)
        residual = matrix - dictionary @ codes
        error = float(np.linalg.norm(residual, ord="fro") ** 2 / max(1, matrix.shape[1]))
        error_history.append(error)
        n_iter = iteration + 1
        if show_progress:
            progress_bar.set_postfix_str(f"recon_err={error:.4g}", refresh=False)
            progress_bar.update(1)
        if previous_error is not None:
            rel_change = abs(previous_error - error) / max(abs(previous_error), 1.0e-12)
            if rel_change < float(tol):
                break
        previous_error = error
    progress_bar.close()

    codes = omp_encode(matrix, dictionary, int(sparsity), tol=tol)
    return KSVDResult(dictionary=dictionary, codes=codes, error_history=error_history, n_iter=n_iter)
