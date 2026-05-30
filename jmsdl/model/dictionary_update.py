from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tqdm.auto import tqdm

from jmsdl.utils.initializer import normalize_columns
from jmsdl.model.sparse_coding import omp_encode


@dataclass
class JMSDLUpdateResult:
    dictionary: np.ndarray
    codes: np.ndarray
    objective_history: list[float]
    similarity_history: list[float]
    n_iter: int


def dictionary_similarity(old_dictionary: np.ndarray, new_dictionary: np.ndarray) -> float:
    old = normalize_columns(old_dictionary)
    new = normalize_columns(new_dictionary)
    if old.shape != new.shape:
        raise ValueError("Dictionaries must have the same shape.")
    return float(np.mean(np.abs(np.sum(old * new, axis=0))))


def _align_with_old_dictionary(dictionary: np.ndarray, old_dictionary: np.ndarray) -> np.ndarray:
    aligned = np.asarray(dictionary, dtype=float).copy()
    signs = np.sign(np.sum(np.asarray(old_dictionary, dtype=float) * aligned, axis=0))
    signs = np.where(signs == 0.0, 1.0, signs)
    aligned *= signs.reshape(1, -1)
    return aligned


def solve_jmsdl_dictionary(
    X_new: np.ndarray,
    codes: np.ndarray,
    old_dictionary: np.ndarray,
    lambda1: float,
    eps: float = 1.0e-10,
) -> np.ndarray:
    """Closed-form dictionary update from Algorithm 1 of JMSDL."""
    X = np.asarray(X_new, dtype=float)
    W = np.asarray(codes, dtype=float)
    D_old = normalize_columns(old_dictionary)
    if X.ndim != 2 or W.ndim != 2 or D_old.ndim != 2:
        raise ValueError("X_new, codes, and old_dictionary must be 2D matrices.")
    if X.shape[0] != D_old.shape[0] or W.shape[0] != D_old.shape[1] or X.shape[1] != W.shape[1]:
        raise ValueError("Matrix dimensions do not match JMSDL update equations.")

    B = W @ W.T
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    F = X @ W.T + 0.5 * float(lambda1) * D_old
    P = F @ eigenvectors
    old_Q = D_old @ eigenvectors

    Q = np.empty_like(P)
    for index, value in enumerate(eigenvalues):
        if abs(float(value)) <= eps:
            Q[:, index] = old_Q[:, index]
        else:
            Q[:, index] = P[:, index] / float(value)

    updated = Q @ eigenvectors.T
    updated = _align_with_old_dictionary(updated, D_old)
    return normalize_columns(updated)


def jmsdl_objective(
    X_new: np.ndarray,
    dictionary: np.ndarray,
    codes: np.ndarray,
    old_dictionary: np.ndarray,
    lambda1: float,
) -> float:
    """JMSDL 目标值 (论文式 4 的前两项)。

    论文式 (4) 含第三项稀疏正则 λ2‖W‖1；本实现按论文 II-A2 用 OMP (L0 约束) 直接求 W，
    等价于以硬稀疏度约束替代 L1 软约束，因此该惩罚不显式出现在目标值里。此处只监控
    重构项 + 保留项的收敛，与 update_dictionary_jmsdl 的迭代停止判据一致。
    """
    residual = np.asarray(X_new, dtype=float) - np.asarray(dictionary, dtype=float) @ np.asarray(codes, dtype=float)
    similarity_term = np.trace(np.eye(dictionary.shape[1]) - normalize_columns(old_dictionary).T @ normalize_columns(dictionary))
    return float(np.linalg.norm(residual, ord="fro") ** 2 + float(lambda1) * similarity_term)


def update_dictionary_jmsdl(
    X_new: np.ndarray,
    old_dictionary: np.ndarray,
    sparsity: int,
    lambda1: float,
    max_iter: int = 30,
    tol: float = 1.0e-5,
    show_progress: bool = False,
    progress_desc: str = "epoch[JMSDL]",
    progress_position: int = 0,
    progress_leave: bool = False,
) -> JMSDLUpdateResult:
    X = np.asarray(X_new, dtype=float)
    D_old = normalize_columns(old_dictionary)
    if X.ndim != 2:
        raise ValueError("Expected a feature-by-sample matrix.")
    if X.shape[0] != D_old.shape[0]:
        raise ValueError("Data and dictionary feature dimensions do not match.")

    D_new = D_old.copy()
    W = omp_encode(X, D_new, int(sparsity), tol=tol)
    objective_history: list[float] = []
    similarity_history: list[float] = []
    previous_objective: float | None = None
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
        D_new = solve_jmsdl_dictionary(X, W, D_old, float(lambda1))
        W = omp_encode(X, D_new, int(sparsity), tol=tol)
        objective = jmsdl_objective(X, D_new, W, D_old, float(lambda1))
        similarity = dictionary_similarity(D_old, D_new)
        objective_history.append(objective)
        similarity_history.append(similarity)
        n_iter = iteration + 1
        if show_progress:
            progress_bar.set_postfix_str(f"obj={objective:.4g}, ds={similarity:.4g}", refresh=False)
            progress_bar.update(1)

        if previous_objective is not None:
            rel_change = abs(previous_objective - objective) / max(abs(previous_objective), 1.0e-12)
            if rel_change < float(tol):
                break
        previous_objective = objective
    progress_bar.close()

    return JMSDLUpdateResult(
        dictionary=D_new,
        codes=W,
        objective_history=objective_history,
        similarity_history=similarity_history,
        n_iter=n_iter,
    )
