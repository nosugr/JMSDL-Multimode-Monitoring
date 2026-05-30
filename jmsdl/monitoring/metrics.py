from __future__ import annotations

import numpy as np

from jmsdl.model.dictionary_update import dictionary_similarity
from jmsdl.monitoring.online import score_samples


def compute_ds(old_dictionary: np.ndarray, new_dictionary: np.ndarray) -> float:
    return dictionary_similarity(old_dictionary, new_dictionary)


def compute_mre(errors: np.ndarray) -> float:
    values = np.asarray(errors, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("errors must not be empty.")
    return float(values.mean())


def compute_mre_by_mode(
    data_by_mode: list[np.ndarray] | tuple[np.ndarray, ...],
    dictionary: np.ndarray,
    sparsity: int,
    tol: float = 1.0e-6,
) -> np.ndarray:
    values: list[float] = []
    for mode_data in data_by_mode:
        errors = score_samples(mode_data, dictionary, int(sparsity), tol=tol)
        values.append(compute_mre(errors))
    return np.asarray(values, dtype=float)


def compute_fdr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.asarray(y_true, dtype=bool)
    predictions = np.asarray(y_pred, dtype=bool)
    faulty_count = int(labels.sum())
    if faulty_count == 0:
        return 0.0
    tp = int(np.logical_and(labels, predictions).sum())
    return tp / faulty_count


def compute_far(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.asarray(y_true, dtype=bool)
    predictions = np.asarray(y_pred, dtype=bool)
    normal_count = int((~labels).sum())
    if normal_count == 0:
        return 0.0
    fp = int(np.logical_and(~labels, predictions).sum())
    return fp / normal_count


def fdr_far(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    return compute_fdr(y_true, y_pred), compute_far(y_true, y_pred)
