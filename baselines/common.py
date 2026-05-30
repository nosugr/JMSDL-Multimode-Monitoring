from __future__ import annotations

import numpy as np

from jmsdl.utils.data_loader import apply_standardizer, fit_standardizer
from jmsdl.utils.initializer import as_feature_by_sample
from jmsdl.model.ksvd import fit_ksvd
from jmsdl.monitoring.offline import compute_reconstruction_errors, kde_threshold
from jmsdl.monitoring.online import score_samples


def sample_by_feature(matrix: np.ndarray, n_features: int | None = None) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("Expected a 2D matrix.")
    if n_features is not None:
        if values.shape[1] == n_features:
            return values
        if values.shape[0] == n_features:
            return values.T
        raise ValueError(f"Cannot infer sample axis for n_features={n_features}.")
    if values.shape[0] >= values.shape[1]:
        return values
    return values.T


class DictionaryMonitorBase:
    def __init__(
        self,
        n_atoms: int = 80,
        sparsity: int = 3,
        alpha: float = 0.99,
        max_iter: int = 30,
        tol: float = 1.0e-5,
        standardize: bool = True,
        random_state: int | None = 0,
    ) -> None:
        self.n_atoms = int(n_atoms)
        self.sparsity = int(sparsity)
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.standardize = bool(standardize)
        self.random_state = random_state
        self.dictionary_: np.ndarray | None = None
        self.threshold_: float | None = None
        self.n_features_: int | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit_standardizer(self, train_data: np.ndarray) -> None:
        """用给定训练数据拟合标准化参数 (每个基线在 fit 起始处调用一次)。"""
        if not self.standardize:
            self.mean_, self.scale_ = None, None
            return
        Y = as_feature_by_sample(train_data, n_features=self.n_features_)
        self.n_features_ = Y.shape[0]
        self.mean_, self.scale_ = fit_standardizer(Y)

    def _standardize(self, matrix: np.ndarray) -> np.ndarray:
        if not self.standardize or self.mean_ is None or self.scale_ is None:
            return matrix
        return apply_standardizer(matrix, self.mean_, self.scale_)

    def _fit_dictionary(
        self,
        train_data: np.ndarray,
        initial_dictionary: np.ndarray | None = None,
        show_progress: bool = False,
        progress_desc: str = "epoch[K-SVD]",
        progress_position: int = 0,
        progress_leave: bool = False,
    ) -> np.ndarray:
        Y = self._standardize(as_feature_by_sample(train_data, n_features=self.n_features_))
        self.n_features_ = Y.shape[0]
        result = fit_ksvd(
            Y,
            n_atoms=self.n_atoms,
            sparsity=self.sparsity,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            initial_dictionary=initial_dictionary,
            show_progress=show_progress,
            progress_desc=progress_desc,
            progress_position=progress_position,
            progress_leave=progress_leave,
        )
        return result.dictionary

    def _set_threshold(self, train_data: np.ndarray) -> None:
        if self.dictionary_ is None:
            raise RuntimeError("Dictionary is not fitted.")
        Y = self._standardize(as_feature_by_sample(train_data, n_features=self.dictionary_.shape[0]))
        errors = compute_reconstruction_errors(Y, self.dictionary_, self.sparsity, tol=self.tol)
        self.threshold_ = kde_threshold(errors, alpha=self.alpha)

    def score_samples(self, samples: np.ndarray) -> np.ndarray:
        if self.dictionary_ is None:
            raise RuntimeError("Monitor is not fitted.")
        Y = self._standardize(as_feature_by_sample(samples, n_features=self.dictionary_.shape[0]))
        return score_samples(Y, self.dictionary_, self.sparsity, tol=self.tol)

    def predict(self, samples: np.ndarray) -> np.ndarray:
        if self.threshold_ is None:
            raise RuntimeError("Threshold is not initialized.")
        return self.score_samples(samples) > self.threshold_
