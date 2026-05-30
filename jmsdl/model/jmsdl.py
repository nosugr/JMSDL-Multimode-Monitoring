from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from jmsdl.model.dictionary_update import (
    JMSDLUpdateResult,
    dictionary_similarity,
    update_dictionary_jmsdl,
)
from jmsdl.utils.data_loader import apply_standardizer, fit_standardizer
from jmsdl.utils.initializer import as_feature_by_sample
from jmsdl.model.ksvd import KSVDResult, fit_ksvd
from jmsdl.model.sparse_coding import omp_encode
from jmsdl.monitoring.offline import compute_reconstruction_errors, kde_threshold
from jmsdl.monitoring.online import score_samples


@dataclass
class JMSDLHyperParams:
    n_atoms: int = 80
    sparsity: int = 3
    update_sparsity_values: tuple[int, ...] = (3, 3, 5)
    lambda_values: tuple[float, ...] = (3.0, 2.5, 2.6)
    initial_max_iter: int = 30
    update_max_iter: int = 30
    tol: float = 1.0e-5
    standardize: bool = True
    random_state: int | None = 0


class JMSDL:
    """Mode-matching and similarity-preserving dictionary learning."""

    def __init__(
        self,
        n_atoms: int = 80,
        sparsity: int = 3,
        lambda_values: list[float] | tuple[float, ...] | None = None,
        update_sparsity_values: list[int] | tuple[int, ...] | None = None,
        initial_max_iter: int = 30,
        update_max_iter: int = 30,
        tol: float = 1.0e-5,
        standardize: bool = True,
        random_state: int | None = 0,
    ) -> None:
        self.n_atoms = int(n_atoms)
        self.sparsity = int(sparsity)
        self.lambda_values = tuple(float(value) for value in (lambda_values or (3.0, 2.5, 2.6)))
        self.update_sparsity_values = (
            tuple(int(value) for value in update_sparsity_values)
            if update_sparsity_values is not None
            else ()
        )
        self.initial_max_iter = int(initial_max_iter)
        self.update_max_iter = int(update_max_iter)
        self.tol = float(tol)
        self.standardize = bool(standardize)
        self.random_state = random_state

        self.dictionaries_: list[np.ndarray] = []
        self.codes_: list[np.ndarray] = []
        self.update_results_: list[JMSDLUpdateResult] = []
        self.initial_result_: KSVDResult | None = None
        self.threshold_: float | None = None
        self.n_features_: int | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    @property
    def dictionary_(self) -> np.ndarray:
        if not self.dictionaries_:
            raise RuntimeError("JMSDL model is not fitted yet.")
        return self.dictionaries_[-1]

    @property
    def ds_history_(self) -> list[float]:
        values: list[float] = []
        for old, new in zip(self.dictionaries_[:-1], self.dictionaries_[1:]):
            values.append(dictionary_similarity(old, new))
        return values

    def _lambda_for_update(self, update_index: int) -> float:
        if not self.lambda_values:
            return 0.0
        return self.lambda_values[min(update_index, len(self.lambda_values) - 1)]

    def _sparsity_for_update(self, update_index: int) -> int:
        if not self.update_sparsity_values:
            return self.sparsity
        return self.update_sparsity_values[min(update_index, len(self.update_sparsity_values) - 1)]

    def _standardize(self, matrix: np.ndarray) -> np.ndarray:
        if not self.standardize or self.mean_ is None or self.scale_ is None:
            return matrix
        return apply_standardizer(matrix, self.mean_, self.scale_)

    def fit(
        self,
        train_modes: list[np.ndarray] | tuple[np.ndarray, ...],
        alpha: float | None = None,
        show_progress: bool = False,
        progress_position: int = 0,
        progress_leave: bool = True,
    ) -> "JMSDL":
        if len(train_modes) == 0:
            raise ValueError("At least one training mode is required.")

        first = as_feature_by_sample(train_modes[0])
        self.n_features_ = first.shape[0]
        modes = [first] + [as_feature_by_sample(mode, n_features=self.n_features_) for mode in train_modes[1:]]

        # 按论文数据尺度，标准化让重构项与保留项可比，是消除灾难性遗忘的关键前处理。
        if self.standardize:
            self.mean_, self.scale_ = fit_standardizer(np.hstack(modes))
            modes = [apply_standardizer(mode, self.mean_, self.scale_) for mode in modes]
        else:
            self.mean_, self.scale_ = None, None

        self.initial_result_ = fit_ksvd(
            modes[0],
            n_atoms=self.n_atoms,
            sparsity=self.sparsity,
            max_iter=self.initial_max_iter,
            tol=self.tol,
            random_state=self.random_state,
            show_progress=show_progress,
            progress_desc="epoch[K-SVD][D1]",
            progress_position=progress_position,
            progress_leave=progress_leave,
        )
        self.dictionaries_ = [self.initial_result_.dictionary]
        self.codes_ = [self.initial_result_.codes]
        self.update_results_ = []

        current_dictionary = self.initial_result_.dictionary
        for update_index, mode in enumerate(modes[1:]):
            result = update_dictionary_jmsdl(
                mode,
                current_dictionary,
                sparsity=self._sparsity_for_update(update_index),
                lambda1=self._lambda_for_update(update_index),
                max_iter=self.update_max_iter,
                tol=self.tol,
                show_progress=show_progress,
                progress_desc=f"epoch[JMSDL][D{update_index + 2}]",
                progress_position=progress_position,
                progress_leave=progress_leave,
            )
            current_dictionary = result.dictionary
            self.update_results_.append(result)
            self.dictionaries_.append(result.dictionary)
            self.codes_.append(result.codes)

        if alpha is not None:
            self.set_threshold(np.hstack(modes), alpha=alpha, _already_standardized=True)
        return self

    def transform(self, Y: np.ndarray) -> np.ndarray:
        matrix = self._standardize(as_feature_by_sample(Y, n_features=self.dictionary_.shape[0]))
        return omp_encode(matrix, self.dictionary_, self.sparsity, tol=self.tol)

    def reconstruct(self, Y: np.ndarray) -> np.ndarray:
        matrix = self._standardize(as_feature_by_sample(Y, n_features=self.dictionary_.shape[0]))
        return self.dictionary_ @ omp_encode(matrix, self.dictionary_, self.sparsity, tol=self.tol)

    def score_samples(self, Y: np.ndarray) -> np.ndarray:
        matrix = self._standardize(as_feature_by_sample(Y, n_features=self.dictionary_.shape[0]))
        return score_samples(matrix, self.dictionary_, self.sparsity, tol=self.tol)

    def mre_by_mode(self, train_modes: list[np.ndarray] | tuple[np.ndarray, ...]) -> np.ndarray:
        """各模态在最终字典下的平均重构误差 (复现论文 Fig.8 的 D_c 行)。

        内部自动套用与训练一致的标准化，调用方传入原始 (任意方向) 模态矩阵即可。
        """
        values: list[float] = []
        for mode in train_modes:
            errors = self.score_samples(mode)
            values.append(float(np.mean(errors)))
        return np.asarray(values, dtype=float)

    def mre_matrix(self, train_modes: list[np.ndarray] | tuple[np.ndarray, ...]) -> np.ndarray:
        """所有字典对所有模态的 MRE 矩阵，形状 (n_dicts, n_modes)。

        第 k 行第 j 列 = 字典 D_{k+1} 对模态 j+1 的平均重构误差，复现论文 Fig.8 的完整矩阵。
        内部自动套用与训练一致的标准化。
        """
        std_modes: list[np.ndarray] = []
        for mode in train_modes:
            m = self._standardize(as_feature_by_sample(mode, n_features=self.dictionaries_[0].shape[0]))
            std_modes.append(m)

        rows: list[list[float]] = []
        for dictionary in self.dictionaries_:
            row: list[float] = []
            for m in std_modes:
                W = omp_encode(m, dictionary, self.sparsity, tol=self.tol)
                row.append(float(np.mean(np.sum((m - dictionary @ W) ** 2, axis=0))))
            rows.append(row)
        return np.asarray(rows, dtype=float)

    def set_threshold(self, train_data: np.ndarray, alpha: float = 0.99, _already_standardized: bool = False) -> float:
        matrix = as_feature_by_sample(train_data, n_features=self.dictionary_.shape[0])
        if not _already_standardized:
            matrix = self._standardize(matrix)
        errors = compute_reconstruction_errors(matrix, self.dictionary_, self.sparsity, tol=self.tol)
        self.threshold_ = kde_threshold(errors, alpha=alpha)
        return self.threshold_

    def predict(self, Y: np.ndarray) -> np.ndarray:
        if self.threshold_ is None:
            raise RuntimeError("Threshold is not set. Call set_threshold() or fit(..., alpha=...).")
        return self.score_samples(Y) > self.threshold_

    def save_npz(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {f"D{index + 1}": dictionary for index, dictionary in enumerate(self.dictionaries_)}
        payload["threshold"] = np.array(np.nan if self.threshold_ is None else self.threshold_, dtype=float)
        payload["ds_history"] = np.asarray(self.ds_history_, dtype=float)
        payload["standardize"] = np.array(self.standardize, dtype=bool)
        if self.mean_ is not None and self.scale_ is not None:
            payload["mean"] = np.asarray(self.mean_, dtype=float)
            payload["scale"] = np.asarray(self.scale_, dtype=float)
        np.savez(destination, **payload)

    @property
    def hyperparams(self) -> JMSDLHyperParams:
        return JMSDLHyperParams(
            n_atoms=self.n_atoms,
            sparsity=self.sparsity,
            update_sparsity_values=self.update_sparsity_values,
            lambda_values=self.lambda_values,
            initial_max_iter=self.initial_max_iter,
            update_max_iter=self.update_max_iter,
            tol=self.tol,
            standardize=self.standardize,
            random_state=self.random_state,
        )
