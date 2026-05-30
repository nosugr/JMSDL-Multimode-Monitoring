from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # 允许 `python baselines/mPCA/mpca_monitor.py` 直接运行
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from tqdm.auto import tqdm

from baselines.common import sample_by_feature
from jmsdl.monitoring.offline import kde_threshold
from jmsdl.utils.data_loader import apply_standardizer, fit_standardizer


class _SinglePCAModel:
    def __init__(self, cpv: float = 0.85, alpha: float = 0.99) -> None:
        self.cpv = float(cpv)
        self.alpha = float(alpha)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.loadings_: np.ndarray | None = None
        self.eigenvalues_: np.ndarray | None = None
        self.t2_threshold_: float | None = None
        self.spe_threshold_: float | None = None

    def fit(self, samples: np.ndarray) -> "_SinglePCAModel":
        X = sample_by_feature(samples)
        mean, scale = fit_standardizer(X.T)
        self.mean_ = mean.T
        self.std_ = scale.T
        Xs = apply_standardizer(X.T, mean, scale).T
        _, singular_values, vt = np.linalg.svd(Xs, full_matrices=False)
        eigenvalues = singular_values**2 / max(1, Xs.shape[0] - 1)
        explained = np.cumsum(eigenvalues) / max(float(eigenvalues.sum()), 1.0e-12)
        n_components = int(np.searchsorted(explained, self.cpv, side="left") + 1)
        self.loadings_ = vt[:n_components].T
        self.eigenvalues_ = np.maximum(eigenvalues[:n_components], 1.0e-12)
        train_scores = self.score_samples(X)
        self.t2_threshold_ = kde_threshold(train_scores["t2"], alpha=self.alpha)
        self.spe_threshold_ = kde_threshold(train_scores["spe"], alpha=self.alpha)
        return self

    def score_samples(self, samples: np.ndarray) -> dict[str, np.ndarray]:
        if self.mean_ is None or self.std_ is None or self.loadings_ is None or self.eigenvalues_ is None:
            raise RuntimeError("PCA model is not fitted.")
        X = sample_by_feature(samples, n_features=self.mean_.shape[1])
        Xs = apply_standardizer(X.T, self.mean_.T, self.std_.T).T
        scores = Xs @ self.loadings_
        reconstruction = scores @ self.loadings_.T
        residual = Xs - reconstruction
        return {
            "t2": np.sum((scores**2) / self.eigenvalues_, axis=1),
            "spe": np.sum(residual**2, axis=1),
        }

    def normalized_fault_score(self, samples: np.ndarray) -> np.ndarray:
        if self.t2_threshold_ is None or self.spe_threshold_ is None:
            raise RuntimeError("PCA thresholds are not fitted.")
        scores = self.score_samples(samples)
        return np.maximum(
            scores["t2"] / max(float(self.t2_threshold_), 1.0e-12),
            scores["spe"] / max(float(self.spe_threshold_), 1.0e-12),
        )


class MPCAMonitor:
    """Multi-model PCA monitor.

    A sample is considered normal if at least one mode-specific PCA model can
    represent it inside that model's control limits.
    """

    def __init__(self, cpv: float = 0.85, alpha: float = 0.99) -> None:
        self.cpv = float(cpv)
        self.alpha = float(alpha)
        self.models_: list[_SinglePCAModel] = []

    def fit(
        self,
        train_modes: list[np.ndarray] | tuple[np.ndarray, ...],
        show_progress: bool = False,
        progress_desc: str = "epoch[mPCA]",
        progress_position: int = 0,
        progress_leave: bool = True,
    ) -> "MPCAMonitor":
        # mPCA 每个模态拟合一个 PCA 模型（SVD 一步到位），进度条按模态推进。
        progress_bar = tqdm(
            total=len(train_modes),
            desc=progress_desc,
            position=progress_position,
            leave=progress_leave,
            dynamic_ncols=True,
            disable=not show_progress,
        )
        self.models_ = []
        for mode_index, mode in enumerate(train_modes, start=1):
            self.models_.append(_SinglePCAModel(self.cpv, self.alpha).fit(mode))
            if show_progress:
                progress_bar.set_postfix_str(f"mode={mode_index}/{len(train_modes)}", refresh=False)
                progress_bar.update(1)
        progress_bar.close()
        return self

    def score_samples(self, samples: np.ndarray) -> np.ndarray:
        if not self.models_:
            raise RuntimeError("MPCAMonitor is not fitted.")
        scores = [model.normalized_fault_score(samples) for model in self.models_]
        return np.min(np.vstack(scores), axis=0)

    def predict(self, samples: np.ndarray) -> np.ndarray:
        return self.score_samples(samples) > 1.0


def main() -> None:
    """独立运行 mPCA 基线：每模态一个 PCA → 监测 → 在本文件夹输出载荷热力图与监测图。

    mPCA 无统一"字典"，故字典热力图改为各模态 PCA 载荷矩阵 (特征 x 主成分) 热力图。
    监测分数已按各自控制限归一化，统一控制限取 1.0。
    """
    from jmsdl.monitoring.metrics import compute_far, compute_fdr
    from jmsdl.utils.data_loader import load_config, load_saved_dataset
    from jmsdl.utils.visualizer import plot_dictionary_heatmap, plot_monitoring_scores

    root = Path(__file__).resolve().parents[2]
    out_dir = Path(__file__).resolve().parent
    config = load_config(root / "config.yaml")
    dataset = load_saved_dataset(root / "data", config=config)
    alpha = float(config.get("monitoring", {}).get("kde_confidence", 0.99))
    baseline_cfg = config.get("baselines", {})

    train_modes = [np.asarray(mode, dtype=float) for mode in dataset["train_modes"]]
    test_all = np.asarray(dataset["test_all"], dtype=float)
    fault_labels = np.asarray(dataset["fault_labels"], dtype=int)
    n_test_per_mode = int(dataset["n_test_per_mode"])
    n_modes = int(dataset["n_modes"])
    boundaries = [index * n_test_per_mode for index in range(n_modes + 1)]

    monitor = MPCAMonitor(cpv=float(baseline_cfg.get("pca_cpv", 0.85)), alpha=alpha)
    monitor.fit(train_modes, show_progress=True, progress_desc="epoch[mPCA]")

    scores = monitor.score_samples(test_all)
    predictions = monitor.predict(test_all)
    fdr = compute_fdr(fault_labels, predictions)
    far = compute_far(fault_labels, predictions)

    # 每模态 PCA 载荷矩阵热力图（特征 x 主成分）。
    for mode_index, pca_model in enumerate(monitor.models_, start=1):
        plot_dictionary_heatmap(
            pca_model.loadings_, out_dir / f"mpca_mode{mode_index}_loadings_heatmap.png",
            title=f"mPCA Mode{mode_index} PCA Loadings",
        )
    plot_monitoring_scores(
        scores, 1.0, fault_labels, out_dir / "mpca_monitoring.png",
        mode_boundaries=boundaries, fdr=fdr, far=far,
        statistic_name="Normalized score", title="mPCA Process Monitoring",
    )
    print(f"[mPCA] FDR={fdr:.4f}  FAR={far:.4f}  (归一化控制限=1.0)")
    print(f"[mPCA] 图已保存到: {out_dir}")


if __name__ == "__main__":
    main()
