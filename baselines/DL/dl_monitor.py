from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from baselines.common import DictionaryMonitorBase
from jmsdl.utils.initializer import as_feature_by_sample


class DLMonitor(DictionaryMonitorBase):
    """Traditional dictionary learning baseline (single global dictionary over all modes)."""

    def __init__(
        self,
        n_atoms: int = 80,
        sparsity: int = 2,
        alpha: float = 0.99,
        max_iter: int = 30,
        tol: float = 1.0e-5,
    ) -> None:
        super().__init__(
            n_atoms=n_atoms,
            sparsity=sparsity,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
        )

    def fit(
        self,
        train_modes: list[np.ndarray] | tuple[np.ndarray, ...] | np.ndarray,
        show_progress: bool = False,
        progress_desc: str = "epoch[DL]",
        progress_position: int = 0,
        progress_leave: bool = True,
    ) -> "DLMonitor":
        # 拼接全部训练模态成一个矩阵，训练单一全局字典。
        if isinstance(train_modes, np.ndarray):
            all_train = as_feature_by_sample(train_modes)
        else:
            if len(train_modes) == 0:
                raise ValueError("At least one mode is required.")
            first = as_feature_by_sample(train_modes[0])
            matrices = [first] + [as_feature_by_sample(mode, n_features=first.shape[0]) for mode in train_modes[1:]]
            all_train = np.hstack(matrices)
        self.fit_standardizer(all_train)
        self.dictionary_ = self._fit_dictionary(
            all_train,
            show_progress=show_progress,
            progress_desc=progress_desc,
            progress_position=progress_position,
            progress_leave=progress_leave,
        )
        self._set_threshold(all_train)
        return self


def main() -> None:
    """独立运行 DL 基线：训练 → 监测 → 在本文件夹输出字典热力图与监测结果图。"""
    from jmsdl.monitoring.metrics import compute_far, compute_fdr
    from jmsdl.model.sparse_coding import omp_encode
    from jmsdl.utils.data_loader import load_config, load_saved_dataset
    from jmsdl.utils.visualizer import (
        plot_dictionary_heatmap,
        plot_monitoring_scores,
        plot_sparse_code_heatmap,
    )

    root = Path(__file__).resolve().parents[2]
    out_dir = Path(__file__).resolve().parent
    config = load_config(root / "config.yaml")
    dataset = load_saved_dataset(root / "data", config=config)
    model_cfg = config.get("model", {})

    train_modes = [np.asarray(mode, dtype=float) for mode in dataset["train_modes"]]
    test_all = np.asarray(dataset["test_all"], dtype=float)
    fault_labels = np.asarray(dataset["fault_labels"], dtype=int)
    n_test_per_mode = int(dataset["n_test_per_mode"])
    n_modes = int(dataset["n_modes"])
    boundaries = [index * n_test_per_mode for index in range(n_modes + 1)]

    # 超参数（n_atoms/sparsity/alpha/tol 等）直接用 DLMonitor.__init__ 的默认值，
    # 要调就改类定义，不再被 config.yaml 覆盖。
    monitor = DLMonitor()
    monitor.standardize = bool(model_cfg.get("standardize", False))
    monitor.random_state = config.get("seed", {}).get("random_state", 0)
    # DL 用全部模态训练一个全局字典。
    monitor.fit(train_modes, show_progress=True, progress_desc="epoch[DL]")

    scores = monitor.score_samples(test_all)
    predictions = monitor.predict(test_all)
    fdr = compute_fdr(fault_labels, predictions)
    far = compute_far(fault_labels, predictions)

    plot_dictionary_heatmap(monitor.dictionary_, out_dir / "dl_dictionary_heatmap.png", title="DL Dictionary")
    # 测试数据在全局字典下的稀疏编码热力图（原子×样本）
    test_matrix = monitor._standardize(as_feature_by_sample(test_all, n_features=monitor.dictionary_.shape[0]))
    test_codes = omp_encode(test_matrix, monitor.dictionary_, monitor.sparsity, tol=monitor.tol)
    plot_sparse_code_heatmap(
        test_codes,
        out_dir / "dl_sparse_codes_heatmap.png",
        mode_boundaries=boundaries,
        fault_labels=fault_labels,
        title="DL Sparse Codes W",
    )
    plot_monitoring_scores(
        scores, float(monitor.threshold_), fault_labels, out_dir / "dl_monitoring.png",
        mode_boundaries=boundaries, fdr=fdr, far=far,
        statistic_name="DRE", title="DL Process Monitoring",
    )
    print(f"[DL] FDR={fdr:.4f}  FAR={far:.4f}  threshold={monitor.threshold_:.6f}")
    print(f"[DL] 图已保存到: {out_dir}")


if __name__ == "__main__":
    main()
