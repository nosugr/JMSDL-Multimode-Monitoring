from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # 允许 `python baselines/DL/dl_monitor.py` 直接运行
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from baselines.common import DictionaryMonitorBase


class DLMonitor(DictionaryMonitorBase):
    """Traditional dictionary learning baseline."""

    def fit(
        self,
        train_samples: np.ndarray,
        show_progress: bool = False,
        progress_desc: str = "epoch[DL]",
        progress_position: int = 0,
        progress_leave: bool = True,
    ) -> "DLMonitor":
        self.fit_standardizer(train_samples)
        self.dictionary_ = self._fit_dictionary(
            train_samples,
            show_progress=show_progress,
            progress_desc=progress_desc,
            progress_position=progress_position,
            progress_leave=progress_leave,
        )
        self._set_threshold(train_samples)
        return self


def main() -> None:
    """独立运行 DL 基线：训练 → 监测 → 在本文件夹输出字典热力图与监测结果图。"""
    from jmsdl.monitoring.metrics import compute_far, compute_fdr
    from jmsdl.utils.data_loader import load_config, load_saved_dataset
    from jmsdl.utils.visualizer import plot_dictionary_heatmap, plot_monitoring_scores

    root = Path(__file__).resolve().parents[2]
    out_dir = Path(__file__).resolve().parent
    config = load_config(root / "config.yaml")
    dataset = load_saved_dataset(root / "data", config=config)
    model_cfg = config.get("model", {})
    alpha = float(config.get("monitoring", {}).get("kde_confidence", 0.99))
    baseline_cfg = config.get("baselines", {})

    train_modes = [np.asarray(mode, dtype=float) for mode in dataset["train_modes"]]
    test_all = np.asarray(dataset["test_all"], dtype=float)
    fault_labels = np.asarray(dataset["fault_labels"], dtype=int)
    n_test_per_mode = int(dataset["n_test_per_mode"])
    n_modes = int(dataset["n_modes"])
    boundaries = [index * n_test_per_mode for index in range(n_modes + 1)]

    monitor = DLMonitor(
        n_atoms=int(model_cfg.get("n_atoms", 80)),
        sparsity=int(model_cfg.get("sparsity", 3)),
        alpha=alpha,
        max_iter=int(baseline_cfg.get("max_iter", 30)),
        tol=float(model_cfg.get("tol", 1.0e-5)),
        random_state=config.get("seed", {}).get("data_random_state", 0),
    )
    # DL 只用第一个模态训练（与论文/对比实验一致）。
    monitor.fit(train_modes[0], show_progress=True, progress_desc="epoch[DL]")

    scores = monitor.score_samples(test_all)
    predictions = monitor.predict(test_all)
    fdr = compute_fdr(fault_labels, predictions)
    far = compute_far(fault_labels, predictions)

    plot_dictionary_heatmap(monitor.dictionary_, out_dir / "dl_dictionary_heatmap.png", title="DL Dictionary")
    plot_monitoring_scores(
        scores, float(monitor.threshold_), fault_labels, out_dir / "dl_monitoring.png",
        mode_boundaries=boundaries, fdr=fdr, far=far,
        statistic_name="IRE", title="DL Process Monitoring",
    )
    print(f"[DL] FDR={fdr:.4f}  FAR={far:.4f}  threshold={monitor.threshold_:.6f}")
    print(f"[DL] 图已保存到: {out_dir}")


if __name__ == "__main__":
    main()
