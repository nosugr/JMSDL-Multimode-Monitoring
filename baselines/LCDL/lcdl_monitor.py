from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # 允许 `python baselines/LCDL/lcdl_monitor.py` 直接运行
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from baselines.common import DictionaryMonitorBase
from jmsdl.utils.initializer import as_feature_by_sample


class LCDLMonitor(DictionaryMonitorBase):
    """Global label-consistent dictionary-learning approximation.

    The original LCDL details are not fully specified in the JMSDL paper. This
    implementation uses a single global dictionary trained on all normal modes.
    """

    def fit(
        self,
        train_modes: list[np.ndarray] | tuple[np.ndarray, ...] | np.ndarray,
        show_progress: bool = False,
        progress_desc: str = "epoch[LCDL]",
        progress_position: int = 0,
        progress_leave: bool = True,
    ) -> "LCDLMonitor":
        if isinstance(train_modes, np.ndarray):
            all_train = train_modes
        else:
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
    """独立运行 LCDL 基线：全局字典训练 → 监测 → 在本文件夹输出热力图与监测图。"""
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

    monitor = LCDLMonitor(
        n_atoms=int(model_cfg.get("n_atoms", 80)),
        sparsity=int(model_cfg.get("sparsity", 3)),
        alpha=alpha,
        max_iter=int(baseline_cfg.get("max_iter", 30)),
        tol=float(model_cfg.get("tol", 1.0e-5)),
        random_state=config.get("seed", {}).get("data_random_state", 0),
    )
    # LCDL 用全部模态训练一个全局字典。
    monitor.fit(train_modes, show_progress=True, progress_desc="epoch[LCDL]")

    scores = monitor.score_samples(test_all)
    predictions = monitor.predict(test_all)
    fdr = compute_fdr(fault_labels, predictions)
    far = compute_far(fault_labels, predictions)

    plot_dictionary_heatmap(monitor.dictionary_, out_dir / "lcdl_dictionary_heatmap.png", title="LCDL Global Dictionary")
    plot_monitoring_scores(
        scores, float(monitor.threshold_), fault_labels, out_dir / "lcdl_monitoring.png",
        mode_boundaries=boundaries, fdr=fdr, far=far,
        statistic_name="IRE", title="LCDL Process Monitoring",
    )
    print(f"[LCDL] FDR={fdr:.4f}  FAR={far:.4f}  threshold={monitor.threshold_:.6f}")
    print(f"[LCDL] 图已保存到: {out_dir}")


if __name__ == "__main__":
    main()
