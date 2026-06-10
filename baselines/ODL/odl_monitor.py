from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from baselines.common import DictionaryMonitorBase
from jmsdl.model.ksvd import KSVDResult, fit_ksvd
from jmsdl.model.sparse_coding import omp_encode
from jmsdl.utils.initializer import (
    as_feature_by_sample,
    normalize_columns,
    reinitialize_dead_atoms,
)


def bcd_dictionary_update(
    dictionary: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    epsilon: float = 1.0e-6,
    max_iter: int = 1,
    tol: float = 1.0e-6,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Mairal Algorithm 2 的块坐标下降 (BCD) 字典更新。

    固定 A、B，逐列刷新原子直到收敛 (论文 Algorithm 2)：

        for c in columns:
            u_c = (b_c - D a_c) / A(c, c) + d_c
            d_c = u_c / max(||u_c||_2, 1)

    其中 a_c / b_c 为 A / B 的第 c 列。A(c,c)≈0 时该原子无样本支撑，跳过更新。
    不再使用整体闭式解 D = B (A+eps I)^{-1}。每轮扫描所有列，相对变化小于 tol 即停。
    """
    D = normalize_columns(dictionary)
    n_atoms = D.shape[1]

    for _ in range(max(1, int(max_iter))):
        D_prev = D.copy()
        for c in range(n_atoms):
            a_cc = A[c, c]
            if abs(a_cc) <= float(epsilon):
                continue
            # u_c = (b_c - D a_c)/A(c,c) + d_c
            u_c = (B[:, c] - D @ A[:, c]) / a_cc + D[:, c]
            norm = float(np.linalg.norm(u_c))
            # Mairal: d_c = u_c / max(||u_c||, 1)，再统一列归一化保持单位范数。
            D[:, c] = u_c / max(norm, 1.0)
        D = normalize_columns(D)
        D, _ = reinitialize_dead_atoms(D, rng=rng)
        change = float(np.linalg.norm(D - D_prev) / max(np.linalg.norm(D_prev), 1.0e-12))
        if change < float(tol):
            break
    return D


def odl_update_step(
    X: np.ndarray,
    dictionary: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    sparsity: int,
    epsilon: float,
    bcd_iter: int = 1,
    tol: float = 1.0e-8,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """对单个模态逐样本做 Mairal Algorithm 2 在线字典学习。

    模态内每来一个样本 x_t：
        i.   w_t = OMP(x_t, D)            固定 D 算稀疏编码
        ii.  A ← A + 0.5 * w_t w_t^T      累加信息矩阵 (0.5 系数同论文)
             B ← B + x_t w_t^T
        iii. BCD 逐列更新字典 D (见 bcd_dictionary_update)
    保持列归一化与死原子重初始化。
    """
    D = normalize_columns(dictionary)
    A_new = np.asarray(A, dtype=float).copy()
    B_new = np.asarray(B, dtype=float).copy()
    n_atoms = D.shape[1]

    W = np.zeros((n_atoms, X.shape[1]), dtype=float)
    for t in range(X.shape[1]):
        x = X[:, t:t + 1]
        w = omp_encode(x, D, int(sparsity), tol=tol)
        W[:, t:t + 1] = w

        A_new += 0.5 * (w @ w.T)
        B_new += x @ w.T

        D = bcd_dictionary_update(
            D, A_new, B_new, epsilon=epsilon, max_iter=bcd_iter, rng=rng
        )

    return D, A_new, B_new, W


class ODLMonitor(DictionaryMonitorBase):
    """Sequential ODL baseline using the A/B accumulator dictionary update."""

    def __init__(
        self,
        n_atoms: int = 80,
        sparsity: int = 3,
        alpha: float = 0.99,
        max_iter: int = 100,
        tol: float = 1.0e-5,
        epsilon: float = 1.0e-6,
        bcd_iter: int = 1,
        batch_size: int = 1,
    ) -> None:
        super().__init__(
            n_atoms=n_atoms,
            sparsity=sparsity,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
        )
        self.epsilon = float(epsilon)
        self.bcd_iter = int(bcd_iter)
        self.batch_size = int(batch_size)
        self.initial_result_: KSVDResult | None = None
        self.dictionaries_: list[np.ndarray] = []
        self.codes_: list[np.ndarray] = []
        self.A_: np.ndarray | None = None
        self.B_: np.ndarray | None = None

    def fit(
        self,
        train_modes: list[np.ndarray] | tuple[np.ndarray, ...],
        show_progress: bool = False,
        progress_desc: str = "epoch[ODL]",
        progress_position: int = 0,
        progress_leave: bool = True,
    ) -> "ODLMonitor":
        if len(train_modes) == 0:
            raise ValueError("At least one mode is required.")

        first = as_feature_by_sample(train_modes[0])
        self.n_features_ = first.shape[0]
        modes = [first] + [
            as_feature_by_sample(mode, n_features=self.n_features_) for mode in train_modes[1:]
        ]
        all_train = np.hstack(modes)

        self.fit_standardizer(all_train)
        standardized_modes = [self._standardize(mode) for mode in modes]

        # 用 X1 初始化字典 D1（K-SVD）。
        self.initial_result_ = fit_ksvd(
            standardized_modes[0],
            n_atoms=self.n_atoms,
            sparsity=self.sparsity,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            show_progress=show_progress,
            progress_desc=f"{progress_desc}[init]",
            progress_position=progress_position,
            progress_leave=progress_leave,
        )

        current_dictionary = self.initial_result_.dictionary
        # 论文 Algorithm 2 初始化信息矩阵：A ← 0.5 * W1 W1^T, B ← X1 W1^T。
        W1 = self.initial_result_.codes
        A = 0.5 * (W1 @ W1.T)
        B = standardized_modes[0] @ W1.T

        self.dictionaries_ = []
        self.codes_ = []
        rng = np.random.default_rng(self.random_state)

        # X1 仅用于 K-SVD 初始化 D1 并贡献初始信息矩阵；从 X2 起按模态顺序、模态内逐样本做 BCD 更新。
        for mode_index, current_mode in enumerate(standardized_modes[1:], start=2):
            current_dictionary, A, B, codes = odl_update_step(
                current_mode,
                current_dictionary,
                A,
                B,
                sparsity=self.sparsity,
                epsilon=self.epsilon,
                bcd_iter=self.bcd_iter,
                tol=self.tol,
                rng=rng,
            )
            self.dictionaries_.append(current_dictionary)
            self.codes_.append(codes)

        self.dictionary_ = current_dictionary
        self.A_ = A
        self.B_ = B
        # 用最终字典重构所有训练数据并用 KDE 得到控制限。
        self._set_threshold(all_train)
        return self

    def monitor_online(
        self,
        X_test: np.ndarray,
        batch_size: int | None = None,
        epsilon: float | None = None,
    ) -> dict[str, object]:
        """在线监测：二分类规则 (normal / abnormal)。

        逐个样本 xj：先用当前字典算 OLRE，再分类并决定是否更新：
            normal   OLRE ≤ 控制限 → 更新字典，不更新控制限；
            abnormal OLRE > 控制限 → 报警，不更新任何东西。

        说明：ODL 原论文逐点处理 (无 batch)，故保留 ``batch_size=1``；该参数仅为接口兼容保留。
        """
        if self.dictionary_ is None or self.A_ is None or self.B_ is None:
            raise RuntimeError("Monitor is not fitted.")
        if self.threshold_ is None:
            raise RuntimeError("Threshold is not initialized.")
        batch_size = self.batch_size if batch_size is None else int(batch_size)
        del batch_size
        epsilon = self.epsilon if epsilon is None else float(epsilon)

        # 统一到标准化后的 特征×样本 空间 (字典在该空间学得)。
        Y = self._standardize(as_feature_by_sample(X_test, n_features=self.dictionary_.shape[0]))
        n_samples = Y.shape[1]

        D_current = normalize_columns(self.dictionary_).copy()
        A_current = self.A_.copy()
        B_current = self.B_.copy()
        threshold_current = float(self.threshold_)
        rng = np.random.default_rng(self.random_state)

        # classification: 0=normal, 1=abnormal。
        NORMAL, ABNORMAL = 0, 1
        scores = np.zeros(n_samples, dtype=float)
        predictions = np.zeros(n_samples, dtype=bool)
        classifications = np.zeros(n_samples, dtype=int)

        for j in range(n_samples):
            x = Y[:, j:j + 1]

            # Step 1: 用当前字典算 OLRE (先监测后更新)。
            w = omp_encode(x, D_current, self.sparsity, tol=self.tol)
            x_hat = D_current @ w
            olre = float(np.sum((x - x_hat) ** 2))
            scores[j] = olre

            # Step 2: 二分类。
            if olre <= threshold_current:
                classification = NORMAL
            else:
                classification = ABNORMAL

            classifications[j] = classification
            predictions[j] = classification == ABNORMAL

            # Step 3: normal 时在线更新字典 (信息矩阵累加 + BCD 逐列更新)。
            if classification == NORMAL:
                A_current += 0.5 * (w @ w.T)
                B_current += x @ w.T
                D_current = bcd_dictionary_update(
                    D_current,
                    A_current,
                    B_current,
                    epsilon=epsilon,
                    max_iter=self.bcd_iter,
                    rng=rng,
                )

        return {
            "scores": scores,
            "predictions": predictions,
            "classifications": classifications,
            "dictionary": D_current,
            "A": A_current,
            "B": B_current,
            "threshold": threshold_current,
        }


def main() -> None:
    """Run the ODL baseline and save its monitoring + catastrophic-forgetting figures."""
    from jmsdl.monitoring.metrics import compute_far, compute_fdr
    from jmsdl.model.sparse_coding import omp_encode
    from jmsdl.utils.data_loader import load_config, load_saved_dataset
    from jmsdl.utils.visualizer import (
        plot_catastrophic_forgetting,
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

    # 超参数（n_atoms/sparsity/alpha/tol/epsilon/rho 等）直接用 ODLMonitor.__init__ 的默认值，
    # 要调就改类定义，不再被 config.yaml 覆盖。
    monitor = ODLMonitor()
    monitor.standardize = bool(model_cfg.get("standardize", False))
    monitor.random_state = config.get("seed", {}).get("random_state", 0)
    monitor.fit(train_modes, show_progress=True, progress_desc="epoch[ODL]")

    # 在线监测：二分类 (normal/abnormal)。
    online = monitor.monitor_online(test_all)
    scores = np.asarray(online["scores"], dtype=float)
    predictions = np.asarray(online["predictions"], dtype=bool)
    online_threshold = float(online["threshold"])
    fdr = compute_fdr(fault_labels, predictions)
    far = compute_far(fault_labels, predictions)

    plot_dictionary_heatmap(monitor.dictionary_, out_dir / "odl_dictionary_heatmap.png", title="ODL Final Dictionary")
    # 测试数据在最终字典下的稀疏编码热力图（原子×样本）
    test_matrix = monitor._standardize(as_feature_by_sample(test_all, n_features=monitor.dictionary_.shape[0]))
    test_codes = omp_encode(test_matrix, monitor.dictionary_, monitor.sparsity, tol=monitor.tol)
    plot_sparse_code_heatmap(
        test_codes,
        out_dir / "odl_sparse_codes_heatmap.png",
        mode_boundaries=boundaries,
        fault_labels=fault_labels,
        title="ODL Sparse Codes W",
    )
    plot_monitoring_scores(
        scores,
        online_threshold,
        fault_labels,
        out_dir / "odl_monitoring.png",
        mode_boundaries=boundaries,
        fdr=fdr,
        far=far,
        statistic_name="OLRE",
        title="ODL Process Monitoring",
    )

    # 灾难性遗忘图：用最终字典回头重构各训练模态（每模态等量样本），早期模态误差最大。
    forget_modes = [np.asarray(mode, dtype=float) for mode in dataset["train_modes"]]
    n_per_mode = min(mode.shape[0] for mode in forget_modes)
    forget_modes = [mode[:n_per_mode] for mode in forget_modes]
    forget_boundaries = [index * n_per_mode for index in range(len(forget_modes) + 1)]
    forget_errors = np.concatenate([monitor.score_samples(mode) for mode in forget_modes])
    plot_catastrophic_forgetting(
        forget_errors,
        forget_boundaries,
        out_dir / "odl_catastrophic_forgetting.png",
        title="ODL Catastrophic Forgetting",
    )
    mode_means = [
        float(forget_errors[forget_boundaries[i]:forget_boundaries[i + 1]].mean())
        for i in range(len(forget_modes))
    ]

    print(f"[ODL] FDR={fdr:.4f}  FAR={far:.4f}  threshold={monitor.threshold_:.6f}")
    print("[ODL] per-mode mean reconstruction error: " + ", ".join(
        f"Mode{i + 1}={value:.4f}" for i, value in enumerate(mode_means)
    ))
    print(f"[ODL] figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
