from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from tqdm.auto import tqdm

from baselines.common import DictionaryMonitorBase
from jmsdl.model.sparse_coding import omp_encode
from jmsdl.utils.initializer import as_feature_by_sample


def _build_atom_ranges(n_atoms: int, n_modes: int) -> list[tuple[int, int]]:
    """把 n_atoms 个原子按模态划分；不能整除时前几个模态各多分一个，保证全部原子被分配。"""
    if n_modes <= 0:
        raise ValueError("n_modes must be positive.")
    base, remainder = divmod(int(n_atoms), int(n_modes))
    ranges: list[tuple[int, int]] = []
    start = 0
    for mode_index in range(n_modes):
        size = base + (1 if mode_index < remainder else 0)
        ranges.append((start, start + size))
        start += size
    return ranges


def _build_label_matrix(
    mode_labels: np.ndarray, atom_ranges: list[tuple[int, int]], n_atoms: int
) -> np.ndarray:
    """构造理想编码标签矩阵 Q (n_atoms, n_samples)：样本所属模态对应的原子组置 1。"""
    labels = np.asarray(mode_labels, dtype=int).ravel()
    Q = np.zeros((int(n_atoms), labels.size), dtype=float)
    for mode_index, (lo, hi) in enumerate(atom_ranges):
        columns = np.flatnonzero(labels == mode_index)
        if columns.size:
            Q[lo:hi, columns] = 1.0
    return Q


def _reinit_extended_atom(
    X_tilde: np.ndarray,
    D_tilde: np.ndarray,
    W: np.ndarray,
    atom_index: int,
    rng: np.random.Generator,
) -> None:
    """重新初始化扩展字典中的死原子：用当前重构残差最大的样本替换它。"""
    residual = X_tilde - D_tilde @ W
    sample_index = int(np.argmax(np.sum(residual**2, axis=0)))
    atom = residual[:, sample_index].copy()
    norm = float(np.linalg.norm(atom))
    if norm <= 1.0e-12:
        atom = rng.standard_normal(X_tilde.shape[0])
        norm = float(np.linalg.norm(atom))
    D_tilde[:, atom_index] = atom / max(norm, 1.0e-12)
    W[atom_index, :] = 0.0


def _extended_ksvd_atom_update(
    X_tilde: np.ndarray,
    D_tilde: np.ndarray,
    W: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """对扩展字典 D~ 做 K-SVD 风格逐原子更新（在扩展空间 [X; sqrt(a)Q] 上）。"""
    D = np.asarray(D_tilde, dtype=float).copy()
    codes = np.asarray(W, dtype=float).copy()
    for atom_index in range(D.shape[1]):
        active = np.flatnonzero(np.abs(codes[atom_index, :]) > 1.0e-12)
        if active.size == 0:
            _reinit_extended_atom(X_tilde, D, codes, atom_index, rng)
            continue
        residual = (
            X_tilde[:, active]
            - D @ codes[:, active]
            + np.outer(D[:, atom_index], codes[atom_index, active])
        )
        if np.linalg.norm(residual) <= 1.0e-12:
            _reinit_extended_atom(X_tilde, D, codes, atom_index, rng)
            continue
        try:
            u, singular_values, vh = np.linalg.svd(residual, full_matrices=False)
        except np.linalg.LinAlgError:
            _reinit_extended_atom(X_tilde, D, codes, atom_index, rng)
            continue
        D[:, atom_index] = u[:, 0]
        codes[atom_index, :] = 0.0
        codes[atom_index, active] = singular_values[0] * vh[0, :]
    return D, codes


class LCDLMonitor(DictionaryMonitorBase):
    """全局标签一致性字典学习基线 (Label-Consistent Dictionary Learning)。

    与 JMSDL 的区别：LCDL 一次性用全部正常模态数据训练一个全局字典，不做在线增量
    更新、不做相似性保持，只验证“全局字典 + 模态标签一致性约束”能否处理多模态监测。

    目标函数::

        min_{D, W, A}  ||X - D W||_F^2 + alpha ||Q - A W||_F^2
        s.t.  ||w_i||_0 <= T

    其中 Q 是理想编码标签矩阵（按模态分配原子组），A 是标签一致性变换矩阵。
    通过扩展为 K-SVD 问题求解::

        Xt = [X; sqrt(alpha) Q],  Dt = [D; sqrt(alpha) A]
        min_{Dt, W} ||Xt - Dt W||_F^2  s.t. ||w_i||_0 <= T

    训练后只取 Dt 上半部分 D 作为监测字典；监测阶段不使用 A 或 Q。
    """

    def __init__(
        self,
        n_atoms: int = 80,
        sparsity: int = 3,
        alpha: float = 0.99,
        max_iter: int = 30,
        tol: float = 1.0e-5,
        label_weight: float = 1.0,
    ) -> None:
        super().__init__(
            n_atoms=n_atoms,
            sparsity=sparsity,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
        )
        # label_weight 即目标函数中的 alpha（标签一致性权重）；
        # 注意与基类 self.alpha（KDE 控制限置信度）区分。
        self.label_weight = float(label_weight)
        self.transform_: np.ndarray | None = None
        self.codes_: np.ndarray | None = None
        self.label_matrix_: np.ndarray | None = None
        self.label_response_: np.ndarray | None = None
        self.train_mode_labels_: np.ndarray | None = None
        self.mode_atom_ranges_: list[tuple[int, int]] | None = None
        self.loss_history_: list[float] = []

    def fit(
        self,
        train_modes: list[np.ndarray] | tuple[np.ndarray, ...] | np.ndarray,
        show_progress: bool = False,
        progress_desc: str = "epoch[LCDL]",
        progress_position: int = 0,
        progress_leave: bool = True,
    ) -> "LCDLMonitor":
        # ---- Step 1: 拼接训练数据为 X (n_features, total_samples)，生成模态标签 ----
        if isinstance(train_modes, np.ndarray):
            modes = [as_feature_by_sample(train_modes)]
        else:
            if len(train_modes) == 0:
                raise ValueError("At least one mode is required.")
            first = as_feature_by_sample(train_modes[0])
            modes = [first] + [
                as_feature_by_sample(mode, n_features=first.shape[0]) for mode in train_modes[1:]
            ]
        n_modes = len(modes)
        all_train = np.hstack(modes)  # (n_features, total_samples) 未标准化
        mode_labels = np.concatenate(
            [np.full(mode.shape[1], mode_index, dtype=int) for mode_index, mode in enumerate(modes)]
        )

        # 标准化（与其它基线 / JMSDL 一致），扩展 K-SVD 在标准化后的 X 上进行。
        self.fit_standardizer(all_train)
        X = self._standardize(as_feature_by_sample(all_train, n_features=self.n_features_))
        n_features, total_samples = X.shape

        # ---- Step 2: 构造理想编码标签矩阵 Q ----
        atom_ranges = _build_atom_ranges(self.n_atoms, n_modes)
        Q = _build_label_matrix(mode_labels, atom_ranges, self.n_atoms)

        # ---- Step 3: 初始化 D（从样本随机抽取并 L2 归一化）与 A（单位阵）----
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(total_samples, size=self.n_atoms, replace=total_samples < self.n_atoms)
        D = X[:, indices].copy()
        norms = np.linalg.norm(D, axis=0, keepdims=True)
        D /= np.where(norms < 1.0e-12, 1.0, norms)
        A = np.eye(self.n_atoms, dtype=float)

        # ---- Step 4: 构造扩展数据 / 扩展字典 ----
        sqrt_alpha = float(np.sqrt(self.label_weight))
        X_tilde = np.vstack([X, sqrt_alpha * Q])
        D_tilde = np.vstack([D, sqrt_alpha * A])

        # ---- Step 5: 交替优化 ----
        W = omp_encode(X_tilde, D_tilde, self.sparsity, tol=self.tol)
        self.loss_history_ = []
        previous_loss: float | None = None

        progress_bar = tqdm(
            total=max(0, int(self.max_iter)),
            desc=progress_desc,
            position=progress_position,
            leave=progress_leave,
            dynamic_ncols=True,
            disable=not show_progress,
        )
        for _ in range(max(0, int(self.max_iter))):
            # 5.1 固定 D~，OMP 更新 W
            W = omp_encode(X_tilde, D_tilde, self.sparsity, tol=self.tol)
            # 5.2 固定 W，K-SVD 风格逐原子更新 D~
            D_tilde, W = _extended_ksvd_atom_update(X_tilde, D_tilde, W, rng)
            # 5.3 归一化：使监测字典 D（上半部分）每列单位范数，同步缩放 W 保持乘积不变
            d_part_norms = np.linalg.norm(D_tilde[:n_features, :], axis=0)
            d_part_norms_safe = np.where(d_part_norms < 1.0e-12, 1.0, d_part_norms)
            D_tilde = D_tilde / d_part_norms_safe[np.newaxis, :]
            W = W * d_part_norms_safe[:, np.newaxis]

            D = D_tilde[:n_features, :]
            A = D_tilde[n_features:, :] / max(sqrt_alpha, 1.0e-12)

            # ---- Step 6: 收敛判断 ----
            recon_loss = float(np.linalg.norm(X - D @ W, ord="fro") ** 2)
            label_loss = float(np.linalg.norm(Q - A @ W, ord="fro") ** 2)
            loss = recon_loss + self.label_weight * label_loss
            self.loss_history_.append(loss)
            if show_progress:
                progress_bar.set_postfix_str(
                    f"loss={loss:.4g} recon={recon_loss:.4g} label={label_loss:.4g}", refresh=False
                )
                progress_bar.update(1)
            if previous_loss is not None:
                rel_change = abs(previous_loss - loss) / max(abs(previous_loss), 1.0e-12)
                if rel_change < self.tol:
                    break
            previous_loss = loss
        progress_bar.close()

        # ---- Step 7: 保存训练输出 ----
        self.dictionary_ = D_tilde[:n_features, :]
        self.transform_ = D_tilde[n_features:, :] / max(sqrt_alpha, 1.0e-12)
        self.codes_ = W
        self.label_matrix_ = Q
        self.label_response_ = self.transform_ @ W
        self.train_mode_labels_ = mode_labels
        self.mode_atom_ranges_ = atom_ranges

        # ---- Step 8: 控制限（与 DL/JMSDL 一致，仅用监测字典 D）----
        self._set_threshold(all_train)
        return self


def main() -> None:
    """独立运行 LCDL 基线：全局标签一致性字典训练 → 监测 → 输出热力图与监测图。"""
    from jmsdl.monitoring.metrics import compute_far, compute_fdr
    from jmsdl.utils.data_loader import load_config, load_saved_dataset
    from jmsdl.utils.visualizer import (
        plot_dictionary_heatmap,
        plot_label_matrix,
        plot_monitoring_scores,
        plot_sparse_code_heatmap,
    )

    root = Path(__file__).resolve().parents[2]
    out_dir = Path(__file__).resolve().parent
    config = load_config(root / "config.yaml")
    dataset = load_saved_dataset(root / "data", config=config)
    model_cfg = config.get("model", {})
    alpha = float(config.get("numerical_simulation", {}).get("kde_confidence", 0.99))
    baseline_cfg = config.get("baselines", {})

    train_modes = [np.asarray(mode, dtype=float) for mode in dataset["train_modes"]]
    test_all = np.asarray(dataset["test_all"], dtype=float)
    fault_labels = np.asarray(dataset["fault_labels"], dtype=int)
    n_train_per_mode = int(dataset["n_train_per_mode"])
    n_test_per_mode = int(dataset["n_test_per_mode"])
    n_modes = int(dataset["n_modes"])
    train_boundaries = [index * n_train_per_mode for index in range(n_modes + 1)]
    boundaries = [index * n_test_per_mode for index in range(n_modes + 1)]

    monitor = LCDLMonitor(
        n_atoms=int(model_cfg.get("n_atoms", 80)),
        sparsity=int(model_cfg.get("sparsity", 3)),
        alpha=alpha,
        tol=float(model_cfg.get("tol", 1.0e-5)),
        label_weight=float(baseline_cfg.get("lcdl_label_weight", 1.0)),
    )
    monitor.standardize = bool(model_cfg.get("standardize", False))
    monitor.random_state = config.get("seed", {}).get("random_state", 0)
    # LCDL 一次性用全部模态训练一个全局标签一致性字典。
    monitor.fit(train_modes, show_progress=True, progress_desc="epoch[LCDL]")

    scores = monitor.score_samples(test_all)
    predictions = monitor.predict(test_all)
    fdr = compute_fdr(fault_labels, predictions)
    far = compute_far(fault_labels, predictions)

    plot_dictionary_heatmap(monitor.dictionary_, out_dir / "lcdl_dictionary_heatmap.png", title="LCDL Global Dictionary")

    if monitor.label_matrix_ is not None:
        plot_label_matrix(
            monitor.label_matrix_,
            out_dir / "lcdl_label_matrix.png",
            mode_boundaries=train_boundaries,
            atom_ranges=monitor.mode_atom_ranges_,
            title="Label Consistency Matrix Q",
        )

    test_matrix = monitor._standardize(as_feature_by_sample(test_all, n_features=monitor.dictionary_.shape[0]))
    test_codes = omp_encode(test_matrix, monitor.dictionary_, monitor.sparsity, tol=monitor.tol)
    plot_sparse_code_heatmap(
        test_codes,
        out_dir / "sparse_codes_heatmap.png",
        mode_boundaries=boundaries,
        atom_ranges=monitor.mode_atom_ranges_,
        fault_labels=fault_labels,
        title="Sparse Codes W",
    )
    plot_monitoring_scores(
        scores, float(monitor.threshold_), fault_labels, out_dir / "lcdl_monitoring.png",
        mode_boundaries=boundaries, fdr=fdr, far=far,
        statistic_name="DRR", title="LCDL Process Monitoring",
    )
    print(f"[LCDL] FDR={fdr:.4f}  FAR={far:.4f}  threshold={monitor.threshold_:.6f}")
    print(f"[LCDL] label_weight(alpha)={monitor.label_weight}  final_loss={monitor.loss_history_[-1]:.4g}")
    print(f"[LCDL] 图已保存到: {out_dir}")


if __name__ == "__main__":
    main()
