"""连续学习下 DL 与 JMSDL 的测试集重构误差对比图（数据表征能力 / 灾难性遗忘验证）。

运行：
    python "main_model/DL vs JMSDL/continual_learning.py"

流程：
- 传统 DL：D1=K-SVD(X1)，D_{n}=在 D_{n-1} 基础上仅用 X_n 微调（不带保留项，复现灾难性遗忘）。
- JMSDL：D1=K-SVD(X1)，D_{n}=JMSDL 序贯更新（带保留项，避免遗忘）。
- 每个阶段字典都对完整四工况测试集 X_test_all 做 OMP 重构，计算逐样本重构误差。

输出：DL vs JMSDL/continual_learning.png（2×4，上排 DRE，下排 IRE）。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from jmsdl.model import JMSDL
from jmsdl.model.ksvd import fit_ksvd
from jmsdl.model.sparse_coding import omp_encode
from jmsdl.utils.data_loader import (
    apply_standardizer,
    fit_standardizer,
    load_config,
    load_saved_dataset,
)
from jmsdl.utils.initializer import as_feature_by_sample


def reconstruction_errors(dictionary: np.ndarray, X_fs: np.ndarray, sparsity: int, tol: float) -> np.ndarray:
    """逐样本重构误差 RE_i = ||x_i - D w_i||_2^2（X_fs 为标准化后的 特征×样本）。"""
    W = omp_encode(X_fs, dictionary, int(sparsity), tol=tol)
    residual = X_fs - dictionary @ W
    return np.sum(residual ** 2, axis=0)


def train_dl_continual(
    std_modes: list[np.ndarray],
    n_atoms: int,
    sparsity: int,
    max_iter: int,
    tol: float,
    random_state: int | None,
) -> list[np.ndarray]:
    """传统 DL 连续学习：每阶段在上一字典基础上仅用当前工况数据继续训练。"""
    dictionaries: list[np.ndarray] = []
    current = None
    for index, mode in enumerate(std_modes):
        result = fit_ksvd(
            mode,
            n_atoms=n_atoms,
            sparsity=sparsity,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            initial_dictionary=current,  # 在旧字典上微调，无保留项 → 灾难性遗忘
            show_progress=True,
            progress_desc=f"epoch[DL][D{index + 1}]",
        )
        current = result.dictionary
        dictionaries.append(current)
    return dictionaries


def _plot_row(
    axes,
    dictionaries: list[np.ndarray],
    X_test_fs: np.ndarray,
    sparsity: int,
    tol: float,
    ylabel: str,
    panel_labels: list[str],
    separators: list[int],
) -> None:
    n_samples = X_test_fs.shape[1]
    x_axis = np.arange(1, n_samples + 1)
    for col, (ax, dictionary, panel) in enumerate(zip(axes, dictionaries, panel_labels)):
        errors = reconstruction_errors(dictionary, X_test_fs, sparsity, tol)
        ax.plot(x_axis, errors, color="tab:blue", linewidth=0.8)
        for boundary in separators:
            ax.axvline(boundary, color="0.7", linewidth=1.0, alpha=0.8)
        ax.set_title(f"$D_{{{col + 1}}}$", fontsize=12)
        ax.set_xlabel("Sample number", style="italic", fontsize=10)
        if col == 0:
            ax.set_ylabel(ylabel, style="italic", fontsize=11)
        ax.set_xlim(0, n_samples)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25, linewidth=0.6)
        # 子图编号 (a)-(h) 放在横轴标签下方
        ax.text(0.5, -0.34, panel, transform=ax.transAxes, ha="center", va="top", fontsize=11)


def main() -> None:
    config = load_config(ROOT / "config.yaml")
    dataset = load_saved_dataset(ROOT / "data", config=config)
    model_cfg = config.get("model", {})
    random_state = config.get("seed", {}).get("random_state", 0)

    n_atoms = int(model_cfg.get("n_atoms", 80))
    sparsity = int(model_cfg.get("sparsity", 3))
    max_iter = int(model_cfg.get("initial_max_iter", 30))
    tol = float(model_cfg.get("tol", 1.0e-5))
    standardize = bool(model_cfg.get("standardize", False))

    n_modes = int(dataset["n_modes"])
    n_test_per_mode = int(dataset["n_test_per_mode"])
    separators = [index * n_test_per_mode for index in range(1, n_modes)]  # 250, 500, 750

    # 训练工况（特征×样本）+ 完整四工况正常测试集（按工况顺序拼接）
    train_modes_fs = [as_feature_by_sample(np.asarray(mode, dtype=float)) for mode in dataset["train_modes"]]
    test_all_fs = as_feature_by_sample(np.asarray(dataset["test_normal"], dtype=float))

    # JMSDL 连续学习：复用 JMSDL 类得到各阶段字典 D1..D4
    jmsdl = JMSDL(
        n_atoms=n_atoms,
        sparsity=sparsity,
        lambda_values=list(model_cfg.get("lambda_values", [3.0, 2.5, 2.6])),
        update_sparsity_values=list(model_cfg.get("update_sparsity_values", [3, 3, 5])),
        initial_max_iter=max_iter,
        update_max_iter=int(model_cfg.get("update_max_iter", 30)),
        tol=tol,
        standardize=standardize,
        random_state=random_state,
    ).fit(dataset["train_modes"], show_progress=True)
    jmsdl_dicts = list(jmsdl.dictionaries_)

    # 复用 JMSDL 的全局标准化参数，保证 DL 与 JMSDL 在同一尺度下比较
    if standardize:
        mean, scale = jmsdl.mean_, jmsdl.scale_
        std_train_modes = [apply_standardizer(mode, mean, scale) for mode in train_modes_fs]
        test_all_std = apply_standardizer(test_all_fs, mean, scale)
    else:
        std_train_modes = train_modes_fs
        test_all_std = test_all_fs

    # 传统 DL 连续学习：各阶段微调字典
    dl_dicts = train_dl_continual(std_train_modes, n_atoms, sparsity, max_iter, tol, random_state)

    # 绘图：2 行 4 列，上排 DL(DRE)，下排 JMSDL(IRE)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    _plot_row(
        axes[0], dl_dicts, test_all_std, sparsity, tol,
        ylabel="DRE", panel_labels=["(a)", "(b)", "(c)", "(d)"], separators=separators,
    )
    _plot_row(
        axes[1], jmsdl_dicts, test_all_std, sparsity, tol,
        ylabel="IRE", panel_labels=["(e)", "(f)", "(g)", "(h)"], separators=separators,
    )

    fig.tight_layout(h_pad=3.5, w_pad=2.0)
    out_path = Path(__file__).resolve().parent / "continual_learning.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"图已保存到: {out_path}")


if __name__ == "__main__":
    main()
