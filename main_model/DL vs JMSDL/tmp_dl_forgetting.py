"""临时脚本：用 DL 连续学习的最终字典 D4 重构训练数据，画灾难性遗忘图。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from jmsdl.model.ksvd import fit_ksvd
from jmsdl.model.sparse_coding import omp_encode
from jmsdl.utils.data_loader import apply_standardizer, fit_standardizer, load_config, load_saved_dataset
from jmsdl.utils.initializer import as_feature_by_sample
from jmsdl.utils.visualizer import plot_catastrophic_forgetting


def reconstruction_errors(dictionary, X_fs, sparsity, tol):
    W = omp_encode(X_fs, dictionary, int(sparsity), tol=tol)
    residual = X_fs - dictionary @ W
    return np.sum(residual ** 2, axis=0)


def train_dl_continual(std_modes, n_atoms, sparsity, max_iter, tol, random_state):
    """传统 DL 连续学习：每阶段在上一字典基础上仅用当前工况数据继续 K-SVD 微调。"""
    current = None
    for index, mode in enumerate(std_modes):
        result = fit_ksvd(
            mode, n_atoms=n_atoms, sparsity=sparsity, max_iter=max_iter, tol=tol,
            random_state=random_state, initial_dictionary=current,
            show_progress=True, progress_desc=f"epoch[DL][D{index + 1}]",
        )
        current = result.dictionary
    return current


def main():
    config = load_config(ROOT / "config.yaml")
    dataset = load_saved_dataset(ROOT / "data", config=config)
    model_cfg = config.get("model", {})
    random_state = config.get("seed", {}).get("random_state", 0)

    n_atoms = int(model_cfg.get("n_atoms", 80))
    sparsity = int(model_cfg.get("sparsity", 3))
    max_iter = int(model_cfg.get("initial_max_iter", 30))
    tol = float(model_cfg.get("tol", 1.0e-5))
    standardize = bool(model_cfg.get("standardize", False))

    # 训练工况（特征×样本），全局标准化后训练 DL。
    train_modes_fs = [as_feature_by_sample(np.asarray(mode, dtype=float)) for mode in dataset["train_modes"]]
    if standardize:
        mean, scale = fit_standardizer(np.hstack(train_modes_fs))
        std_train_modes = [apply_standardizer(mode, mean, scale) for mode in train_modes_fs]
    else:
        std_train_modes = train_modes_fs

    # DL 连续学习得到最终字典 D4。
    D4 = train_dl_continual(std_train_modes, n_atoms, sparsity, max_iter, tol, random_state)

    # 用 D4 重构各训练工况，按工况顺序拼接逐样本重构误差。
    n_per_mode = min(mode.shape[1] for mode in std_train_modes)
    boundaries = [index * n_per_mode for index in range(len(std_train_modes) + 1)]
    errors = np.concatenate([
        reconstruction_errors(D4, mode[:, :n_per_mode], sparsity, tol) for mode in std_train_modes
    ])

    out_path = Path(__file__).resolve().parent / "dl_catastrophic_forgetting.png"
    plot_catastrophic_forgetting(errors, boundaries, out_path, title="DL Catastrophic Forgetting")

    mode_means = [
        float(errors[boundaries[i]:boundaries[i + 1]].mean()) for i in range(len(std_train_modes))
    ]
    print("[DL] per-mode mean reconstruction error (train data, D4): " + ", ".join(
        f"Mode{i + 1}={value:.4f}" for i, value in enumerate(mode_means)
    ))
    print(f"图已保存到: {out_path}")


if __name__ == "__main__":
    main()
