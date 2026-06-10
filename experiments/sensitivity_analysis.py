"""JMSDL 保留项参数 lambda1 敏感性分析（独立模块）。

运行：
    python experiments/sensitivity_analysis.py

输出（outputs/sensitivity_analysis/）：
- lambda1_ds_raw.csv          每次 run、每个 lambda1 的字典相似度 ds
- lambda1_ds_summary.csv      各 lambda1 的 ds 均值/标准差
- lambda1_ds_curve.png        lambda1 - ds 曲线（带误差棒）
- dictionary_diff_heatmaps.png  各 lambda1 下 (Do - Dn) 差值热力图面板（取第 1 次 run）
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from experiments._common import dataset_from_files, model_params, samples_to_features, write_table
from jmsdl.model.dictionary_update import dictionary_similarity, update_dictionary_jmsdl
from jmsdl.utils.initializer import normalize_columns
from jmsdl.model.ksvd import fit_ksvd
from jmsdl.utils.data_loader import apply_standardizer, fit_standardizer, load_config


def run_sensitivity_analysis(
    config: dict, show_progress: bool = True, data_dir: str | Path = ROOT / "data"
) -> tuple[pd.DataFrame, dict[float, np.ndarray]]:
    """返回 (ds 明细表, 第 1 次 run 各 lambda1 对应的 (Do-Dn) 差值矩阵)。"""
    sensitivity = config.get("sensitivity_analysis", {})
    lambda_values = [float(value) for value in sensitivity.get("lambda1_values", [0.5, 1.0, 2.0, 3.0])]
    n_runs = int(sensitivity.get("n_runs", 5))
    params = model_params(config)
    seed_cfg = config.get("seed", {})
    base_seed = int(seed_cfg.get("random_state", 0))

    rows: list[dict[str, float | int]] = []
    diff_matrices: dict[float, np.ndarray] = {}
    dataset = dataset_from_files(data_dir, config=config)
    train_modes = [samples_to_features(np.asarray(mode, dtype=float)) for mode in dataset["train_modes"]]
    if bool(params.get("standardize", False)):
        mean, scale = fit_standardizer(np.hstack(train_modes))
        train_modes = [apply_standardizer(mode, mean, scale) for mode in train_modes]

    progress_bar = tqdm(
        total=n_runs * len(lambda_values),
        desc="sensitivity[run x lambda1]",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for run in range(n_runs):
        initial = fit_ksvd(
            train_modes[0],
            n_atoms=int(params["n_atoms"]),
            sparsity=int(params["sparsity"]),
            max_iter=int(params["initial_max_iter"]),
            tol=float(params["tol"]),
            random_state=base_seed + run,
        )
        d_old = normalize_columns(initial.dictionary)
        for lambda1 in lambda_values:
            updated = update_dictionary_jmsdl(
                train_modes[1],
                initial.dictionary,
                sparsity=int(params["sparsity"]),
                lambda1=lambda1,
                max_iter=int(params["update_max_iter"]),
                tol=float(params["tol"]),
            )
            rows.append(
                {
                    "run": run,
                    "lambda1": lambda1,
                    "ds": dictionary_similarity(initial.dictionary, updated.dictionary),
                }
            )
            if run == 0:
                diff_matrices[lambda1] = d_old - normalize_columns(updated.dictionary)
            if show_progress:
                progress_bar.set_postfix_str(f"run={run + 1}/{n_runs}, lambda1={lambda1:g}", refresh=False)
                progress_bar.update(1)
    progress_bar.close()
    return pd.DataFrame(rows), diff_matrices


def plot_sensitivity(summary: pd.DataFrame, output_path: Path) -> None:
    """ds 随 lambda1 变化曲线：蓝色折线 + 圆点 + 红色误差棒（论文 Fig.5 风格）。"""
    grouped = summary.groupby("lambda1")["ds"].agg(["mean", "std"]).reset_index()
    x = grouped["lambda1"].to_numpy(dtype=float)
    mean = grouped["mean"].to_numpy(dtype=float)
    std = grouped["std"].fillna(0.0).to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.errorbar(
        x, mean, yerr=std,
        color="tab:blue", linewidth=2.0, marker="o", markersize=7,
        markerfacecolor="white", markeredgecolor="tab:blue", markeredgewidth=1.6,
        ecolor="tab:red", elinewidth=1.4, capsize=5, capthick=1.4, zorder=3,
    )
    ax.set_xlabel(r"$\lambda_1$", fontsize=13)
    ax.set_ylabel(r"$d_s$", fontsize=13, rotation=0, labelpad=14)
    ax.set_xticks(x)
    ax.margins(x=0.05)
    ax.grid(alpha=0.3, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_dn_minus_do_heatmaps(diff_matrices: dict[float, np.ndarray], output_path: Path) -> None:
    """各 lambda1 下 (Do - Dn) 差值热力图面板（论文 Fig.6 思路 + 本项目字典热力图样式）。

    lambda1 越大、保留越强、差值越接近 0；少数被重写的原子列呈现明显差值。
    配色用 MATLAB jet（与论文 Fig.6 一致）：0 值为绿，正值黄→红、负值青→蓝，对称色标 (±max)。
    """
    if not diff_matrices:
        return
    lambda_values = sorted(diff_matrices.keys())
    n = len(lambda_values)
    n_cols = min(4, n)
    n_rows = int(np.ceil(n / n_cols))
    n_features, n_atoms = next(iter(diff_matrices.values())).shape
    vmax = max(float(np.max(np.abs(matrix))) for matrix in diff_matrices.values()) or 1.0
    feature_labels = [str(index) for index in range(1, n_features + 1)]

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.6 * n_cols, 3.0 * n_rows),
        squeeze=False, layout="constrained",
    )
    image = None
    for position, lambda1 in enumerate(lambda_values):
        ax = axes[position // n_cols][position % n_cols]
        image = ax.imshow(
            diff_matrices[lambda1], aspect="auto", cmap="jet",
            vmin=-vmax, vmax=vmax, interpolation="nearest",
        )
        ax.set_title(rf"$\lambda_1 = {lambda1:g}$", fontsize=11)

        # y 轴显示特征编号（1 基），x 轴隐藏刻度（与 plot_dictionary_heatmap 一致）
        ax.set_yticks(np.arange(n_features))
        ax.set_yticklabels(feature_labels, fontsize=6)
        ax.set_xticks([])

        # 细灰网格描出每个单元格边界
        ax.set_xticks(np.arange(-0.5, n_atoms, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_features, 1), minor=True)
        ax.grid(which="minor", color="0.15", linestyle="-", linewidth=0.35, alpha=0.85)
        ax.tick_params(axis="both", which="major", length=0)
        ax.tick_params(axis="both", which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("0.15")
    # 关闭多余子图
    for position in range(n, n_rows * n_cols):
        axes[position // n_cols][position % n_cols].axis("off")

    if image is not None:
        fig.colorbar(image, ax=axes, fraction=0.046, pad=0.02)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    config = load_config(ROOT / "config.yaml")
    out_dir = ROOT / "outputs" / "sensitivity_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    result, diff_matrices = run_sensitivity_analysis(config, show_progress=True)
    write_table(result, out_dir / "lambda1_ds_raw.csv")
    summary = result.groupby("lambda1")["ds"].agg(["mean", "std"]).reset_index()
    write_table(summary, out_dir / "lambda1_ds_summary.csv")
    plot_sensitivity(result, out_dir / "lambda1_ds_curve.png")
    plot_dn_minus_do_heatmaps(diff_matrices, out_dir / "dictionary_diff_heatmaps.png")

    print(summary.to_string(index=False))
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
