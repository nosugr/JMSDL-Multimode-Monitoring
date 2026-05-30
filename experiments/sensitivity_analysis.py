"""JMSDL 保留项参数 lambda1 敏感性分析（独立模块）。

运行：
    python experiments/sensitivity_analysis.py

输出（outputs/sensitivity_analysis/）：
- fig5_sensitivity_raw.csv        每次 run、每个 lambda1 的字典相似度 ds
- fig5_sensitivity_summary.csv    各 lambda1 的 ds 均值/标准差
- fig5_sensitivity.png            lambda1 - ds 曲线（带误差棒）
- dn_minus_do_heatmaps.png        各 lambda1 下 |Dn - Do| 差值热力图面板（取第 1 次 run）
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
from jmsdl.utils.data_loader import load_config


def run_sensitivity_analysis(
    config: dict, show_progress: bool = True, data_dir: str | Path = ROOT / "data"
) -> tuple[pd.DataFrame, dict[float, np.ndarray]]:
    """返回 (ds 明细表, 第 1 次 run 各 lambda1 对应的 |Dn-Do| 差值矩阵)。"""
    sensitivity = config.get("sensitivity_analysis", {})
    lambda_values = [float(value) for value in sensitivity.get("lambda1_values", [0.5, 1.0, 2.0, 3.0])]
    n_runs = int(sensitivity.get("n_runs", 5))
    params = model_params(config)
    seed_cfg = config.get("seed", {})
    base_seed = int(seed_cfg.get("data_random_state", 0))

    rows: list[dict[str, float | int]] = []
    diff_matrices: dict[float, np.ndarray] = {}
    dataset = dataset_from_files(data_dir, config=config)
    train_modes = [samples_to_features(np.asarray(mode, dtype=float)) for mode in dataset["train_modes"]]

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
                diff_matrices[lambda1] = np.abs(normalize_columns(updated.dictionary) - d_old)
            if show_progress:
                progress_bar.set_postfix_str(f"run={run + 1}/{n_runs}, lambda1={lambda1:g}", refresh=False)
                progress_bar.update(1)
    progress_bar.close()
    return pd.DataFrame(rows), diff_matrices


def plot_sensitivity(summary: pd.DataFrame, output_path: Path) -> None:
    grouped = summary.groupby("lambda1")["ds"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.errorbar(grouped["lambda1"], grouped["mean"], yerr=grouped["std"].fillna(0.0), marker="o", capsize=4)
    ax.set_xlabel("lambda1")
    ax.set_ylabel("Dictionary similarity ds")
    ax.set_title("JMSDL sensitivity analysis")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_dn_minus_do_heatmaps(diff_matrices: dict[float, np.ndarray], output_path: Path) -> None:
    """各 lambda1 下 |Dn - Do| 差值热力图面板：lambda1 越大、保留越强、差值越小。"""
    if not diff_matrices:
        return
    lambda_values = sorted(diff_matrices.keys())
    n = len(lambda_values)
    n_cols = min(4, n)
    n_rows = int(np.ceil(n / n_cols))
    vmax = max(float(np.percentile(matrix, 99)) for matrix in diff_matrices.values()) or 1.0

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 3.0 * n_rows), squeeze=False)
    image = None
    for position, lambda1 in enumerate(lambda_values):
        ax = axes[position // n_cols][position % n_cols]
        image = ax.imshow(
            diff_matrices[lambda1], aspect="auto", cmap="magma", vmin=0.0, vmax=vmax,
            interpolation="nearest", origin="upper",
        )
        ax.set_title(f"lambda1 = {lambda1:g}", fontsize=10)
        ax.set_xlabel("Atom")
        ax.set_ylabel("Feature")
    # 关闭多余子图
    for position in range(n, n_rows * n_cols):
        axes[position // n_cols][position % n_cols].axis("off")

    fig.suptitle("|Dn - Do| per lambda1 (run 1)", fontsize=13)
    if image is not None:
        fig.colorbar(image, ax=axes, fraction=0.02, pad=0.02)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    config = load_config(ROOT / "config.yaml")
    out_dir = ROOT / "outputs" / "sensitivity_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    result, diff_matrices = run_sensitivity_analysis(config, show_progress=True)
    write_table(result, out_dir / "fig5_sensitivity_raw.csv")
    summary = result.groupby("lambda1")["ds"].agg(["mean", "std"]).reset_index()
    write_table(summary, out_dir / "fig5_sensitivity_summary.csv")
    plot_sensitivity(result, out_dir / "fig5_sensitivity.png")
    plot_dn_minus_do_heatmaps(diff_matrices, out_dir / "dn_minus_do_heatmaps.png")

    print(summary.to_string(index=False))
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
