"""JMSDL 保留项 lambda1 逐步标定（解决灾难性遗忘）。

按论文 Fig.5/8 思路，对每个序贯更新步 (D_k -> D_{k+1}) 在对数网格上扫 lambda1，
用「训练数据」的 MRE 判据贪心选 lambda：新模态 MRE 要低，且已学模态 MRE 不被拔高。
选定一步后固定该 lambda，再标定下一步。全程只用训练/验证数据，不碰测试集（无泄漏）。

运行：
    python experiments/calibrate_lambda.py

输出（outputs/lambda_calibration/）：
- lambda_calibration_grid.csv   每步每个候选 lambda 的 new_mre / old_mre_max / 选中标记
- lambda_calibration_curve.png  各步 new_mre 与 old_mre_max 随 lambda 的曲线
- 终端打印推荐的 lambda_values（可直接填回 config.yaml）
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
from jmsdl.model.dictionary_update import update_dictionary_jmsdl
from jmsdl.model.ksvd import fit_ksvd
from jmsdl.model.sparse_coding import omp_encode
from jmsdl.utils.data_loader import apply_standardizer, fit_standardizer, load_config

# 对数网格：覆盖 O(1) 到 O(100)，找出保留项真正起作用的量级
LAMBDA_GRID = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
# 旧模态 MRE 相对 D_k 的允许恶化比例（超过即认为遗忘）
OLD_MRE_TOLERANCE = 0.10


def _mre(dictionary: np.ndarray, mode: np.ndarray, sparsity: int, tol: float) -> float:
    """字典对单个（已标准化的）模态的平均重构误差。"""
    W = omp_encode(mode, dictionary, sparsity, tol=tol)
    return float(np.mean(np.sum((mode - dictionary @ W) ** 2, axis=0)))


def calibrate(config: dict, show_progress: bool = True, data_dir: str | Path = ROOT / "data") -> tuple[pd.DataFrame, list[float]]:
    params = model_params(config)
    seed = int(config.get("seed", {}).get("random_state", 0))
    n_atoms = int(params["n_atoms"])
    init_sparsity = int(params["sparsity"])
    update_sparsity = [int(v) for v in params["update_sparsity_values"]]
    init_iter = int(params["initial_max_iter"])
    upd_iter = int(params["update_max_iter"])
    tol = float(params["tol"])

    dataset = dataset_from_files(data_dir, config=config)
    train_modes = [samples_to_features(np.asarray(mode, dtype=float)) for mode in dataset["train_modes"]]
    if bool(params.get("standardize", False)):
        mean, scale = fit_standardizer(np.hstack(train_modes))
        train_modes = [apply_standardizer(mode, mean, scale) for mode in train_modes]
    n_modes = len(train_modes)

    # D1：K-SVD 初始字典（固定）
    initial = fit_ksvd(
        train_modes[0], n_atoms=n_atoms, sparsity=init_sparsity,
        max_iter=init_iter, tol=tol, random_state=seed,
    )

    rows: list[dict[str, float | int | bool]] = []
    chosen_lambdas: list[float] = []
    current_dictionary = initial.dictionary

    progress = tqdm(
        total=(n_modes - 1) * len(LAMBDA_GRID),
        desc="calibrate[step x lambda]", dynamic_ncols=True, disable=not show_progress,
    )
    for step in range(n_modes - 1):
        new_mode = train_modes[step + 1]
        sp = update_sparsity[min(step, len(update_sparsity) - 1)] if update_sparsity else init_sparsity
        # D_k 在已学模态上的基线 MRE（防遗忘参照）
        baseline_old = [_mre(current_dictionary, train_modes[j], sp, tol) for j in range(step + 1)]

        candidates: list[dict[str, float]] = []
        for lambda1 in LAMBDA_GRID:
            updated = update_dictionary_jmsdl(
                new_mode, current_dictionary, sparsity=sp, lambda1=lambda1,
                max_iter=upd_iter, tol=tol,
            )
            new_mre = _mre(updated.dictionary, new_mode, sp, tol)
            old_mres = [_mre(updated.dictionary, train_modes[j], sp, tol) for j in range(step + 1)]
            old_mre_max = max(old_mres)
            # 旧模态相对基线的最大恶化比例
            old_increase = max(
                (old_mres[j] - baseline_old[j]) / (baseline_old[j] + 1e-12) for j in range(step + 1)
            )
            candidates.append({
                "lambda1": lambda1,
                "new_mre": new_mre,
                "old_mre_max": old_mre_max,
                "old_increase": old_increase,
                "dictionary": updated.dictionary,
            })
            rows.append({
                "step": step + 1, "update": f"D{step + 1}->D{step + 2}",
                "lambda1": lambda1, "new_mre": new_mre,
                "old_mre_max": old_mre_max, "old_increase": old_increase,
            })
            if show_progress:
                progress.set_postfix_str(f"D{step+2}, lambda={lambda1:g}", refresh=False)
                progress.update(1)

        # 选择：在「旧模态恶化不超过容差」的候选里取 new_mre 最小者；
        # 若无候选满足容差，取一个折中目标 new_mre + old_increase 惩罚最小者。
        feasible = [c for c in candidates if c["old_increase"] <= OLD_MRE_TOLERANCE]
        if feasible:
            best = min(feasible, key=lambda c: c["new_mre"])
        else:
            best = min(candidates, key=lambda c: c["new_mre"] + 10.0 * max(0.0, c["old_increase"]))
        chosen_lambdas.append(float(best["lambda1"]))
        for row in rows:
            if row["step"] == step + 1 and row["lambda1"] == best["lambda1"]:
                row["chosen"] = True
        current_dictionary = best["dictionary"]
    progress.close()

    for row in rows:
        row.setdefault("chosen", False)
    return pd.DataFrame(rows), chosen_lambdas


def plot_calibration(table: pd.DataFrame, output_path: Path) -> None:
    """各更新步：new_mre 与 old_mre_max 随 lambda 的曲线（对数 x 轴）。"""
    steps = sorted(table["step"].unique())
    n = len(steps)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.0), squeeze=False)
    for idx, step in enumerate(steps):
        ax = axes[0][idx]
        sub = table[table["step"] == step].sort_values("lambda1")
        x = sub["lambda1"].to_numpy(dtype=float)
        ax.plot(x, sub["new_mre"], "o-", color="tab:blue", label="new mode MRE", markerfacecolor="white")
        ax.plot(x, sub["old_mre_max"], "s-", color="tab:red", label="old modes MRE (max)", markerfacecolor="white")
        chosen = sub[sub["chosen"]]
        if not chosen.empty:
            ax.axvline(float(chosen["lambda1"].iloc[0]), color="0.4", ls="--", lw=1.2,
                       label=f"chosen λ={float(chosen['lambda1'].iloc[0]):g}")
        ax.set_xscale("log")
        ax.set_xlabel(r"$\lambda_1$", fontsize=12)
        ax.set_ylabel("MRE", fontsize=12)
        ax.set_title(f"{sub['update'].iloc[0]}", fontsize=11)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    config = load_config(ROOT / "config.yaml")
    out_dir = ROOT / "outputs" / "lambda_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    table, chosen = calibrate(config, show_progress=True)
    write_table(table, out_dir / "lambda_calibration_grid.csv")
    plot_calibration(table, out_dir / "lambda_calibration_curve.png")

    print(table.to_string(index=False))
    print(f"\n推荐 lambda_values: {chosen}")
    print(f"可填回 config.yaml 的 model.lambda_values")
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
