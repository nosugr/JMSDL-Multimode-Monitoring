"""独立运行 JMSDL 模型：序贯训练 + 监测，结果输出到脚本同目录 main_model/run_jmsdl/。

运行：
    python main_model/run_jmsdl/run_jmsdl.py

输出（main_model/run_jmsdl/）：
- jmsdl_model.npz                 训练好的各阶段字典 + 控制限
- jmsdl_scores.csv                逐样本监测分数与故障标签
- D1..D4_dictionary_heatmap.png   各阶段字典热力图
- jmsdl_sparse_codes_heatmap.png  测试数据在最终字典 D4 下的稀疏编码热力图
- jmsdl_monitoring.png            监测结果图（分数 + 控制限 + 故障阴影 + FDR/FAR）
- jmsdl_mre_matrix.png            各字典对各模态的 MRE 折线图（断轴，复现论文 Fig.8）
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
import pandas as pd

from experiments._common import (
    dataset_from_files,
    model_params,
    monitoring_confidence,
    samples_to_features,
    write_table,
)
from jmsdl.model import JMSDL
from jmsdl.monitoring.metrics import compute_far, compute_fdr
from jmsdl.utils.data_loader import load_config
from jmsdl.utils.visualizer import (
    plot_dictionary_heatmap,
    plot_monitoring_scores,
    plot_sparse_code_heatmap,
)


def run_jmsdl(config: dict, show_progress: bool = True, data_dir: str | Path = ROOT / "data") -> dict[str, object]:
    dataset = dataset_from_files(data_dir, config=config)
    train_modes = [samples_to_features(np.asarray(mode, dtype=float)) for mode in dataset["train_modes"]]
    test_all = samples_to_features(np.asarray(dataset["test_all"], dtype=float))
    fault_labels = np.asarray(dataset["fault_labels"], dtype=int)
    n_modes = int(dataset["n_modes"])
    n_test_per_mode = int(dataset["n_test_per_mode"])
    boundaries = [index * n_test_per_mode for index in range(n_modes + 1)]

    model = JMSDL(**model_params(config)).fit(
        train_modes, alpha=monitoring_confidence(config), show_progress=show_progress
    )
    predictions = model.predict(test_all)
    scores = model.score_samples(test_all)
    mre_mat = model.mre_matrix(train_modes)

    return {
        "model": model,
        "scores": scores,
        "predictions": predictions,
        "fault_labels": fault_labels,
        "boundaries": boundaries,
        "test_all": test_all,
        "fdr": compute_fdr(fault_labels, predictions),
        "far": compute_far(fault_labels, predictions),
        "mre_matrix": mre_mat,
    }


def _plot_mre_matrix(mre_matrix: np.ndarray, output_path: Path) -> None:
    """各字典对各模态的 MRE 折线图（复现论文 Fig.8 样式）。

    mre_matrix: (n_dicts, n_modes)，第 k 行 = 字典 D_{k+1} 对各模态的 MRE。
    """
    n_dicts, n_modes = mre_matrix.shape
    modes = np.arange(1, n_modes + 1)

    # 柔和色系，细线，空心 marker
    palette = [
        ("#e0605f", "^"),   # D1 柔红 · 正三角
        ("#4f8fd0", "o"),   # D2 柔蓝 · 圆
        ("#5cb87a", "D"),   # D3 柔绿 · 菱形
        ("#b07cc6", "s"),   # D4 柔紫 · 方块
    ]

    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafc")
    for k in range(n_dicts):
        color, marker = palette[k % len(palette)]
        ax.plot(
            modes, mre_matrix[k],
            color=color, marker=marker,
            markersize=7, linewidth=1.8,
            markerfacecolor="white", markeredgewidth=1.5,
            alpha=0.92,
            label=f"$D_{{{k + 1}}}$",
        )

    ax.set_xticks(modes)
    ax.set_xlim(modes[0] - 0.15, modes[-1] + 0.15)
    ax.tick_params(labelsize=10, length=0)
    ax.grid(alpha=0.22, linewidth=0.7, linestyle="--")
    ax.set_axisbelow(True)
    # 去掉上、右边框，留下更清爽的坐标系
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.9)
        ax.spines[spine].set_color("0.4")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Mode", style="italic", fontsize=12, labelpad=6)
    ax.set_ylabel("MRE", style="italic", fontsize=12, labelpad=6)
    ax.legend(fontsize=11, loc="upper left", framealpha=0.9,
              edgecolor="0.8", handlelength=1.8, ncol=2, columnspacing=1.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    config = load_config(ROOT / "config.yaml")
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_jmsdl(config, show_progress=True)
    model: JMSDL = result["model"]  # type: ignore[assignment]

    # 模型与表格
    model.save_npz(out_dir / "jmsdl_model.npz")
    write_table(
        pd.DataFrame(
            {
                "sample": np.arange(len(result["scores"])),
                "score": result["scores"],
                "fault": result["fault_labels"],
                "predicted_fault": np.asarray(result["predictions"], dtype=int),
            }
        ),
        out_dir / "jmsdl_scores.csv",
    )

    # 各阶段字典热力图
    for index, dictionary in enumerate(model.dictionaries_, start=1):
        plot_dictionary_heatmap(
            dictionary, out_dir / f"D{index}_dictionary_heatmap.png", title=f"JMSDL Dictionary D{index}"
        )

    # 测试数据在最终字典 D4 下的稀疏编码热力图（原子×样本）
    test_codes = model.transform(np.asarray(result["test_all"], dtype=float))
    plot_sparse_code_heatmap(
        test_codes,
        out_dir / "jmsdl_sparse_codes_heatmap.png",
        mode_boundaries=result["boundaries"],
        fault_labels=np.asarray(result["fault_labels"], dtype=int),
        title="JMSDL Sparse Codes W",
    )

    # 监测结果图
    plot_monitoring_scores(
        np.asarray(result["scores"], dtype=float),
        float(model.threshold_),
        np.asarray(result["fault_labels"], dtype=int),
        out_dir / "jmsdl_monitoring.png",
        mode_boundaries=result["boundaries"],
        fdr=float(result["fdr"]),
        far=float(result["far"]),
        statistic_name="IRE",
        title="JMSDL Process Monitoring",
    )

    # MRE 矩阵折线图（断轴，复现论文 Fig.8）
    _plot_mre_matrix(np.asarray(result["mre_matrix"], dtype=float), out_dir / "jmsdl_mre_matrix.png")

    print("JMSDL run complete.")
    print(f"  FDR: {result['fdr']:.4f}")
    print(f"  FAR: {result['far']:.4f}")
    print(f"  Threshold: {model.threshold_:.6f}")
    print(f"  Outputs: {out_dir}")


if __name__ == "__main__":
    main()
