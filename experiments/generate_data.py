
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from jmsdl.utils.data_loader import generate_from_config, load_config
from jmsdl.utils.visualizer import (
    plot_mode_mean_cov,
    plot_multimode_pca_scatter,
    plot_multimode_timeseries,
    plot_test_data_heatmap,
    plot_test_fault_timeseries,
    plot_test_pca_scatter,
)

# 状态向量分布参数（论文式26固定，与 data_loader 默认值一致）
S1_MEAN, S1_STD = 2.0, 1.0
S2_MEAN, S2_STD = 3.0, 1.0


def save_observation_matrices_text(
    observation_matrices: list[np.ndarray],
    path: Path,
    train_modes: list[np.ndarray] | None = None,
) -> None:
    """把实际使用的观测矩阵 A_i 存成文本（论文式26: x = A_i·s + e）。"""
    n_features, state_dim = observation_matrices[0].shape
    lines: list[str] = []
    lines.append("观测矩阵 A_i（论文式(26): x = A_i · s + e）")
    lines.append(f"生成方式: scale_i * np.random.default_rng(40).normal(0, 1, size=({n_features}, {state_dim}))，连续抽 {len(observation_matrices)} 个，各模态乘以不同尺度因子")
    lines.append(f"形状: 每个 {n_features} x {state_dim}  (n_features x state_dim)")
    lines.append("说明: 与 data/ 下已保存数据所用的完全一致，确定性可复现")
    lines.append("=" * 70)
    for i, A in enumerate(observation_matrices):
        col_norms = np.linalg.norm(A, axis=0)
        col_norm_str = ", ".join(f"{v:.4f}" for v in col_norms)
        lines.append(f"\n# Mode {i + 1}  A_{i + 1}")
        lines.append(f"#   |元素|均值={np.mean(np.abs(A)):.4f}  Frobenius范数={np.linalg.norm(A):.4f}  "
                     f"条件数={np.linalg.cond(A):.4f}  列范数=[{col_norm_str}]")
        for row in A:
            lines.append("   " + "  ".join(f"{v: .6f}" for v in row))

    if train_modes is not None:
        # 每个模态训练样本的逐变量均值与方差
        lines.append("\n" + "=" * 70)
        lines.append("各模态训练样本统计量（逐变量，按列）")
        lines.append("=" * 70)
        for i, mode in enumerate(train_modes):
            mode = np.asarray(mode, dtype=float)
            mean = mode.mean(axis=0)
            var = mode.var(axis=0)
            lines.append(f"\n# Mode {i + 1}  (n_samples={mode.shape[0]}, n_features={mode.shape[1]})")
            lines.append("  均值: " + "  ".join(f"{v: .6f}" for v in mean))
            lines.append("  方差: " + "  ".join(f"{v: .6f}" for v in var))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_state_distribution(path: Path, n_samples: int, random_state: int) -> None:
    """画 s1/s2 抽样分布直方图与理论正态密度对比。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(random_state)
    s1 = rng.normal(S1_MEAN, S1_STD, size=n_samples)
    s2 = rng.normal(S2_MEAN, S2_STD, size=n_samples)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, data, mean, std, name, color in (
        (axes[0], s1, S1_MEAN, S1_STD, "s1", "#3b7dd8"),
        (axes[1], s2, S2_MEAN, S2_STD, "s2", "#d8703b"),
    ):
        ax.hist(data, bins=40, density=True, alpha=0.6, color=color, edgecolor="white",
                label=f"sampled (n={n_samples})")
        grid = np.linspace(data.min(), data.max(), 300)
        pdf = np.exp(-0.5 * ((grid - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
        ax.plot(grid, pdf, "k--", lw=1.6, label=f"N({mean:.0f}, {std:.0f}) theory")
        ax.axvline(mean, color="red", ls=":", lw=1.2, label=f"mean={mean:.0f}")
        ax.set_title(f"State variable {name} ~ N({mean:.0f}, {std:.0f})")
        ax.set_xlabel(name)
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("State vector s sampling distribution (Eq.26)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_datasets(dataset: dict[str, object], train_dir: Path, test_dir: Path) -> None:
    """把训练/测试数据落盘为 CSV。"""
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    train_modes = dataset["train_modes"]
    assert isinstance(train_modes, list)
    for mode_index, mode_data in enumerate(train_modes):
        pd.DataFrame(np.asarray(mode_data, dtype=float)).to_csv(
            train_dir / f"train_mode{mode_index + 1}.csv", index=False
        )
    pd.DataFrame(np.asarray(dataset["train_all"], dtype=float)).to_csv(
        train_dir / "train_all.csv", index=False
    )
    pd.DataFrame({"mode": np.asarray(dataset["train_mode_labels"], dtype=int)}).to_csv(
        train_dir / "train_mode_labels.csv", index=False
    )

    pd.DataFrame(np.asarray(dataset["test_normal"], dtype=float)).to_csv(
        test_dir / "test_normal.csv", index=False
    )
    pd.DataFrame(np.asarray(dataset["test_all"], dtype=float)).to_csv(
        test_dir / "test_all.csv", index=False
    )
    pd.DataFrame(
        {
            "mode": np.asarray(dataset["test_mode_labels"], dtype=int),
            "fault": np.asarray(dataset["fault_labels"], dtype=int),
        }
    ).to_csv(test_dir / "test_labels.csv", index=False)


def main() -> None:
    config = load_config(ROOT / "config.yaml")
    dataset = generate_from_config(config)

    train_dir = ROOT / "data" / "train"
    test_dir = ROOT / "data" / "test"
    a_matrices_dir = ROOT / "data" / "A_matrices"
    a_matrices_dir.mkdir(parents=True, exist_ok=True)
    save_datasets(dataset, train_dir, test_dir)

    # 保存实际使用的观测矩阵 A 文本 + 状态向量 s 抽样分布图
    observation_matrices = [np.asarray(A, dtype=float) for A in dataset["observation_matrices"]]
    train_modes = [np.asarray(mode, dtype=float) for mode in dataset["train_modes"]]
    save_observation_matrices_text(observation_matrices, a_matrices_dir / "A_matrices.txt", train_modes)
    random_state = int(config.get("seed", {}).get("random_state", 42))

    n_modes = int(dataset["n_modes"])
    n_train_per_mode = int(dataset["n_train_per_mode"])
    mode_names = [f"Mode{index + 1}" for index in range(n_modes)]
    timeseries_dims = tuple(config["numerical_simulation"]["timeseries_dims"])

    # 图存放在 data/train 旁
    first, second = plot_multimode_pca_scatter(
        np.asarray(dataset["train_all"], dtype=float),
        np.asarray(dataset["train_mode_labels"], dtype=int),
        train_dir / "pca_multimode_scatter.png",
        mode_names=mode_names,
    )

    boundaries = [mode_index * n_train_per_mode for mode_index in range(n_modes + 1)]
    plot_multimode_timeseries(
        np.asarray(dataset["train_all"], dtype=float),
        boundaries,
        train_dir / "multimode_timeseries.png",
        dims=timeseries_dims,
        mode_names=mode_names,
    )

    # 各模态均值柱状图 + 协方差热力图，直观对比模态间分布差异
    plot_mode_mean_cov(
        [np.asarray(mode, dtype=float) for mode in dataset["train_modes"]],
        a_matrices_dir / "mode_mean_cov.png",
        mode_names=mode_names,
    )

    # 状态向量 s1/s2 抽样分布图
    plot_state_distribution(
        a_matrices_dir / "s_distribution.png",
        n_modes * n_train_per_mode,
        random_state,
    )

    # 测试数据热力图：JSSDL 样式（原始数值对称着色 + 故障变量行红框 + 红虚线 + 文字标注）
    n_test_per_mode = int(dataset["n_test_per_mode"])
    test_boundaries = [mode_index * n_test_per_mode for mode_index in range(n_modes + 1)]
    plot_test_data_heatmap(
        np.asarray(dataset["test_all"], dtype=float),
        int(dataset["fault_feature"]),
        np.asarray(dataset["fault_labels"], dtype=int),
        test_dir / "test_data_heatmap.png",
        mode_boundaries=test_boundaries,
    )

    # 测试数据时序折线图：故障变量 x2 上的阶跃 + 模态分界 + 故障区间阴影
    plot_test_fault_timeseries(
        np.asarray(dataset["test_all"], dtype=float),
        int(dataset["fault_feature"]),
        np.asarray(dataset["fault_labels"], dtype=int),
        test_dir / "test_fault_timeseries.png",
        mode_boundaries=test_boundaries,
        test_normal=np.asarray(dataset["test_normal"], dtype=float),
        mode_names=mode_names,
    )

    # 测试数据 PCA 散点图：模态上色 + 正常/故障不同 marker
    plot_test_pca_scatter(
        np.asarray(dataset["test_all"], dtype=float),
        np.asarray(dataset["test_mode_labels"], dtype=int),
        np.asarray(dataset["fault_labels"], dtype=int),
        test_dir / "test_pca_scatter.png",
        mode_names=mode_names,
    )

    print("数据生成完成：")
    print(f"  训练数据：{train_dir}")
    print(f"  测试数据：{test_dir}")
    print(f"  观测矩阵/分布图：{a_matrices_dir}")
    print(f"  PCA 散点图选用主成分：PC{first + 1} vs PC{second + 1}")
    print(f"  图已保存到：{train_dir}、{test_dir} 与 {a_matrices_dir}")


if __name__ == "__main__":
    main()
