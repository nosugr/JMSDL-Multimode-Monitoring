"""多模态数据可视化。

提供两类图：
1. PCA 二维散点图（plot_multimode_pca_scatter）—— 自动选取对模态区分度最高的两个主成分，
   按模态上色并圈出置信椭圆，复现论文 Fig.1 的多模态可视化效果。
2. 多模态时序折线图（plot_multimode_timeseries）—— 选若干维度沿样本号绘制，用虚线分隔各模态，
   展示模态切换时数据分布的跳变。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Rectangle

MODE_COLORS = ["tab:blue", "tab:orange", "tab:green", "#a76de0"]


def _ensure_parent(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _pca_scores(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """对数据做（仅可视化用的）PCA，返回主成分得分与各主成分方差解释比例。"""
    X = np.asarray(data, dtype=float)
    centered = X - X.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    scores = centered @ vt.T
    variances = singular_values**2
    explained_ratio = variances / variances.sum() if variances.sum() > 0 else variances
    return scores, explained_ratio


def _component_discriminability(component_scores: np.ndarray, labels: np.ndarray) -> float:
    """单个主成分的类间/类内方差比（Fisher 比），用于衡量对模态的区分度。"""
    values = np.asarray(component_scores, dtype=float)
    label_values = np.asarray(labels)
    overall_mean = values.mean()
    between = 0.0
    within = 0.0
    for mode in np.unique(label_values):
        group = values[label_values == mode]
        between += group.size * (group.mean() - overall_mean) ** 2
        within += float(((group - group.mean()) ** 2).sum())
    if within <= 1.0e-12:
        return float("inf")
    return between / within


def _select_discriminative_components(scores: np.ndarray, labels: np.ndarray, k: int = 2) -> list[int]:
    """挑选区分度最高的 k 个主成分（返回主成分索引，已按区分度降序）。"""
    ratios = [_component_discriminability(scores[:, j], labels) for j in range(scores.shape[1])]
    order = np.argsort(ratios)[::-1]
    return [int(index) for index in order[:k]]


def _add_confidence_ellipse(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    edgecolor: str = "red",
    n_std: float = 2.4,
    linewidth: float = 1.8,
) -> None:
    """在散点簇外围添加协方差置信椭圆。"""
    if x.size < 3:
        return
    covariance = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = np.clip(eigenvalues[order], a_min=0.0, a_max=None)
    eigenvectors = eigenvectors[:, order]
    angle = float(np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])))
    width, height = 2.0 * n_std * np.sqrt(eigenvalues)
    ellipse = Ellipse(
        (float(x.mean()), float(y.mean())),
        width=width,
        height=height,
        angle=angle,
        fill=False,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    ax.add_patch(ellipse)


def plot_multimode_pca_scatter(
    data: np.ndarray,
    mode_labels: np.ndarray,
    output_path: str | Path,
    mode_names: list[str] | None = None,
    draw_ellipses: bool = True,
    title: str = "Multimode Data in Principal Component Space",
    figsize: tuple[float, float] = (7.0, 6.0),
) -> tuple[int, int]:
    """绘制多模态 PCA 散点图，返回所选两个主成分的索引 (0 基)。"""
    path = _ensure_parent(output_path)
    labels = np.asarray(mode_labels)
    scores, explained_ratio = _pca_scores(data)
    selected = _select_discriminative_components(scores, labels, k=2)
    first, second = selected[0], selected[1]
    xs = scores[:, first]
    ys = scores[:, second]

    unique_modes = np.unique(labels)
    fig, ax = plt.subplots(figsize=figsize)
    for position, mode in enumerate(unique_modes):
        mask = labels == mode
        color = MODE_COLORS[position % len(MODE_COLORS)]
        name = mode_names[position] if mode_names is not None else f"Mode{int(mode) + 1}"
        ax.scatter(xs[mask], ys[mask], s=12, color=color, alpha=0.7, label=name)
        if draw_ellipses:
            _add_confidence_ellipse(ax, xs[mask], ys[mask], edgecolor="red")

    ax.set_xlabel(f"PC{first + 1} ({explained_ratio[first] * 100:.1f}%)")
    ax.set_ylabel(f"PC{second + 1} ({explained_ratio[second] * 100:.1f}%)")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return first, second


def plot_multimode_timeseries(
    data: np.ndarray,
    mode_boundaries: list[int] | np.ndarray,
    output_path: str | Path,
    dims: tuple[int, ...] = (0, 1),
    mode_names: list[str] | None = None,
    title: str = "Multimode data",
    figsize: tuple[float, float] = (12.0, 5.0),
) -> None:
    """绘制多模态时序折线图。

    mode_boundaries: 各模态分界点（含首尾），如 [0, 1000, 2000, 3000, 4000]。
    dims: 要展示的维度索引 (0 基)。
    """
    path = _ensure_parent(output_path)
    X = np.asarray(data, dtype=float)
    n_samples = X.shape[0]
    x_axis = np.arange(n_samples)
    boundaries = list(mode_boundaries)
    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    fig, ax = plt.subplots(figsize=figsize)
    for position, dim in enumerate(dims):
        ax.plot(
            x_axis,
            X[:, dim],
            linewidth=0.8,
            color=palette[position % len(palette)],
            label=f"Dimension {dim + 1}",
        )

    # 内部模态分界虚线
    for boundary in boundaries[1:-1]:
        ax.axvline(boundary, color="red", linestyle="--", linewidth=1.2, alpha=0.9)

    # 顶部留出空白，使模态名称单独占一行、不与数据/图例重叠
    selected_values = X[:, list(dims)]
    y_min = float(np.min(selected_values))
    y_max = float(np.max(selected_values))
    y_range = y_max - y_min if y_max > y_min else 1.0
    ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.18 * y_range)

    # 模态名称标注（放在数据上方的留白区）
    y_label = y_max + 0.06 * y_range
    for segment in range(len(boundaries) - 1):
        center = (boundaries[segment] + boundaries[segment + 1]) / 2.0
        name = mode_names[segment] if mode_names is not None else f"Mode{segment + 1}"
        ax.text(center, y_label, name, color="red", fontsize=11, ha="center", va="bottom")

    ax.set_xlabel("Sample number")
    ax.set_ylabel("Value")
    ax.set_title(title, pad=28)
    ax.set_xlim(0, n_samples)
    # 图例放到坐标区上方、横向排列（标题之下），避免遮挡曲线
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=len(dims), frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _contiguous_runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """从 0/1 标记数组中找出所有连续为 1 的区间，返回 [(start, end), ...]（end 不含）。"""
    mask = np.asarray(flags).astype(bool)
    runs: list[tuple[int, int]] = []
    start = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, mask.size))
    return runs


def plot_test_data_heatmap(
    data: np.ndarray,
    fault_feature: int,
    fault_labels: np.ndarray,
    output_path: str | Path,
    mode_boundaries: list[int] | np.ndarray | None = None,
    baseline: np.ndarray | None = None,
    center: bool = False,
    title: str = "Raw Test Data (Fault Highlighted)",
    figsize: tuple[float, float] = (12.0, 4.8),
    cmap: str = "coolwarm",
) -> None:
    """绘制测试数据热力图并标出故障（样式与 JSSDL plot_fault_annotated_heatmap 一致）。

    样式要点（照搬 JSSDL）：
    - 直接使用原始数值，按 ±max_abs 对称着色（不做去均值、不做分位裁剪）。
    - cmap 默认 coolwarm。
    - 故障变量行用红色实线框 + 故障起点红色虚线 + 白底红框 "Fault feature = N" 文字标注。
    - 使用默认 colorbar。

    参数
    ----
    data: (n_samples, n_features) 测试数据（一般传 test_faulty）。
    fault_feature: 故障变量索引（0 基）。
    fault_labels: (n_samples,) 0 正常 / 1 故障，用于定位故障样本区间。
    mode_boundaries / baseline / center: 保留以兼容旧调用，默认不参与 JSSDL 样式绘制。
    """
    path = _ensure_parent(output_path)
    X = np.asarray(data, dtype=float)
    n_samples, n_features = X.shape

    if center:
        reference = np.asarray(baseline, dtype=float) if baseline is not None else X
        X = X - reference.mean(axis=0, keepdims=True)

    matrix = X.T  # (n_features, n_samples)，行=变量，列=样本
    label_values = np.asarray(fault_labels, dtype=int).ravel()
    feature_index = int(fault_feature)
    if not 0 <= feature_index < n_features:
        raise ValueError(f"fault_feature 必须在 [0, {n_features - 1}]，得到 {feature_index}。")

    max_abs = float(np.max(np.abs(matrix))) if matrix.size else 0.0

    fig, ax = plt.subplots(figsize=figsize)
    if max_abs > 0.0:
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-max_abs, vmax=max_abs)
    else:
        image = ax.imshow(matrix, aspect="auto", cmap=cmap)

    # 故障变量行的红框 + 故障起点红虚线（每个连续故障区间各一处，适配多模态多段故障）
    fault_runs = _contiguous_runs(label_values)
    for run_start, run_end in fault_runs:
        ax.axvline(run_start - 0.5, color="red", linestyle="--", linewidth=1.5, alpha=0.9)
        rectangle = Rectangle(
            (run_start - 0.5, feature_index - 0.5),
            run_end - run_start,
            1.0,
            fill=False,
            edgecolor="red",
            linewidth=2.0,
        )
        ax.add_patch(rectangle)

    # 文字标注故障变量（显示为 1 基，白底红框，放在故障变量行上方）
    if fault_runs:
        first_start = fault_runs[0][0]
        ax.text(
            first_start,
            max(feature_index - 1.1, 0.0),
            f"Fault feature = {feature_index + 1}",
            color="red",
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "red", "alpha": 0.75, "pad": 2},
        )

    ax.set_yticks(np.arange(n_features))
    ax.set_yticklabels(np.arange(1, n_features + 1))
    ax.set_xlabel("Sample")
    ax.set_ylabel("Feature")
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_test_fault_timeseries(
    test_faulty: np.ndarray,
    fault_feature: int,
    fault_labels: np.ndarray,
    output_path: str | Path,
    mode_boundaries: list[int] | np.ndarray,
    test_normal: np.ndarray | None = None,
    mode_names: list[str] | None = None,
    title: str = "Multimode test data (fault on x2)",
    figsize: tuple[float, float] = (12.0, 5.0),
) -> None:
    """测试数据时序折线图（增强版）：重点展示故障变量上的阶跃。

    风格与训练图 plot_multimode_timeseries 保持统一（线宽、调色板、坐标轴标签、
    图例位置、红虚线分界 + 顶部红色模态名），在此基础上叠加两个故障展示元素：
    - 浅红背景阴影（axvspan）标出每个模态末尾的故障样本区间。
    - 若给定 test_normal，则叠加同一变量的正常曲线作为对照。

    参数
    ----
    test_faulty: (n_samples, n_features) 含故障的测试数据。
    fault_feature: 故障变量索引（0 基），论文设定为 1（即 x2）。
    fault_labels: (n_samples,) 0 正常 / 1 故障。
    mode_boundaries: 各模态分界点（含首尾，如 [0,250,500,750,1000]）。
    test_normal: 可选，(n_samples, n_features) 全正常测试数据，用作对照基线。
    """
    path = _ensure_parent(output_path)
    faulty = np.asarray(test_faulty, dtype=float)
    feature_index = int(fault_feature)
    n_samples = faulty.shape[0]
    x_axis = np.arange(n_samples)
    boundaries = list(mode_boundaries)
    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    fig, ax = plt.subplots(figsize=figsize)

    # 故障区间浅红阴影（只在每段第一次出现时加图例）
    fault_runs = _contiguous_runs(fault_labels)
    for run_position, (run_start, run_end) in enumerate(fault_runs):
        ax.axvspan(
            run_start,
            run_end,
            color="red",
            alpha=0.12,
            label="Fault interval" if run_position == 0 else None,
        )

    # 正常对照曲线（可选）—— 用调色板第二色，与训练图维度配色一致
    if test_normal is not None:
        normal = np.asarray(test_normal, dtype=float)
        ax.plot(
            x_axis,
            normal[:, feature_index],
            linewidth=0.8,
            color=palette[1],
            alpha=0.7,
            label=f"Dimension {feature_index + 1} (normal)",
        )

    # 故障变量主曲线 —— 用调色板首色，与训练图风格一致
    ax.plot(
        x_axis,
        faulty[:, feature_index],
        linewidth=0.8,
        color=palette[0],
        label=f"Dimension {feature_index + 1} (with fault)",
    )

    # 内部模态分界虚线
    for boundary in boundaries[1:-1]:
        ax.axvline(boundary, color="red", linestyle="--", linewidth=1.2, alpha=0.9)

    # 顶部留出空白，使模态名称单独占一行、不与数据/图例重叠
    series = faulty[:, feature_index]
    y_min = float(series.min())
    y_max = float(series.max())
    y_range = y_max - y_min if y_max > y_min else 1.0
    ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.18 * y_range)

    # 模态名称标注（放在数据上方的留白区）
    y_label = y_max + 0.06 * y_range
    for segment in range(len(boundaries) - 1):
        center = (boundaries[segment] + boundaries[segment + 1]) / 2.0
        name = mode_names[segment] if mode_names is not None else f"Mode{segment + 1}"
        ax.text(center, y_label, name, color="red", fontsize=11, ha="center", va="bottom")

    ax.set_xlabel("Sample number")
    ax.set_ylabel("Value")
    ax.set_title(title, pad=28)
    ax.set_xlim(0, n_samples)
    # 图例放到坐标区上方、横向排列（标题之下），避免遮挡曲线
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_test_pca_scatter(
    test_faulty: np.ndarray,
    mode_labels: np.ndarray,
    fault_labels: np.ndarray,
    output_path: str | Path,
    mode_names: list[str] | None = None,
    draw_ellipses: bool = True,
    title: str = "Test Data in Principal Component Space (Normal vs Fault)",
    figsize: tuple[float, float] = (7.5, 6.0),
) -> tuple[int, int]:
    """测试数据 PCA 散点图：按模态上色，用不同 marker 区分正常/故障。

    - 颜色 = 模态；正常用圆点 'o'，故障用叉 'x'。
    - 置信椭圆基于各模态的正常样本绘制，故障点落在椭圆外即体现故障可分。
    - 主成分选取沿用 _select_discriminative_components（按模态区分度）。

    返回所选两个主成分索引（0 基）。
    """
    path = _ensure_parent(output_path)
    X = np.asarray(test_faulty, dtype=float)
    modes = np.asarray(mode_labels)
    faults = np.asarray(fault_labels).astype(bool)

    scores, explained_ratio = _pca_scores(X)
    selected = _select_discriminative_components(scores, modes, k=2)
    first, second = selected[0], selected[1]
    xs = scores[:, first]
    ys = scores[:, second]

    unique_modes = np.unique(modes)
    fig, ax = plt.subplots(figsize=figsize)
    for position, mode in enumerate(unique_modes):
        color = MODE_COLORS[position % len(MODE_COLORS)]
        name = mode_names[position] if mode_names is not None else f"Mode{int(mode) + 1}"
        normal_mask = (modes == mode) & (~faults)
        fault_mask = (modes == mode) & faults

        ax.scatter(
            xs[normal_mask], ys[normal_mask],
            s=14, color=color, alpha=0.7, marker="o", label=f"{name} normal",
        )
        ax.scatter(
            xs[fault_mask], ys[fault_mask],
            s=28, color=color, alpha=0.9, marker="x", linewidths=1.2,
            label=f"{name} fault",
        )
        # 椭圆基于该模态正常样本
        if draw_ellipses:
            _add_confidence_ellipse(ax, xs[normal_mask], ys[normal_mask], edgecolor=color)

    ax.set_xlabel(f"PC{first + 1} ({explained_ratio[first] * 100:.1f}%)")
    ax.set_ylabel(f"PC{second + 1} ({explained_ratio[second] * 100:.1f}%)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return first, second


def plot_dictionary_heatmap(
    dictionary: np.ndarray,
    output_path: str | Path,
    title: str = "Learned Dictionary",
    cmap: str = "viridis",
    figsize: tuple[float, float] = (8.5, 5.2),
    feature_labels: list[int | str] | None = None,
    show_title: bool = True,
) -> None:
    """字典矩阵热力图（行=特征，列=原子）。

    画法照搬 JSSDL：viridis 配色 + 对称色标 (±max) + 每个单元格描细灰边，
    视觉上比红蓝发散色标更清爽。各基线 (DL/LCDL/ODL) 与 JMSDL 各阶段字典统一用此函数。
    """
    path = _ensure_parent(output_path)
    values = np.asarray(dictionary, dtype=float)
    if values.ndim != 2:
        raise ValueError("dictionary must be a 2D (features x atoms) matrix.")
    n_features, n_atoms = values.shape
    max_abs = float(np.max(np.abs(values))) if values.size else 0.0

    if feature_labels is None:
        labels = [str(index) for index in range(1, n_features + 1)]
    else:
        labels = [str(label) for label in feature_labels]
        if len(labels) != n_features:
            raise ValueError(
                f"Expected {n_features} feature labels for the dictionary heatmap, got {len(labels)}."
            )

    fig, ax = plt.subplots(figsize=figsize)
    if max_abs > 0.0:
        image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=-max_abs, vmax=max_abs, interpolation="nearest")
    else:
        image = ax.imshow(values, aspect="auto", cmap=cmap, interpolation="nearest")

    ax.set_yticks(np.arange(n_features))
    ax.set_yticklabels(labels)
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    if show_title:
        ax.set_title(title)

    # 细灰网格描出每个单元格边界
    ax.set_xticks(np.arange(-0.5, n_atoms, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_features, 1), minor=True)
    ax.grid(which="minor", color="0.15", linestyle="-", linewidth=0.35, alpha=0.85)
    ax.tick_params(axis="both", which="major", length=0)
    ax.tick_params(axis="both", which="minor", bottom=False, left=False)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.15")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_monitoring_scores(
    scores: np.ndarray,
    threshold: float,
    fault_labels: np.ndarray,
    output_path: str | Path,
    mode_boundaries: list[int] | np.ndarray | None = None,
    fdr: float | None = None,
    far: float | None = None,
    statistic_name: str = "IRE",
    title: str = "Process Monitoring",
    figsize: tuple[float, float] = (12.0, 5.0),
) -> None:
    """监测结果图：逐样本监测统计量 + 控制限 + 故障区间阴影 + FDR/FAR 标注。

    参数
    ----
    scores: (n_samples,) 监测统计量（JMSDL/基线的 IRE，mPCA 的归一化分数）。
    threshold: 控制限（红色水平线）；对已归一化分数 (mPCA) 传 1.0。
    fault_labels: (n_samples,) 0 正常 / 1 故障，用浅红阴影标出连续故障段。
    mode_boundaries: 可选模态分界点（含首尾），内部分界画灰色虚线。
    fdr/far: 可选，若给定则在角上标注。
    """
    path = _ensure_parent(output_path)
    values = np.asarray(scores, dtype=float).ravel()
    faults = np.asarray(fault_labels, dtype=int)
    n_samples = values.size
    x_axis = np.arange(n_samples)

    fig, ax = plt.subplots(figsize=figsize)

    # 故障区间浅红阴影
    for run_position, (run_start, run_end) in enumerate(_contiguous_runs(faults)):
        ax.axvspan(
            run_start, run_end, color="red", alpha=0.12,
            label="Fault interval" if run_position == 0 else None,
        )

    ax.plot(x_axis, values, linewidth=0.8, color="tab:blue", label=statistic_name)
    ax.axhline(
        float(threshold), color="red", linestyle="--", linewidth=1.4,
        label=f"Control limit = {float(threshold):.4g}",
    )

    if mode_boundaries is not None:
        for boundary in list(mode_boundaries)[1:-1]:
            ax.axvline(boundary, color="black", linestyle="-", linewidth=1.0, alpha=0.35)

    if fdr is not None or far is not None:
        parts = []
        if fdr is not None:
            parts.append(f"FDR = {float(fdr):.3f}")
        if far is not None:
            parts.append(f"FAR = {float(far):.3f}")
        ax.text(
            0.012, 0.97, "\n".join(parts), transform=ax.transAxes,
            ha="left", va="top", fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="red"),
        )

    ax.set_xlabel("Sample number")
    ax.set_ylabel(statistic_name)
    ax.set_title(title)
    ax.set_xlim(0, n_samples)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
