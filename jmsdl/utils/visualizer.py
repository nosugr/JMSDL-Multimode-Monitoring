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


def _heatmap_limit(values: np.ndarray, floor: float = 1.0) -> float:
    matrix = np.asarray(values, dtype=float)
    if matrix.size == 0:
        return float(floor)
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return float(floor)
    limit = float(np.percentile(np.abs(finite), 99.0))
    return max(limit, float(floor), 1.0e-12)


def _matrix_ticks(size: int, max_ticks: int = 5) -> tuple[np.ndarray, list[str]]:
    if size <= 0:
        return np.asarray([], dtype=int), []
    if size <= max_ticks:
        labels = np.arange(1, size + 1, dtype=int)
    else:
        step = int(np.ceil(size / max(1, max_ticks - 1)))
        labels = [1, *range(step, size, step), size]
        labels = np.asarray(labels, dtype=int)
    return labels - 1, [str(int(label)) for label in labels]


def _set_matrix_axis_ticks(ax: plt.Axes, n_rows: int, n_columns: int) -> None:
    y_positions, y_labels = _matrix_ticks(n_rows, max_ticks=5)
    x_positions, x_labels = _matrix_ticks(n_columns, max_ticks=6)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.tick_params(axis="both", which="major", length=0)


def _set_jssdl_heatmap_ticks(ax: plt.Axes, n_rows: int, n_columns: int) -> None:
    if n_columns >= 3000:
        x_step = 1000
    elif n_columns >= 600:
        x_step = 200
    else:
        x_step = max(1, int(np.ceil(n_columns / 5)))
    y_step = 20 if n_rows >= 60 else max(1, int(np.ceil(n_rows / 5)))
    x_ticks = np.arange(0, n_columns, x_step, dtype=int)
    y_ticks = np.arange(0, n_rows + 1, y_step, dtype=int)
    y_ticks = y_ticks[y_ticks < n_rows]
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([str(int(value)) for value in x_ticks])
    ax.set_yticklabels([str(int(value)) for value in y_ticks])


def _plot_jssdl_code_heatmap(
    matrix: np.ndarray,
    output_path: str | Path,
    title: str,
    xlabel: str = "Sample",
    ylabel: str = "Atom",
    figsize: tuple[float, float] = (13.5, 4.2),
) -> None:
    path = _ensure_parent(output_path)
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("matrix must be a 2D array.")

    max_abs = float(np.max(np.abs(values))) if values.size else 0.0
    fig, ax = plt.subplots(figsize=figsize)
    if max_abs > 0.0:
        image = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-max_abs, vmax=max_abs)
    else:
        image = ax.imshow(values, aspect="auto", cmap="coolwarm")

    _set_jssdl_heatmap_ticks(ax, values.shape[0], values.shape[1])
    fig.colorbar(image, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _ranges_from_boundaries(
    boundaries: list[int] | np.ndarray | None,
    size: int,
) -> list[tuple[int, int]]:
    if boundaries is None:
        return [(0, int(size))]
    values = [int(value) for value in list(boundaries)]
    ranges: list[tuple[int, int]] = []
    for start, end in zip(values[:-1], values[1:]):
        lo = max(0, min(int(size), start))
        hi = max(0, min(int(size), end))
        if hi > lo:
            ranges.append((lo, hi))
    return ranges or [(0, int(size))]


def _block_means(
    matrix: np.ndarray,
    row_ranges: list[tuple[int, int]],
    column_ranges: list[tuple[int, int]],
) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    summary = np.zeros((len(row_ranges), len(column_ranges)), dtype=float)
    for row_index, (row_start, row_end) in enumerate(row_ranges):
        for column_index, (column_start, column_end) in enumerate(column_ranges):
            block = values[row_start:row_end, column_start:column_end]
            summary[row_index, column_index] = float(np.mean(block)) if block.size else 0.0
    return summary


def _plot_grouped_block_bars(
    ax: plt.Axes,
    values: np.ndarray,
    title: str,
    ylabel: str,
    ylim: tuple[float, float],
    colors: list[str],
) -> None:
    matrix = np.asarray(values, dtype=float)
    n_groups, n_modes = matrix.shape
    x_positions = np.arange(n_modes, dtype=float)
    total_width = 0.78
    bar_width = total_width / max(1, n_groups)
    offsets = (np.arange(n_groups, dtype=float) - (n_groups - 1) / 2.0) * bar_width

    for group_index in range(n_groups):
        ax.bar(
            x_positions + offsets[group_index],
            matrix[group_index, :],
            width=bar_width * 0.9,
            color=colors[group_index % len(colors)],
            alpha=0.82,
            label=f"Atom group {group_index + 1}",
        )

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Sample mode")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(index + 1) for index in range(n_modes)])
    ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _draw_heatmap_guides(
    ax: plt.Axes,
    mode_boundaries: list[int] | np.ndarray | None = None,
    atom_ranges: list[tuple[int, int]] | None = None,
    color: str = "white",
    alpha: float = 0.9,
) -> None:
    if mode_boundaries is not None:
        for boundary in list(mode_boundaries)[1:-1]:
            ax.axvline(boundary - 0.5, color=color, linestyle=":", linewidth=1.1, alpha=alpha)
    if atom_ranges is not None:
        for _, hi in atom_ranges[:-1]:
            ax.axhline(hi - 0.5, color=color, linestyle="-", linewidth=1.0, alpha=alpha)


def plot_sparse_code_heatmap(
    codes: np.ndarray,
    output_path: str | Path,
    mode_boundaries: list[int] | np.ndarray | None = None,
    atom_ranges: list[tuple[int, int]] | None = None,
    fault_labels: np.ndarray | None = None,
    title: str = "Sparse Codes X",
    figsize: tuple[float, float] = (8.8, 6.0),
    cmap: str = "coolwarm",
    absolute: bool = False,
) -> None:
    """Plot sparse codes W with optional mode and fault annotations."""
    path = _ensure_parent(output_path)
    W = np.asarray(codes, dtype=float)
    if W.ndim != 2:
        raise ValueError("codes must be a 2D (atoms x samples) matrix.")
    del mode_boundaries, atom_ranges, fault_labels, cmap
    display = np.abs(W) if absolute else W
    _plot_jssdl_code_heatmap(display, path, title=title, xlabel="Sample", ylabel="Atom", figsize=figsize)


def plot_mode_match_confusion(
    true_modes: np.ndarray,
    matched_modes: np.ndarray,
    output_path: str | Path,
    n_modes: int | None = None,
    normalize: bool = True,
    mode_names: list[str] | None = None,
    title: str = "Mode-Matching Confusion Matrix",
    figsize: tuple[float, float] = (5.6, 4.8),
) -> np.ndarray:
    """模态匹配混淆矩阵热力图：行=真实模态，列=匹配模态。

    normalize=True 时按行归一化（每行=该真实模态样本的匹配比例，对角线即匹配准确率），
    单元格上叠加数值标注。返回原始计数矩阵 (n_modes, n_modes)。
    """
    path = _ensure_parent(output_path)
    true_labels = np.asarray(true_modes, dtype=int).ravel()
    matched_labels = np.asarray(matched_modes, dtype=int).ravel()
    if true_labels.size != matched_labels.size:
        raise ValueError("true_modes and matched_modes must have the same length.")

    k = int(n_modes) if n_modes is not None else int(max(true_labels.max(), matched_labels.max()) + 1)
    counts = np.zeros((k, k), dtype=int)
    for t, m in zip(true_labels, matched_labels):
        counts[t, m] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    display = counts / np.where(row_sums < 1, 1, row_sums) if normalize else counts.astype(float)

    names = mode_names if mode_names is not None else [f"Mode {i + 1}" for i in range(k)]

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    image = ax.imshow(
        display, cmap="Blues", vmin=0.0, vmax=1.0 if normalize else display.max(),
    )

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Match proportion" if normalize else "Sample count", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # 单元格数值标注：归一化显示比例，原始计数显示个数；按背景深浅自动反色。
    threshold = (display.max() + display.min()) / 2.0
    for i in range(k):
        for j in range(k):
            text = f"{display[i, j]:.2f}" if normalize else f"{counts[i, j]:d}"
            ax.text(
                j, i, text, ha="center", va="center", fontsize=11,
                color="white" if display[i, j] > threshold else "#222222",
            )

    ax.set_xticks(np.arange(k))
    ax.set_yticks(np.arange(k))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Matched mode", fontsize=11)
    ax.set_ylabel("True mode", fontsize=11)
    ax.set_title(title, fontsize=13)

    # 浅灰网格分隔单元格，弱化但清晰。
    ax.set_xticks(np.arange(-0.5, k, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, k, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    fig.savefig(path, dpi=300)
    plt.close(fig)
    return counts


def plot_label_response(
    label_response: np.ndarray,
    output_path: str | Path,
    label_matrix: np.ndarray | None = None,
    mode_boundaries: list[int] | np.ndarray | None = None,
    atom_ranges: list[tuple[int, int]] | None = None,
    mode_names: list[str] | None = None,
    title: str = "Reconstructed Label Response A·W",
    figsize: tuple[float, float] = (10.0, 6.0),
) -> None:
    """可视化变换后的标签响应 A·W (n_atoms, n_samples)，并与理想 Q 对比。

    LCDL 训练含约束 ||Q - A·W||²；这里把实际学到的 A·W 画出来，看它逼近理想 Q 的程度。
    若给定 label_matrix(Q)，则上图 A·W、下图残差 Q - A·W，便于直接看差异。
    A·W 为连续值，用对称色标（以 0 为中心）。
    """
    path = _ensure_parent(output_path)
    AW = np.asarray(label_response, dtype=float)
    if AW.ndim != 2:
        raise ValueError("label_response must be a 2D (atoms x samples) matrix.")
    n_atoms, n_samples = AW.shape

    Q = None if label_matrix is None else np.asarray(label_matrix, dtype=float)
    n_rows = 2 if Q is not None else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, constrained_layout=True, squeeze=False)

    def _draw(ax, matrix, sub_title, cmap, vmin, vmax, cbar_label):
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        if mode_boundaries is not None:
            for boundary in list(mode_boundaries)[1:-1]:
                ax.axvline(boundary - 0.5, color="#555555", linestyle="--", linewidth=0.8, alpha=0.7)
        if atom_ranges is not None:
            for _, hi in atom_ranges[:-1]:
                ax.axhline(hi - 0.5, color="#555555", linestyle="--", linewidth=0.8, alpha=0.7)
        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label(cbar_label, fontsize=10)
        cbar.ax.tick_params(labelsize=9)
        ax.set_xlabel("Training sample", fontsize=11)
        ax.set_ylabel("Atom index", fontsize=11)
        ax.tick_params(axis="both", labelsize=10)
        ax.set_title(sub_title, fontsize=12, pad=20)

    # 上图：理想标签矩阵 Q（配色不变，对称色标；无 Q 时退回画 A·W）
    top_matrix = Q if Q is not None else AW
    top_title = "Ideal Q" if Q is not None else "A·W (learned)"
    top_cbar = "Ideal code" if Q is not None else "Response"
    limit = float(np.max(np.abs(top_matrix))) if top_matrix.size else 1.0
    limit = limit if limit > 1.0e-12 else 1.0
    _draw(axes[0, 0], top_matrix, top_title, "RdBu_r", -limit, limit, top_cbar)

    # 模态名标注在最上方子图顶端
    if mode_boundaries is not None:
        bounds = list(mode_boundaries)
        names = mode_names if mode_names is not None else [f"Mode {i + 1}" for i in range(len(bounds) - 1)]
        for index in range(len(bounds) - 1):
            center = (bounds[index] + bounds[index + 1]) / 2.0
            axes[0, 0].text(center, -1.2, names[index], ha="center", va="bottom", fontsize=9, color="#333333")

    # 下图：残差 Q - A·W
    if Q is not None:
        residual = Q - AW
        rlimit = float(np.max(np.abs(residual))) if residual.size else 1.0
        rlimit = rlimit if rlimit > 1.0e-12 else 1.0
        mae = float(np.mean(np.abs(residual)))
        _draw(
            axes[1, 0], residual,
            f"Residual Q - A·W   (MAE={mae:.3g})",
            "RdBu_r", -rlimit, rlimit, "Q - A·W",
        )

    fig.suptitle(title, fontsize=13)
    fig.savefig(path, dpi=300, bbox_inches="tight")
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
    figsize: tuple[float, float] = (10.0, 4.0),
) -> None:
    """监测结果图（JSSDL 风格）：监测统计量实线 + 红虚线控制限 + 故障样本橙色散点。

    样式照搬 JSSDL plot_monitoring_scores：
    - 实线主曲线（蓝），线宽 1.4。
    - 红色虚线控制限，标签 "Threshold"。
    - 故障样本用橙色散点叠在曲线上（不画区间阴影）。
    - 若给定 fdr/far，则以文字标注在坐标轴上方（FAR 左、FDR 右）。
    - 可选 mode_boundaries：内部模态分界用灰色点状竖线标出。

    参数
    ----
    scores: (n_samples,) 监测统计量（JMSDL/基线的 IRE，mPCA 的归一化分数）。
    threshold: 控制限（红色水平线）；对已归一化分数 (mPCA) 传 1.0。
    fault_labels: (n_samples,) 0 正常 / 1 故障，故障样本画橙色散点。
    mode_boundaries: 可选模态分界点（含首尾），内部分界画灰色点状线。
    fdr/far: 可选，若给定则在坐标轴上方标注。
    """
    path = _ensure_parent(output_path)
    values = np.asarray(scores, dtype=float).ravel()
    faults = np.asarray(fault_labels, dtype=int).ravel()
    n_samples = values.size
    x_axis = np.arange(1, n_samples + 1)

    annotate = fdr is not None or far is not None

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x_axis, values, linewidth=1.4, color="tab:blue", label=statistic_name)
    ax.axhline(
        float(threshold), color="tab:red", linestyle="--", linewidth=1.5,
        label=f"Control limit = {float(threshold):.4g}",
    )

    fault_regions = np.where(faults == 1)[0]
    if fault_regions.size > 0:
        ax.scatter(
            x_axis[fault_regions], values[fault_regions],
            s=10, c="tab:orange", alpha=0.6, label="Faulty samples",
        )

    if mode_boundaries is not None:
        for boundary in list(mode_boundaries)[1:-1]:
            ax.axvline(boundary, color="0.5", linestyle=":", linewidth=1.0, alpha=0.75)

    if annotate:
        if far is not None:
            ax.text(0.25, 1.03, f"FAR: {float(far) * 100:.1f}%", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=11)
        if fdr is not None:
            ax.text(0.75, 1.03, f"FDR: {float(fdr) * 100:.1f}%", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=11)
    else:
        ax.set_title(title)

    ax.set_xlabel("Sample number")
    ax.set_ylabel(statistic_name)
    ax.set_xlim(1, n_samples)
    ax.legend()
    ax.grid(alpha=0.3)
    if annotate:
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_catastrophic_forgetting(
    errors: np.ndarray,
    mode_boundaries: list[int] | np.ndarray,
    output_path: str | Path,
    mode_names: list[str] | None = None,
    title: str = "ODL Catastrophic Forgetting",
    figsize: tuple[float, float] = (10.0, 4.6),
) -> None:
    """逐样本重构误差分段折线图，凸显灾难性遗忘。

    用最终字典回头重构各历史模态时，越早的模态误差越大。各模态分段用不同颜色，
    不画控制限、不标 FAR/FDR，图例置于右上角。
    """
    path = _ensure_parent(output_path)
    values = np.asarray(errors, dtype=float).ravel()
    n_samples = values.size
    x_axis = np.arange(n_samples)
    boundaries = [int(b) for b in mode_boundaries]
    palette = ["#e8000b", "#1f3df0", "#00c4c4", "#9bb7e0"]

    fig, ax = plt.subplots(figsize=figsize)
    for segment in range(len(boundaries) - 1):
        start, end = boundaries[segment], boundaries[segment + 1]
        color = palette[segment % len(palette)]
        name = mode_names[segment] if mode_names is not None else f"Mode{segment + 1}"
        ax.plot(x_axis[start:end], values[start:end], linewidth=0.9, color=color, label=name)

    ax.set_xlabel("Sample Number")
    ax.set_ylabel("Reconstruction Error")
    ax.set_title(title)
    ax.set_xlim(0, n_samples)
    ax.set_ylim(0, float(values.max()) * 1.05 if values.size else 1.0)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_mode_mean_cov(
    train_modes: list[np.ndarray] | tuple[np.ndarray, ...],
    output_path: str | Path,
    mode_names: list[str] | None = None,
    title: str = "Per-Mode Mean and Covariance",
    figsize: tuple[float, float] | None = None,
) -> None:
    """各训练模态的均值柱状图 + 协方差热力图，用于直观对比模态间分布差异。

    上排为每个模态的协方差矩阵热力图（对称色标），下排为对应的均值向量柱状图，
    便于观察“均值差异”（整体位置）与“协方差差异”（形状/方向）。
    """
    path = _ensure_parent(output_path)
    modes = [np.asarray(mode, dtype=float) for mode in train_modes]
    if len(modes) == 0:
        raise ValueError("At least one mode is required.")
    n_modes = len(modes)
    names = mode_names if mode_names is not None else [f"Mode{i + 1}" for i in range(n_modes)]
    if figsize is None:
        figsize = (5.0 * n_modes, 9.0)

    fig, axes = plt.subplots(2, n_modes, figsize=figsize, squeeze=False)
    for index, mode_data in enumerate(modes):
        mu = mode_data.mean(axis=0)
        cov = np.cov(mode_data, rowvar=False)
        n_features = mu.size
        color = MODE_COLORS[index % len(MODE_COLORS)]

        # 上排：协方差热力图（对称色标）
        ax_cov = axes[0, index]
        max_abs = float(np.max(np.abs(cov))) if cov.size else 0.0
        if max_abs > 0.0:
            image = ax_cov.imshow(cov, cmap="coolwarm", vmin=-max_abs, vmax=max_abs, interpolation="nearest")
        else:
            image = ax_cov.imshow(cov, cmap="coolwarm", interpolation="nearest")
        _set_matrix_axis_ticks(ax_cov, n_features, n_features)
        ax_cov.set_title(f"{names[index]} covariance", fontsize=11)
        fig.colorbar(image, ax=ax_cov, fraction=0.046, pad=0.03)

        # 下排：均值柱状图
        ax_mean = axes[1, index]
        ax_mean.bar(np.arange(n_features), mu, color=color, alpha=0.85)
        ax_mean.axhline(0.0, color="0.5", linewidth=0.8)
        ax_mean.set_title(f"{names[index]} mean", fontsize=11)
        ax_mean.set_xlabel("Dimension")
        ax_mean.grid(axis="y", alpha=0.25)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=200)
    plt.close(fig)
