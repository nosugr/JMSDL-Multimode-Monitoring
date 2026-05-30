from __future__ import annotations

import numpy as np

from jmsdl.monitoring.online import score_samples

try:
    from scipy.stats import gaussian_kde
except Exception:  # pragma: no cover
    gaussian_kde = None


def compute_reconstruction_errors(
    Y: np.ndarray,
    dictionary: np.ndarray,
    sparsity: int,
    tol: float = 1.0e-6,
) -> np.ndarray:
    """训练样本的逐样本重构误差 (IRE)。与 online.score_samples 等价，

    此处仅作离线语义封装 (建控制限用)，统一复用同一份 OMP 编码 + 残差实现避免重复。
    """
    return score_samples(Y, dictionary, int(sparsity), tol=tol)


def kde_threshold(errors: np.ndarray, alpha: float = 0.99, grid_size: int = 4096) -> float:
    """用 KDE 估计重构误差分布，取置信度 alpha 对应的分位点作为控制限。

    论文式 (18) 写作 ∫_{-∞}^{Rtr} f(R)dR = α/2；按上下文 (α 为置信水平，如 0.99) 该处
    α/2 应为笔误——取 0.495 分位作控制限会让绝大多数正常样本被判故障。这里按惯例直接用
    α 分位 (单侧上控制限)，更符合"控制限覆盖 α 比例正常数据"的本意。
    """
    values = np.asarray(errors, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("Errors array must not be empty.")
    confidence = float(alpha)
    if not 0.0 < confidence < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    if values.size == 1 or np.allclose(values, values[0]):
        return float(values.max())
    if gaussian_kde is None:
        return float(np.quantile(values, confidence))

    std = float(values.std(ddof=0))
    if std <= 1.0e-12:
        return float(values.max())

    try:
        kde = gaussian_kde(values)
        lower = max(0.0, float(values.min() - 0.1 * std))
        upper = float(values.max() + 3.0 * std)
        grid = np.linspace(lower, upper, int(grid_size))
        density = np.maximum(kde(grid), 0.0)
    except Exception:
        return float(np.quantile(values, confidence))

    # 梯形累积积分得到 CDF（含网格步长），再归一化消除积分区间截断带来的偏差。
    cdf = np.concatenate([[0.0], np.cumsum((density[1:] + density[:-1]) * 0.5 * np.diff(grid))])
    if cdf[-1] <= 0.0:
        return float(np.quantile(values, confidence))
    cdf /= cdf[-1]
    index = min(int(np.searchsorted(cdf, confidence, side="left")), grid.size - 1)
    return float(grid[index])
