from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from tqdm.auto import tqdm

from baselines.common import sample_by_feature
from jmsdl.monitoring.offline import kde_threshold
from jmsdl.utils.data_loader import apply_standardizer, fit_standardizer


def _logsumexp(log_values: np.ndarray, axis: int) -> np.ndarray:
    peak = np.max(log_values, axis=axis, keepdims=True)
    out = peak + np.log(np.sum(np.exp(log_values - peak), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


@dataclass
class _PCAComponent:
    """PCA 混合模型的单个高斯分量。

    EM 期间用带岭正则的全协方差高斯（单调收敛、规避奇异）；EM 收敛后对协方差做一次
    PCA，得到载荷 P 与特征值 Λ 供 T²/SPE 监测。混合权重 π 与两个控制限随后填入。
    """

    mean: np.ndarray                       # (m,) 分量均值
    covariance: np.ndarray                 # (m, m) 带岭全协方差 Σ
    weight: float                          # 混合权重 π
    precision: np.ndarray = None           # (m, m) Σ⁻¹（EM/匹配密度用）
    logdet: float = 0.0                    # log|2πΣ|
    loadings: np.ndarray = None            # (m, l) 保留主元载荷 P（事后 PCA）
    eigenvalues: np.ndarray = None         # (l,) 对应特征值 Λ
    t2_threshold: float = 1.0
    spe_threshold: float = 1.0

    def __post_init__(self) -> None:
        m = self.covariance.shape[0]
        self.precision = np.linalg.inv(self.covariance)
        sign, logabsdet = np.linalg.slogdet(self.covariance)
        self.logdet = float(m * np.log(2.0 * np.pi) + logabsdet)

    def log_density(self, X: np.ndarray) -> np.ndarray:
        """全协方差高斯对数密度 log g(x|μ,Σ)（EM 责任与在线后验匹配共用）。"""
        diff = X - self.mean
        quad = np.einsum("ij,jk,ik->i", diff, self.precision, diff)
        return -0.5 * (quad + self.logdet)

    def fit_pca(self, cpv: float, reg: float) -> None:
        """对分量协方差做 PCA，按 CPV 保留主元，供监测统计量使用（式10-13）。"""
        eigvals, eigvecs = np.linalg.eigh(self.covariance)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 1.0e-12)
        eigvecs = eigvecs[:, order]
        explained = np.cumsum(eigvals) / float(eigvals.sum())
        n_comp = max(1, min(int(np.searchsorted(explained, cpv, side="left") + 1), eigvals.size))
        self.loadings = eigvecs[:, :n_comp]
        self.eigenvalues = eigvals[:n_comp]

    def t2(self, X: np.ndarray) -> np.ndarray:
        """主空间统计量 T² = tᵀΛ⁻¹t = (x-μ)ᵀPΛ⁻¹Pᵀ(x-μ)（式15）。"""
        scores = (X - self.mean) @ self.loadings
        return np.sum(scores**2 / self.eigenvalues, axis=1)

    def spe(self, X: np.ndarray) -> np.ndarray:
        """残差空间统计量 SPE = ||(I-PPᵀ)(x-μ)||²（式16）。"""
        centered = X - self.mean
        reconstruction = (centered @ self.loadings) @ self.loadings.T
        residual = centered - reconstruction
        return np.sum(residual**2, axis=1)


class MPCAMonitor:
    """PCA 混合模型 (PCA Mixture Model, Xu et al. 2014) 多模态过程监测。

    每个工况模态对应一个高斯分量；用 EM 估计混合参数，并对每个分量协方差做
    PCA 去相关降维（式10-13）以规避高维共线性导致的协方差奇异。在线监测时，
    先用贝叶斯后验把样本匹配到后验概率最大的分量（式23-24），再用该分量的归一化
    T²/SPE 判定（式15-16,25），任一超过统一控制限 1.0 即报警。

    说明：分量个数 K 固定为训练模态数（项目已知模态数，故跳过论文的 BYY 自动选 K）；
    控制限沿用项目统一的 KDE 估计，便于与其他基线公平对比。
    """

    def __init__(
        self,
        cpv: float = 0.85,
        alpha: float = 0.99,
        max_iter: int = 100,
        tol: float = 1.0e-4,
        reg: float = 1.0e-6,
    ) -> None:
        self.cpv = float(cpv)
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.reg = float(reg)
        self.standardize = False
        self.random_state: int | None = 0

        self.components_: list[_PCAComponent] = []
        self.n_features_: int | None = None
        self.mean_: np.ndarray | None = None      # 全局标准化均值 (feature,)
        self.scale_: np.ndarray | None = None      # 全局标准化尺度 (feature,)

    # ---------- 标准化 ----------
    def _standardize(self, samples: np.ndarray) -> np.ndarray:
        """样本(行)→ 标准化后的样本(行)。复用项目按特征(列)标准化的参数。"""
        X = sample_by_feature(samples, n_features=self.n_features_)
        if not self.standardize:
            return X
        Xs = apply_standardizer(X.T, self.mean_, self.scale_).T      # 在特征-样本布局上标准化
        return Xs

    # ---------- 由加权样本构造一个全协方差高斯分量 ----------
    def _make_component(
        self, X: np.ndarray, weights: np.ndarray, weight: float
    ) -> _PCAComponent:
        total = float(weights.sum())
        total = total if total > 1.0e-12 else 1.0e-12
        mean = (weights[:, None] * X).sum(axis=0) / total
        centered = X - mean
        cov = (centered * weights[:, None]).T @ centered / total
        cov = np.atleast_2d(cov) + self.reg * np.eye(cov.shape[0])     # 岭正则防奇异
        return _PCAComponent(mean=mean, covariance=cov, weight=weight)

    def _log_responsibilities(self, X: np.ndarray) -> np.ndarray:
        """返回 (n, K) 的对数后验责任（已归一化）。"""
        log_joint = np.column_stack(
            [np.log(max(c.weight, 1.0e-12)) + c.log_density(X) for c in self.components_]
        )
        log_norm = _logsumexp(log_joint, axis=1)
        return log_joint - log_norm[:, None]

    # ---------- 拟合 ----------
    def fit(
        self,
        train_modes: list[np.ndarray] | tuple[np.ndarray, ...],
        show_progress: bool = False,
        progress_desc: str = "epoch[mPCA]",
        progress_position: int = 0,
        progress_leave: bool = True,
    ) -> "MPCAMonitor":
        modes = [np.asarray(mode, dtype=float) for mode in train_modes]
        self.n_features_ = int(sample_by_feature(modes[0]).shape[1])

        # 全局标准化参数：在所有模态汇总数据上拟合一次。
        pooled_raw = np.vstack([sample_by_feature(mode, n_features=self.n_features_) for mode in modes])
        if self.standardize:
            self.mean_, self.scale_ = fit_standardizer(pooled_raw.T)
        else:
            self.mean_, self.scale_ = None, None

        X = self._standardize(pooled_raw)                    # (N, m) 标准化样本
        n_samples = X.shape[0]
        K = len(modes)

        # 用各模态数据初始化 K 个分量（好初值，避免 EM 局部最优 + 保证分量与真实模态对齐）。
        self.components_ = []
        for mode in modes:
            Xi = self._standardize(sample_by_feature(mode, n_features=self.n_features_))
            ones = np.ones(Xi.shape[0])
            self.components_.append(
                self._make_component(Xi, ones, weight=Xi.shape[0] / n_samples)
            )

        # EM 迭代：E 步算后验责任，M 步更新 π/μ/Σ（全协方差，单调收敛）。
        progress_bar = tqdm(
            total=self.max_iter, desc=progress_desc, position=progress_position,
            leave=progress_leave, dynamic_ncols=True, disable=not show_progress,
        )
        prev_ll = -np.inf
        for iteration in range(self.max_iter):
            log_resp = self._log_responsibilities(X)         # (N, K)
            resp = np.exp(log_resp)

            log_joint = np.column_stack(
                [np.log(max(c.weight, 1.0e-12)) + c.log_density(X) for c in self.components_]
            )
            log_likelihood = float(_logsumexp(log_joint, axis=1).sum())

            new_components: list[_PCAComponent] = []
            for k in range(K):
                rk = resp[:, k]
                weight = float(rk.sum()) / n_samples
                weight = max(weight, 1.0e-6)
                new_components.append(self._make_component(X, rk, weight))
            self.components_ = new_components

            if show_progress:
                progress_bar.set_postfix_str(f"ll={log_likelihood:.1f}", refresh=False)
                progress_bar.update(1)

            if np.abs(log_likelihood - prev_ll) < self.tol * (1.0 + abs(prev_ll)):
                break
            prev_ll = log_likelihood
        progress_bar.close()

        # EM 收敛后，对各分量协方差做一次 PCA（式10-13），得到监测用的载荷与特征值。
        for component in self.components_:
            component.fit_pca(self.cpv, self.reg)

        # 各分量控制限：用硬分配（后验 argmax）到该分量的训练样本，KDE 定 T²/SPE 限。
        assignment = np.argmax(self._log_responsibilities(X), axis=1)
        for k, component in enumerate(self.components_):
            members = X[assignment == k]
            if members.shape[0] < 2:
                members = X                                   # 极端兜底：分量样本过少则用全部
            component.t2_threshold = float(kde_threshold(component.t2(members), alpha=self.alpha))
            component.spe_threshold = float(kde_threshold(component.spe(members), alpha=self.alpha))
        return self

    # ---------- 在线匹配与监测 ----------
    def match_modes(self, samples: np.ndarray) -> np.ndarray:
        """贝叶斯后验匹配：每个样本归到后验概率最大的分量（式23-24）。"""
        if not self.components_:
            raise RuntimeError("MPCAMonitor is not fitted.")
        X = self._standardize(samples)
        return np.argmax(self._log_responsibilities(X), axis=1)

    def _matched_normalized(self, samples: np.ndarray, kind: str) -> np.ndarray:
        X = self._standardize(samples)
        matched = np.argmax(self._log_responsibilities(X), axis=1)
        out = np.empty(X.shape[0], dtype=float)
        for k, component in enumerate(self.components_):
            mask = matched == k
            if not np.any(mask):
                continue
            Xk = X[mask]
            if kind == "t2":
                out[mask] = component.t2(Xk) / max(component.t2_threshold, 1.0e-12)
            else:
                out[mask] = component.spe(Xk) / max(component.spe_threshold, 1.0e-12)
        return out

    def normalized_t2(self, samples: np.ndarray) -> np.ndarray:
        """匹配分量的归一化 T²（主空间统计量 / 该分量控制限），控制限统一为 1.0。"""
        if not self.components_:
            raise RuntimeError("MPCAMonitor is not fitted.")
        return self._matched_normalized(samples, "t2")

    def normalized_spe(self, samples: np.ndarray) -> np.ndarray:
        """匹配分量的归一化 SPE（残差空间统计量 / 该分量控制限），控制限统一为 1.0。"""
        if not self.components_:
            raise RuntimeError("MPCAMonitor is not fitted.")
        return self._matched_normalized(samples, "spe")

    def score_samples(self, samples: np.ndarray) -> np.ndarray:
        """综合监测分数：归一化 T² 与 SPE 取较大值（式25，任一超 1 即故障）。"""
        return np.maximum(self.normalized_t2(samples), self.normalized_spe(samples))

    def predict(self, samples: np.ndarray) -> np.ndarray:
        return self.score_samples(samples) > 1.0


def main() -> None:
    """独立运行 mPCA 基线：GMM+PCA 混合模型 → 监测 → 输出监测图。

    监测分数已按各分量控制限归一化，统一控制限取 1.0。
    """
    from jmsdl.monitoring.metrics import compute_far, compute_fdr
    from jmsdl.utils.data_loader import load_config, load_saved_dataset
    from jmsdl.utils.visualizer import plot_mode_match_confusion, plot_monitoring_scores

    root = Path(__file__).resolve().parents[2]
    out_dir = Path(__file__).resolve().parent

    config = load_config(root / "config.yaml")
    dataset = load_saved_dataset(root / "data", config=config)

    train_modes = [np.asarray(mode, dtype=float) for mode in dataset["train_modes"]]
    test_all = np.asarray(dataset["test_all"], dtype=float)
    fault_labels = np.asarray(dataset["fault_labels"], dtype=int)
    test_mode_labels = np.asarray(dataset["test_mode_labels"], dtype=int)

    n_test_per_mode = int(dataset["n_test_per_mode"])
    n_modes = int(dataset["n_modes"])
    boundaries = [index * n_test_per_mode for index in range(n_modes + 1)]

    # 超参数（cpv/alpha/max_iter/tol/reg）直接用 MPCAMonitor.__init__ 的默认值，
    # 要调就改类定义，不再被 config.yaml 覆盖。
    monitor = MPCAMonitor()
    monitor.standardize = bool(config.get("model", {}).get("standardize", False))
    monitor.random_state = config.get("seed", {}).get("random_state", 0)

    monitor.fit(
        train_modes,
        show_progress=True,
        progress_desc="epoch[mPCA]",
    )

    scores = monitor.score_samples(test_all)
    predictions = monitor.predict(test_all)

    fdr = compute_fdr(fault_labels, predictions)
    far = compute_far(fault_labels, predictions)

    # 主空间 T² 与残差空间 SPE 的归一化统计量，各自统一控制限为 1.0。
    t2_scores = monitor.normalized_t2(test_all)
    spe_scores = monitor.normalized_spe(test_all)

    t2_pred = t2_scores > 1.0
    spe_pred = spe_scores > 1.0

    t2_fdr = compute_fdr(fault_labels, t2_pred)
    t2_far = compute_far(fault_labels, t2_pred)

    spe_fdr = compute_fdr(fault_labels, spe_pred)
    spe_far = compute_far(fault_labels, spe_pred)

    # 模态匹配混淆矩阵：行=真实模态，列=贝叶斯后验匹配分量，对角线为匹配准确率。
    matched_modes = monitor.match_modes(test_all)
    plot_mode_match_confusion(
        test_mode_labels,
        matched_modes,
        out_dir / "mpca_mode_match_confusion.png",
        n_modes=n_modes,
        title="Mode-Matching Confusion Matrix",
    )

    # 主空间监测图 T²。
    plot_monitoring_scores(
        t2_scores,
        1.0,
        fault_labels,
        out_dir / "mpca_monitoring_T2.png",
        mode_boundaries=boundaries,
        fdr=t2_fdr,
        far=t2_far,
        statistic_name="nT2",
        title="mPCA Process Monitoring (Principal Space)",
    )

    # 残差空间监测图 SPE。
    plot_monitoring_scores(
        spe_scores,
        1.0,
        fault_labels,
        out_dir / "mpca_monitoring_SPE.png",
        mode_boundaries=boundaries,
        fdr=spe_fdr,
        far=spe_far,
        statistic_name="nSPE",
        title="mPCA Process Monitoring (Residual Space)",
    )

    print(f"[mPCA] 综合 FDR={fdr:.4f}  FAR={far:.4f}  (T²/SPE 任一超限即报警)")
    print(f"[mPCA] T²(主空间)  FDR={t2_fdr:.4f}  FAR={t2_far:.4f}")
    print(f"[mPCA] SPE(残差空间) FDR={spe_fdr:.4f}  FAR={spe_far:.4f}")
    print(f"[mPCA] 图已保存到: {out_dir}")


if __name__ == "__main__":
    main()
