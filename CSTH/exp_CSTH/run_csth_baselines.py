"""CSTH 多模态过程监测实验：JMSDL + 四个基线方法对比。

论文 CSTH 实验参数 (Table I/III):
- 3 个模态, 每模态 1000 训练 + 250 正常测试 + 250 故障测试
- 故障: Mode1 Level+1, Mode2 Flow*1.07, Mode3 Temp*1.1

重要说明:
  CSTH 只有 3 个过程变量 (Flow, Level, Temperature)。论文给出的 K=80, T=4 是
  针对 20 维数值仿真数据的参数。在 3 维数据上，80 原子的字典极度过完备，任何 3 维
  向量都能被精确重构（误差为机器精度零），监测完全失效。

  论文 CSTH 实验 (Section III-B) 原文参数:
    K=80, sparsity=4, lambda1_D2=0.05, lambda1_D3=0.28
  但由于 dim=3 << K=80 且 sparsity=4 > dim=3，OMP 精确求解，IRE≡0。

  合理适配方案: 降低原子数使字典不至于过度完备，同时 sparsity < dim。
  这里采用 K=10, sparsity=2（保证稀疏约束在 3 维空间有意义），lambda 按论文比例缩放。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from jmsdl.model.jmsdl import JMSDL
from jmsdl.monitoring.metrics import compute_fdr, compute_far
from jmsdl.utils.visualizer import (
    plot_dictionary_heatmap,
    plot_monitoring_scores,
)
from baselines.common import DictionaryMonitorBase
from baselines.DL.dl_monitor import DLMonitor
from baselines.LCDL.lcdl_monitor import LCDLMonitor
from baselines.ODL.odl_monitor import ODLMonitor
from baselines.mPCA.mpca_monitor import MPCAMonitor

# ============ 配置 ============
DATA_DIR = ROOT / "CSTH" / "data"
OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_TRAIN_PER_MODE = 1000
N_TEST_PER_MODE = 250
N_MODES = 3
ALPHA = 0.99  # KDE 置信度

# 适配 3 维 CSTH 数据的超参数
# 关键约束: sparsity 必须 < n_features(3), 否则 OMP 精确重构, IRE≡0
# T=1 时每个样本只能用 1 个原子近似, 方向偏差(故障)产生显著残差
N_ATOMS = 80        # 3 维数据用 8 原子
SPARSITY = 1       # 稀疏度=1, 保证有非零重构误差
LAMBDA_VALUES = (0.05, 0.28)       # 保持论文 lambda 值
UPDATE_SPARSITY_VALUES = (1, 1)    # D2, D3 稀疏度

SEED = 0


def load_csth_data():
    """加载 CSTH 数据集，返回训练模态列表和测试数据。"""
    train_df = pd.read_csv(DATA_DIR / "train_data.csv")
    test_df = pd.read_csv(DATA_DIR / "test_data.csv")

    train_all = train_df.values  # (3000, 3)
    test_data = test_df[["Flow", "Level", "Temperature"]].values  # (1500, 3)
    test_labels_raw = test_df["label"].values  # 1=正常, 0=故障

    # 拆分训练模态
    train_modes = [
        train_all[i * N_TRAIN_PER_MODE:(i + 1) * N_TRAIN_PER_MODE]
        for i in range(N_MODES)
    ]

    # 故障标签: 论文约定 1=故障, 0=正常
    fault_labels = (test_labels_raw == 0).astype(int)

    # 测试集模态分界 (6段: 3正常 + 3故障, 每段250)
    test_boundaries = [i * N_TEST_PER_MODE for i in range(N_MODES * 2 + 1)]

    return train_modes, test_data, fault_labels, test_boundaries


def run_jmsdl(train_modes, test_data, fault_labels, test_boundaries):
    """运行 JMSDL 方法。"""
    print("=" * 50)
    print("[JMSDL] Training...")

    model = JMSDL(
        n_atoms=N_ATOMS,
        sparsity=SPARSITY,
        lambda_values=LAMBDA_VALUES,
        update_sparsity_values=UPDATE_SPARSITY_VALUES,
        initial_max_iter=30,
        update_max_iter=30,
        tol=1e-5,
        standardize=True,
        random_state=SEED,
    )
    model.fit(train_modes, alpha=ALPHA, show_progress=True)

    scores = model.score_samples(test_data)
    predictions = model.predict(test_data)
    fdr = compute_fdr(fault_labels, predictions)
    far = compute_far(fault_labels, predictions)

    # 输出
    plot_dictionary_heatmap(
        model.dictionary_, OUT_DIR / "jmsdl_dictionary.png", title="JMSDL Final Dictionary D3"
    )
    plot_monitoring_scores(
        scores, float(model.threshold_), fault_labels, OUT_DIR / "jmsdl_monitoring.png",
        mode_boundaries=test_boundaries, fdr=fdr, far=far,
        statistic_name="IRE", title="JMSDL Process Monitoring (CSTH)",
    )

    print(f"[JMSDL] FDR={fdr:.4f}  FAR={far:.4f}  threshold={model.threshold_:.6f}")
    print(f"[JMSDL] ds_history: {model.ds_history_}")
    return {"method": "JMSDL", "FDR": fdr, "FAR": far}


def run_dl(train_modes, test_data, fault_labels, test_boundaries):
    """运行 DL 基线 (全局单字典)。"""
    print("=" * 50)
    print("[DL] Training...")

    monitor = DLMonitor(n_atoms=N_ATOMS, sparsity=SPARSITY, alpha=ALPHA)
    monitor.standardize = True
    monitor.random_state = SEED
    monitor.fit(train_modes, show_progress=True, progress_desc="epoch[DL]")

    scores = monitor.score_samples(test_data)
    predictions = monitor.predict(test_data)
    fdr = compute_fdr(fault_labels, predictions)
    far = compute_far(fault_labels, predictions)

    plot_dictionary_heatmap(
        monitor.dictionary_, OUT_DIR / "dl_dictionary.png", title="DL Dictionary"
    )
    plot_monitoring_scores(
        scores, float(monitor.threshold_), fault_labels, OUT_DIR / "dl_monitoring.png",
        mode_boundaries=test_boundaries, fdr=fdr, far=far,
        statistic_name="DRE", title="DL Process Monitoring (CSTH)",
    )

    print(f"[DL] FDR={fdr:.4f}  FAR={far:.4f}  threshold={monitor.threshold_:.6f}")
    return {"method": "DL", "FDR": fdr, "FAR": far}


def run_lcdl(train_modes, test_data, fault_labels, test_boundaries):
    """运行 LCDL 基线 (全局标签一致性字典)。"""
    print("=" * 50)
    print("[LCDL] Training...")

    monitor = LCDLMonitor(n_atoms=N_ATOMS, sparsity=SPARSITY, alpha=ALPHA)
    monitor.standardize = True
    monitor.random_state = SEED
    monitor.fit(train_modes, show_progress=True, progress_desc="epoch[LCDL]")

    scores = monitor.score_samples(test_data)
    predictions = monitor.predict(test_data)
    fdr = compute_fdr(fault_labels, predictions)
    far = compute_far(fault_labels, predictions)

    plot_dictionary_heatmap(
        monitor.dictionary_, OUT_DIR / "lcdl_dictionary.png", title="LCDL Dictionary"
    )
    plot_monitoring_scores(
        scores, float(monitor.threshold_), fault_labels, OUT_DIR / "lcdl_monitoring.png",
        mode_boundaries=test_boundaries, fdr=fdr, far=far,
        statistic_name="DRR", title="LCDL Process Monitoring (CSTH)",
    )

    print(f"[LCDL] FDR={fdr:.4f}  FAR={far:.4f}  threshold={monitor.threshold_:.6f}")
    return {"method": "LCDL", "FDR": fdr, "FAR": far}


def run_odl(train_modes, test_data, fault_labels, test_boundaries):
    """运行 ODL 基线 (在线字典学习, 无保留项)。"""
    print("=" * 50)
    print("[ODL] Training...")

    monitor = ODLMonitor(n_atoms=N_ATOMS, sparsity=SPARSITY, alpha=ALPHA)
    monitor.standardize = True
    monitor.random_state = SEED
    monitor.fit(train_modes, show_progress=True, progress_desc="epoch[ODL]")

    scores = monitor.score_samples(test_data)
    predictions = monitor.predict(test_data)
    fdr = compute_fdr(fault_labels, predictions)
    far = compute_far(fault_labels, predictions)

    plot_dictionary_heatmap(
        monitor.dictionary_, OUT_DIR / "odl_dictionary.png", title="ODL Dictionary"
    )
    plot_monitoring_scores(
        scores, float(monitor.threshold_), fault_labels, OUT_DIR / "odl_monitoring.png",
        mode_boundaries=test_boundaries, fdr=fdr, far=far,
        statistic_name="OLRE", title="ODL Process Monitoring (CSTH)",
    )

    print(f"[ODL] FDR={fdr:.4f}  FAR={far:.4f}  threshold={monitor.threshold_:.6f}")
    return {"method": "ODL", "FDR": fdr, "FAR": far}


def run_mpca(train_modes, test_data, fault_labels, test_boundaries):
    """运行 mPCA 基线 (PCA 混合模型)。"""
    print("=" * 50)
    print("[mPCA] Training...")

    monitor = MPCAMonitor(cpv=0.85, alpha=ALPHA)
    monitor.standardize = True
    monitor.random_state = SEED
    monitor.fit(train_modes, show_progress=True, progress_desc="epoch[mPCA]")

    scores = monitor.score_samples(test_data)
    predictions = monitor.predict(test_data)
    fdr = compute_fdr(fault_labels, predictions)
    far = compute_far(fault_labels, predictions)

    # mPCA 统一控制限为 1.0
    plot_monitoring_scores(
        scores, 1.0, fault_labels, OUT_DIR / "mpca_monitoring.png",
        mode_boundaries=test_boundaries, fdr=fdr, far=far,
        statistic_name="max(nT²,nSPE)", title="mPCA Process Monitoring (CSTH)",
    )

    print(f"[mPCA] FDR={fdr:.4f}  FAR={far:.4f}")
    return {"method": "mPCA", "FDR": fdr, "FAR": far}


def main():
    print("Loading CSTH data...")
    train_modes, test_data, fault_labels, test_boundaries = load_csth_data()
    print(f"  Train: {N_MODES} modes x {N_TRAIN_PER_MODE} samples, {train_modes[0].shape[1]} features")
    print(f"  Test: {len(test_data)} samples (normal={int((fault_labels==0).sum())}, fault={int((fault_labels==1).sum())})")
    print()

    results = []
    results.append(run_jmsdl(train_modes, test_data, fault_labels, test_boundaries))
    results.append(run_mpca(train_modes, test_data, fault_labels, test_boundaries))
    results.append(run_dl(train_modes, test_data, fault_labels, test_boundaries))
    results.append(run_lcdl(train_modes, test_data, fault_labels, test_boundaries))
    results.append(run_odl(train_modes, test_data, fault_labels, test_boundaries))

    # 汇总结果表
    print("\n" + "=" * 50)
    print("CSTH Experiment Results Summary")
    print("=" * 50)
    df = pd.DataFrame(results)
    df = df.set_index("method")
    print(df.to_string())
    df.to_csv(OUT_DIR / "csth_results.csv")
    print(f"\nResults saved to: {OUT_DIR / 'csth_results.csv'}")
    print(f"Figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
