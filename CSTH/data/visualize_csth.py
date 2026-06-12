"""CSTH 数据可视化脚本，调用 jmsdl.utils.visualizer 已有函数。"""

import sys
from pathlib import Path

# 项目根目录加入路径
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from jmsdl.utils.visualizer import (
    plot_multimode_pca_scatter,
    plot_multimode_timeseries,
    plot_test_data_heatmap,
    plot_test_fault_timeseries,
    plot_test_pca_scatter,
)

DATA_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = DATA_DIR

# 每个 mode 1000 训练 + 250 测试样本
N_TRAIN_PER_MODE = 1000
N_TEST_PER_MODE = 250
N_MODES = 3
MODE_NAMES = ["Mode 1", "Mode 2", "Mode 3"]

# ========== 加载数据 ==========
train = pd.read_csv(DATA_DIR / "train_data.csv").values  # (3000, 3)
test = pd.read_csv(DATA_DIR / "test_data.csv")
test_data = test[["Flow", "Level", "Temperature"]].values  # (1500, 4) -> (1500, 3)
test_labels = test["label"].values  # 1=正常, 0=故障

# 构造模态标签 (训练: 0,0,...,1,1,...,2,2,...)
train_mode_labels = np.repeat(np.arange(N_MODES), N_TRAIN_PER_MODE)

# 测试集结构: 前750=正常(250*3模态), 后750=故障(250*3模态)
# 模态标签: mode1 normal + mode2 normal + mode3 normal + mode1 fault + mode2 fault + mode3 fault
test_mode_labels = np.tile(np.repeat(np.arange(N_MODES), N_TEST_PER_MODE), 2)

# 故障标签转为 0/1 (0=正常, 1=故障) —— visualizer 约定
fault_flags = (test_labels == 0).astype(int)

# 测试集模态分界 (6段，每段250)
test_boundaries = [0, 250, 500, 750, 1000, 1250, 1500]
test_mode_names_full = ["Mode1 N", "Mode2 N", "Mode3 N", "Mode1 F", "Mode2 F", "Mode3 F"]

# ========== 1. 训练数据 PCA 散点图 ==========
print("Plotting: train PCA scatter...")
plot_multimode_pca_scatter(
    data=train,
    mode_labels=train_mode_labels,
    output_path=OUTPUT_DIR / "train_pca_scatter.png",
    mode_names=MODE_NAMES,
    title="CSTH Training Data - PCA Scatter",
)

# ========== 2. 训练数据时序图 ==========
print("Plotting: train timeseries...")
plot_multimode_timeseries(
    data=train,
    mode_boundaries=[0, 1000, 2000, 3000],
    output_path=OUTPUT_DIR / "train_timeseries.png",
    dims=(0, 1, 2),
    mode_names=MODE_NAMES,
    title="CSTH Training Data - Multimode Time Series",
)

# ========== 3. 测试数据热力图 ==========
# 故障变量: Mode1=Level(col1), Mode2=Flow(col0), Mode3=Temp(col2)
# 这里展示最明显的故障维度; 用 Level(idx=1) 作为主展示（Mode1 的 +1 偏置最明显）
print("Plotting: test data heatmap...")
plot_test_data_heatmap(
    data=test_data,
    fault_feature=1,  # Level (0-based)
    fault_labels=fault_flags,
    output_path=OUTPUT_DIR / "test_data_heatmap.png",
    title="CSTH Test Data Heatmap (Fault Highlighted)",
)

# ========== 4. 各模态故障对比时序图 ==========
# 测试集结构: 前750正常(3×250) + 后750故障(3×250)
# 为了直观对比每个 mode 的故障变量，单独展示每个 mode 的正常 vs 故障
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Mode 1 故障: Level + 1
# Mode 2 故障: Flow * 1.07
# Mode 3 故障: Temperature * 1.1
fault_info = [
    {"mode": 1, "var_idx": 1, "var_name": "Level",       "fault_desc": "Level + 1"},
    {"mode": 2, "var_idx": 0, "var_name": "Flow",        "fault_desc": "Flow × 1.07"},
    {"mode": 3, "var_idx": 2, "var_name": "Temperature", "fault_desc": "Temp × 1.1"},
]

print("Plotting: per-mode fault comparison...")
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
for ax, info in zip(axes, fault_info):
    mode_idx = info["mode"] - 1  # 0-based
    var_idx = info["var_idx"]
    # 正常段: rows [mode_idx*250 : (mode_idx+1)*250]
    normal_seg = test_data[mode_idx * 250 : (mode_idx + 1) * 250, var_idx]
    # 故障段: rows [750 + mode_idx*250 : 750 + (mode_idx+1)*250]
    fault_seg = test_data[750 + mode_idx * 250 : 750 + (mode_idx + 1) * 250, var_idx]

    x = np.arange(250)
    ax.plot(x, normal_seg, linewidth=0.9, color="tab:blue", label="Normal")
    ax.plot(x, fault_seg, linewidth=0.9, color="tab:red", label="Fault")
    ax.set_title(f"Mode {info['mode']}: {info['fault_desc']}", fontsize=11)
    ax.set_ylabel(info["var_name"])
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

axes[-1].set_xlabel("Sample")
fig.suptitle("CSTH Fault vs Normal Comparison (per mode)", fontsize=13, y=0.98)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "test_fault_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ========== 5. 完整测试集 3 变量时序总览 ==========
print("Plotting: test all-variable timeseries...")
plot_multimode_timeseries(
    data=test_data,
    mode_boundaries=test_boundaries,
    output_path=OUTPUT_DIR / "test_timeseries_all.png",
    dims=(0, 1, 2),
    mode_names=test_mode_names_full,
    title="CSTH Test Data - All Variables",
)

# ========== 6. 测试数据 PCA 散点图 ==========
print("Plotting: test PCA scatter...")
plot_test_pca_scatter(
    test_faulty=test_data,
    mode_labels=test_mode_labels,
    fault_labels=fault_flags,
    output_path=OUTPUT_DIR / "test_pca_scatter.png",
    mode_names=MODE_NAMES,
    title="CSTH Test Data - PCA Scatter (Normal vs Fault)",
)

print(f"\nDone! All plots saved to: {OUTPUT_DIR}")
