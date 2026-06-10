# JMSDL-Multimode-Monitoring

本项目用于复现论文 **Adaptive Multimode Process Monitoring Based on Mode-Matching and Similarity-Preserving Dictionary Learning** 中的数值仿真实验部分。项目重点复现 JMSDL 在多模态过程监测中的两个核心能力：

- **模态匹配**：模型能够只使用新模态数据，自适应学习新出现的工作模态。
- **相似性保持**：模型在学习新模态时尽量保持旧字典能力，缓解灾难性遗忘。

当前项目只覆盖论文中的数值仿真实验，不复现 CSTH 仿真实验和真实焙烧过程实验。

## 当前进度

已实现内容：

- 论文式 (26) 的多模态数值数据生成。
- 训练集、测试集、故障注入与标签保存。
- 多模态 PCA 散点图、训练时序图、测试热力图、故障时序图等可视化。
- OMP 稀疏编码。
- K-SVD 初始字典学习。
- JMSDL 序贯字典更新，包括模态匹配项和相似性保持项。
- KDE 全局控制限、在线 IRE 评分、故障判定。
- 评价指标：`ds`、`MRE`、`FDR`、`FAR`。
- 基线方法：`mPCA`、`DL`、`LCDL` 近似实现、`ODL` 近似实现。
- 实验入口：JMSDL 单独运行、数值仿真多方法对比、`lambda1` 敏感性分析、DL 与 JMSDL 连续学习对比。
- 单元测试，覆盖数据生成、稀疏编码、字典学习、监测指标、基线方法、实验入口模块与可视化函数。

## 项目结构

```text
JMSDL-Multimode-Monitoring/
├── config.yaml                  # 实验参数配置
├── README.md                    # 项目说明
├── requirements.txt             # Python 依赖
├── paper_text.txt               # 论文文本辅助材料
│
├── documents/                   # 论文与项目计划书
│   └── 项目计划书.md
│
├── jmsdl/                       # JMSDL 核心包
│   ├── model/                   # OMP、K-SVD、JMSDL 更新、模型类
│   ├── monitoring/              # 控制限、在线评分、评价指标
│   └── utils/                   # 数据生成与可视化
│
├── baselines/                   # 对比方法
│   ├── mPCA/                    # 多模型 PCA
│   ├── DL/                      # 传统字典学习
│   ├── LCDL/                    # 全局字典学习近似实现
│   └── ODL/                     # 无保留项的序贯字典学习近似实现
│
├── experiments/                 # 实验脚本
│   ├── generate_data.py         # 生成数据和基础可视化
│   ├── exp_numerical.py         # 多方法数值仿真对比
│   └── sensitivity_analysis.py  # lambda1 敏感性分析
│
├── main_model/                  # JMSDL 主模型相关脚本与产物
│   ├── run_jmsdl/               # JMSDL 单独训练与监测（脚本与模型、分数、字典/监测/MRE 图同目录）
│   │   └── run_jmsdl.py
│   └── DL vs JMSDL/             # DL 与 JMSDL 连续学习对比（灾难性遗忘验证）
│       └── continual_learning.py  # 各阶段字典对全工况测试集的重构误差对比图
│
├── data/                        # 生成的数据
│   ├── train/
│   └── test/
│
├── outputs/                     # 实验输出（各实验脚本写入独立子目录）
│   ├── exp_numerical/           # 多方法对比 FDR/FAR 表
│   └── sensitivity_analysis/    # lambda1 敏感性分析表与图
│
└── tests/                       # 单元测试
```

## 环境安装

建议使用虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

如果本机已有合适的 Python 环境，也可以直接安装依赖：

```powershell
pip install -r requirements.txt
```

主要依赖包括：

- `numpy`
- `scipy`
- `matplotlib`
- `pandas`
- `pyyaml`
- `tqdm`
- `pytest`

## 配置说明

项目由 `config.yaml` 驱动。当前主要配置分为五组：

```yaml
seed:
  random_state: 0
```

`seed.random_state` 控制状态变量和噪声等数据生成与模型初始化的随机性。各模态观测矩阵 `A_i` 使用独立的固定种子（`generate_multimode_dataset` 的 `observation_matrix_seed` 默认参数，值为 40），以保证模态结构在不同 `random_state` 下稳定。

```yaml
numerical_simulation:
  n_features: 20
  state_dim: 2
  n_modes: 4
  n_train_per_mode: 1000
  n_test_per_mode: 250
  n_fault_per_mode: 125
  fault_feature: 9
  fault_bias: 3.3
  timeseries_dims: [0, 1]
  kde_confidence: 0.99
```

`numerical_simulation` 控制数值仿真数据。数据生成公式为：

```text
x = A_i · s + e
```

其中 `A_i` 是第 `i` 个模态的观测矩阵，`s` 是二维状态向量，`e` 是高斯噪声。`fault_feature` 是注入偏置故障的变量索引（0 基），`fault_bias` 是叠加的偏置量。论文式 (26) 默认在 x2 上叠加 +4；本项目在调试故障可分性时调整为在第 10 个变量（`fault_feature: 9`）上叠加 `+3.3`，可按需改回论文设定。`kde_confidence` 控制 KDE 全局控制限的置信度。状态向量分布（`s1~N(2,1)`、`s2~N(3,1)`）与噪声标准差按论文式 (26) 固定，不在 config 中暴露。

```yaml
model:
  n_atoms: 80
  sparsity: 3
  update_sparsity_values: [3, 3, 5]
  lambda_values: [3.0, 2.5, 2.6]
  initial_max_iter: 30
  update_max_iter: 30
  tol: 1.0e-5
```

`model` 控制 JMSDL 字典学习。第 1 个模态用 K-SVD 学习初始字典（稀疏度 `sparsity`），后续模态按 `lambda_values` 依次做 JMSDL 更新。

- `update_sparsity_values`：每次 JMSDL 更新所用的稀疏度。论文中 D2、D3 稀疏度为 3，D4 为 5，故默认 `[3, 3, 5]`。长度不足时用最后一个值补齐。
- 数据默认按特征做 z-score 标准化。论文数据 `x = A·s + e` 的状态向量非零均值、各特征量级不一，若不标准化，重构项 `‖Xn − DnW‖²` 量级约为保留项 `λ1·tr(I − Doᵀ Dn)` 的 10 倍以上，导致论文给定的 `λ1`（2~3）几乎不起作用、出现灾难性遗忘。标准化使两项量级可比，是该方法生效的关键前处理。基线方法（DL/LCDL/ODL）使用各自训练数据拟合标准化参数，保证对比公平。

对比方法（mPCA/DL/LCDL/ODL）的参数（如 PCA 累计方差贡献率、迭代次数）采用各自实现的默认值，不在 `config.yaml` 中暴露。

```yaml
sensitivity_analysis:
  lambda1_values: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
  n_runs: 5
```

`sensitivity_analysis` 控制 `lambda1` 敏感性分析。

## 数据生成

运行：

```powershell
python experiments/generate_data.py
```

输出：

```text
data/train/train_mode1.csv
data/train/train_mode2.csv
data/train/train_mode3.csv
data/train/train_mode4.csv
data/train/train_all.csv
data/train/train_mode_labels.csv

data/test/test_normal.csv
data/test/test_faulty.csv
data/test/test_all.csv
data/test/test_labels.csv
```

同时会生成基础可视化图：

```text
data/train/pca_multimode_scatter.png
data/train/multimode_timeseries.png
data/test/test_data_heatmap.png
data/test/test_fault_timeseries.png
data/test/test_pca_scatter.png
```

注意：CSV 文件采用“样本 × 特征”格式保存；JMSDL 核心算法内部采用论文中的“特征 × 样本”矩阵约定，实验脚本会自动转置。

## 运行 JMSDL

运行：

```powershell
python main_model/run_jmsdl/run_jmsdl.py
```

该脚本会：

- 生成或读取配置对应的数据对象。
- 按模态顺序训练 JMSDL。
- 用训练集重构误差计算 KDE 控制限。
- 对含故障测试集计算 IRE。
- 输出 FDR/FAR 和每个样本的监测分数。
- 绘制各阶段字典热力图、监测结果图、各字典对各模态的 MRE 折线图（复现论文 Fig.8）。

输出文件（均位于脚本同目录 `main_model/run_jmsdl/`）：

```text
main_model/run_jmsdl/jmsdl_model.npz              训练好的各阶段字典 D1~D4 + 控制限
main_model/run_jmsdl/jmsdl_scores.csv             逐样本监测分数、真实标签与预测标签
main_model/run_jmsdl/D1_dictionary_heatmap.png    各阶段字典热力图（D1~D4）
main_model/run_jmsdl/D2_dictionary_heatmap.png
main_model/run_jmsdl/D3_dictionary_heatmap.png
main_model/run_jmsdl/D4_dictionary_heatmap.png
main_model/run_jmsdl/jmsdl_monitoring.png         监测结果图（分数 + 控制限 + 故障散点 + FDR/FAR）
main_model/run_jmsdl/jmsdl_mre_matrix.png         各字典对各模态的 MRE 折线图
```

## 多方法对比实验

运行：

```powershell
python experiments/exp_numerical.py
```

该脚本会比较：

- `JMSDL`
- `mPCA`
- `DL`
- `LCDL`
- `ODL`

输出：

```text
outputs/exp_numerical/fig9_fdr_far.csv
```

其中 `FDR` 是故障检测率，越高越好；`FAR` 是误报率，越低越好。结果以 CSV 表格形式输出，不再生成柱状对比图。

## lambda1 敏感性分析

运行：

```powershell
python experiments/sensitivity_analysis.py
```

该脚本会改变 JMSDL 保留项参数 `lambda1`，多次运行并计算新旧字典相似度 `ds`。

输出：

```text
outputs/sensitivity_analysis/lambda1_ds_raw.csv
outputs/sensitivity_analysis/lambda1_ds_summary.csv
outputs/sensitivity_analysis/lambda1_ds_curve.png
outputs/sensitivity_analysis/dictionary_diff_heatmaps.png
```

## DL 与 JMSDL 连续学习对比

运行：

```powershell
python "main_model/DL vs JMSDL/continual_learning.py"
```

该脚本复现论文中"灾难性遗忘"的对比验证：

- 传统 `DL`：`D1` 由 K-SVD 学得，之后每个阶段在上一字典基础上**只用当前模态数据**继续微调（无相似性保持项）。
- `JMSDL`：`D1` 同样由 K-SVD 学得，之后按序贯方式做 JMSDL 更新（带相似性保持项）。
- 两种方法的每个阶段字典都对**完整四工况测试集**做 OMP 重构，计算逐样本重构误差。

两者共用 JMSDL 拟合得到的全局标准化参数，保证在同一尺度下比较。预期现象：传统 DL 在学完后续模态后会遗忘早期模态（早期样本段重构误差升高），而 JMSDL 各阶段对全部模态都保持较低重构误差。

输出：

```text
main_model/DL vs JMSDL/continual_learning.png   2×4 面板，上排 DL（DRE），下排 JMSDL（IRE）
```

## 测试

运行：

```powershell
python -m pytest
```

当前测试覆盖：

- 数据生成参数校验（`test_data_loader.py`）。
- OMP 稀疏编码（`test_sparse_coding.py`）。
- K-SVD 字典学习、JMSDL 序贯训练和评分（`test_ksvd_and_jmsdl.py`）。
- KDE 控制限与监测指标（`test_monitoring.py`）。
- mPCA、DL、LCDL、ODL 基线方法（`test_baselines.py`）。
- 实验入口模块端到端冒烟测试（`test_experiment_modules.py`，小规模 config 跑通 run_jmsdl / exp_numerical / sensitivity_analysis）。
- 可视化函数产图（`test_visualizer.py`，字典热力图、监测图、稀疏编码热力图）。

## 实现说明

### JMSDL 更新

JMSDL 的核心目标是同时满足：

```text
min ||X_n - D_n W||_F^2 + lambda1 * tr(I - D_o^T D_n)
```

其中：

- `X_n` 是新模态数据。
- `D_o` 是旧字典。
- `D_n` 是更新后的新字典。
- `W` 是新模态数据在新字典下的稀疏编码。

实现流程：

1. 初始 `D_n = D_o`。
2. 固定 `D_n`，用 OMP 求 `W`。
3. 固定 `W`，按论文闭式解更新 `D_n`。
4. 对字典原子归一化。
5. 重复迭代直到达到最大迭代次数或收敛。

### 关键实现决策

以下几点是论文未明确给出、或本项目在复现中做出的具体选择，集中说明便于查阅：

- **数据标准化（关键）**：论文数据 `x = A·s + e` 的状态向量 `s1~N(2,1)`、`s2~N(3,1)` 非零均值，使重构项量级远大于保留项。实测在不标准化时，重构项约为 `λ1` 保留项的 10 倍以上，论文给定的 `λ1`（2~3）几乎不起作用，序贯更新会退化成"在新模态上重学字典"，出现严重的灾难性遗忘（最终字典只能表示最后一个模态）。本项目默认对数据按特征做 z-score 标准化，使两项量级可比后，论文给定的 `λ1` 才能真正平衡"学新模态"与"保留旧模态"，最终字典对四个模态的 MRE 才趋于均衡。这是该方法能复现出抗遗忘效果的前提。
- **按步稀疏度**：论文中 D2、D3 稀疏度为 3，D4 为 5。本项目用 `model.update_sparsity_values: [3, 3, 5]` 支持每步不同稀疏度，而非全程固定。
- **稀疏正则项（L1 → L0）**：论文式 (4) 含稀疏项 `λ2‖W‖₁`，本项目按论文 II-A2 节用 OMP（L0 硬稀疏约束）直接求 `W`，等价于以硬稀疏度替代 L1 软约束。因此 `λ2` 不作为显式超参数出现，收敛监控的目标值只含重构项与保留项。
- **KDE 控制限的置信度**：论文式 (18) 写作 `∫f(R)dR = α/2`，按上下文（`α` 为置信水平，如 0.99）该处 `α/2` 应为笔误——取 0.495 分位会让绝大多数正常样本被判故障。本项目按惯例直接取 `α` 分位作为单侧上控制限。

### 基线方法

`DL` 直接使用 K-SVD 字典学习。

`mPCA` 使用每个模态一个 PCA 模型的多模型监测方式。样本只要被任一模态 PCA 模型判为正常，就视为正常。

`LCDL` 和 `ODL` 在 JMSDL 原论文中没有给出完整可复现细节，因此本项目采用近似实现：

- `LCDL`：用所有正常模态训练一个全局字典。
- `ODL`：按模态顺序持续更新字典，但不使用 JMSDL 的相似性保持项。

这两个基线用于趋势对比，不保证逐数值复刻原论文。所有字典类方法（DL/LCDL/ODL）与 JMSDL 使用一致的标准化策略（各自在其训练数据上拟合标准化参数），以保证对比公平。

## 已知限制

- **故障检测率（FDR）偏低**：当前数值仿真下，JMSDL 的 FDR 明显低于预期（约 0.02 量级）。经定位，这与字典学习的抗遗忘能力无关（MRE 矩阵已验证遗忘被消除），而是故障可分性问题：论文式 (26) 的默认故障（20 维中仅 1 维叠加 +4 偏置）对重构误差 IRE 的抬升很弱（故障样本 IRE 中位数仅略高于正常样本），且单一全局 KDE 控制限会被表示最差模态的长尾抬高，从而淹没故障信号。此问题随随机种子与各模态 MRE 平衡度变化，属于阈值鲁棒性 / 故障幅度与噪声水平匹配的范畴，尚未解决。可能的改进方向：采用逐模态或更鲁棒的控制限，或重新核对故障幅度与噪声水平。
- **抗遗忘能力（核心结论）已复现**：标准化 + 按步稀疏度修复后，最终字典对各模态的 MRE 趋于均衡（不再只表示最后一个模态），论文关于"相似性保持消除灾难性遗忘"的核心结论可复现。

## 注意事项

- 原论文未提供随机种子，因此本项目不追求逐数值复刻，而是关注趋势与结论一致性。
- 当前数值数据严格按论文式 (26) 独立采样，同一模态内不额外加入时间趋势。
- 修改 `timeseries_dims` 只影响时序图可视化，不影响训练数据和算法输入。
- 若完整运行默认配置，耗时会明显高于测试中的小规模配置。
