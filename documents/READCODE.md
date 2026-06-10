# JMSDL 多模态过程监测 —— 代码详解（READCODE）

本文面向想读懂整个项目代码的人，逐个文件说明：**这段代码是干什么的、内部实现流程是什么、和其他代码怎么配合调用**。

论文：*Adaptive Multimode Process Monitoring Based on Mode-Matching and Similarity-Preserving Dictionary Learning*（JMSDL）。本项目只复现其中的**数值仿真实验**部分。

---

## 0. 一句话总览

项目做的事情可以拆成四步：

1. **造数据**：按论文式 (26) `x = A_i·s + e` 生成 4 个工作模态的多模态数据，并在测试集注入故障。
2. **学字典**：第 1 个模态用 K-SVD 学一个初始字典；后续模态用 JMSDL 闭式解**只拿新模态数据**增量更新字典，同时通过相似性保持项尽量不忘旧模态（缓解灾难性遗忘）。
3. **做监测**：用训练集的重构误差经 KDE 估一个全局控制限；在线对每个样本算重构误差（IRE），超过控制限即判为故障。
4. **比对评估**：和 mPCA / DL / LCDL / ODL 等基线对比 FDR、FAR；并做 lambda1 敏感性分析。

核心数据流：

```
config.yaml
   │  (load_config)
   ▼
data_loader.load_saved_dataset ──► 数据集 dict（train_modes / test_all / fault_labels ...）
   │
   ├──► visualizer.*           生成 PCA/时序/热力图
   │
   ▼
JMSDL.fit(train_modes)
   │   ├─ 模态1: fit_ksvd ────────────► 初始字典 D1
   │   └─ 模态2..N: update_dictionary_jmsdl ─► D2, D3, ... （增量更新）
   │   └─ set_threshold ─► compute_reconstruction_errors + kde_threshold ─► 控制限
   ▼
JMSDL.predict(test_faulty)
   │   └─ score_samples (omp_encode → 重构残差平方和 IRE) > threshold
   ▼
metrics.compute_fdr / compute_far ──► 评价结果
```

所有矩阵约定：**论文核心算法内部用「特征 × 样本」（feature-by-sample）**；而 **CSV 和数据生成用「样本 × 特征」（sample-by-feature）**。两者之间靠 `as_feature_by_sample` / `samples_to_features` 转置衔接，这是读代码时最容易混淆的点，务必记牢。

---

## 1. 目录与职责划分

```
jmsdl/                 核心算法包
├── model/             字典学习算法层（OMP、K-SVD、JMSDL 更新、模型外壳）
├── monitoring/        监测层（控制限、在线评分、评价指标）
└── utils/             数据生成与可视化

baselines/             对比方法（mPCA / DL / LCDL / ODL）
experiments/           实验入口脚本（数据生成、跑 JMSDL、多方法对比、敏感性分析）
DL vs JMSDL/           DL 与 JMSDL 连续学习对比脚本（灾难性遗忘验证）
tests/                 单元测试
config.yaml            统一参数配置
data/ outputs/         数据与结果产物
```

依赖方向（谁调用谁）：

```
experiments  ─┬─►  jmsdl.model.JMSDL
              ├─►  baselines.*
              └─►  jmsdl.utils.data_loader / visualizer

baselines    ──►  jmsdl.model.ksvd / monitoring.*   （复用核心算法）

jmsdl.model.JMSDL ─► ksvd ─► sparse_coding(OMP) ─► initializer
                  └─► dictionary_update ─► sparse_coding(OMP) ─► initializer
                  └─► monitoring.offline / online
```

注意一个细节：`jmsdl.model.jmsdl` 反向 import 了 `jmsdl.monitoring` 里的函数（控制限、评分）。也就是说监测逻辑被「内联」进了模型类，方便 `model.predict()` 直接用。

---

## 2. 核心算法层 `jmsdl/model/`

### 2.1 `initializer.py` —— 基础工具与字典初始化

这是被所有人依赖的最底层模块,提供以下函数:

- **`normalize_columns(matrix)`**：把矩阵每一列归一化为单位 L2 范数（字典原子必须是单位向量）。范数过小的列保持不变，避免除零。几乎每个字典操作后都会调它。
- **`as_feature_by_sample(matrix, n_features=None)`**：**矩阵方向统一器**。把任意 2D 矩阵转成「特征 × 样本」。
  - 给了 `n_features` 时：哪一维等于 `n_features` 就以那一维当特征轴，必要时转置。
  - 没给时：用启发式「行数 ≤ 列数则认为已经是特征×样本」。
  - 作用：让算法层不关心调用方传进来的是行样本还是列样本。
- **`fit_standardizer(matrix)` / `apply_standardizer(matrix, mean, scale)`**：**z-score 标准化（关键修复）**。对「特征×样本」矩阵按特征（行）估计均值和标准差，再做标准化。论文数据 `x = A·s + e` 状态非零均值，不标准化时重构项量级远大于保留项（约 10 倍以上），论文给的 `λ1` 失效、出现灾难性遗忘。标准化使两项可比，是 JMSDL 抗遗忘能力得以复现的前提。函数位于 `utils.data_loader`，被 `JMSDL`、基线基类和 mPCA 共用。
- **`initialize_svd_dictionary(Y, n_atoms)`**：K-SVD 默认初始化。对中心化数据做 SVD，取前几个左奇异向量当原子；不够 `n_atoms` 时，在主子空间里按奇异值加权随机线性组合补足。
- **`initialize_dictionary_from_data(Y, n_atoms)`**：另一种初始化，随机抽样本列当原子并加一点噪声。

调用关系：被 `ksvd.py`、`dictionary_update.py`、`jmsdl.py` 以及 `baselines/common.py` 共同使用。

### 2.2 `sparse_coding.py` —— OMP 稀疏编码（整个项目的计算核心）

提供正交匹配追踪（Orthogonal Matching Pursuit）。给定数据 `Y` 和字典 `D`，求稀疏系数 `W`，使 `Y ≈ D·W` 且每列非零元素不超过 `sparsity` 个。

- **`_omp_single(y, D, sparsity, ...)`**：对单个样本向量做 OMP。流程：
  1. 把字典各原子归一化（内部副本）。
  2. 迭代最多 `sparsity` 次：每次选与当前残差相关性最大的原子加入活跃集；
  3. 在活跃集上用最小二乘（可选 ridge 正则）解系数，更新残差；
  4. 残差足够小或相关性低于阈值就提前停。
  5. 最后把系数除回原子范数，还原到未归一化字典的坐标。
- **`omp_encode(Y, D, sparsity, ...)`**：对外接口。支持 1D 单样本或 2D「特征×样本」矩阵，后者逐列调用 `_omp_single`，拼成系数矩阵 `W`（形状 `n_atoms × n_samples`）。

谁在用它：K-SVD 每次迭代要算编码、JMSDL 更新每次迭代要算编码、在线评分/重构误差都要算编码。**它是被调用最频繁的函数。**

### 2.3 `ksvd.py` —— K-SVD 初始字典学习

用于学习**第一个模态**的初始字典 `D1`。`KSVDResult` 数据类返回 `dictionary / codes / error_history / n_iter`。

- **`fit_ksvd(Y, n_atoms, sparsity, ...)`** 主流程（经典 K-SVD 的「稀疏编码 + 逐原子更新」交替）：
  1. 初始化字典（默认 SVD 初始化，或外部传入 `initial_dictionary`）。
  2. 循环 `max_iter` 次：
     - OMP 求编码 `W`；
     - `_update_dictionary` 逐个原子做 SVD 更新；
     - 算 Frobenius 重构误差，记录历史；相对变化小于 `tol` 即收敛退出。
  3. 最后再编码一次返回。
- **`_update_dictionary(...)`**：K-SVD 的关键。对每个原子，只看用到它的样本，扣掉其他原子的贡献得到残差矩阵，对残差做 SVD，用第一左奇异向量更新该原子、第一右奇异向量×奇异值更新对应系数。
- **`_reinitialize_atom(...)`**：处理「死原子」（没有样本使用它）。用当前重构残差最大的样本替换它，避免原子浪费。

  4. **进度条**：`fit_ksvd` 接收 `show_progress / progress_desc / progress_position / progress_leave` 四个可选参数（默认 `show_progress=False`），用 `tqdm.auto` 在迭代循环里逐轮刷新重构误差。因为 DL/LCDL/ODL 三个基线都复用 `fit_ksvd`，进度条挂在这一层，三个基线就都自动有了——这是「进度条放在底层循环」的根本原因（详见第 11 节）。

谁在用它：`JMSDL.fit` 学初始字典；`baselines/common.py` 的 `DLMonitor`/`ODLMonitor`/`LCDLMonitor` 也用它学字典；敏感性分析脚本也直接调它。

### 2.4 `dictionary_update.py` —— JMSDL 增量更新（论文 Algorithm 1，全项目灵魂）

实现「只用新模态数据更新字典，同时保持与旧字典相似」。目标函数：

```
min over D_n, W   ||X_n - D_n·W||_F^2  +  lambda1 · tr(I - D_o^T·D_n)
```

- 第一项 `||X_n - D_n W||²`：**模态匹配**，新字典要能重构新模态数据。
- 第二项 `lambda1·tr(I - D_oᵀD_n)`：**相似性保持**，让新字典各原子尽量与旧字典对应原子方向一致（点积接近 1），缓解灾难性遗忘。`lambda1` 越大越偏向保留旧字典。

关键函数：

- **`dictionary_similarity(D_old, D_new)` → ds**：两字典逐列归一化后，对应原子点积绝对值的平均。衡量「字典改变了多少」，是敏感性分析和监测指标 `ds` 的基础。
- **`solve_jmsdl_dictionary(X_new, W, D_old, lambda1)`**：固定 `W` 时字典的**闭式解**。
  1. `B = W·Wᵀ`，对 `B` 做特征分解 `B = VΛVᵀ`；
  2. `F = X·Wᵀ + 0.5·lambda1·D_old`；投影到特征向量空间 `P = F·V`；
  3. 逐特征值求解 `Q[:,i] = P[:,i]/λ_i`（λ_i 近 0 时退化为沿用旧字典对应列，保证数值稳定）；
  4. 转回 `D = Q·Vᵀ`，再做符号对齐 + 归一化。
- **`_align_with_old_dictionary(...)`**：消除 SVD/特征分解带来的符号翻转歧义，让新原子方向和旧原子同向（点积非负），保证相似性度量有意义。
- **`jmsdl_objective(...)`**：算上面那个目标函数值,用于监控收敛。论文式 (4) 还含稀疏项 `λ2‖W‖₁`,本项目按论文 II-A2 用 OMP（L0 硬稀疏约束）求 `W`,等价以硬稀疏度替代 L1,因此该目标值只含重构项 + 保留项,不含 `λ2` 项。
- **`update_dictionary_jmsdl(X_new, D_old, sparsity, lambda1, ...)`** 主流程（交替优化）：
  1. 初始 `D_new = D_old`，OMP 编码得 `W`；
  2. 循环：用闭式解更新 `D_new` → 重新 OMP 编码 `W` → 算目标值和相似度；
  3. 目标相对变化小于 `tol` 即收敛。返回 `JMSDLUpdateResult`（字典、编码、目标历史、相似度历史、迭代数）。
  4. 同样接收 `show_progress` 等四个进度条参数，在迭代循环里用 `tqdm` 逐轮刷新目标值与相似度 `ds`。

谁在用它：`JMSDL.fit` 对第 2 个及以后的模态逐个调用；敏感性分析脚本直接调它扫 `lambda1`。

### 2.5 `jmsdl.py` —— JMSDL 模型外壳（对外主类）

把上面所有零件组装成一个 sklearn 风格的模型类。

- **`JMSDLHyperParams`**：超参数数据类（原子数、稀疏度、按步稀疏度序列、lambda 序列、迭代数、tol、随机种子）。
- **`JMSDL` 类**关键属性：`dictionaries_`（每个模态训练后的字典列表）、`codes_`、`update_results_`、`initial_result_`、`threshold_`（控制限）、`mean_`/`scale_`（标准化参数）。
- **`fit(train_modes, alpha=None, show_progress=False, ...)`** 训练主流程：
  0. 透传进度条：`show_progress=True` 时，初始 K-SVD 显示一条 `epoch[K-SVD][D1]` 进度条，之后每个模态更新各显示一条 `epoch[JMSDL][D2]`、`[D3]`…，分别下沉到 `fit_ksvd` 和 `update_dictionary_jmsdl` 的循环里渲染。
  1. 把每个模态统一成「特征×样本」；
  2. 在所有模态拼接数据上 `fit_standardizer` 估标准化参数,并对各模态标准化（**抗遗忘的关键前处理**）；
  3. 模态 0：`fit_ksvd` 学初始字典，存入 `dictionaries_[0]`；
  4. 模态 1..N：依次 `update_dictionary_jmsdl`，每步的 `lambda1` 由 `_lambda_for_update` 取、每步的稀疏度由 `_sparsity_for_update` 从 `update_sparsity_values` 取（论文 D4=5）；逐步累积 `dictionaries_`；
  5. 若给了 `alpha`，把标准化后的所有模态数据拼起来 `set_threshold` 定控制限。
- **`_lambda_for_update` / `_sparsity_for_update`**：按更新步取 `lambda1` 和稀疏度,超出序列长度就用最后一个值。
- **`_standardize(matrix)`**：用训练阶段拟合的 `mean_/scale_` 对输入做一致变换。
- **`transform` / `reconstruct`**：先标准化再 OMP 编码 / 重构。
- **`score_samples(Y)`**：先标准化,再调 `monitoring.online.score_samples`,返回每个样本的重构误差平方和（IRE，监测统计量）。
- **`mre_by_mode(train_modes)`**：各模态在最终字典下的平均重构误差,内部自动套用与训练一致的标准化（复现论文 Fig.8 的 D_c 行）。调用方传原始模态矩阵即可,避免"用原始数据评估标准化空间字典"的错误。
- **`mre_matrix(train_modes)`**：所有字典对所有模态的 MRE 矩阵,形状 `(n_dicts, n_modes)`,第 k 行第 j 列 = 字典 `D_{k+1}` 对模态 j+1 的平均重构误差,复现论文 Fig.8 的完整矩阵；内部同样自动标准化。`run_jmsdl` 用它画 MRE 折线图。
- **`set_threshold(train_data, alpha)`**：算训练误差 + KDE 控制限,存到 `threshold_`；内部 `_already_standardized` 标志避免对已标准化数据二次标准化。
- **`predict(Y)`**：`score_samples > threshold_`，返回布尔故障标记。
- **`dictionary_`（属性）**：最新字典；**`ds_history_`（属性）**：相邻模态字典的相似度序列。
- **`save_npz(path)`**：把各模态字典 `D1,D2,...`、控制限、ds 历史、标准化开关与 `mean/scale` 存成 npz。

谁在用它：`experiments/run_jmsdl.py`、`experiments/exp_numerical.py`。

### 2.6 `model/__init__.py`

导出 `JMSDL` 等公共符号，让外部 `from jmsdl.model import JMSDL` 即可。

---

## 3. 监测层 `jmsdl/monitoring/`

### 3.1 `offline.py` —— 离线：重构误差 + KDE 控制限

- **`compute_reconstruction_errors(Y, D, sparsity)`**：训练数据的逐样本重构误差 `||y - D·w||²`（IRE 的离线版本）。为避免重复,内部直接复用 `online.score_samples`（两者实现等价,只是语义角色不同：建控制限 vs 在线打分）。
- **`kde_threshold(errors, alpha=0.99)`**：用核密度估计（`scipy.stats.gaussian_kde`）拟合误差分布,用**梯形积分**在网格上累积出 CDF（含网格步长,再归一化）,取 `alpha` 分位点作为控制限。带多重退化保护：误差全相等/标准差过小直接取最大值；scipy 不可用或 KDE 失败回退到经验分位数 `np.quantile`。注：论文式 (18) 的 `α/2` 经判断为笔误,本实现按惯例用 `α` 分位作单侧上控制限。

谁在用它：`JMSDL.set_threshold`、`baselines/common.py`、`mPCA` 的 T²/SPE 控制限。

### 3.2 `online.py` —— 在线：评分与判定

- **`encode_samples(Y, D, sparsity)`**：OMP 编码（在线版，薄封装）。
- **`score_samples(Y, D, sparsity)`**：编码后算残差平方和 → 每个样本的 IRE 监测分数。
- **`detect_fault(scores, threshold)`**：`scores > threshold` 返回布尔数组。

`offline` 与 `online` 的重构误差计算其实等价，区别只是语义角色（建控制限 vs 在线打分）。`JMSDL.score_samples` 直接复用 `online.score_samples`。

### 3.3 `metrics.py` —— 评价指标

- **`compute_ds`**：转调 `dictionary_similarity`，新旧字典相似度。
- **`compute_mre(errors)`**：平均重构误差（Mean Reconstruction Error）。
- **`compute_mre_by_mode(data_by_mode, D, sparsity)`**：逐模态算 MRE，衡量「最终字典对每个模态的表示能力」——这是检验「相似性保持/没遗忘旧模态」的关键指标。注：实验脚本现改用 `JMSDL.mre_by_mode`（标准化感知）来算,避免对标准化空间字典误用原始数据；本函数保留作通用工具。
- **`compute_fdr(y_true, y_pred)`**：故障检测率 = 检出故障数 / 实际故障数（越高越好）。
- **`compute_far(y_true, y_pred)`**：误报率 = 误报数 / 实际正常数（越低越好）。
- **`fdr_far(...)`**：一次返回 (FDR, FAR)。

谁在用它：所有实验脚本算最终指标。

---

## 4. 工具层 `jmsdl/utils/`

### 4.1 `data_loader.py` —— 数据生成（实验的起点）

按论文式 (26) `x = A_i·s + e` 造多模态数据，输出全部为「样本 × 特征」。

- **`load_config(path)`**：读 `config.yaml`。
- **`_generate_mode_samples(...)`**：单模态采样。状态 `s1~N(2,·)`、`s2~N(3,·)`，经该模态观测矩阵 `A_i` 线性映射加高斯噪声。**不同模态用不同 `A_i`，所以分布不同 → 构成多模态。**
- **`_inject_fault_bias(samples, fault_feature, fault_bias)`**：在指定变量上叠加偏置故障。函数默认值为第 2 个变量 +4（论文设定），实际取值由 `config.yaml` 的 `fault_feature`/`fault_bias` 决定（当前为第 10 个变量 `+3.3`）。
- **`generate_multimode_dataset(...)`**：核心生成函数。
  - 观测矩阵用独立种子 `observation_matrix_seed` 生成，保证模态结构在不同数据种子下稳定；
  - 为每个模态生成训练段和测试段；
  - 训练：`train_modes`（列表，供序贯训练）、`train_all`（堆叠）、`train_mode_labels`；
  - 测试：`test_normal`（全正常）、`test_faulty`（每个模态测试段末尾注入故障）、`fault_labels`（0/1）；
  - 还返回 `observation_matrices` 等元信息。
- **`generate_from_config(config)`**：从 config 字典取参数调上面的函数。状态分布和噪声参数按论文固定、不在 config 暴露。
- **`load_saved_dataset(data_dir, config)`**：从 `data/train` 和 `data/test` 读取已保存 CSV，训练/测试实验统一使用这个入口。

谁在用它：`experiments/generate_data.py` 使用生成函数落盘；`experiments/_common.dataset_from_files` → 训练/测试实验脚本读取已保存数据。

### 4.2 `visualizer.py` —— 可视化

纯绘图（matplotlib，Agg 后端无界面），不参与算法。主要函数：

- **`plot_multimode_pca_scatter`**：训练数据 PCA 散点图。亮点是 `_select_discriminative_components` 用 Fisher 比（类间/类内方差）自动挑出**对模态区分度最高的两个主成分**，再按模态上色并画置信椭圆，复现论文 Fig.1。
- **`plot_multimode_timeseries`**：训练数据时序折线图，红虚线分隔模态、顶部标模态名。
- **`plot_test_data_heatmap`**：测试数据热力图，红框框出故障变量行、红虚线标故障起点。
- **`plot_test_fault_timeseries`**：故障变量时序图，浅红阴影标故障区间，可叠加正常对照曲线。
- **`plot_test_pca_scatter`**：测试数据 PCA 散点，按模态上色、正常用圆点/故障用叉，椭圆基于正常样本。
- **`plot_dictionary_heatmap(D, output_path, title=...)`**（公用）：字典/载荷矩阵热力图（行=特征，列=原子）。画法照搬 JSSDL —— **viridis 配色 + 对称色标 ±max + 每个单元格描细灰边**，左侧标特征编号、底部不显示原子刻度。被所有基线（DL/LCDL/ODL 字典、mPCA 各模态载荷）与 JMSDL 各阶段字典（D1–D4）共用，统一风格。
- **`plot_monitoring_scores(scores, threshold, fault_labels, output_path, ...)`**（公用）：监测结果图（JSSDL 风格）—— 监测统计量蓝色实线 + 红色虚线控制限 + 故障样本橙色散点（不画区间阴影），可选模态分界灰色点线，并可在坐标轴上方标注 `FDR/FAR`。被所有基线 main()、`run_jmsdl`、`exp_numerical` 共用。
- **`plot_sparse_code_heatmap(codes, output_path, ...)`**：稀疏编码 `W`（原子×样本）热力图，coolwarm 对称着色，供分析编码稀疏结构用。
- **`plot_mode_match_confusion(true_modes, matched_modes, output_path, ...)`**：模态匹配混淆矩阵（行=真实模态、列=匹配模态），可按行归一化并叠加数值标注，对角线即匹配准确率。
- **`plot_label_response(label_response, output_path, label_matrix=None, ...)`**：变换后标签响应 A·W（原子×样本）可视化，上图 A·W、下图绝对残差 |Q - A·W|，Blues 着色，展示 LCDL 标签一致性约束 ||Q - A·W||² 的逼近程度。
- 辅助：`_pca_scores`（可视化用 PCA）、`_add_confidence_ellipse`（协方差置信椭圆）、`_contiguous_runs`（从 0/1 标记找连续故障区间）。

谁在用它：`experiments/generate_data.py`（前 5 个函数）；四个基线 main()、`run_jmsdl`、`exp_numerical`、`DL vs JMSDL/continual_learning.py`（`plot_dictionary_heatmap`/`plot_monitoring_scores` 等公用函数）；`plot_sparse_code_heatmap`/`plot_mode_match_confusion`/`plot_label_response` 主要供分析与基线（如 LCDL 标签响应、mPCA 模态匹配）使用。

---

## 5. 基线方法 `baselines/`

### 5.1 `common.py` —— 基线共享基类

- **`sample_by_feature(matrix, n_features)`**：和 `as_feature_by_sample` 相反，统一成「样本 × 特征」（mPCA 需要）。
- **`DictionaryMonitorBase`**：字典类基线的共享逻辑。封装 `fit_standardizer`（在各基线 `fit` 起始处用其训练数据拟合 z-score 参数,与 JMSDL 一致以保证对比公平）、`_standardize`、`_fit_dictionary`（先标准化再调 `fit_ksvd`,支持传入初始字典做 warm-start）、`_set_threshold`（标准化后算 KDE 控制限）、`score_samples`、`predict`。DL/LCDL/ODL 都继承它,只需各自实现 `fit`,并在开头各调一次 `fit_standardizer`。

### 5.2 `DL/dl_monitor.py` —— 传统字典学习

`DLMonitor`：**只用第一个模态**数据学一个 K-SVD 字典做监测。代表「不适应新模态」的基线。

### 5.3 `LCDL/lcdl_monitor.py` —— 全局字典（近似实现）

`LCDLMonitor`：把**所有模态**数据拼在一起学一个全局字典。代表「一次性看全部模态」的上界式参照。论文未给完整细节，此为近似实现。

### 5.4 `ODL/odl_monitor.py` —— 无保留项的序贯字典（近似实现）

`ODLMonitor`：和 JMSDL 一样按模态顺序增量更新，但**用 K-SVD warm-start 续训、没有相似性保持项**。它是 JMSDL 的直接对照——用来凸显「相似性保持项」的作用（预期会出现灾难性遗忘）。

### 5.5 `mPCA/mpca_monitor.py` —— PCA 混合模型 (PCA Mixture Model, Xu et al. 2014)

- **`_PCAComponent`**：混合模型的单个高斯分量。EM 期间用带岭正则的全协方差高斯（单调收敛、规避奇异）；EM 收敛后对协方差做一次 PCA，按 `cpv` 选主元，得载荷 `P` 与特征值 `Λ` 供 T²/SPE 监测，并用 KDE 给两者定控制限。
- **`MPCAMonitor`**：每个工况模态对应一个分量，分量数 `K` 固定为训练模态数（已知模态，跳过论文的 BYY 自动选 K）。`fit` 全局标准化后用各模态数据初始化 K 个分量做 EM 好初值，跑 EM 估计 `π/μ/Σ`，再对各分量协方差做 PCA 并定 KDE 控制限。在线时 `match_modes` 用**贝叶斯后验 argmax**把样本匹配到最可能的分量（式23-24），再用该分量的归一化 T²/SPE 判定（式15-16,25），任一超过统一控制限 1.0 即报警。

### 5.6 `baselines/__init__.py`

导出 `DLMonitor / LCDLMonitor / ODLMonitor / MPCAMonitor`。

---

## 6. 实验入口 `experiments/`

### 6.1 `_common.py` —— 实验公共工具

所有实验脚本的共享层：

- `samples_to_features(samples)`：`.T` 转置（样本×特征 → 特征×样本），喂给算法层前的标准动作。
- `dataset_from_files(data_dir, config)`：转调 `load_saved_dataset`，读取 `data/` 下已保存数据。
- `model_params(config)`：从 config 抽出 `JMSDL` 构造参数（含把随机种子接到 `random_state`）。
- `monitoring_confidence(config)`：取 KDE 置信度。
- `ensure_output_dirs(root)`：建 `outputs/checkpoints|figures|tables`。
- `write_table(frame, path)`：存 CSV。

### 6.2 `generate_data.py` —— 生成数据 + 基础可视化

读 config → 生成数据集 → 把训练/测试 CSV 和标签写到 `data/` → 调 `visualizer.*` 画 5 张图到 `data/train` 和 `data/test`。是整个流程的第一步。

### 6.3 `run_jmsdl.py` —— 单独训练并监测 JMSDL

`run_jmsdl(config)` 流程：
1. 生成数据并转成特征×样本；
2. `JMSDL(**params).fit(train_modes, alpha=...)` 训练 + 定控制限；
3. `predict` / `score_samples` 在含故障测试集上评分；
4. `model.mre_matrix` 算各字典对各模态的 MRE 矩阵（标准化感知），`compute_fdr/far` 算指标。

`main()` 把模型存 npz、把逐样本分数表写盘,并画三类图：各阶段字典热力图（`plot_dictionary_heatmap`）、监测结果图（`plot_monitoring_scores`）、各字典对各模态的 MRE 折线图（`_plot_mre_matrix`,断轴风格复现论文 Fig.8）。

输出（统一写到 `outputs/run_jmsdl/`）：`jmsdl_model.npz`、`jmsdl_scores.csv`、`D1..D4_dictionary_heatmap.png`、`jmsdl_monitoring.png`、`jmsdl_mre_matrix.png`。

### 6.4 `exp_numerical.py` —— 多方法对比（论文 Fig.9）

`run_numerical_experiment(config)`：构造 5 个监测器（JMSDL / mPCA / DL / LCDL / ODL），各自 `fit` 后在同一含故障测试集上 `predict`，算 FDR/FAR 汇成 DataFrame。

注意调用差异：JMSDL 收「特征×样本」，mPCA 收「样本×特征」，DL/LCDL/ODL 各按其 `fit` 约定喂数据（脚本里已分别处理 `train_modes_samples` 与 `train_modes_features`）。

输出：`fig9_fdr_far.csv` 表格（位于 `outputs/exp_numerical/`），仅以表格形式给出各方法的 FDR/FAR，不再生成柱状对比图。

### 6.5 `sensitivity_analysis.py` —— lambda1 敏感性分析（论文 Fig.5）

`run_sensitivity_analysis(config)`：多次运行（每次换数据种子），每次用 K-SVD 学模态 1 的初始字典，然后对模态 2 用一系列 `lambda1` 值各做一次 `update_dictionary_jmsdl`，记录新旧字典相似度 `ds`。

`plot_sensitivity` 画 `ds` 随 `lambda1` 的均值±标准差曲线。输出：`lambda1_ds_raw.csv`、`lambda1_ds_summary.csv`、`lambda1_ds_curve.png`、`dictionary_diff_heatmaps.png`。预期结论：`lambda1` 越大，`ds` 越高（字典越像旧字典，保留越强）。

### 6.6 `DL vs JMSDL/continual_learning.py` —— 连续学习下的灾难性遗忘对比

独立于 `experiments/` 的对比脚本（在仓库根目录的 `DL vs JMSDL/` 下，用 `python "DL vs JMSDL/continual_learning.py"` 运行）。目的是直观验证「相似性保持项消除灾难性遗忘」：

- **传统 DL 连续学习**（`train_dl_continual`）：`D1 = fit_ksvd(X1)`；之后每个阶段都把上一字典当 `initial_dictionary` 传给 `fit_ksvd`，**只用当前模态数据**继续微调，没有保留项 → 会遗忘旧模态。
- **JMSDL 连续学习**：直接复用 `JMSDL.fit` 得到各阶段字典 `D1..D4`（带保留项）。
- 两条路径共用 **JMSDL 拟合出的全局标准化参数**（`model.mean_/scale_`），保证 DL 与 JMSDL 在同一尺度下比较；每个阶段字典都对完整四工况正常测试集 `test_normal`（标准化后）做 OMP 重构，算逐样本重构误差。

输出：`DL vs JMSDL/continual_learning.png`，2×4 面板，上排 DL（DRE）、下排 JMSDL（IRE），列对应 `D1..D4`，竖线标模态分界。预期：DL 上排随阶段推进，早期模态样本段误差明显抬高（遗忘）；JMSDL 下排各阶段对全部模态都保持较低误差。

---

## 7. 测试 `tests/`

| 文件 | 覆盖内容 |
|------|----------|
| `test_data_loader.py` | 数据生成形状、标签、故障注入、参数校验 |
| `test_sparse_coding.py` | OMP 编码正确性与稀疏度约束 |
| `test_ksvd_and_jmsdl.py` | K-SVD 收敛、JMSDL 序贯训练与评分 |
| `test_monitoring.py` | KDE 控制限、FDR/FAR/MRE 等指标 |
| `test_baselines.py` | mPCA / DL / LCDL / ODL 基线 |
| `test_experiment_modules.py` | 实验入口端到端冒烟（小 config 跑通 run_jmsdl / exp_numerical / sensitivity_analysis） |
| `test_visualizer.py` | 可视化函数产图（字典热力图、监测图、稀疏编码热力图） |

运行：`python -m pytest`。

---

## 8. 端到端调用链（把所有文件串起来）

以 `python experiments/run_jmsdl.py` 为例，完整调用栈：

```
run_jmsdl.main
└─ load_config(config.yaml)                         # utils.data_loader
└─ ensure_output_dirs                               # experiments._common
└─ run_jmsdl(config)
   ├─ dataset_from_files → load_saved_dataset                    # utils.data_loader
   ├─ samples_to_features (转置)                      # experiments._common
   ├─ JMSDL.fit(train_modes, alpha)                  # model.jmsdl
   │   ├─ as_feature_by_sample                       # model.initializer
   │   ├─ fit_standardizer + apply_standardizer      # utils.data_loader (z-score, 抗遗忘关键)
   │   ├─ fit_ksvd(模态1)                             # model.ksvd
   │   │   ├─ initialize_svd_dictionary              # model.initializer
   │   │   ├─ omp_encode (每轮)                        # model.sparse_coding
   │   │   └─ _update_dictionary (SVD 逐原子)          # model.ksvd
   │   ├─ update_dictionary_jmsdl(模态2..N)           # model.dictionary_update (按步 lambda1/sparsity)
   │   │   ├─ omp_encode                             # model.sparse_coding
   │   │   └─ solve_jmsdl_dictionary (闭式解)          # model.dictionary_update
   │   └─ set_threshold
   │       ├─ compute_reconstruction_errors          # monitoring.offline (→ online.score_samples)
   │       └─ kde_threshold                          # monitoring.offline
   ├─ JMSDL.predict / score_samples                  # 先标准化 → monitoring.online.score_samples
   └─ model.mre_by_mode / compute_fdr / compute_far   # monitoring.metrics
└─ model.save_npz + write_table                      # 落盘 outputs/
```

理解这条链，基本就掌握了整个项目：**数据生成 → 转置 → K-SVD 初始字典 → JMSDL 增量更新（OMP 编码 + 闭式解交替）→ KDE 控制限 → 在线 IRE 评分 → FDR/FAR 评估**。

---

## 9. 读代码建议顺序

1. `model/sparse_coding.py`（OMP，最底层、被调用最多）
2. `model/initializer.py`（矩阵方向约定，避免后面被转置绕晕）
3. `model/ksvd.py`（初始字典）
4. `model/dictionary_update.py`（JMSDL 灵魂：目标函数 + 闭式解）
5. `model/jmsdl.py`（组装外壳）
6. `monitoring/offline.py` + `online.py` + `metrics.py`（监测与评估）
7. `utils/data_loader.py`（数据怎么来的）
8. `experiments/*`（怎么把上面所有东西跑起来）
9. `baselines/*`（对照方法）

---

## 10. 关键实现决策与已知限制

读代码时这几点容易踩坑或产生疑问,集中说明:

### 关键实现决策

- **数据标准化是 JMSDL 生效的前提**。论文数据 `x = A·s + e` 状态非零均值,不标准化时重构项 `‖Xn−DnW‖²` 量级约为保留项 `λ1·tr(I−DoᵀDn)` 的 10 倍以上,论文给的 `λ1`（2~3）几乎不起作用,序贯更新退化成"在新模态上重学字典",最终字典只能表示最后一个模态（灾难性遗忘）。默认 z-score 标准化后,两项量级可比,论文 `λ1` 才能平衡新旧模态,MRE 矩阵从"对角"变为均衡。基线 DL/LCDL/ODL 用一致的标准化策略以保证公平对比。
- **按步稀疏度**：`update_sparsity_values=[3,3,5]`,对应论文 D2/D3=3、D4=5。
- **L1 → L0**：论文式 (4) 的稀疏项 `λ2‖W‖₁` 用 OMP 的 L0 硬约束替代,`jmsdl_objective` 只含重构项 + 保留项。
- **KDE 控制限置信度**：论文式 (18) 的 `α/2` 判断为笔误,实现按惯例用 `α` 分位作单侧上控制限;CDF 用梯形积分。

### 已知限制

- **FDR 偏低（未解决）**：当前数值仿真下 JMSDL 的故障检测率明显偏低（约 0.02 量级）。这与抗遗忘能力无关（MRE 已验证遗忘消除）,而是故障可分性问题——论文式 (26) 的默认故障（20 维仅 1 维 +4 偏置）对 IRE 抬升很弱,且单一全局 KDE 控制限会被表示最差模态的长尾抬高,淹没故障信号。属阈值鲁棒性 / 故障幅度匹配范畴,随种子与各模态 MRE 平衡度变化。可能改进:逐模态或更鲁棒的控制限,或重新核对故障幅度与噪声水平。
- **抗遗忘核心结论已复现**：标准化 + 按步稀疏度修复后,最终字典对各模态 MRE 趋于均衡,论文"相似性保持消除灾难性遗忘"的核心结论可复现。
