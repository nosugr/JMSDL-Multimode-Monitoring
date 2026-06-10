
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(path: str | Path = "config.yaml") -> dict:
    """读取 YAML 配置文件。"""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def fit_standardizer(matrix: np.ndarray, eps: float = 1.0e-12) -> tuple[np.ndarray, np.ndarray]:
    """Fit feature-wise z-score statistics for a feature-by-sample matrix."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("Expected a 2D feature-by-sample matrix.")
    mean = values.mean(axis=1, keepdims=True)
    scale = values.std(axis=1, keepdims=True)
    scale = np.where(scale < eps, 1.0, scale)
    return mean, scale


def apply_standardizer(matrix: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Apply feature-wise z-score statistics fitted by fit_standardizer."""
    return (np.asarray(matrix, dtype=float) - mean) / scale


def _read_csv_matrix(path: Path, dtype: type = float) -> np.ndarray:
    return pd.read_csv(path).to_numpy(dtype=dtype)


def _mode_file_index(path: Path) -> int:
    suffix = path.stem.removeprefix("train_mode")
    return int(suffix)


def load_saved_dataset(data_dir: str | Path = "data", config: dict | None = None) -> dict[str, object]:
    """Load the saved train/test CSV files under data_dir."""
    root = Path(data_dir)
    train_dir = root / "train"
    test_dir = root / "test"

    train_mode_files = sorted(
        (path for path in train_dir.glob("train_mode*.csv") if path.stem.removeprefix("train_mode").isdigit()),
        key=_mode_file_index,
    )
    train_modes = [_read_csv_matrix(path, float) for path in train_mode_files]
    train_all = _read_csv_matrix(train_dir / "train_all.csv", float)
    train_mode_labels = pd.read_csv(train_dir / "train_mode_labels.csv")["mode"].to_numpy(dtype=int)

    test_normal = _read_csv_matrix(test_dir / "test_normal.csv", float)
    test_all = _read_csv_matrix(test_dir / "test_all.csv", float)
    test_labels = pd.read_csv(test_dir / "test_labels.csv")
    test_mode_labels = test_labels["mode"].to_numpy(dtype=int)
    fault_labels = test_labels["fault"].to_numpy(dtype=int)

    n_modes = len(train_modes)
    n_train_per_mode = int(train_modes[0].shape[0]) if train_modes else 0
    if test_mode_labels.size:
        counts = pd.Series(test_mode_labels).value_counts(sort=False)
        n_test_per_mode = int(counts.iloc[0])
    else:
        n_test_per_mode = 0

    simulation = (config or {}).get("numerical_simulation", {})
    return {
        "train_modes": train_modes,
        "train_all": train_all,
        "train_mode_labels": train_mode_labels,
        "test_modes": [test_normal[test_mode_labels == mode] for mode in range(n_modes)],
        "test_normal": test_normal,
        "test_mode_labels": test_mode_labels,
        "test_all": test_all,
        "fault_labels": fault_labels,
        "fault_feature": int(simulation.get("fault_feature", 1)),
        "n_modes": int(n_modes),
        "n_test_per_mode": int(n_test_per_mode),
        "n_train_per_mode": int(n_train_per_mode),
    }


def _generate_mode_samples(
    n_samples: int,
    observation_matrix: np.ndarray,
    s1_mean: float,
    s1_std: float,
    s2_mean: float,
    s2_std: float,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """按式 26 为单个模态生成样本，返回 (n_samples, n_features)。"""
    n_features = observation_matrix.shape[0]
    s1 = rng.normal(s1_mean, s1_std, size=n_samples)
    s2 = rng.normal(s2_mean, s2_std, size=n_samples)
    state = np.vstack([s1, s2])                                  # (state_dim, n_samples)
    noise = rng.normal(0.0, noise_std, size=(n_features, n_samples))
    samples = observation_matrix @ state + noise                # (n_features, n_samples)
    return samples.T                                            # (n_samples, n_features)


def _inject_fault_bias(samples: np.ndarray, fault_feature: int, fault_bias: float) -> np.ndarray:
    """在指定变量上叠加偏置故障。"""
    faulted = np.asarray(samples, dtype=float).copy()
    n_features = faulted.shape[1]
    feature_index = int(fault_feature)
    if not 0 <= feature_index < n_features:
        raise ValueError(f"fault_feature 必须在 [0, {n_features - 1}]，得到 {feature_index}。")
    faulted[:, feature_index] += float(fault_bias)
    return faulted


def generate_multimode_dataset(
    n_features: int = 20,
    state_dim: int = 2,
    n_modes: int = 4,
    n_train_per_mode: int = 1000,
    n_test_per_mode: int = 250,
    n_fault_per_mode: int = 125,
    s1_mean: float = 2.0,
    s1_std: float = 1.0,
    s2_mean: float = 3.0,
    s2_std: float = 1.0,
    noise_std: float = 0.31622776601683794,
    fault_feature: int = 1,
    fault_bias: float = 4.0,
    random_state: int | None = 0,
    observation_matrix_seed: int | None = 40,
) -> dict[str, object]:
    """生成完整的多模态数据集。

    返回字典包含：
    - train_modes: 长度 n_modes 的列表，每个 (n_train_per_mode, n_features)，供序贯训练
    - train_all: (n_modes*n_train_per_mode, n_features)，按模态顺序堆叠
    - train_mode_labels: (n_modes*n_train_per_mode,)，每个样本所属模态 (0 基)
    - test_modes: 长度 n_modes 的列表，每个 (n_test_per_mode, n_features)，全部正常
    - test_normal: (n_modes*n_test_per_mode, n_features)，按模态顺序堆叠（表示能力实验用）
    - test_mode_labels: (n_modes*n_test_per_mode,)
    - test_all: 与 test_normal 同形，每个模态测试段末尾 n_fault_per_mode 个样本注入故障（监测实验用）
    - fault_labels: (n_modes*n_test_per_mode,)，0 正常 / 1 故障
    - observation_matrices: 长度 n_modes 的列表，每个 (n_features, state_dim)
    """
    if state_dim != 2:
        raise ValueError("当前数据生成系统按论文设定固定 state_dim=2（s1, s2）。")
    if not 0 <= n_fault_per_mode <= n_test_per_mode:
        raise ValueError(
            "n_fault_per_mode 必须满足 0 <= n_fault_per_mode <= n_test_per_mode，"
            f"得到 n_fault_per_mode={n_fault_per_mode}, n_test_per_mode={n_test_per_mode}。"
        )

    # 观测矩阵用独立种子，保证各模态结构在不同 random_state 下稳定
    obs_rng = np.random.default_rng(observation_matrix_seed)
    observation_matrices = [
        obs_rng.normal(0.0, 1.0, size=(n_features, state_dim)) for _ in range(n_modes)
    ]

    rng = np.random.default_rng(random_state)
    train_modes: list[np.ndarray] = []
    test_modes: list[np.ndarray] = []
    for observation_matrix in observation_matrices:
        train_modes.append(
            _generate_mode_samples(
                n_train_per_mode, observation_matrix,
                s1_mean, s1_std, s2_mean, s2_std, noise_std, rng,
            )
        )
        test_modes.append(
            _generate_mode_samples(
                n_test_per_mode, observation_matrix,
                s1_mean, s1_std, s2_mean, s2_std, noise_std, rng,
            )
        )

    train_all = np.vstack(train_modes)
    train_mode_labels = np.repeat(np.arange(n_modes), n_train_per_mode)

    test_normal = np.vstack(test_modes)
    test_mode_labels = np.repeat(np.arange(n_modes), n_test_per_mode)

    # 故障测试集：每个模态测试段末尾 n_fault_per_mode 个样本注入故障
    test_all = test_normal.copy()
    fault_labels = np.zeros(n_modes * n_test_per_mode, dtype=int)
    for mode_index in range(n_modes):
        segment_start = mode_index * n_test_per_mode
        fault_start = segment_start + (n_test_per_mode - n_fault_per_mode)
        fault_end = segment_start + n_test_per_mode
        test_all[fault_start:fault_end] = _inject_fault_bias(
            test_all[fault_start:fault_end], fault_feature, fault_bias
        )
        fault_labels[fault_start:fault_end] = 1

    return {
        "train_modes": train_modes,
        "train_all": train_all,
        "train_mode_labels": train_mode_labels,
        "test_modes": test_modes,
        "test_normal": test_normal,
        "test_mode_labels": test_mode_labels,
        "test_all": test_all,
        "fault_labels": fault_labels,
        "observation_matrices": observation_matrices,
        "fault_feature": int(fault_feature),
        "n_modes": int(n_modes),
        "n_test_per_mode": int(n_test_per_mode),
        "n_train_per_mode": int(n_train_per_mode),
    }


def generate_from_config(config: dict) -> dict[str, object]:
    """从 config 字典构造数据集，便于实验脚本调用。

    注：状态向量分布参数 (s1/s2 的均值与标准差) 与噪声标准差按论文式 26 固定，
    不在 config 中暴露，直接采用 generate_multimode_dataset 的默认值。
    """
    seed_cfg = config.get("seed", {})
    sim = config["numerical_simulation"]
    return generate_multimode_dataset(
        n_features=sim["n_features"],
        state_dim=sim["state_dim"],
        n_modes=sim["n_modes"],
        n_train_per_mode=sim["n_train_per_mode"],
        n_test_per_mode=sim["n_test_per_mode"],
        n_fault_per_mode=sim["n_fault_per_mode"],
        noise_std=sim.get("noise_std", 0.31622776601683794),
        fault_feature=sim["fault_feature"],
        fault_bias=sim["fault_bias"],
        random_state=seed_cfg.get("random_state", 0),
    )
