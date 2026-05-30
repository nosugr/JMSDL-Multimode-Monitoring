
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
    plot_multimode_pca_scatter,
    plot_multimode_timeseries,
    plot_test_data_heatmap,
    plot_test_fault_timeseries,
    plot_test_pca_scatter,
)


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
    pd.DataFrame(np.asarray(dataset["test_faulty"], dtype=float)).to_csv(
        test_dir / "test_faulty.csv", index=False
    )
    pd.DataFrame(np.asarray(dataset["test_faulty"], dtype=float)).to_csv(
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
    save_datasets(dataset, train_dir, test_dir)

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

    # 测试数据热力图：JSSDL 样式（原始数值对称着色 + 故障变量行红框 + 红虚线 + 文字标注）
    n_test_per_mode = int(dataset["n_test_per_mode"])
    test_boundaries = [mode_index * n_test_per_mode for mode_index in range(n_modes + 1)]
    plot_test_data_heatmap(
        np.asarray(dataset["test_faulty"], dtype=float),
        int(dataset["fault_feature"]),
        np.asarray(dataset["fault_labels"], dtype=int),
        test_dir / "test_data_heatmap.png",
        mode_boundaries=test_boundaries,
    )

    # 测试数据时序折线图：故障变量 x2 上的阶跃 + 模态分界 + 故障区间阴影
    plot_test_fault_timeseries(
        np.asarray(dataset["test_faulty"], dtype=float),
        int(dataset["fault_feature"]),
        np.asarray(dataset["fault_labels"], dtype=int),
        test_dir / "test_fault_timeseries.png",
        mode_boundaries=test_boundaries,
        test_normal=np.asarray(dataset["test_normal"], dtype=float),
        mode_names=mode_names,
    )

    # 测试数据 PCA 散点图：模态上色 + 正常/故障不同 marker
    plot_test_pca_scatter(
        np.asarray(dataset["test_faulty"], dtype=float),
        np.asarray(dataset["test_mode_labels"], dtype=int),
        np.asarray(dataset["fault_labels"], dtype=int),
        test_dir / "test_pca_scatter.png",
        mode_names=mode_names,
    )

    print("数据生成完成：")
    print(f"  训练数据：{train_dir}")
    print(f"  测试数据：{test_dir}")
    print(f"  PCA 散点图选用主成分：PC{first + 1} vs PC{second + 1}")
    print(f"  图已保存到：{train_dir} 与 {test_dir}")


if __name__ == "__main__":
    main()
