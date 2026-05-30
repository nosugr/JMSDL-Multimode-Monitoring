from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from jmsdl.utils.data_loader import load_saved_dataset


ROOT = Path(__file__).resolve().parents[1]


def samples_to_features(samples: np.ndarray) -> np.ndarray:
    return np.asarray(samples, dtype=float).T


def dataset_from_files(data_dir: str | Path = ROOT / "data", config: dict | None = None) -> dict[str, object]:
    return load_saved_dataset(data_dir, config=config)


def model_params(config: dict) -> dict[str, object]:
    model = config.get("model", {})
    seed = config.get("seed", {})
    return {
        "n_atoms": int(model.get("n_atoms", 80)),
        "sparsity": int(model.get("sparsity", 3)),
        "lambda_values": list(model.get("lambda_values", [3.0, 2.5, 2.6])),
        "update_sparsity_values": list(model.get("update_sparsity_values", [3, 3, 5])),
        "initial_max_iter": int(model.get("initial_max_iter", 30)),
        "update_max_iter": int(model.get("update_max_iter", 30)),
        "tol": float(model.get("tol", 1.0e-5)),
        "random_state": seed.get("data_random_state", 0),
    }


def monitoring_confidence(config: dict) -> float:
    return float(config.get("monitoring", {}).get("kde_confidence", 0.99))


def ensure_output_dirs(root: Path = ROOT) -> dict[str, Path]:
    paths = {
        "checkpoints": root / "outputs" / "checkpoints",
        "figures": root / "outputs" / "figures",
        "tables": root / "outputs" / "tables",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
