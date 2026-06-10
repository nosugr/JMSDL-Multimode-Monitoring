from __future__ import annotations

import numpy as np
import pytest


def _tiny_config() -> dict:
    """小规模 config，让实验模块在测试里能秒级跑完。"""
    return {
        "seed": {"random_state": 0},
        "numerical_simulation": {
            "n_features": 6,
            "state_dim": 2,
            "n_modes": 3,
            "n_train_per_mode": 24,
            "n_test_per_mode": 8,
            "n_fault_per_mode": 4,
            "fault_feature": 1,
            "fault_bias": 4.0,
            "timeseries_dims": [0, 1],
            "kde_confidence": 0.95,
        },
        "model": {
            "n_atoms": 8,
            "sparsity": 2,
            "update_sparsity_values": [2, 2],
            "lambda_values": [2.0, 2.0],
            "initial_max_iter": 2,
            "update_max_iter": 2,
            "tol": 1.0e-5,
        },
        "sensitivity_analysis": {"lambda1_values": [1.0, 2.0], "n_runs": 2},
    }


def _tiny_data_dir(tmp_path) -> object:
    from experiments.generate_data import save_datasets
    from jmsdl.utils.data_loader import generate_from_config

    data_dir = tmp_path / "data"
    dataset = generate_from_config(_tiny_config())
    save_datasets(dataset, data_dir / "train", data_dir / "test")
    return data_dir


def test_run_jmsdl_returns_metrics(tmp_path) -> None:
    from main_model.run_jmsdl.run_jmsdl import run_jmsdl

    result = run_jmsdl(_tiny_config(), show_progress=False, data_dir=_tiny_data_dir(tmp_path))

    assert {"model", "scores", "predictions", "fdr", "far", "boundaries"} <= set(result)
    assert np.asarray(result["scores"]).shape == np.asarray(result["fault_labels"]).shape
    assert 0.0 <= float(result["fdr"]) <= 1.0
    assert 0.0 <= float(result["far"]) <= 1.0


def test_run_numerical_experiment_lists_all_methods(tmp_path) -> None:
    from experiments.exp_numerical import run_numerical_experiment

    frame = run_numerical_experiment(_tiny_config(), show_progress=False, data_dir=_tiny_data_dir(tmp_path))

    assert set(frame["method"]) == {"JMSDL", "mPCA", "DL", "LCDL", "ODL"}
    assert {"FDR", "FAR"} <= set(frame.columns)


def test_run_sensitivity_analysis_returns_ds_and_diffs(tmp_path) -> None:
    from experiments.sensitivity_analysis import run_sensitivity_analysis

    frame, diff_matrices = run_sensitivity_analysis(
        _tiny_config(), show_progress=False, data_dir=_tiny_data_dir(tmp_path)
    )

    assert set(frame["lambda1"]) == {1.0, 2.0}
    assert set(diff_matrices.keys()) == {1.0, 2.0}
    for matrix in diff_matrices.values():
        assert np.all(matrix >= 0.0)
