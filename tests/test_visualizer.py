from __future__ import annotations

import numpy as np

from jmsdl.utils.visualizer import (
    plot_dictionary_heatmap,
    plot_monitoring_scores,
    plot_sparse_code_heatmap,
)


def test_plot_dictionary_heatmap_writes_file(tmp_path) -> None:
    rng = np.random.default_rng(0)
    dictionary = rng.standard_normal((20, 80))
    output = tmp_path / "dict.png"

    plot_dictionary_heatmap(dictionary, output, title="Test Dictionary")

    assert output.exists() and output.stat().st_size > 0


def test_plot_monitoring_scores_writes_file(tmp_path) -> None:
    rng = np.random.default_rng(1)
    scores = np.abs(rng.standard_normal(40))
    fault_labels = np.zeros(40, dtype=int)
    fault_labels[30:] = 1
    output = tmp_path / "monitor.png"

    plot_monitoring_scores(
        scores, threshold=1.5, fault_labels=fault_labels, output_path=output,
        mode_boundaries=[0, 20, 40], fdr=0.8, far=0.05,
    )

    assert output.exists() and output.stat().st_size > 0


def test_plot_sparse_code_heatmap_writes_file(tmp_path) -> None:
    rng = np.random.default_rng(3)
    codes = rng.standard_normal((8, 24))
    fault_labels = np.zeros(24, dtype=int)
    fault_labels[18:] = 1
    output = tmp_path / "sparse_codes_heatmap.png"

    plot_sparse_code_heatmap(
        codes,
        output,
        mode_boundaries=[0, 12, 24],
        atom_ranges=[(0, 4), (4, 8)],
        fault_labels=fault_labels,
    )

    assert output.exists() and output.stat().st_size > 0
