from __future__ import annotations

import numpy as np

from jmsdl.monitoring import compute_far, compute_fdr, compute_reconstruction_errors, detect_fault, kde_threshold


def test_reconstruction_errors_and_threshold_are_finite() -> None:
    dictionary = np.eye(3)
    samples = np.eye(3)

    errors = compute_reconstruction_errors(samples, dictionary, sparsity=1)
    threshold = kde_threshold(errors + 0.01, alpha=0.95)

    assert errors.shape == (3,)
    assert np.all(np.isfinite(errors))
    assert np.isfinite(threshold)


def test_fdr_far_metrics() -> None:
    labels = np.array([0, 0, 1, 1])
    predictions = np.array([0, 1, 1, 0])

    assert compute_fdr(labels, predictions) == 0.5
    assert compute_far(labels, predictions) == 0.5
    np.testing.assert_array_equal(detect_fault(np.array([0.2, 2.0]), 1.0), np.array([False, True]))
