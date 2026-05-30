from __future__ import annotations

import numpy as np

from jmsdl.model.sparse_coding import omp_encode


def test_omp_recovers_sparse_code_with_identity_dictionary() -> None:
    dictionary = np.eye(4)
    true_code = np.array([1.5, 0.0, -2.0, 0.0])
    sample = dictionary @ true_code

    estimated = omp_encode(sample, dictionary, sparsity=2)

    np.testing.assert_allclose(estimated, true_code, atol=1.0e-10)


def test_omp_batch_shape_and_sparsity() -> None:
    dictionary = np.eye(5)
    samples = np.eye(5)

    codes = omp_encode(samples, dictionary, sparsity=1)

    assert codes.shape == (5, 5)
    assert np.all(np.sum(np.abs(codes) > 1.0e-12, axis=0) <= 1)
