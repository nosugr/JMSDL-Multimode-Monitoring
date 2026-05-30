from __future__ import annotations

import numpy as np

from jmsdl.model import JMSDL, dictionary_similarity, fit_ksvd, update_dictionary_jmsdl


def _mode(seed: int, n_samples: int = 24) -> np.ndarray:
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(2, n_samples))
    mixing = rng.normal(size=(6, 2))
    return mixing @ latent + 0.03 * rng.normal(size=(6, n_samples))


def test_fit_ksvd_returns_normalized_dictionary_and_sparse_codes() -> None:
    data = _mode(0)

    result = fit_ksvd(data, n_atoms=5, sparsity=2, max_iter=2, random_state=1)

    assert result.dictionary.shape == (6, 5)
    assert result.codes.shape == (5, data.shape[1])
    np.testing.assert_allclose(np.linalg.norm(result.dictionary, axis=0), np.ones(5), atol=1.0e-8)
    assert np.all(np.sum(np.abs(result.codes) > 1.0e-8, axis=0) <= 2)


def test_jmsdl_update_keeps_dictionary_shape_and_reports_similarity() -> None:
    old = fit_ksvd(_mode(1), n_atoms=5, sparsity=2, max_iter=1, random_state=2).dictionary
    new_data = _mode(2)

    result = update_dictionary_jmsdl(new_data, old, sparsity=2, lambda1=2.0, max_iter=2)

    assert result.dictionary.shape == old.shape
    assert result.codes.shape == (5, new_data.shape[1])
    assert 0.0 <= dictionary_similarity(old, result.dictionary) <= 1.0
    assert len(result.objective_history) >= 1


def test_jmsdl_fits_sequential_modes_and_scores_samples() -> None:
    modes = [_mode(3), _mode(4), _mode(5)]
    model = JMSDL(
        n_atoms=6,
        sparsity=2,
        lambda_values=[1.0, 1.2],
        initial_max_iter=1,
        update_max_iter=1,
        random_state=3,
    )

    model.fit(modes, alpha=0.95)
    scores = model.score_samples(modes[0])
    predictions = model.predict(modes[0])

    assert len(model.dictionaries_) == 3
    assert len(model.ds_history_) == 2
    assert model.dictionary_.shape == (6, 6)
    assert scores.shape == (modes[0].shape[1],)
    assert predictions.shape == (modes[0].shape[1],)
