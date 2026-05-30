from __future__ import annotations

import numpy as np

from baselines import DLMonitor, LCDLMonitor, MPCAMonitor, ODLMonitor


def _sample_mode(seed: int, n_samples: int = 28) -> np.ndarray:
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n_samples, 2))
    mixing = rng.normal(size=(2, 5))
    return latent @ mixing + 0.05 * rng.normal(size=(n_samples, 5))


def test_dictionary_baselines_fit_and_predict_shapes() -> None:
    modes = [_sample_mode(0), _sample_mode(1), _sample_mode(2)]
    test = _sample_mode(3, n_samples=7)

    monitors = [
        DLMonitor(n_atoms=6, sparsity=2, max_iter=1, alpha=0.95, random_state=0).fit(modes[0]),
        LCDLMonitor(n_atoms=6, sparsity=2, max_iter=1, alpha=0.95, random_state=0).fit(modes),
        ODLMonitor(n_atoms=6, sparsity=2, max_iter=1, alpha=0.95, random_state=0).fit(modes),
    ]

    for monitor in monitors:
        scores = monitor.score_samples(test)
        predictions = monitor.predict(test)
        assert scores.shape == (test.shape[0],)
        assert predictions.shape == (test.shape[0],)


def test_mpca_monitor_fit_and_predict_shape() -> None:
    modes = [_sample_mode(4), _sample_mode(5)]
    test = _sample_mode(6, n_samples=6)

    monitor = MPCAMonitor(cpv=0.85, alpha=0.95).fit(modes)
    scores = monitor.score_samples(test)
    predictions = monitor.predict(test)

    assert scores.shape == (test.shape[0],)
    assert predictions.shape == (test.shape[0],)
