from __future__ import annotations

import numpy as np

from baselines import DLMonitor, LCDLMonitor, MPCAMonitor, ODLMonitor
from jmsdl.model.ksvd import KSVDResult


def _sample_mode(seed: int, n_samples: int = 28) -> np.ndarray:
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n_samples, 2))
    mixing = rng.normal(size=(2, 5))
    return latent @ mixing + 0.05 * rng.normal(size=(n_samples, 5))


def test_dictionary_baselines_fit_and_predict_shapes() -> None:
    modes = [_sample_mode(0), _sample_mode(1), _sample_mode(2)]
    test = _sample_mode(3, n_samples=7)

    monitors = [
        DLMonitor(n_atoms=6, sparsity=2, max_iter=1, alpha=0.95).fit(modes[0]),
        LCDLMonitor(n_atoms=6, sparsity=2, max_iter=1, alpha=0.95).fit(modes),
        ODLMonitor(n_atoms=6, sparsity=2, max_iter=1, alpha=0.95).fit(modes),
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


def test_odl_uses_x1_init_then_ab_accumulator_updates(monkeypatch) -> None:
    from baselines.ODL import odl_monitor

    modes = [_sample_mode(7, 8), _sample_mode(8, 10), _sample_mode(9, 12)]
    seen_shapes: list[tuple[int, int]] = []
    seen_warm_starts: list[bool] = []

    def fake_fit_ksvd(
        Y: np.ndarray,
        n_atoms: int,
        sparsity: int,
        max_iter: int = 30,
        tol: float = 1.0e-5,
        random_state: int | None = None,
        initial_dictionary: np.ndarray | None = None,
        init: str = "svd",
        show_progress: bool = False,
        progress_desc: str = "epoch[K-SVD]",
        progress_position: int = 0,
        progress_leave: bool = False,
    ) -> KSVDResult:
        del sparsity, max_iter, tol, random_state, init
        del show_progress, progress_desc, progress_position, progress_leave
        seen_shapes.append(Y.shape)
        seen_warm_starts.append(initial_dictionary is not None)
        dictionary = np.eye(Y.shape[0], int(n_atoms), dtype=float)
        codes = np.zeros((int(n_atoms), Y.shape[1]), dtype=float)
        return KSVDResult(dictionary=dictionary, codes=codes, error_history=[], n_iter=0)

    monkeypatch.setattr(odl_monitor, "fit_ksvd", fake_fit_ksvd)

    monitor = odl_monitor.ODLMonitor(n_atoms=5, sparsity=2, max_iter=2).fit(modes)

    # K-SVD 只用于 X1 的初始化，后续模态全部走 A/B 累计更新。
    assert seen_shapes == [(5, 8)]
    assert seen_warm_starts == [False]
    assert len(monitor.dictionaries_) == len(modes)
    assert monitor.A_.shape == (5, 5)
    assert monitor.B_.shape == (5, 5)
    # 死原子已随机重初始化，最终字典每列都是单位范数（不再有零列）。
    norms = np.linalg.norm(monitor.dictionary_, axis=0)
    assert np.allclose(norms, 1.0, atol=1.0e-6)


def test_odl_monitor_online_monitors_before_updating() -> None:
    modes = [_sample_mode(0), _sample_mode(1), _sample_mode(2)]
    test = _sample_mode(3, n_samples=9)

    monitor = ODLMonitor(n_atoms=6, sparsity=2, max_iter=2, alpha=0.95).fit(modes)
    final_dictionary = monitor.dictionary_.copy()

    result = monitor.monitor_online(test, batch_size=1, consecutive_flag=2)
    scores = np.asarray(result["scores"], dtype=float)
    predictions = np.asarray(result["predictions"], dtype=bool)

    assert scores.shape == (test.shape[0],)
    assert predictions.shape == (test.shape[0],)
    # 第一个样本在更新前用 D_final 监测，应与固定字典打分一致（先监测后更新）。
    fixed_scores = monitor.score_samples(test)
    assert np.isclose(scores[0], fixed_scores[0], atol=1.0e-8)
    # 字典列应保持单位范数（死原子已重初始化）。
    norms = np.linalg.norm(np.asarray(result["dictionary"], dtype=float), axis=0)
    assert np.allclose(norms, 1.0, atol=1.0e-6)
    # 训练态字典不应被在线监测改动。
    assert np.allclose(monitor.dictionary_, final_dictionary)


def test_odl_monitor_online_consecutive_flag_rule() -> None:
    modes = [_sample_mode(4), _sample_mode(5)]
    test = _sample_mode(6, n_samples=7)

    monitor = ODLMonitor(n_atoms=6, sparsity=2, max_iter=2, alpha=0.95).fit(modes)
    result = monitor.monitor_online(test, batch_size=1, consecutive_flag=2)

    classifications = np.asarray(result["classifications"], dtype=int)
    predictions = np.asarray(result["predictions"], dtype=bool)

    # 三分类取值合法，且 abnormal(=2) 当且仅当 prediction 为 True。
    assert set(np.unique(classifications)).issubset({0, 1, 2})
    assert np.array_equal(predictions, classifications == 2)
    # 连续超限规则：abnormal 必须紧接在一个超限样本(time-varying 或 abnormal)之后，
    # 故首样本不可能直接判为 abnormal。
    assert classifications[0] != 2
    for j in range(1, len(classifications)):
        if classifications[j] == 2:
            assert classifications[j - 1] in (1, 2)


def test_odl_monitor_online_all_normal_updates_without_alarm() -> None:
    modes = [_sample_mode(7), _sample_mode(8)]
    test = _sample_mode(9, n_samples=5)

    monitor = ODLMonitor(n_atoms=6, sparsity=2, max_iter=2, alpha=0.95).fit(modes)
    final_dictionary = monitor.dictionary_.copy()
    # 控制限抬到极大 → 所有样本判为 normal：更新字典、不报警、不动控制限。
    monitor.threshold_ = 1.0e18
    result = monitor.monitor_online(test, batch_size=1, consecutive_flag=2)

    classifications = np.asarray(result["classifications"], dtype=int)
    assert np.all(classifications == 0)
    assert not np.any(np.asarray(result["predictions"], dtype=bool))
    # normal 样本会更新字典。
    assert not np.allclose(result["dictionary"], final_dictionary)
    # normal 不更新控制限，全程保持初始值。
    assert np.allclose(np.asarray(result["threshold_history"], dtype=float), 1.0e18)
