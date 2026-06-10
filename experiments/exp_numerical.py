from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from baselines import DLMonitor, LCDLMonitor, MPCAMonitor, ODLMonitor
from experiments._common import (
    dataset_from_files,
    model_params,
    monitoring_confidence,
    samples_to_features,
    write_table,
)
from jmsdl.model import JMSDL
from jmsdl.monitoring.metrics import compute_far, compute_fdr
from jmsdl.utils.data_loader import load_config


def run_numerical_experiment(
    config: dict, show_progress: bool = True, data_dir: str | Path = ROOT / "data"
) -> pd.DataFrame:
    dataset = dataset_from_files(data_dir, config=config)
    train_modes_samples = [np.asarray(mode, dtype=float) for mode in dataset["train_modes"]]
    train_modes_features = [samples_to_features(mode) for mode in train_modes_samples]
    test_all_samples = np.asarray(dataset["test_all"], dtype=float)
    test_all_features = samples_to_features(test_all_samples)
    fault_labels = np.asarray(dataset["fault_labels"], dtype=int)

    alpha = monitoring_confidence(config)
    params = model_params(config)
    standardize = bool(params["standardize"])
    random_state = params["random_state"]

    monitors: list[tuple[str, object]] = []
    monitors.append(
        ("JMSDL", JMSDL(**params).fit(train_modes_features, alpha=alpha, show_progress=show_progress))
    )

    mpca_monitor = MPCAMonitor(alpha=alpha)
    mpca_monitor.standardize = standardize
    mpca_monitor.random_state = random_state
    mpca_monitor.fit(train_modes_samples, show_progress=show_progress, progress_desc="epoch[mPCA]")
    monitors.append(
        ("mPCA", mpca_monitor)
    )

    dl_monitor = DLMonitor(
        n_atoms=int(params["n_atoms"]),
        sparsity=int(params["sparsity"]),
        alpha=alpha,
        tol=float(params["tol"]),
    )
    dl_monitor.standardize = standardize
    dl_monitor.random_state = random_state
    dl_monitor.fit(train_modes_samples, show_progress=show_progress, progress_desc="epoch[DL]")
    monitors.append(
        ("DL", dl_monitor)
    )

    lcdl_monitor = LCDLMonitor(
        n_atoms=int(params["n_atoms"]),
        sparsity=int(params["sparsity"]),
        alpha=alpha,
        tol=float(params["tol"]),
    )
    lcdl_monitor.standardize = standardize
    lcdl_monitor.random_state = random_state
    lcdl_monitor.fit(train_modes_features, show_progress=show_progress, progress_desc="epoch[LCDL]")
    monitors.append(
        ("LCDL", lcdl_monitor)
    )

    odl_monitor = ODLMonitor(
        n_atoms=int(params["n_atoms"]),
        sparsity=int(params["sparsity"]),
        alpha=alpha,
        tol=float(params["tol"]),
    )
    odl_monitor.standardize = standardize
    odl_monitor.random_state = random_state
    odl_monitor.fit(train_modes_features, show_progress=show_progress, progress_desc="epoch[ODL]")
    monitors.append(
        ("ODL", odl_monitor)
    )

    rows: list[dict[str, float | str]] = []
    for name, monitor in monitors:
        if name == "JMSDL":
            predictions = monitor.predict(test_all_features)  # type: ignore[attr-defined]
        else:
            predictions = monitor.predict(test_all_samples)  # type: ignore[attr-defined]
        rows.append(
            {
                "method": name,
                "FDR": compute_fdr(fault_labels, predictions),
                "FAR": compute_far(fault_labels, predictions),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    config = load_config(ROOT / "config.yaml")
    out_dir = ROOT / "outputs" / "exp_numerical"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_numerical_experiment(config, show_progress=True)
    write_table(result, out_dir / "fig9_fdr_far.csv")
    print(result.to_string(index=False))
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
