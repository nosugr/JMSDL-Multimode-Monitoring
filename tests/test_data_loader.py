import pytest

from jmsdl.utils.data_loader import generate_multimode_dataset


@pytest.mark.parametrize("n_fault_per_mode", [-1, 6])
def test_generate_multimode_dataset_rejects_invalid_fault_count(n_fault_per_mode):
    with pytest.raises(ValueError, match="n_fault_per_mode"):
        generate_multimode_dataset(
            n_features=3,
            n_train_per_mode=2,
            n_test_per_mode=5,
            n_fault_per_mode=n_fault_per_mode,
        )
