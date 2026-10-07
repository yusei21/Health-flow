import numpy as np
import pytest

from app.ml.splits import SplitStrategy, cv_folds, split_train_test


def make_data(n_patients: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    groups = np.repeat([f"p{i}" for i in range(n_patients)], rng.integers(1, 5, n_patients))
    labels = rng.choice(["PRIMARY_CARE", "URGENT_CARE", "EMERGENCY"], size=len(groups))
    return labels, groups


def test_group_split_never_shares_a_patient_between_train_and_test() -> None:
    labels, groups = make_data()
    split = split_train_test(labels, groups, random_state=42)
    assert split.strategy is SplitStrategy.STRATIFIED_GROUP
    assert not set(groups[split.train_index]) & set(groups[split.test_index])
    assert len(split.train_index) + len(split.test_index) == len(labels)
    assert 0.1 < len(split.test_index) / len(labels) < 0.3


def test_group_cv_folds_never_share_a_patient() -> None:
    labels, groups = make_data()
    folds = list(cv_folds(labels, groups, random_state=42))
    assert len(folds) == 5
    for train, validation in folds:
        assert not set(groups[train]) & set(groups[validation])
    all_validation = np.concatenate([validation for _, validation in folds])
    assert sorted(all_validation) == list(range(len(labels)))


def test_split_is_deterministic() -> None:
    labels, groups = make_data()
    first = split_train_test(labels, groups, random_state=42)
    second = split_train_test(labels, groups, random_state=42)
    assert np.array_equal(first.test_index, second.test_index)


@pytest.mark.parametrize("groups", [None])
def test_without_groups_uses_stratified_row_split(groups: None) -> None:
    labels, _ = make_data()
    split = split_train_test(labels, groups, random_state=42)
    assert split.strategy is SplitStrategy.STRATIFIED_ROW
    assert len(list(cv_folds(labels[split.train_index], None, 42))) == 5
