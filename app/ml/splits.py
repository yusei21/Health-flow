"""Train/test split and cross-validation folds.

When examples carry a patient `group_id`, every split is grouped: one patient never
appears on both sides (prevents leakage between visits of the same patient). Without
groups (synthetic rows are independent fictional patients) splits are stratified by row.
The held-out test set is used only for the final evaluation, never for model selection.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

TEST_FOLDS = 5  # one of five stratified group folds → ~20% held-out test
CV_FOLDS = 5


class SplitStrategy(StrEnum):
    STRATIFIED_ROW = "stratified_row_80_20__cv_stratified_kfold_5"
    STRATIFIED_GROUP = "stratified_group_by_patient_80_20__cv_stratified_group_kfold_5"


@dataclass(frozen=True)
class DataSplit:
    strategy: SplitStrategy
    train_index: NDArray[np.intp]
    test_index: NDArray[np.intp]


def split_train_test(
    labels: NDArray[np.str_], groups: NDArray[np.str_] | None, random_state: int
) -> DataSplit:
    indices = np.arange(len(labels))
    if groups is None:
        train_index, test_index = train_test_split(
            indices, test_size=0.2, stratify=labels, random_state=random_state
        )
        return DataSplit(SplitStrategy.STRATIFIED_ROW, train_index, test_index)
    splitter = StratifiedGroupKFold(n_splits=TEST_FOLDS, shuffle=True, random_state=random_state)
    train_index, test_index = next(splitter.split(indices, labels, groups))
    return DataSplit(SplitStrategy.STRATIFIED_GROUP, train_index, test_index)


def cv_folds(
    labels: NDArray[np.str_], groups: NDArray[np.str_] | None, random_state: int
) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]:
    indices = np.arange(len(labels))
    if groups is None:
        yield from StratifiedKFold(CV_FOLDS, shuffle=True, random_state=random_state).split(
            indices, labels
        )
    else:
        yield from StratifiedGroupKFold(CV_FOLDS, shuffle=True, random_state=random_state).split(
            indices, labels, groups
        )
