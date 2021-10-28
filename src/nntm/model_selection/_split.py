"""
The `nntm.model_selection._split` module includes classes and
functions to split the data based on a preset strategy.
"""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Gael Varoquaux <gael.varoquaux@normalesup.org>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Raghav RV <rvraghav93@gmail.com>
#         Leandro Hermida <hermidal@cs.umd.edu>
#         Rodion Martynov <marrodon@gmail.com>
# License: BSD 3 clause
# Author: Timo Sutterer <hi@timo-sutterer.de>
# License: MIT

import logging
import numbers
import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import indexable
from ..utils.validation import _num_samples

logger = logging.getLogger(__name__)

__all__ = ["PurgedKFold"]


class PurgedKFold(BaseCrossValidator):
    """Purged K-Folds cross-validator

    Provides train/test indices to split data in train/test sets. Split
    dataset into k consecutive folds. Training observations overlapping
    in time with test observations are purged.

    Optionally, the eras that immediately follow the test set can be
    eliminated using the `embargo` argument.

    Data is assumed to be contiguous (shuffle=False).

    References
    ----------
    .. [1] `Marcos Lopez de Prado (2018). Advances in Financial Machine
            Learning. Chapter 7 (Cross-Validation in Finance).`_
    .. [2] `Super Massive Data Release: Deep Dive
            <https://forum.numer.ai/t/super-massive-data-release-deep-dive/4053>`_
    """

    def __init__(self, n_splits=5, target_days=20, embargo=None):
        if not isinstance(n_splits, numbers.Integral):
            raise ValueError(
                "The number of folds must be of Integral type. "
                f"`n_splits={n_splits}` of type {type(n_splits)} was passed."
            )
        n_splits = int(n_splits)

        if n_splits <= 1:
            raise ValueError(
                "k-fold cross-validation requires at least one "
                "train/test split by setting `n_splits=2` or more, "
                f"got `n_splits={n_splits}`."
            )

        if not isinstance(target_days, numbers.Integral):
            raise ValueError(
                "The number of target days must be of Integral type. "
                f"`target_days={target_days}` of type {type(target_days)} was passed."
            )
        target_days = int(target_days)

        if target_days % 5 != 0:
            raise ValueError(
                "The number of target days has to be a multiple of 5. "
                f"`target_days={target_days}` was passed."
            )

        if embargo:
            if not isinstance(embargo, float):
                raise ValueError(
                    "Embargo must be of float type. "
                    f"`embargo={embargo}` of type {type(embargo)} was passed."
                )

            if not 0.0 < embargo < 1.0:
                raise ValueError(
                    "Embargo must be between 0.0 and 1.0. "
                    f"`embargo={embargo}` was passed."
                )

        self.n_splits = n_splits
        self.target_days = target_days
        self.embargo = embargo

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.
        groups : array-like of shape (n_samples,), default=None
            Eras for the samples used while splitting the dataset into
            train/test set. This parameter is not required when X is
            a pandas DataFrame containing an `era` column.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if isinstance(X, np.ndarray) and not groups:
            raise ValueError("`groups` parameter is required when X is a numpy array")

        if isinstance(X, pd.DataFrame) and not groups and "era" not in X.columns:
            raise ValueError("`groups` parameter is required when X has no era column")

        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        if self.n_splits > n_samples:
            raise ValueError(
                (
                    f"Cannot have number of splits n_splits={self.n_splits} greater "
                    f"than the number of samples: n_samples={n_samples}."
                )
            )

        eras = np.fromiter(self._get_eras(X, groups=groups), dtype=int)
        target_weeks = self.target_days // 5
        eras_target_release = np.array([era + target_weeks - 1 for era in eras])

        embargo_era_count = 0
        if self.embargo:
            era_count = len(set(eras))
            embargo_era_count = int(round(era_count * self.embargo))

        indices = np.arange(_num_samples(X))
        for test_index_mask in self._iter_test_masks(X, y, groups):
            test_index = indices[test_index_mask]
            test_era_min = min(eras[test_index])
            test_era_max = max(eras[test_index])
            test_era_target_release_max = max(eras_target_release[test_index])

            train_index = indices[np.logical_not(test_index_mask)]
            for idx, train_era, train_era_target_release in zip(
                train_index, eras[train_index], eras_target_release[train_index]
            ):
                purge = not (
                    train_era_target_release < test_era_min
                    or train_era > test_era_target_release_max
                )
                embargo = test_era_max <= train_era <= test_era_max + embargo_era_count

                if purge or embargo:
                    train_index = train_index[train_index != idx]

            yield train_index, test_index

    def _iter_test_indices(self, X=None, y=None, groups=None):
        """Generates integer indices corresponding to test sets."""
        n_samples = _num_samples(X)
        indices = np.arange(n_samples)

        n_splits = self.n_splits

        # Fold sizes depend on n_samples (not n_eras)
        fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
        fold_sizes[: n_samples % n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            yield indices[start:stop]
            current = stop

    def _get_eras(self, X, groups=None):
        """Generates integer eras."""
        eras = groups
        if not groups:
            eras = X["era"].tolist()

        for era in eras:
            yield int(era)

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator
        Parameters
        ----------
        X : object
            Always ignored, exists for sklearn compatibility.
        y : object
            Always ignored, exists for sklearn compatibility.
        groups : object
            Always ignored, exists for sklearn compatibility.
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits
