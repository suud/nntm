"""Utilities for meta-estimators"""
# Author: Joel Nothman
#         Andreas Mueller
# License: BSD
# Author: Timo Sutterer <hi@timo-sutterer.de>
# License: MIT

import numpy as np
from ..utils import _safe_indexing
from ..base import _is_pairwise


def _safe_split(estimator, X, y, eras, indices, train_indices=None):
    """Create subset of dataset and properly handle kernels.
    Slice X, y according to indices for cross-validation, but take care of
    precomputed kernel-matrices or pairwise affinities / distances.
    If ``estimator._pairwise is True``, X needs to be square and
    we slice rows and columns. If ``train_indices`` is not None,
    we slice rows using ``indices`` (assumed the test set) and columns
    using ``train_indices``, indicating the training set.
    .. deprecated:: 0.24
        The _pairwise attribute is deprecated in 0.24. From 1.1
        (renaming of 0.26) and onward, this function will check for the
        pairwise estimator tag.
    Labels y will always be indexed only along the first axis.
    Parameters
    ----------
    estimator : object
        Estimator to determine whether we should slice only rows or rows and
        columns.
    X : array-like, sparse matrix or iterable
        Data to be indexed. If ``estimator._pairwise is True``,
        this needs to be a square array-like or sparse matrix.
    y : array-like, sparse matrix or iterable
        Targets to be indexed.
    eras : TODO
    indices : array of int
        Rows to select from X and y.
        If ``estimator._pairwise is True`` and ``train_indices is None``
        then ``indices`` will also be used to slice columns.
    train_indices : array of int or None, default=None
        If ``estimator._pairwise is True`` and ``train_indices is not None``,
        then ``train_indices`` will be use to slice the columns of X.
    Returns
    -------
    X_subset : array-like, sparse matrix or list
        Indexed data.
    y_subset : array-like, sparse matrix or list
        Indexed targets.
    eras_subset : TODO
    """
    if _is_pairwise(estimator):
        if not hasattr(X, "shape"):
            raise ValueError(
                "Precomputed kernels or affinity matrices have "
                "to be passed as arrays or sparse matrices."
            )
        # X is a precomputed square kernel matrix
        if X.shape[0] != X.shape[1]:
            raise ValueError("X should be a square kernel matrix")
        if train_indices is None:
            X_subset = X[np.ix_(indices, indices)]
        else:
            X_subset = X[np.ix_(indices, train_indices)]
    else:
        X_subset = _safe_indexing(X, indices)

    if y is not None:
        y_subset = _safe_indexing(y, indices)
    else:
        y_subset = None

    if eras is not None:
        eras_subset = _safe_indexing(eras, indices)
    else:
        eras_subset = None

    return X_subset, y_subset, eras_subset
