"""Base classes for all estimators."""

# Author: Gael Varoquaux <gael.varoquaux@normalesup.org>
# License: BSD 3 clause
# Author: Timo Sutterer <hi@timo-sutterer.de>
# License: MIT

from .utils._tags import _safe_tags


def _is_pairwise(estimator):
    """Returns True if estimator is pairwise.
    Parameters
    ----------
    estimator : object
        Estimator object to test.
    Returns
    -------
    out : bool
        True if the estimator is pairwise and False otherwise.
    """
    pairwise_tag = _safe_tags(estimator, key="pairwise")

    return pairwise_tag
