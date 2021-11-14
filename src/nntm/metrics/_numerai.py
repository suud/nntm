"""Metrics to assess performance on Numerai tournament predictions."""

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_consistent_length

__all__ = ["corr_score"]


def corr_score(y_true, y_pred, eras):
    """Spearman's rank correlation coefficient.

    This score is referred to as correlation (corr) in the Numerai main
    tournament.

    Parameters
    ----------
    TODO
    Returns
    -------
    TODO

    References
    ----------
    https://docs.numer.ai/tournament/learn#scoring
    https://github.com/numerai/example-scripts/blob/96677b980ffa1bf17f62f8f3d9695a1aac054a4a/analysis_and_tips.ipynb
    """
    if eras is None:
        check_consistent_length(y_true, y_pred)
        rank_pred = y_pred.rank(pct=True, method="first")
    else:
        check_consistent_length(y_true, y_pred, eras)

        if isinstance(y_pred, np.ndarray):
            y_pred = pd.Series(y_pred)

        rank_pred = y_pred.groupby(eras).apply(
            lambda x: x.rank(pct=True, method="first")
        )
    return np.corrcoef(y_true, rank_pred)[0, 1]
