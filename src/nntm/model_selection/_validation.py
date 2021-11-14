"""
The `nntm.model_selection._validation` module includes classes and
functions to validate the model.
"""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Gael Varoquaux <gael.varoquaux@normalesup.org>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Raghav RV <rvraghav93@gmail.com>
#         Michal Karbownik <michakarbownik@gmail.com>
# License: BSD 3 clause
# Author: Timo Sutterer <hi@timo-sutterer.de>
# License: MIT

import time
import logging
import numbers
import warnings
from contextlib import suppress
from traceback import format_exc
import numpy as np
from joblib import Parallel
from joblib import logger as logger_joblib
from sklearn.utils import indexable
from sklearn.base import clone
from sklearn.utils.fixes import delayed
from ..metrics import check_scoring
from ..metrics._scorer import _MultimetricScorer, _PredictWithErasScorer
from ..utils.validation import _check_fit_params
from ..utils.metaestimators import _safe_split
from ._split import check_cv, _num_samples

logger = logging.getLogger(__name__)

__all__ = ["validation_curve"]


def validation_curve(
    estimator,
    X,
    y,
    *,
    param_name,
    param_range,
    groups=None,
    cv=None,
    target_days=20,
    embargo=None,
    scoring="corr",
    n_jobs=None,
    pre_dispatch="all",
    verbose=0,
    error_score=np.nan,
    fit_params=None,
):
    """Validation curve.
    Determine training and test scores for varying parameter values.
    Compute scores for an estimator with different values of a specified
    parameter. This is similar to grid search with one parameter.
    However, this will also compute training scores and is merely a
    utility for plotting the results.
    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    X : array-like of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        Target relative to X for classification or regression;
        None for unsupervised learning.
    param_name : str
        Name of the parameter that will be varied.
    param_range : array-like of shape (n_values,)
        The values of the parameter that will be evaluated.
    groups : array-like of shape (n_samples,), default=None
        Eras for the samples used while splitting the dataset into
        train/test set. Also required for some scoring functions.
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold purged cross validation,
        - integer, to specify the number of folds for purged cross
          validation,
        - An iterable yielding (train, test) splits as arrays of
          indices.
    target_days : int, default=20
        Days between the observation of samples and the target.
    embargo : float between 0.0 and 1.0, default=None
        Relative number of eras to be purged after every test set.
        (`embargo` * `total_era_count`) eras are embargoed.
    scoring : str or callable, default='corr'
        A str or a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
    n_jobs : int, default=None
        Number of jobs to run in parallel. Training the estimator and
        computing the score are parallelized over the combinations of
        each parameter value and each cross-validation split.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors. See
        :term:`Glossary <n_jobs>` for more details.
    pre_dispatch : int or str, default='all'
        Number of predispatched jobs for parallel execution (default is
        all). The option can reduce the allocated memory. The str can
        be an expression like '2*n_jobs'.
    verbose : int, default=0
        Controls the verbosity: the higher, the more messages.
    fit_params : dict, default=None
        Parameters to pass to the fit method of the estimator.
    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator
        fitting. If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised.
    Returns
    -------
    train_scores : array of shape (n_ticks, n_cv_folds)
        Scores on training sets.
    test_scores : array of shape (n_ticks, n_cv_folds)
        Scores on test set.
    """
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, target_days=target_days, embargo=embargo)
    scorer = check_scoring(estimator, scoring=scoring)
    eras = groups if isinstance(scorer, _PredictWithErasScorer) else None

    parallel = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch, verbose=verbose)
    results = parallel(
        delayed(_fit_and_score)(
            clone(estimator),
            X,
            y,
            scorer,
            train,
            test,
            verbose,
            parameters={param_name: v},
            fit_params=fit_params,
            eras=eras,
            return_train_score=True,
            error_score=error_score,
        )
        # NOTE do not change order of iteration to allow one time cv splitters
        for train, test in cv.split(X, y, groups)
        for v in param_range
    )
    n_params = len(param_range)

    results = _aggregate_score_dicts(results)
    train_scores = results["train_scores"].reshape(-1, n_params).T
    test_scores = results["test_scores"].reshape(-1, n_params).T

    return train_scores, test_scores


def _fit_and_score(
    estimator,
    X,
    y,
    scorer,
    train,
    test,
    verbose,
    parameters,
    fit_params,
    eras=None,
    return_train_score=False,
    return_parameters=False,
    return_n_test_samples=False,
    return_times=False,
    return_estimator=False,
    split_progress=None,
    candidate_progress=None,
    error_score=np.nan,
):

    """Fit estimator and compute scores for a given dataset split.
    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    X : array-like of shape (n_samples, n_features)
        The data to fit.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        The target variable to try to predict in the case of
        supervised learning.
    scorer : A single callable or dict mapping scorer name to the callable
        If it is a single callable, the return value for ``train_scores`` and
        ``test_scores`` is a single float.
        For a dict, it should be one mapping the scorer name to the scorer
        callable object / function.
        The callable object / fn should have signature
        ``scorer(estimator, X, y)``.
    train : array-like of shape (n_train_samples,)
        Indices of training samples.
    test : array-like of shape (n_test_samples,)
        Indices of test samples.
    verbose : int
        The verbosity level.
    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised.
    parameters : dict or None
        Parameters to be set on the estimator.
    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.
    eras : TODO
    return_train_score : bool, default=False
        Compute and return score on training set.
    return_parameters : bool, default=False
        Return parameters that has been used for the estimator.
    split_progress : {list, tuple} of int, default=None
        A list or tuple of format (<current_split_id>, <total_num_of_splits>).
    candidate_progress : {list, tuple} of int, default=None
        A list or tuple of format
        (<current_candidate_id>, <total_number_of_candidates>).
    return_n_test_samples : bool, default=False
        Whether to return the ``n_test_samples``.
    return_times : bool, default=False
        Whether to return the fit/score times.
    return_estimator : bool, default=False
        Whether to return the fitted estimator.
    Returns
    -------
    result : dict with the following attributes
        train_scores : dict of scorer name -> float
            Score on training set (for all the scorers),
            returned only if `return_train_score` is `True`.
        test_scores : dict of scorer name -> float
            Score on testing set (for all the scorers).
        n_test_samples : int
            Number of test samples.
        fit_time : float
            Time spent for fitting in seconds.
        score_time : float
            Time spent for scoring in seconds.
        parameters : dict or None
            The parameters that have been evaluated.
        estimator : estimator object
            The fitted estimator.
        fit_error : str or None
            Traceback str if the fit failed, None if the fit succeeded.
    """
    if not isinstance(error_score, numbers.Number) and error_score != "raise":
        raise ValueError(
            "error_score must be the string 'raise' or a numeric value. "
            "(Hint: if using 'raise', please make sure that it has been "
            "spelled correctly.)"
        )
    if isinstance(scorer, _PredictWithErasScorer) and eras is None:
        raise ValueError("`eras` can not be None when using `_PredictWithErasScorer`.")

    progress_msg = ""
    if verbose > 2:
        if split_progress is not None:
            progress_msg = f" {split_progress[0]+1}/{split_progress[1]}"
        if candidate_progress and verbose > 9:
            progress_msg += f"; {candidate_progress[0]+1}/{candidate_progress[1]}"

    if verbose > 1:
        if parameters is None:
            params_msg = ""
        else:
            sorted_keys = sorted(parameters)  # Ensure deterministic o/p
            params_msg = ", ".join(f"{k}={parameters[k]}" for k in sorted_keys)
    if verbose > 9:
        start_msg = f"[CV{progress_msg}] START {params_msg}"
        print(f"{start_msg}{(80 - len(start_msg)) * '.'}")

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_fit_params(X, fit_params, train)

    if parameters is not None:
        # clone after setting parameters in case any parameters
        # are estimators (like pipeline steps)
        # because pipeline doesn't clone steps in fit
        cloned_parameters = {}
        for k, v in parameters.items():
            cloned_parameters[k] = clone(v, safe=False)

        estimator = estimator.set_params(**cloned_parameters)

    start_time = time.time()

    X_train, y_train, eras_train = _safe_split(estimator, X, y, eras, train)
    X_test, y_test, eras_test = _safe_split(estimator, X, y, eras, test, train)

    result = {}
    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)

    except Exception:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == "raise":
            raise
        elif isinstance(error_score, numbers.Number):
            if isinstance(scorer, dict):
                test_scores = {name: error_score for name in scorer}
                if return_train_score:
                    train_scores = test_scores.copy()
            else:
                test_scores = error_score
                if return_train_score:
                    train_scores = error_score
        result["fit_error"] = format_exc()
    else:
        result["fit_error"] = None

        fit_time = time.time() - start_time
        test_scores = _score(estimator, X_test, y_test, eras_test, scorer, error_score)
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            train_scores = _score(
                estimator, X_train, y_train, eras_train, scorer, error_score
            )

    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = f"[CV{progress_msg}] END "
        result_msg = params_msg + (";" if params_msg else "")
        if verbose > 2:
            if isinstance(test_scores, dict):
                for scorer_name in sorted(test_scores):
                    result_msg += f" {scorer_name}: ("
                    if return_train_score:
                        scorer_scores = train_scores[scorer_name]
                        result_msg += f"train={scorer_scores:.3f}, "
                    result_msg += f"test={test_scores[scorer_name]:.3f})"
            else:
                result_msg += ", score="
                if return_train_score:
                    result_msg += f"(train={train_scores:.3f}, test={test_scores:.3f})"
                else:
                    result_msg += f"{test_scores:.3f}"
        result_msg += f" total time={logger_joblib.short_format_time(total_time)}"

        # Right align the result_msg
        end_msg += "." * (80 - len(end_msg) - len(result_msg))
        end_msg += result_msg
        print(end_msg)

    result["test_scores"] = test_scores
    if return_train_score:
        result["train_scores"] = train_scores
    if return_n_test_samples:
        result["n_test_samples"] = _num_samples(X_test)
    if return_times:
        result["fit_time"] = fit_time
        result["score_time"] = score_time
    if return_parameters:
        result["parameters"] = parameters
    if return_estimator:
        result["estimator"] = estimator
    return result


def _score(estimator, X_test, y_test, eras_test, scorer, error_score="raise"):
    """Compute the score(s) of an estimator on a given test set.
    Will return a dict of floats if `scorer` is a dict, otherwise a single
    float is returned.
    """
    if isinstance(scorer, dict):
        # will cache method calls if needed. scorer() returns a dict
        scorer = _MultimetricScorer(**scorer)

    try:
        if y_test is None:
            scores = scorer(estimator, X_test)
        else:
            scores = scorer(estimator, X_test, y_test, eras=eras_test)
    except Exception:
        if error_score == "raise":
            raise
        else:
            if isinstance(scorer, _MultimetricScorer):
                scores = {name: error_score for name in scorer._scorers}
            else:
                scores = error_score
            warnings.warn(
                "Scoring failed. The score on this train-test partition for "
                f"these parameters will be set to {error_score}. Details: \n"
                f"{format_exc()}",
                UserWarning,
            )

    error_msg = "scoring must return a number, got %s (%s) instead. (scorer=%s)"
    if isinstance(scores, dict):
        for name, score in scores.items():
            if hasattr(score, "item"):
                with suppress(ValueError):
                    # e.g. unwrap memmapped scalars
                    score = score.item()
            if not isinstance(score, numbers.Number):
                raise ValueError(error_msg % (score, type(score), name))
            scores[name] = score
    else:  # scalar
        if hasattr(scores, "item"):
            with suppress(ValueError):
                # e.g. unwrap memmapped scalars
                scores = scores.item()
        if not isinstance(scores, numbers.Number):
            raise ValueError(error_msg % (scores, type(scores), scorer))
    return scores


def _aggregate_score_dicts(scores):
    """Aggregate the list of dict to dict of np ndarray
    The aggregated output of _aggregate_score_dicts will be a list of dict
    of form [{'prec': 0.1, 'acc':1.0}, {'prec': 0.1, 'acc':1.0}, ...]
    Convert it to a dict of array {'prec': np.array([0.1 ...]), ...}
    Parameters
    ----------
    scores : list of dict
        List of dicts of the scores for all scorers. This is a flat list,
        assumed originally to be of row major order.
    Example
    -------
    >>> scores = [{'a': 1, 'b':10}, {'a': 2, 'b':2}, {'a': 3, 'b':3},
    ...           {'a': 10, 'b': 10}]                         # doctest: +SKIP
    >>> _aggregate_score_dicts(scores)                        # doctest: +SKIP
    {'a': array([1, 2, 3, 10]),
     'b': array([10, 2, 3, 10])}
    """
    return {
        key: np.asarray([score[key] for score in scores])
        if isinstance(scores[0][key], numbers.Number)
        else [score[key] for score in scores]
        for key in scores[0]
    }
