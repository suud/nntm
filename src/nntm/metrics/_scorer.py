"""
The `nntm.metrics.scorer` submodule implements a flexible interface
for model selection and evaluation using arbitrary score functions.

A scorer object is a callable that can be passed as the `scoring`
argument, to specify how a model should be evaluated.

The signature of the call is `(estimator, X, y, *)` where `estimator`
is the model to be evaluated, `X` is the test data and `y` is the
ground truth labeling (or `None` in the case of unsupervised models).
"""

# Authors: Andreas Mueller <amueller@ais.uni-bonn.de>
#          Lars Buitinck
#          Arnaud Joly <arnaud.v.joly@gmail.com>
# License: Simplified BSD
# Author: Timo Sutterer <hi@timo-sutterer.de>
# License: MIT

from functools import partial
from collections import Counter
from collections.abc import Iterable
import numpy as np
from sklearn.base import is_regressor
from sklearn.metrics import SCORERS
from sklearn.utils.multiclass import type_of_target
from ._numerai import corr_score

__all__ = ["get_scorer", "check_scoring"]


def _cached_call(cache, estimator, method, *args, **kwargs):
    """Call estimator with method and args and kwargs."""
    if cache is None:
        return getattr(estimator, method)(*args, **kwargs)

    try:
        return cache[method]
    except KeyError:
        result = getattr(estimator, method)(*args, **kwargs)
        cache[method] = result
        return result


class _MultimetricScorer:
    """Callable for multimetric scoring used to avoid repeated calls
    to `predict_proba`, `predict`, and `decision_function`.
    `_MultimetricScorer` will return a dictionary of scores corresponding to
    the scorers in the dictionary. Note that `_MultimetricScorer` can be
    created with a dictionary with one key  (i.e. only one actual scorer).
    Parameters
    ----------
    scorers : dict
        Dictionary mapping names to callable scorers.
    """

    def __init__(self, **scorers):
        self._scorers = scorers

    def __call__(self, estimator, *args, **kwargs):
        """Evaluate predicted target values."""
        scores = {}
        cache = {} if self._use_cache(estimator) else None
        cached_call = partial(_cached_call, cache)

        for name, scorer in self._scorers.items():
            if isinstance(scorer, _BaseScorer):
                score = scorer._score(cached_call, estimator, *args, **kwargs)
            else:
                score = scorer(estimator, *args, **kwargs)
            scores[name] = score
        return scores

    def _use_cache(self, estimator):
        """Return True if using a cache is beneficial.
        Caching may be beneficial when one of these conditions holds:
          - `_ProbaScorer` will be called twice.
          - `_PredictScorer` will be called twice.
          - `_ThresholdScorer` will be called twice.
          - `_ThresholdScorer` and `_PredictScorer` are called and
             estimator is a regressor.
          - `_ThresholdScorer` and `_ProbaScorer` are called and
             estimator does not have a `decision_function` attribute.
        """
        if len(self._scorers) == 1:  # Only one scorer
            return False

        counter = Counter([type(v) for v in self._scorers.values()])

        if any(
            counter[known_type] > 1
            for known_type in [_PredictScorer, _ProbaScorer, _ThresholdScorer]
        ):
            return True

        if counter[_ThresholdScorer]:
            if is_regressor(estimator) and counter[_PredictScorer]:
                return True
            elif counter[_ProbaScorer] and not hasattr(estimator, "decision_function"):
                return True
        return False


class _BaseScorer:
    def __init__(self, score_func, sign, kwargs):
        self._kwargs = kwargs
        self._score_func = score_func
        self._sign = sign

    @staticmethod
    def _check_pos_label(pos_label, classes):
        if pos_label not in list(classes):
            raise ValueError(f"pos_label={pos_label} is not a valid label: {classes}")

    def _select_proba_binary(self, y_pred, classes):
        """Select the column of the positive label in `y_pred` when
        probabilities are provided.
        Parameters
        ----------
        y_pred : ndarray of shape (n_samples, n_classes)
            The prediction given by `predict_proba`.
        classes : ndarray of shape (n_classes,)
            The class labels for the estimator.
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Probability predictions of the positive class.
        """
        if y_pred.shape[1] == 2:
            pos_label = self._kwargs.get("pos_label", classes[1])
            self._check_pos_label(pos_label, classes)
            col_idx = np.flatnonzero(classes == pos_label)[0]
            return y_pred[:, col_idx]

        err_msg = (
            f"Got predict_proba of shape {y_pred.shape}, but need "
            f"classifier with two classes for {self._score_func.__name__} "
            "scoring"
        )
        raise ValueError(err_msg)

    def __repr__(self):
        kwargs_string = "".join(
            [", %s=%s" % (str(k), str(v)) for k, v in self._kwargs.items()]
        )
        return "make_scorer(%s%s%s%s)" % (
            self._score_func.__name__,
            "" if self._sign > 0 else ", greater_is_better=False",
            self._factory_args(),
            kwargs_string,
        )

    def __call__(self, estimator, X, y_true, sample_weight=None, eras=None):
        """Evaluate predicted target values for X relative to y_true.
        Parameters
        ----------
        estimator : object
            Trained estimator to use for scoring. Must have a predict_proba
            method; the output of that is used to compute the score.
        X : {array-like, sparse matrix}
            Test data that will be fed to estimator.predict.
        y_true : array-like
            Gold standard target values for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        eras : TODO
        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        return self._score(
            partial(_cached_call, None),
            estimator,
            X,
            y_true,
            sample_weight=sample_weight,
            eras=eras,
        )

    def _factory_args(self):
        """Return non-default make_scorer arguments for repr."""
        return ""


class _PredictScorer(_BaseScorer):
    def _score(self, method_caller, estimator, X, y_true, sample_weight=None, **kwargs):
        """Evaluate predicted target values for X relative to y_true.
        Parameters
        ----------
        method_caller : callable
            Returns predictions given an estimator, method name, and other
            arguments, potentially caching results.
        estimator : object
            Trained estimator to use for scoring. Must have a `predict`
            method; the output of that is used to compute the score.
        X : {array-like, sparse matrix}
            Test data that will be fed to estimator.predict.
        y_true : array-like
            Gold standard target values for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """

        y_pred = method_caller(estimator, "predict", X)
        if sample_weight is not None:
            return self._sign * self._score_func(
                y_true, y_pred, sample_weight=sample_weight, **self._kwargs
            )
        else:
            return self._sign * self._score_func(y_true, y_pred, **self._kwargs)


class _ProbaScorer(_BaseScorer):
    def _score(self, method_caller, clf, X, y, sample_weight=None, **kwargs):
        """Evaluate predicted probabilities for X relative to y_true.
        Parameters
        ----------
        method_caller : callable
            Returns predictions given an estimator, method name, and other
            arguments, potentially caching results.
        clf : object
            Trained classifier to use for scoring. Must have a `predict_proba`
            method; the output of that is used to compute the score.
        X : {array-like, sparse matrix}
            Test data that will be fed to clf.predict_proba.
        y : array-like
            Gold standard target values for X. These must be class labels,
            not probabilities.
        sample_weight : array-like, default=None
            Sample weights.
        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """

        y_type = type_of_target(y)
        y_pred = method_caller(clf, "predict_proba", X)
        if y_type == "binary" and y_pred.shape[1] <= 2:
            # `y_type` could be equal to "binary" even in a multi-class
            # problem: (when only 2 class are given to `y_true` during scoring)
            # Thus, we need to check for the shape of `y_pred`.
            y_pred = self._select_proba_binary(y_pred, clf.classes_)
        if sample_weight is not None:
            return self._sign * self._score_func(
                y, y_pred, sample_weight=sample_weight, **self._kwargs
            )
        else:
            return self._sign * self._score_func(y, y_pred, **self._kwargs)

    def _factory_args(self):
        return ", needs_proba=True"


class _ThresholdScorer(_BaseScorer):
    def _score(self, method_caller, clf, X, y, sample_weight=None, **kwargs):
        """Evaluate decision function output for X relative to y_true.
        Parameters
        ----------
        method_caller : callable
            Returns predictions given an estimator, method name, and other
            arguments, potentially caching results.
        clf : object
            Trained classifier to use for scoring. Must have either a
            decision_function method or a predict_proba method; the output of
            that is used to compute the score.
        X : {array-like, sparse matrix}
            Test data that will be fed to clf.decision_function or
            clf.predict_proba.
        y : array-like
            Gold standard target values for X. These must be class labels,
            not decision function values.
        sample_weight : array-like, default=None
            Sample weights.
        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """

        y_type = type_of_target(y)
        if y_type not in ("binary", "multilabel-indicator"):
            raise ValueError("{0} format is not supported".format(y_type))

        if is_regressor(clf):
            y_pred = method_caller(clf, "predict", X)
        else:
            try:
                y_pred = method_caller(clf, "decision_function", X)

                if isinstance(y_pred, list):
                    # For multi-output multi-class estimator
                    y_pred = np.vstack([p for p in y_pred]).T
                elif y_type == "binary" and "pos_label" in self._kwargs:
                    self._check_pos_label(self._kwargs["pos_label"], clf.classes_)
                    if self._kwargs["pos_label"] == clf.classes_[0]:
                        # The implicit positive class of the binary classifier
                        # does not match `pos_label`: we need to invert the
                        # predictions
                        y_pred *= -1

            except (NotImplementedError, AttributeError):
                y_pred = method_caller(clf, "predict_proba", X)

                if y_type == "binary":
                    y_pred = self._select_proba_binary(y_pred, clf.classes_)
                elif isinstance(y_pred, list):
                    y_pred = np.vstack([p[:, -1] for p in y_pred]).T

        if sample_weight is not None:
            return self._sign * self._score_func(
                y, y_pred, sample_weight=sample_weight, **self._kwargs
            )
        else:
            return self._sign * self._score_func(y, y_pred, **self._kwargs)

    def _factory_args(self):
        return ", needs_threshold=True"


class _PredictWithErasScorer(_BaseScorer):
    def _score(
        self, method_caller, estimator, X, y_true, sample_weight=None, eras=None
    ):
        """Evaluate predicted target values for X relative to y_true.
        Parameters
        ----------
        method_caller : callable
            Returns predictions given an estimator, method name, and other
            arguments, potentially caching results.
        estimator : object
            Trained estimator to use for scoring. Must have a `predict`
            method; the output of that is used to compute the score.
        X : {array-like, sparse matrix}
            Test data that will be fed to estimator.predict.
        y_true : array-like
            Gold standard target values for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        eras : TODO
        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        if eras is None:
            raise ValueError("`eras` are required for `_PredictWithErasScorer`.")

        y_pred = method_caller(estimator, "predict", X)
        if sample_weight is not None:
            return self._sign * self._score_func(
                y_true, y_pred, sample_weight=sample_weight, **self._kwargs
            )
        else:
            return self._sign * self._score_func(y_true, y_pred, eras, **self._kwargs)

    def _factory_args(self):
        return ", needs_eras=True"


def get_scorer(scoring):
    """Get a scorer from string.
    Read more in the :ref:`User Guide <scoring_parameter>`.
    Parameters
    ----------
    scoring : str or callable
        Scoring method as string. If callable it is returned as is.
    Returns
    -------
    scorer : callable
        The scorer.
    """
    if isinstance(scoring, str):
        try:
            scorer = SCORERS[scoring]
        except KeyError:
            raise ValueError(
                f"{scoring} is not a valid scoring value. "
                "Use sorted(nntm.metrics.SCORERS.keys()) "
                "to get valid options."
            )
    else:
        scorer = scoring
    return scorer


def _passthrough_scorer(estimator, *args, **kwargs):
    """Function that wraps estimator.score"""
    return estimator.score(*args, **kwargs)


def check_scoring(estimator, scoring=None, *, allow_none=False):
    """Determine scorer from user options.
    A TypeError will be thrown if the estimator cannot be scored.
    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    scoring : str or callable, default=None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        If None, the provided estimator object's `score` method is used.
    allow_none : bool, default=False
        If no scoring is specified and the estimator has no score function, we
        can either return None or raise an exception.
    Returns
    -------
    scoring : callable
        A scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
    """
    if not hasattr(estimator, "fit"):
        raise TypeError(
            "estimator should be an estimator implementing 'fit' method, %r was passed"
            % estimator
        )
    if isinstance(scoring, str):
        return get_scorer(scoring)
    elif callable(scoring):
        # Heuristic to ensure user has not passed a metric
        module = getattr(scoring, "__module__", None)
        if (
            hasattr(module, "startswith")
            and module.startswith("sklearn.metrics.")
            and not module.startswith("sklearn.metrics._scorer")
            and not module.startswith("sklearn.metrics.tests.")
        ):
            raise ValueError(
                f"scoring value {scoring} looks like it is a metric "
                "function rather than a scorer. A scorer should "
                "require an estimator as its first parameter. "
                "Please use `make_scorer` to convert a metric "
                "to a scorer."
            )
        return get_scorer(scoring)
    elif scoring is None:
        if hasattr(estimator, "score"):
            return _passthrough_scorer
        elif allow_none:
            return None
        else:
            raise TypeError(
                "If no scoring is specified, the estimator passed should "
                f"have a 'score' method. The estimator {estimator} does not."
            )
    elif isinstance(scoring, Iterable):
        raise ValueError(
            "For evaluating multiple scores, use "
            "sklearn.model_selection.cross_validate instead. "
            "{0} was passed.".format(scoring)
        )
    else:
        raise ValueError(
            "scoring value should either be a callable, string or None."
            f"{scoring} was passed"
        )


def make_scorer(
    score_func,
    *,
    greater_is_better=True,
    needs_proba=False,
    needs_threshold=False,
    needs_eras=False,
    **kwargs,
):
    """Make a scorer from a performance metric or loss function.
    This factory function wraps scoring functions for use in
    :class:`~sklearn.model_selection.GridSearchCV` and
    :func:`~sklearn.model_selection.cross_val_score`.
    It takes a score function, such as :func:`~sklearn.metrics.accuracy_score`,
    :func:`~sklearn.metrics.mean_squared_error`,
    :func:`~sklearn.metrics.adjusted_rand_index` or
    :func:`~sklearn.metrics.average_precision`
    and returns a callable that scores an estimator's output.
    The signature of the call is `(estimator, X, y)` where `estimator`
    is the model to be evaluated, `X` is the data and `y` is the
    ground truth labeling (or `None` in the case of unsupervised models).
    Read more in the :ref:`User Guide <scoring>`.
    Parameters
    ----------
    score_func : callable
        Score function (or loss function) with signature
        ``score_func(y, y_pred, **kwargs)``.
    greater_is_better : bool, default=True
        Whether score_func is a score function (default), meaning high is good,
        or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the score_func.
    needs_proba : bool, default=False
        Whether score_func requires predict_proba to get probability estimates
        out of a classifier.
        If True, for binary `y_true`, the score function is supposed to accept
        a 1D `y_pred` (i.e., probability of the positive class, shape
        `(n_samples,)`).
    needs_threshold : bool, default=False
        Whether score_func takes a continuous decision certainty.
        This only works for binary classification using estimators that
        have either a decision_function or predict_proba method.
        If True, for binary `y_true`, the score function is supposed to accept
        a 1D `y_pred` (i.e., probability of the positive class or the decision
        function, shape `(n_samples,)`).
        For example ``average_precision`` or the area under the roc curve
        can not be computed using discrete predictions alone.
    needs_eras : bool, default=False
        TODO
    **kwargs : additional arguments
        Additional parameters to be passed to score_func.
    Returns
    -------
    scorer : callable
        Callable object that returns a scalar score; greater is better.
    Examples
    --------
    >>> from sklearn.metrics import fbeta_score, make_scorer
    >>> ftwo_scorer = make_scorer(fbeta_score, beta=2)
    >>> ftwo_scorer
    make_scorer(fbeta_score, beta=2)
    >>> from sklearn.model_selection import GridSearchCV
    >>> from sklearn.svm import LinearSVC
    >>> grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]},
    ...                     scoring=ftwo_scorer)
    Notes
    -----
    If `needs_proba=False` and `needs_threshold=False`, the score
    function is supposed to accept the output of :term:`predict`. If
    `needs_proba=True`, the score function is supposed to accept the
    output of :term:`predict_proba` (For binary `y_true`, the score function is
    supposed to accept probability of the positive class). If
    `needs_threshold=True`, the score function is supposed to accept the
    output of :term:`decision_function` or :term:`predict_proba` when
    :term:`decision_function` is not present.
    """
    sign = 1 if greater_is_better else -1
    if needs_proba and needs_threshold:
        raise ValueError(
            "Set either needs_proba or needs_threshold to True, but not both."
        )
    if needs_proba and needs_eras:
        raise ValueError("Set either needs_proba or needs_eras to True, but not both.")
    if needs_threshold and needs_eras:
        raise ValueError(
            "Set either needs_threshold or needs_eras to True, but not both."
        )
    if needs_proba:
        cls = _ProbaScorer
    elif needs_threshold:
        cls = _ThresholdScorer
    elif needs_eras:
        cls = _PredictWithErasScorer
    else:
        cls = _PredictScorer
    return cls(score_func, sign, kwargs)


SCORERS["corr"] = make_scorer(corr_score, needs_eras=True)
