"""Numerai main tournament dataset.

The Numerai main tournament dataset is made of high quality financial
data that has been cleaned, regularized and obfuscated.

The original dataset is available from

    https://numerai-public-datasets.s3-us-west-2.amazonaws.com/

A detailed description of the dataset is available from

    https://forum.numer.ai/t/super-massive-data-release-deep-dive/4053


The dataset is divided into:

Training data:
--------------
- 2412105 samples over 574 eras (1-574)
- 1050 features
- 20 different targets

Test data:
----------
- 1407586 samples over 278 eras (575-852)
- 1050 features
- no target values provided

Validation data:
----------------
- 539658 samples over 105 eras (857-961)
- 1050 features
- 20 different targets

Live data:
----------
- changes weekly
- number of samples may vary, one era
- 1050 features
- no target values provided

Tournament data:
----------------
- test data and live data

More data may be added from time to time.

References
----------

https://docs.numer.ai/tournament/learn
https://forum.numer.ai/t/super-massive-data-release-deep-dive/4053
https://github.com/numerai/example-scripts/blob/0a8c4f764a3aee3b7c1709058dd1488b26bd5f01/analysis_and_tips.ipynb
https://github.com/uuazed/numerapi

"""

import json
import logging
from os.path import exists, splitext
from os import makedirs, remove
from typing import List

import pandas as pd
from numerapi import NumerAPI
from sklearn.datasets import get_data_home
from sklearn.utils import Bunch

logger = logging.getLogger(__name__)


def _get_numerai_fetcher(filename_float32, filename_int8, name, has_rounds=False):
    """Return fetch function for passed files"""

    def fetch_numerai(
        *,
        data_home=None,
        download_if_missing=True,
        keep=False,
        return_X_y=False,
        target="target",
        as_frame=False,
        columns=None,
        int8=True,
        round_num=None,
    ):
        # Make sure required columns will be read
        if columns:
            columns = list(set(columns + ["era", "data_type"]))

        # Get round (if dataset supports rounds)
        napi = NumerAPI()
        if round_num and not has_rounds:
            raise ValueError("`round_num` given for a dataset without rounds.")
        if not round_num and has_rounds:
            round_num = napi.get_current_round()
            logger.info(f"Using current round={round_num}")

        # Get file locations
        data_home = get_data_home(data_home=data_home)
        if not exists(data_home):
            makedirs(data_home)
        filename = filename_float32
        if int8:
            filename = filename_int8
        filepath = "/".join([data_home, filename])
        if round_num:
            suffix = "_" + str(round_num)
            filename_with_suffix = _add_filename_suffix(filename, suffix)
            filepath = "/".join([data_home, filename_with_suffix])
        else:
            filepath = "/".join([data_home, filename])

        # Download and read dataset
        if not exists(filepath):
            if not download_if_missing:
                raise IOError("Data not found and `download_if_missing` is False")

            logger.info(f"Downloading Numerai {name} data to {filepath}")
            napi.download_dataset("/".join(["v3", filename]), dest_path=filepath)

            df = pd.read_parquet(filepath, columns=columns)
            if not keep:
                remove(filepath)
        else:
            df = pd.read_parquet(filepath, columns=columns)

        # Get feature and target columns
        feature_names = _get_feature_names(df)
        target_names = _get_target_names(df)

        # Enforce expected data type for features
        dtype = _get_dtype(is_int8=int8)
        df[feature_names] = df[feature_names].astype(dtype)

        X = df[feature_names]
        # One entry per target
        target_dict = {tn: df[tn] for tn in target_names}
        # All targets in one DataFrame
        targets = df[target_names]

        id_ = df.index
        era = df.era
        data_type = df.data_type

        if not as_frame:
            # Convert to numpy
            X = X.to_numpy(dtype=dtype)
            target_dict = {k: v.to_numpy() for k, v in target_dict.items()}
            targets = targets.to_numpy()
            id_ = id_.to_numpy()
            era = era.to_numpy()
            data_type = data_type.to_numpy()
            df = None

        if return_X_y:
            return X, target_dict[target]

        return Bunch(
            data=X,
            **target_dict,
            targets=targets,
            id=id_,
            era=era,
            data_type=data_type,
            feature_names=feature_names,
            target_names=target_names,
            int8=int8,
            DESCR=f"Numerai main tournament: {name} data",
            round_num=round_num,
            frame=df,
        )

    return fetch_numerai


def fetch_numerai_training(*args, **kwargs):
    """Load the Numerai training dataset.

    Parameters
    ----------
    data_home : str, default=None
        Specify another download and cache folder for the datasets. By
        default all data is stored in `~/scikit_learn_data` subfolders.

    download_if_missing : bool, default=True
        If False, raise an IOError if the data is not locally available
        instead of trying to download the data from the source bucket.

    keep : bool, default=False
        If True, does not remove the downloaded file from disk after
        reading it.

    return_X_y : bool, default=False
        If True, returns `(data.data, data.target)` instead of a Bunch
        object. A custom target column can be selected through the
        `target` parameter.

    target : str, default="target"
        Target column to return as `y` when `return_X_y=True`.

    as_frame : bool, default=False
        If True, `data` and `targets` are pandas DataFrames. `target`,
        `target_<name>`, `id`, `era` and `data_type` are pandas Series.
        `frame` will be given.

    columns : list, default=None
        If not None, only these columns will be read from the file.
        `index`, `era` and `data_type` columns are always read.

    int8 : bool, default=True
        If True, the feature columns will use the `int8` data type
        instead of `float32`. Target columns are always `float32`.

    Returns
    -------
    dataset : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : {ndarray, DataFrame} of shape (2412105, 1050)
            Each row corresponding to the `feature_names` in order.
            When `as_frame=True`, `data` is a pandas DataFrame.

        target : {ndarray, Series} of shape (2412105,)
            When `as_frame=True`, `target` is a pandas Series.

        targets : {ndarray, DataFrame} of shape (2412105, 21)
            When `as_frame=True`, `targets` is a pandas DataFrame.

        target_<name> : {ndarray, Series} of shape (2412105,)
            See `target_names` for available targets.
            When `as_frame=True`, `target_<name>` is a pandas Series.

        id : {ndarray, Series} of shape (2412105,)
            `id` of each row in `data`.
            When `as_frame=True`, `id` is a pandas Series.

        era : {ndarray, Series} of shape (2412105,)
            `era` of each row in `data`.
            When `as_frame=True`, `era` is a pandas Series.

        data_type : {ndarray, Series} of shape (2412105,)
            `data_type` of each row in `data`.
            When `as_frame=True`, `data_type` is a pandas Series.

        feature_names : list of length 1050
            List of ordered feature names used in the dataset.

        target_names : list of length 21
            List of ordered target names used in the dataset.

        int8 : bool
            True when features use `int8` data type.

        DESCR : string
            Description of the dataset.

        frame : DataFrame if `as_frame=True`
            Only present when `as_frame=True`. Pandas DataFrame with
            `era`, `data_type`, features and targets.

        (data, target) : tuple if `return_X_y=True`
            Only present when `return_X_y=True`. `target` corresponds
            to the column set by the `target` attribute.
    """
    fetcher = _get_numerai_fetcher(
        "numerai_training_data.parquet",
        "numerai_training_data_int8.parquet",
        "training",
    )
    return fetcher(*args, **kwargs)


def fetch_numerai_test(*args, **kwargs):
    """Load the Numerai test dataset.

    Parameters
    ----------
    data_home : str, default=None
        Specify another download and cache folder for the datasets. By
        default all data is stored in `~/scikit_learn_data` subfolders.

    download_if_missing : bool, default=True
        If False, raise an IOError if the data is not locally available
        instead of trying to download the data from the source bucket.

    keep : bool, default=False
        If True, does not remove the downloaded file from disk after
        reading it.

    return_X_y : bool, default=False
        If True, returns `(data.data, data.target)` instead of a Bunch
        object. A custom target column can be selected through the
        `target` parameter.

    target : str, default="target"
        Target column to return as `y` when `return_X_y=True`.

    as_frame : bool, default=False
        If True, `data` and `targets` are pandas DataFrames. `target`,
        `target_<name>`, `id`, `era` and `data_type` are pandas Series.
        `frame` will be given.

    columns : list, default=None
        If not None, only these columns will be read from the file.
        `index`, `era` and `data_type` columns are always read.

    int8 : bool, default=True
        If True, the feature columns will use the `int8` data type
        instead of `float32`. Target columns are always `float32`.

    Returns
    -------
    dataset : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : {ndarray, DataFrame} of shape (1407586, 1050)
            Each row corresponding to the `feature_names` in order.
            When `as_frame=True`, `data` is a pandas DataFrame.

        target : {ndarray, Series} of shape (1407586,)
            When `as_frame=True`, `target` is a pandas Series.

        targets : {ndarray, DataFrame} of shape (1407586, 21)
            When `as_frame=True`, `targets` is a pandas DataFrame.

        target_<name> : {ndarray, Series} of shape (1407586,)
            See `target_names` for available targets.
            When `as_frame=True`, `target_<name>` is a pandas Series.

        id : {ndarray, Series} of shape (1407586,)
            `id` of each row in `data`.
            When `as_frame=True`, `id` is a pandas Series.

        era : {ndarray, Series} of shape (1407586,)
            `era` of each row in `data`.
            When `as_frame=True`, `era` is a pandas Series.

        data_type : {ndarray, Series} of shape (1407586,)
            `data_type` of each row in `data`.
            When `as_frame=True`, `data_type` is a pandas Series.

        feature_names : list of length 1050
            List of ordered feature names used in the dataset.

        target_names : list of length 21
            List of ordered target names used in the dataset.

        int8 : bool
            True when features use `int8` data type.

        DESCR : string
            Description of the dataset.

        frame : DataFrame if `as_frame=True`
            Only present when `as_frame=True`. Pandas DataFrame with
            `era`, `data_type`, features and targets.

        (data, target) : tuple if `return_X_y=True`
            Only present when `return_X_y=True`. `target` corresponds
            to the column set by the `target` attribute.
    """
    fetcher = _get_numerai_fetcher(
        "numerai_test_data.parquet",
        "numerai_test_data_int8.parquet",
        "test",
    )
    return fetcher(*args, **kwargs)


def fetch_numerai_validation(*args, **kwargs):
    """Load the Numerai validation dataset.

    Parameters
    ----------
    data_home : str, default=None
        Specify another download and cache folder for the datasets. By
        default all data is stored in `~/scikit_learn_data` subfolders.

    download_if_missing : bool, default=True
        If False, raise an IOError if the data is not locally available
        instead of trying to download the data from the source bucket.

    keep : bool, default=False
        If True, does not remove the downloaded file from disk after
        reading it.

    return_X_y : bool, default=False
        If True, returns `(data.data, data.target)` instead of a Bunch
        object. A custom target column can be selected through the
        `target` parameter.

    target : str, default="target"
        Target column to return as `y` when `return_X_y=True`.

    as_frame : bool, default=False
        If True, `data` and `targets` are pandas DataFrames. `target`,
        `target_<name>`, `id`, `era` and `data_type` are pandas Series.
        `frame` will be given.

    columns : list, default=None
        If not None, only these columns will be read from the file.
        `index`, `era` and `data_type` columns are always read.

    int8 : bool, default=True
        If True, the feature columns will use the `int8` data type
        instead of `float32`. Target columns are always `float32`.

    Returns
    -------
    dataset : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : {ndarray, DataFrame} of shape (539658, 1050)
            Each row corresponding to the `feature_names` in order.
            When `as_frame=True`, `data` is a pandas DataFrame.

        target : {ndarray, Series} of shape (539658,)
            When `as_frame=True`, `target` is a pandas Series.

        targets : {ndarray, DataFrame} of shape (539658, 21)
            When `as_frame=True`, `targets` is a pandas DataFrame.

        target_<name> : {ndarray, Series} of shape (539658,)
            See `target_names` for available targets.
            When `as_frame=True`, `target_<name>` is a pandas Series.

        id : {ndarray, Series} of shape (539658,)
            `id` of each row in `data`.
            When `as_frame=True`, `id` is a pandas Series.

        era : {ndarray, Series} of shape (539658,)
            `era` of each row in `data`.
            When `as_frame=True`, `era` is a pandas Series.

        data_type : {ndarray, Series} of shape (539658,)
            `data_type` of each row in `data`.
            When `as_frame=True`, `data_type` is a pandas Series.

        feature_names : list of length 1050
            List of ordered feature names used in the dataset.

        target_names : list of length 21
            List of ordered target names used in the dataset.

        int8 : bool
            True when features use `int8` data type.

        DESCR : string
            Description of the dataset.

        frame : DataFrame if `as_frame=True`
            Only present when `as_frame=True`. Pandas DataFrame with
            `era`, `data_type`, features and targets.

        (data, target) : tuple if `return_X_y=True`
            Only present when `return_X_y=True`. `target` corresponds
            to the column set by the `target` attribute.
    """
    fetcher = _get_numerai_fetcher(
        "numerai_validation_data.parquet",
        "numerai_validation_data_int8.parquet",
        "validation",
    )
    return fetcher(*args, **kwargs)


def fetch_numerai_live(*args, **kwargs):
    """Load the Numerai live dataset.

    Parameters
    ----------
    data_home : str, default=None
        Specify another download and cache folder for the datasets. By
        default all data is stored in `~/scikit_learn_data` subfolders.

    download_if_missing : bool, default=True
        If False, raise an IOError if the data is not locally available
        instead of trying to download the data from the source bucket.

    keep : bool, default=False
        If True, does not remove the downloaded file from disk after
        reading it.

    return_X_y : bool, default=False
        If True, returns `(data.data, data.target)` instead of a Bunch
        object. A custom target column can be selected through the
        `target` parameter.

    target : str, default="target"
        Target column to return as `y` when `return_X_y=True`.

    as_frame : bool, default=False
        If True, `data` and `targets` are pandas DataFrames. `target`,
        `target_<name>`, `id`, `era` and `data_type` are pandas Series.
        `frame` will be given.

    columns : list, default=None
        If not None, only these columns will be read from the file.
        `index`, `era` and `data_type` columns are always read.

    int8 : bool, default=True
        If True, the feature columns will use the `int8` data type
        instead of `float32`. Target columns are always `float32`.

    round_num : int, default=None
        Tournament round to download. If None, current round will be
        downloaded.

    Returns
    -------
    dataset : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : {ndarray, DataFrame} of shape (n_samples, 1050)
            Each row corresponding to the `feature_names` in order.
            When `as_frame=True`, `data` is a pandas DataFrame.

        target : {ndarray, Series} of shape (n_samples,)
            When `as_frame=True`, `target` is a pandas Series.

        targets : {ndarray, DataFrame} of shape (n_samples, 21)
            When `as_frame=True`, `targets` is a pandas DataFrame.

        target_<name> : {ndarray, Series} of shape (n_samples,)
            See `target_names` for available targets.
            When `as_frame=True`, `target_<name>` is a pandas Series.

        id : {ndarray, Series} of shape (n_samples,)
            `id` of each row in `data`.
            When `as_frame=True`, `id` is a pandas Series.

        era : {ndarray, Series} of shape (n_samples,)
            `era` of each row in `data`.
            When `as_frame=True`, `era` is a pandas Series.

        data_type : {ndarray, Series} of shape (n_samples,)
            `data_type` of each row in `data`.
            When `as_frame=True`, `data_type` is a pandas Series.

        feature_names : list of length 1050
            List of ordered feature names used in the dataset.

        target_names : list of length 21
            List of ordered target names used in the dataset.

        int8 : bool
            True when features use `int8` data type.

        DESCR : string
            Description of the dataset.

        round_num : int
            Round number of the dataset.

        frame : DataFrame if `as_frame=True`
            Only present when `as_frame=True`. Pandas DataFrame with
            `era`, `data_type`, features and targets.

        (data, target) : tuple if `return_X_y=True`
            Only present when `return_X_y=True`. `target` corresponds
            to the column set by the `target` attribute.

    Notes
    -----
    Data changes weekly.
    """
    fetcher = _get_numerai_fetcher(
        "numerai_live_data.parquet",
        "numerai_live_data_int8.parquet",
        "live",
        has_rounds=True,
    )
    return fetcher(*args, **kwargs)


def fetch_numerai_tournament(*args, **kwargs):
    """Load the Numerai tournament dataset.

    Parameters
    ----------
    data_home : str, default=None
        Specify another download and cache folder for the datasets. By
        default all data is stored in `~/scikit_learn_data` subfolders.

    download_if_missing : bool, default=True
        If False, raise an IOError if the data is not locally available
        instead of trying to download the data from the source bucket.

    keep : bool, default=False
        If True, does not remove the downloaded file from disk after
        reading it.

    return_X_y : bool, default=False
        If True, returns `(data.data, data.target)` instead of a Bunch
        object. A custom target column can be selected through the
        `target` parameter.

    target : str, default="target"
        Target column to return as `y` when `return_X_y=True`.

    as_frame : bool, default=False
        If True, `data` and `targets` are pandas DataFrames. `target`,
        `target_<name>`, `id`, `era` and `data_type` are pandas Series.
        `frame` will be given.

    columns : list, default=None
        If not None, only these columns will be read from the file.
        `index`, `era` and `data_type` columns are always read.

    int8 : bool, default=True
        If True, the feature columns will use the `int8` data type
        instead of `float32`. Target columns are always `float32`.

    round_num : int, default=None
        Tournament round to download. If None, current round will be
        downloaded.

    Returns
    -------
    dataset : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : {ndarray, DataFrame} of shape (n_samples, 1050)
            Each row corresponding to the `feature_names` in order.
            When `as_frame=True`, `data` is a pandas DataFrame.

        target : {ndarray, Series} of shape (n_samples,)
            When `as_frame=True`, `target` is a pandas Series.

        targets : {ndarray, DataFrame} of shape (n_samples, 21)
            When `as_frame=True`, `targets` is a pandas DataFrame.

        target_<name> : {ndarray, Series} of shape (n_samples,)
            See `target_names` for available targets.
            When `as_frame=True`, `target_<name>` is a pandas Series.

        id : {ndarray, Series} of shape (n_samples,)
            `id` of each row in `data`.
            When `as_frame=True`, `id` is a pandas Series.

        era : {ndarray, Series} of shape (n_samples,)
            `era` of each row in `data`.
            When `as_frame=True`, `era` is a pandas Series.

        data_type : {ndarray, Series} of shape (n_samples,)
            `data_type` of each row in `data`.
            When `as_frame=True`, `data_type` is a pandas Series.

        feature_names : list of length 1050
            List of ordered feature names used in the dataset.

        target_names : list of length 21
            List of ordered target names used in the dataset.

        int8 : bool
            True when features use `int8` data type.

        DESCR : string
            Description of the dataset.

        round_num : int
            Round number of the dataset.

        frame : DataFrame if `as_frame=True`
            Only present when `as_frame=True`. Pandas DataFrame with
            `era`, `data_type`, features and targets.

        (data, target) : tuple if `return_X_y=True`
            Only present when `return_X_y=True`. `target` corresponds
            to the column set by the `target` attribute.

    Notes
    -----
    Data changes weekly.
    """
    fetcher = _get_numerai_fetcher(
        "numerai_tournament_data.parquet",
        "numerai_tournament_data_int8.parquet",
        "tournament",
        has_rounds=True,
    )
    return fetcher(*args, **kwargs)


def _get_numerai_prediction_fetcher(fname, name, has_rounds=False):
    """Return prediction fetch function for passed files"""

    def fetch_numerai_predictions(
        *,
        data_home=None,
        download_if_missing=True,
        return_y=False,
        as_frame=False,
        round_num=None,
    ):
        # Get round (if predictions support rounds)
        napi = NumerAPI()
        if round_num and not has_rounds:
            raise ValueError("`round_num` given for predictions without rounds.")
        if not round_num and has_rounds:
            round_num = napi.get_current_round()
            logger.info(f"Using current round={round_num}")

        # Get file locations
        data_home = get_data_home(data_home=data_home)
        filename = fname
        if not exists(data_home):
            makedirs(data_home)
        if round_num:
            suffix = "_" + str(round_num)
            filename_with_suffix = _add_filename_suffix(filename, suffix)
            filepath = "/".join([data_home, filename_with_suffix])
        else:
            filepath = "/".join([data_home, filename])

        # Download and read dataset
        if not exists(filepath):
            if not download_if_missing:
                raise IOError("Data not found and `download_if_missing` is False")

            logger.info(f"Downloading Numerai {name} predictions to {filepath}")
            napi.download_dataset("/".join(["v3", filename]), dest_path=filepath)

            df = pd.read_parquet(filepath)
            remove(filepath)
        else:
            df = pd.read_parquet(filepath)

        y = df.prediction
        id_ = df.index

        if not as_frame:
            # Convert to numpy
            y = y.to_numpy()
            id_ = id_.to_numpy()
            df = None

        if return_y:
            return y

        return Bunch(
            prediction=y,
            id=id_,
            DESCR=f"Numerai main tournament: {name} predictions",
            round_num=round_num,
            frame=df,
        )

    return fetch_numerai_predictions


def fetch_numerai_example_predictions(*args, **kwargs):
    """Load the Numerai example predictions.

    Parameters
    ----------
    data_home : str, default=None
        Specify another download and cache folder for the datasets. By
        default all data is stored in `~/scikit_learn_data` subfolders.

    download_if_missing : bool, default=True
        If False, raise an IOError if the data is not locally available
        instead of trying to download the data from the source bucket.

    keep : bool, default=False
        If True, does not remove the downloaded file from disk after
        reading it.

    return_y : bool, default=False.
        If True, returns `prediction` instead of a Bunch object.

    as_frame : bool, default=False
        If True, `prediction` and `id` are pandas Series. `frame` will
        be given.

    round_num : int, default=None
        Prediction round to download. If None, current round will be
        downloaded.

    Returns
    -------
    dataset : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        prediction : {ndarray, Series} of shape (n_samples,)
            When `as_frame=True`, `prediction` is a pandas Series.

        id : {ndarray, Series} of shape (n_samples,)
            `id` of each `prediction` row.
            When `as_frame=True`, `id` is a pandas Series.

        round_num : int
            Round number of the predictions.

        frame : DataFrame if `as_frame=True`
            Only present when `as_frame=True`. Pandas DataFrame with
            `prediction`.

        y : {ndarray, Series} of shape (n_samples,) if `return_y=True`
            Only present when `return_y=True`. When `as_frame=True`,
            `y` is a pandas Series.

    Notes
    -----
    Data changes weekly.
    """
    fetcher = _get_numerai_prediction_fetcher(
        "example_predictions.parquet",
        "example",
        has_rounds=True,
    )
    return fetcher(*args, **kwargs)


def fetch_numerai_example_validation_predictions(*args, **kwargs):
    """Load the Numerai example validation predictions.

    Parameters
    ----------
    data_home : str, default=None
        Specify another download and cache folder for the datasets. By
        default all data is stored in `~/scikit_learn_data` subfolders.

    download_if_missing : bool, default=True
        If False, raise an IOError if the data is not locally available
        instead of trying to download the data from the source bucket.

    keep : bool, default=False
        If True, does not remove the downloaded file from disk after
        reading it.

    return_y : bool, default=False.
        If True, returns `prediction` instead of a Bunch object.

    as_frame : bool, default=False
        If True, `prediction` and `id` are pandas Series. `frame` will
        be given.

    Returns
    -------
    dataset : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        prediction : {ndarray, Series} of shape (539658,)
            When `as_frame=True`, `prediction` is a pandas Series.

        id : {ndarray, Series} of shape (539658,)
            `id` of each `prediction` row.
            When `as_frame=True`, `id` is a pandas Series.

        frame : DataFrame if `as_frame=True`
            Only present when `as_frame=True`. Pandas DataFrame with
            `prediction`.

        y : {ndarray, Series} of shape (539658,) if `return_y=True`
            Only present when `return_y=True`. When `as_frame=True`,
            `y` is a pandas Series.
    """
    fetcher = _get_numerai_prediction_fetcher(
        "example_validation_predictions.parquet",
        "example validation",
    )
    return fetcher(*args, **kwargs)


def fetch_numerai_feature_metadata(
    *, data_home=None, download_if_missing=True, keep=False
):
    """Load the Numerai feature metadata.

    Parameters
    ----------
    data_home : str, default=None
        Specify another download and cache folder for the feature
        metadata. By default all data is stored in
        `~/scikit_learn_data` subfolders.

    download_if_missing : bool, default=True
        If False, raise an IOError if the data is not locally available
        instead of trying to download the data from the source bucket.

    keep : bool, default=False
        If True, does not remove the downloaded file from disk after
        reading it.

    Returns
    -------
    feature_metadata : dict
        Dictionary with the following keys.

        feature_sets : dict
            Dictionary containing lists of feature names as values.

        feature_stats : dict
            Dictionary with feature names as keys and a dictionary
            of different statistics about each feature as values.

    References
    ----------

    https://forum.numer.ai/t/october-2021-updates/4384

    """
    # Get file locations
    data_home = get_data_home(data_home=data_home)
    filename = "features.json"
    if not exists(data_home):
        makedirs(data_home)
    filepath = "/".join([data_home, filename])

    # Download and read dataset
    if not exists(filepath):
        if not download_if_missing:
            raise IOError("Data not found and `download_if_missing` is False")

        logger.info(f"Downloading Numerai feature metadata to {filepath}")
        napi = NumerAPI()
        napi.download_dataset("/".join(["v3", filename]), dest_path=filepath)

        feature_metadata_dict = _read_json_file(filepath)
        if not keep:
            remove(filepath)
    else:
        feature_metadata_dict = _read_json_file(filepath)

    return feature_metadata_dict


def submit_numerai_tournament(
    prediction,
    model_id=None,
    public_id=None,
    secret_key=None,
    data_home=None,
    keep=False,
    version=None,
):
    """Submit Numerai main tournament prediction for current round.

    Parameters
    ----------
    prediction : {list, Series}
        Predicted values. Requires same order as example predictions.

    model_id : str, default=None
        Target model UUID. Required for accounts with multiple models.
        See https://numer.ai/models

    public_id : str, default=None
        ID of an API key. Needs `Upload submissions` scope.
        See https://numer.ai/account -> AUTOMATION.

    secret_key : str, default=None
        Secret of an API key. Needs `Upload submissions` scope.
        See https://numer.ai/account -> AUTOMATION.

    data_home : str, default=None
        Specify another download and cache folder for the predictions.
        By default all data is stored in `~/scikit_learn_data`
        subfolders.

    keep : bool, default=False
        If True, does not remove the prediction csv file from disk
        after uploading it.
    """
    if version is not None:
        logger.warning("Argument 'version' is deprecated.")

    data_home = get_data_home(data_home=data_home)
    if not exists(data_home):
        makedirs(data_home)

    napi = NumerAPI(public_id=public_id, secret_key=secret_key)
    current_round_num = napi.get_current_round()
    example_predictions = fetch_numerai_example_predictions(
        data_home=data_home, as_frame=True, round_num=current_round_num
    )
    assert example_predictions.round_num == current_round_num

    # rank from 0 to 1 to meet upload requirements
    prediction = pd.Series(prediction).rank(pct=True).values

    # create csv of predictions
    prediction_df = example_predictions.frame
    prediction_df["prediction"] = prediction
    filename = f"predictions_{current_round_num}_{model_id}.csv"
    filepath = "/".join([data_home, filename])
    prediction_df.to_csv(filepath)

    # submit predictions
    napi.upload_predictions(filepath, model_id=model_id)

    # remove prediction file
    if not keep:
        remove(filepath)


def _get_feature_names(df: pd.DataFrame) -> List[str]:
    """Get list of `df`s feature column names."""
    return [c for c in df if c.startswith("feature_")]


def _get_target_names(df: pd.DataFrame) -> List[str]:
    """Get list of `df`s target column names."""
    return [c for c in df if c.startswith("target")]


def _get_dtype(is_int8: bool) -> str:
    """Return data type. Either int8 or float32."""
    if is_int8:
        return "int8"
    else:
        return "float32"


def _add_filename_suffix(filepath: str, suffix: str) -> str:
    """Insert suffix between current filename and extension."""
    filepath, file_extension = splitext(filepath)
    return filepath + suffix + file_extension


def _read_json_file(filepath: str) -> dict:
    """Return content of json file at filepath."""
    with open(filepath) as json_file:
        content = json.load(json_file)

    return content
