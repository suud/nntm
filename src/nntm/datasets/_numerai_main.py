"""Numerai main tournament dataset.

The Numerai main tournament dataset is made of high quality financial data
that has been cleaned, regularized and obfuscated.

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
from os.path import exists
from os import makedirs, remove
from typing import List

import logging
import pandas as pd
from numerapi import NumerAPI
from sklearn.datasets import get_data_home
from sklearn.utils import Bunch

logger = logging.getLogger(__name__)


def fetch_numerai_training(
    *,
    data_home=None,
    download_if_missing=True,
    return_X_y=False,
    target="target",
    as_frame=False,
    columns=None,
    int8=True,
    na_value=None,
):
    """Load the Numerai training dataset.

    Parameters
    ----------
    data_home : str, default=None
        Specify another download and cache folder for the datasets. By default
        all data is stored in `~/scikit_learn_data` subfolders.

    download_if_missing : bool, default=True
        If False, raise an IOError if the data is not locally available
        instead of trying to download the data from the source bucket.

    return_X_y : bool, default=False.
        If True, returns `(data.data, data.target)` instead of a Bunch
        object. A custom target column can be selected through the
        `target` parameter.

    target : str, default="target"
        Target column to return as `y` when `return_X_y=True`.

    as_frame : bool, default=False
        If True, data is a pandas DataFrame, targets are pandas Series,
        info a pandas DataFrame and frame will be given.

    columns : list, default=None
        If not None, only these columns will be read from the file.
        `index`, `era` and `data_type` columns are always read.

    int8 : bool, default=True
        If True, the feature columns will use the `int8` data type
        instead of `float32`. Target columns are always `float32`.

    na_value : Any, default=None
        The value to use for missing feature values (NaNs) in the dataset.
        By default, `2` will be used when `int8=True` or `0.5`
        when `int8=False`.

    Returns
    -------
    dataset : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : {ndarray, DataFrame} of shape (2412105, 1050)
            Each row corresponding to the `feature_names` in order.
            When `as_frame=True`, `data` is a pandas DataFrame.

        target : {ndarray, Series} of shape (2412105,)
            When `as_frame=True`, `target` is a pandas Series.

        target_<name> : {ndarray, Series} of shape (2412105,)
            See `target_names` for available targets.
            When `as_frame=True`, `target_<name>` is a pandas Series.

        info : {ndarray, DataFrame} of shape (2412105, 3)
            Each row corresponding to `id`, `era`, `data_type` in order.
            `era` uses the `int` data type (as opposed to the original dataset
            where `era`s are strings).
            When `as_frame=True`, `info` is a pandas DataFrame.

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
            `era` uses the `int` data type (as opposed to the original dataset
            where `era`s are strings).

        (data, target) : tuple if `return_X_y=True`
            Only present when `return_X_y=True`
            `target` corresponds to the column set by the `target` attribute.

    Notes
    -----
    `target` corresponds to `target_nomi_20`.

    """
    # make sure required columns will be read
    if columns:
        columns = list(set(columns + ["era", "data_type"]))

    # get file locations
    data_home = get_data_home(data_home=data_home)
    if not exists(data_home):
        makedirs(data_home)
    filename = "numerai_training_data.parquet"
    if int8:
        filename = "numerai_training_data_int8.parquet"
    filepath = "/".join([data_home, filename])

    # download and read dataset
    if not exists(filepath):
        if not download_if_missing:
            raise IOError("Data not found and `download_if_missing` is False")

        logger.info(f"Downloading Numerai training data to {filepath}")
        napi = NumerAPI()
        napi.download_dataset(filename, dest_path=filepath)

        df = pd.read_parquet(filepath, columns=columns)
        remove(filepath)
    else:
        df = pd.read_parquet(filepath, columns=columns)

    # get feature and target columns
    feature_names = _get_feature_names(df)
    target_names = _get_target_names(df)

    # replace NaNs
    na_value = _get_na_value(na_value=na_value, is_int8=int8)
    dtype = _get_dtype(is_int8=int8)
    df[feature_names] = df[feature_names].fillna(na_value)

    # ensure expected data type for features
    df[feature_names] = df[feature_names].astype(dtype)

    # convert era's to int
    df["era"] = df["era"].astype(int)

    X = df[feature_names]
    target_dict = {tn: df[tn] for tn in target_names}
    info = pd.DataFrame(
        {
            "id": df.index,
            "era": df.era,
            "data_type": df.data_type,
        }
    )

    if not as_frame:
        # convert to numpy
        X = X.to_numpy(dtype=dtype)
        target_dict = {k: v.to_numpy() for k, v in target_dict.items()}
        info = info.to_numpy()
        df = None

    if return_X_y:
        return X, target_dict[target]

    return Bunch(
        data=X,
        **target_dict,
        info=info,
        feature_names=feature_names,
        target_names=target_names,
        int8=int8,
        DESCR="Numerai main tournament: training data",
        frame=df,
    )


def fetch_numerai_validation(
    *,
    data_home=None,
    download_if_missing=True,
    return_X_y=False,
    target="target",
    as_frame=False,
    columns=None,
    int8=True,
    na_value=None,
):
    """Load the Numerai validation dataset.

    Parameters
    ----------
    data_home : str, default=None
        Specify another download and cache folder for the datasets. By default
        all data is stored in `~/scikit_learn_data` subfolders.

    download_if_missing : bool, default=True
        If False, raise an IOError if the data is not locally available
        instead of trying to download the data from the source bucket.

    return_X_y : bool, default=False.
        If True, returns `(data.data, data.target)` instead of a Bunch
        object. A custom target column can be selected through the
        `target` parameter.

    target : str, default="target"
        Target column to return as `y` when `return_X_y=True`.

    as_frame : bool, default=False
        If True, data is a pandas DataFrame, targets are pandas Series,
        info a pandas DataFrame and frame will be given.

    columns : list, default=None
        If not None, only these columns will be read from the file.
        `index`, `era` and `data_type` columns are always read.

    int8 : bool, default=True
        If True, the feature columns will use the `int8` data type
        instead of `float32`. Target columns are always `float32`.

    na_value : Any, default=None
        The value to use for missing feature values (NaNs) in the dataset.
        By default, `2` will be used when `int8=True` or `0.5`
        when `int8=False`.

    Returns
    -------
    dataset : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : {ndarray, DataFrame} of shape (2412105, 1050)
            Each row corresponding to the `feature_names` in order.
            When `as_frame=True`, `data` is a pandas DataFrame.

        target : {ndarray, Series} of shape (2412105,)
            When `as_frame=True`, `target` is a pandas Series.

        target_<name> : {ndarray, Series} of shape (2412105,)
            See `target_names` for available targets.
            When `as_frame=True`, `target_<name>` is a pandas Series.

        info : {ndarray, DataFrame} of shape (2412105, 3)
            Each row corresponding to `id`, `era`, `data_type` in order.
            `era` uses the `int` data type (as opposed to the original dataset
            where `era`s are strings).
            When `as_frame=True`, `info` is a pandas DataFrame.

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
            `era` uses the `int` data type (as opposed to the original dataset
            where `era`s are strings).

        (data, target) : tuple if `return_X_y=True`
            Only present when `return_X_y=True`
            `target` corresponds to the column set by the `target` attribute.

    Notes
    -----
    `target` corresponds to `target_nomi_20`.

    """
    # make sure required columns will be read
    if columns:
        columns = list(set(columns + ["era", "data_type"]))

    # get file locations
    data_home = get_data_home(data_home=data_home)
    if not exists(data_home):
        makedirs(data_home)
    filename = "numerai_validation_data.parquet"
    if int8:
        filename = "numerai_validation_data_int8.parquet"
    filepath = "/".join([data_home, filename])

    # download and read dataset
    if not exists(filepath):
        if not download_if_missing:
            raise IOError("Data not found and `download_if_missing` is False")

        logger.info(f"Downloading Numerai validation data to {filepath}")
        napi = NumerAPI()
        napi.download_dataset(filename, dest_path=filepath)

        df = pd.read_parquet(filepath, columns=columns)
        remove(filepath)
    else:
        df = pd.read_parquet(filepath, columns=columns)

    # get feature and target columns
    feature_names = _get_feature_names(df)
    target_names = _get_target_names(df)

    # replace NaNs
    na_value = _get_na_value(na_value=na_value, is_int8=int8)
    dtype = _get_dtype(is_int8=int8)
    df[feature_names] = df[feature_names].fillna(na_value)

    # ensure expected data type for features
    df[feature_names] = df[feature_names].astype(dtype)

    # convert era's to int
    df["era"] = df["era"].astype(int)

    X = df[feature_names]
    target_dict = {tn: df[tn] for tn in target_names}
    info = pd.DataFrame(
        {
            "id": df.index,
            "era": df.era,
            "data_type": df.data_type,
        }
    )

    if not as_frame:
        # convert to numpy
        X = X.to_numpy(dtype=dtype)
        target_dict = {k: v.to_numpy() for k, v in target_dict.items()}
        info = info.to_numpy()
        df = None

    if return_X_y:
        return X, target_dict[target]

    return Bunch(
        data=X,
        **target_dict,
        info=info,
        feature_names=feature_names,
        target_names=target_names,
        int8=int8,
        DESCR="Numerai main tournament: validation data",
        frame=df,
    )


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


def _get_na_value(na_value, is_int8: bool):
    """Get na_value or acceptable default."""
    if na_value is not None:
        return na_value

    if is_int8:
        return 2
    else:
        return 0.5
