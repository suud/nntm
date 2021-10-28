"""Test the split module"""
import secrets
import pytest
import numpy as np
import pandas as pd
from nntm.model_selection import PurgedKFold

ERAS_TOTAL = 100
SAMPLES_PER_ERA = 3
SAMPLES_TOTAL = ERAS_TOTAL * SAMPLES_PER_ERA
eras = [str(i + 1) for i in range(ERAS_TOTAL) for _ in range(SAMPLES_PER_ERA)]
feat_a = [1 for _ in range(SAMPLES_TOTAL)]
feat_b = [3 for _ in range(SAMPLES_TOTAL)]
unique_ids = [secrets.token_hex(nbytes=8) for _ in range(SAMPLES_TOTAL)]
X_np = np.array([feat_a, feat_b]).T
y_np = np.array([0.5 for _ in range(SAMPLES_TOTAL)])
X_df = pd.DataFrame(
    {
        "id": unique_ids,
        "era": eras,
        "feature_a": feat_a,
        "feature_b": feat_b,
    }
).set_index("id")
y_df = pd.Series([0.5 for _ in range(SAMPLES_TOTAL)])


def test_purgedkfold_params():
    # Check defaults
    pkfold = PurgedKFold()
    assert str(pkfold) == "PurgedKFold(embargo=None, n_splits=5, target_days=20)"
    assert pkfold.get_n_splits() == 5
    # Let's use custom values
    pkfold = PurgedKFold(n_splits=9, target_days=60)
    assert str(pkfold) == "PurgedKFold(embargo=None, n_splits=9, target_days=60)"
    assert pkfold.get_n_splits() == 9
    # n_splits has to be an integer > 1
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=5.2)
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=1)
    # target_days has to be and integer and multiple of 5
    with pytest.raises(ValueError):
        PurgedKFold(target_days=60.0)
    with pytest.raises(ValueError):
        PurgedKFold(target_days=19)
    # embargo has to be a float between 0.0 and 1.0
    with pytest.raises(ValueError):
        PurgedKFold(embargo=5)
    with pytest.raises(ValueError):
        PurgedKFold(embargo=1.3)


def test_purgedkfold_np():
    # When X is a numpy array, the split method requires a groups
    # parameter to know what has to be purged
    pkfold = PurgedKFold()
    with pytest.raises(ValueError):
        list(pkfold.split(X=X_np))

    # Test number of purged samples
    check_purged_splits(X_np, y_np, eras)


def test_purgedkfold_df():
    # When X is a pandas dataframe that has no "era" column, the split
    # method requires a group parameter to know what has to be purged
    pkfold = PurgedKFold()
    X_df_without_era_col = X_df.drop(columns="era")
    with pytest.raises(ValueError):
        list(pkfold.split(X=X_df_without_era_col))

    # Test number of purged samples
    check_purged_splits(X_df, y_df, None)


def test_purgedkfold_embargo():
    eras_int = [int(era) for era in eras]
    era_count = len(set(eras_int))
    era_max = max(eras_int)

    for target_days, purge_n_eras in [(20, 3), (60, 11)]:
        for embargo in [0.01, 0.05, 0.1, 0.2, 0.5]:
            for n_splits in [2, 5, 10]:
                pkfold_embargo = PurgedKFold(
                    n_splits=n_splits, target_days=target_days, embargo=embargo
                )
                pkfold = PurgedKFold(n_splits=n_splits, target_days=target_days)
                for (train_idx_embargo, test_idx_embargo), (train_idx, test_idx) in zip(
                    pkfold_embargo.split(X=X_df), pkfold.split(X=X_df)
                ):
                    # Test data is not affected
                    assert np.array_equal(test_idx_embargo, test_idx)

                    # Embargo applies only after test set
                    test_eras = get_int_era_set(X_df, test_idx)
                    if era_max in test_eras:
                        # No embargo if test set is the last split
                        assert np.array_equal(train_idx_embargo, train_idx)
                    else:
                        # Make sure the right amount of eras has been left out
                        train_eras = get_int_era_set(X_df, train_idx)
                        emb_train_eras = get_int_era_set(X_df, train_idx_embargo)
                        # Embargo is given as a percentage of all eras
                        embargo_only = int(round(embargo * era_count))
                        # Embargo overlaps with purge
                        embargo_only = max(0, embargo_only - purge_n_eras)
                        # Embargo can only remove as much as exists
                        trailing_eras = [e for e in train_eras if e > max(test_eras)]
                        embargo_only = min(len(trailing_eras), embargo_only)

                        assert len(train_eras) == len(emb_train_eras) + embargo_only


def check_purged_splits(X, y, groups):
    """Make sure the correct amount of samples gets purged."""
    n_samples = len(X)

    # https://forum.numer.ai/t/super-massive-data-release-deep-dive/4053
    # You need to sample every 4th era to get non-overlapping eras
    # with the 20 day targets, but every 12th era to get
    # non-overlapping eras with the 60 day targets
    for target_days, purge_n_eras in [(20, 3), (60, 11)]:
        for n_splits in [2, 5, 10, 99]:
            # Nothing should be purged from the test set
            expected_test_set_length = np.full(
                n_splits, n_samples // n_splits, dtype=int
            )
            expected_test_set_length[: n_samples % n_splits] += 1

            pkfold = PurgedKFold(n_splits=n_splits, target_days=target_days)
            for (train_idx, test_idx), test_set_length in zip(
                pkfold.split(X=X, y=y, groups=groups), expected_test_set_length
            ):
                assert test_set_length > 0
                assert len(test_idx) == test_set_length

                test_era_set = get_int_era_set(X, test_idx, groups)
                test_era_min, test_era_max = min(test_era_set), max(test_era_set)
                train_era_set = get_int_era_set(X, train_idx, groups)

                # There is no intersection between test and train eras
                assert test_era_set.intersection(train_era_set) == set()

                # Eras immediately before and after test set were purged
                for era in range(test_era_min - purge_n_eras, test_era_min + 1):
                    assert era not in train_era_set
                for era in range(test_era_max, test_era_max + purge_n_eras + 1):
                    assert era not in train_era_set

                # Not too many eras were purged
                eras_left_count = len(test_era_set.union(train_era_set))
                eras_left_expected = ERAS_TOTAL - (purge_n_eras * 2)
                assert eras_left_count >= eras_left_expected


def get_int_era_set(X, idx, groups=None):
    """Return set of eras from samples under idx.
    Eras are converted to int.
    """
    if groups:
        groups = np.array(groups)
        era_list = groups[idx]
    else:
        X_sub = X.iloc[idx]
        era_list = X_sub.era

    return set([int(era) for era in era_list])
