## 1.3.0 (2021-11-04)

### Feat

- add argument to keep downloaded file

## 1.2.2 (2021-11-04)

### Fix

- explicitly check if groups is None

## 1.2.1 (2021-11-04)

### Fix

- explicitly check if groups is None

## 1.2.0 (2021-10-30)

### Feat

- add constants for feature and target names

## 1.1.1 (2021-10-30)

### Fix

- add suffix only to filepath

## 1.1.0 (2021-10-30)

### Feat

- add fetcher for feature metadata
- add argument to fetch custom round

## 1.0.0 (2021-10-29)

### Feat

- add fetcher for validation predictions
- add fetcher for example predictions
- add fetcher for live data
- add fetcher for test data
- add fetcher for tournament data
- add targets attribute
- remove na_value argument
- use original data types
- return metadata as separate attributes

### BREAKING CHANGE

- remove support for NaN value replacement in fetchers.
- don't convert `era`s to int.
- replace the dataset's `info` attribute by `id`, `era`
and `data_type` attributes.

## 0.3.0 (2021-10-28)

### Feat

- add PurgedKFold cross-validator

### Perf

- fill NaNs only when necessary

## 0.2.0 (2021-10-08)

### Feat

- add fetcher for validation data

## 0.1.2 (2021-10-07)

### Fix

- support earlier python versions

## 0.1.1 (2021-10-07)

### Refactor

- remove icons

## 0.1.0 (2021-10-07)

### Feat

- add fetcher for training data
