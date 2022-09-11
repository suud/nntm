## 1.6.2 (2022-09-11)

### Fix

- use version prefix when downloading dataset

## 1.6.1 (2022-06-19)

### Fix

- remove submission version

## 1.6.0 (2021-12-05)

### Feat

- add function to submit tournament prediction

## 1.5.0 (2021-11-30)

### Feat

- add round_num attribute to datasets
- add metrics to public api

## 1.4.1 (2021-11-14)

### Fix

- set factory args
- don't pass eras to predict method
- pass eras to scorers on call
- convert ndarray to series

## 1.4.0 (2021-11-11)

### Feat

- add validation_curve
- add numerai correlation score
- add utility function to build cv

### Refactor

- simplify function

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
