[![PyPI version](https://img.shields.io/pypi/v/nntm.svg)](https://pypi.python.org/pypi/nntm/)
[![PyPI status](https://img.shields.io/pypi/status/nntm.svg)](https://pypi.python.org/pypi/nntm/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/nntm.svg)](https://pypi.python.org/pypi/nntm/)
[![PyPI license](https://img.shields.io/pypi/l/nntm.svg)](https://pypi.python.org/pypi/nntm/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# nntm
A set of modules for the [Numerai tournament](https://numer.ai/tournament).

## Installation
```sh
pip install nntm==1.4.1
```

## Usage
```python
from nntm.datasets import (
    fetch_numerai_training,
    fetch_numerai_tournament,
    COLUMN_NAMES_SMALL,
)
from sklearn.linear_model import LinearRegression

# Leave out some columns to save RAM
columns = COLUMN_NAMES_SMALL

# Fit
X_train, y_train = fetch_numerai_training(return_X_y=True, columns=columns)
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
X_tourn, _ = fetch_numerai_tournament(return_X_y=True, columns=columns)
y_pred = model.predict(X_tourn)
```