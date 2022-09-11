[![PyPI version](https://img.shields.io/pypi/v/nntm.svg)](https://pypi.python.org/pypi/nntm/)
[![PyPI status](https://img.shields.io/pypi/status/nntm.svg)](https://pypi.python.org/pypi/nntm/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/nntm.svg)](https://pypi.python.org/pypi/nntm/)
[![PyPI license](https://img.shields.io/pypi/l/nntm.svg)](https://pypi.python.org/pypi/nntm/)
[![Documentation Status](https://img.shields.io/readthedocs/nntm)](https://nntm.readthedocs.io/en/latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Warning**
>
> nntm can only be used with the **v3** dataset for now

# nntm
A set of modules for the [Numerai tournament](https://numer.ai/tournament).

## Installation
```sh
pip install nntm==1.6.2
```

## Usage
```python
from getpass import getpass
from nntm.datasets import (
    fetch_numerai_training,
    fetch_numerai_tournament,
    submit_numerai_tournament,
    COLUMN_NAMES_SMALL,
)
from sklearn.linear_model import LinearRegression

# Leave some columns out to save RAM
columns = COLUMN_NAMES_SMALL

# Fit
X_train, y_train = fetch_numerai_training(return_X_y=True, columns=columns)
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
X_tourn, _ = fetch_numerai_tournament(return_X_y=True, columns=columns)
y_pred = model.predict(X_tourn)

# Submit
model_id = input("Model ID (numer.ai/models):")
public_id = input("API Key Public ID (numer.ai/account):")
secret_key = getpass("API Key Secret (numer.ai/account):")
submit_numerai_tournament(
    y_pred, model_id=model_id, public_id=public_id, secret_key=secret_key
)
```

## Development
### Run Tests
```sh
pytest
```

### Docs
```sh
apt-get install -y python3-sphinx
pip install -r docs/requirements.txt
pip install .
cd docs
```
#### Generate api documentation from docstrings

```sh
sphinx-apidoc -f -o source/ ../src/nntm/
```

#### Build html documentation
```sh
make html
```
