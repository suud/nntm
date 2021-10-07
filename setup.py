#! /usr/bin/env python
"""Install packages as defined in this file into the Python environment."""
from setuptools import setup, find_packages

# The version of this tool is based on the following steps:
# https://packaging.python.org/guides/single-sourcing-package-version/
VERSION = {}

with open("./nntm/__init__.py") as fp:
    # pylint: disable=W0122
    exec(fp.read(), VERSION)

setup(
    name="nntm",
    author="Timo Sutterer",
    author_email="hi@timo-sutterer.de",
    url="https://github.com/suud/nntm",
    description="A set of python modules for the Numerai competition",
    version=VERSION.get("__version__", "0.0.0"),
    packages=find_packages(where=".", exclude=["tests"]),
    install_requires=[
        "numerapi>=2.9.0",
        "pandas>=1.3.3",
        "pyarrow>=5.0.0",
        "scikit-learn>=1.0",
        "setuptools>=45.0",
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Intended Audience :: Science/Research",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
    ],
)
