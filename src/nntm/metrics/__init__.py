"""
The `nntm.metrics` module includes score functions, performance metrics,
pairwise metrics and distance computations.
"""

from ._numerai import corr_score

__all__ = ["corr_score"]
