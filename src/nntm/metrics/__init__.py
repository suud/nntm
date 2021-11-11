"""
The `nntm.metrics` module includes score functions, performance metrics,
pairwise metrics and distance computations.
"""

from ._numerai import corr_score
from ._scorer import get_scorer, check_scoring

__all__ = ["corr_score", "get_scorer", "check_scoring"]
