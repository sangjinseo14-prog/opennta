"""MLE-based size-distribution reconstruction methods for NTA.

Both distributors maximize the same Gamma log-likelihood; they differ only in
how f(d) is represented.

* :class:`FTLADistributor` -- parametric family (Saveyn et al. 2010).
* :class:`IterativeDistributor` -- non-parametric bin weights (Walker 2012).
"""

from .families import DistributionFamily, LognormalFamily, available_families, get_family
from .ftla_distributor import FTLADistributor
from .iterative_distributor import IterativeDistributor
from .likelihood import (
    chi_squared,
    em_iterate,
    em_step,
    log_likelihood,
    log_prob_matrix,
)
from .track_stats import TrackStats, extract_track_stats
from .types import PhysicalParams

__all__ = [
    "DistributionFamily",
    "FTLADistributor",
    "IterativeDistributor",
    "LognormalFamily",
    "PhysicalParams",
    "TrackStats",
    "available_families",
    "chi_squared",
    "em_iterate",
    "em_step",
    "extract_track_stats",
    "get_family",
    "log_likelihood",
    "log_prob_matrix",
]
