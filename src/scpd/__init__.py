"""
SCPD: Single-Cell Pseudodynamics

Fit density dynamics from 1D state-coordinate snapshots of single-cell data.
Estimates diffusion D(s), drift v(s), and net growth g(s) along normalized state coordinate s ∈ [0,1].
"""

__version__ = "0.1.0"

from .preprocess import prepare_inputs, find_robust_root, compute_normalized_pseudotime
from .results import PreparedData, PseudodynamicsResult, LandmarkInfo, Diagnostics
from .solver import PseudodynamicsModel

__all__ = [
    "__version__",
    "prepare_inputs",
    "find_robust_root",
    "compute_normalized_pseudotime",
    "PreparedData",
    "PseudodynamicsResult",
    "PseudodynamicsModel",
    "LandmarkInfo",
    "Diagnostics",
]

