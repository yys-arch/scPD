"""
Utility functions for scpd.

Provides common utilities for data processing, normalization, and similarity computation.
"""

import numpy as np
from typing import Tuple, Optional


def normalize_to_unit(
    values: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
) -> Tuple[np.ndarray, float, float]:
    """
    Normalize values to [0, 1] range.
    
    Parameters
    ----------
    values : ndarray
        Values to normalize.
    vmin, vmax : float, optional
        Min/max for normalization. If None, computed from data.
        
    Returns
    -------
    normalized : ndarray
        Normalized values in [0, 1].
    vmin, vmax : float
        The min/max used for normalization.
    """
    values = np.asarray(values)
    
    if vmin is None:
        vmin = np.min(values)
    if vmax is None:
        vmax = np.max(values)
    
    if vmax > vmin:
        normalized = (values - vmin) / (vmax - vmin)
    else:
        normalized = np.zeros_like(values) + 0.5
    
    return np.clip(normalized, 0, 1), vmin, vmax


def bin_by_quantile(
    values: np.ndarray,
    n_bins: int = 6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin values by quantile into equal-sized groups.
    
    Parameters
    ----------
    values : ndarray of shape (n,)
        Values to bin.
    n_bins : int, default=6
        Number of bins.
        
    Returns
    -------
    bin_labels : ndarray of shape (n,)
        Integer bin labels (0 to n_bins-1).
    bin_edges : ndarray of shape (n_bins + 1,)
        Bin edges.
    """
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(values, quantiles)
    
    # Handle edge case where multiple edges are the same
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < n_bins + 1:
        bin_edges = np.linspace(np.min(values), np.max(values), n_bins + 1)
    
    bin_labels = np.digitize(values, bin_edges[1:-1])
    
    return bin_labels, bin_edges


def correlation_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Pearson correlation coefficient.
    
    Parameters
    ----------
    x, y : ndarray
        Arrays to correlate.
        
    Returns
    -------
    r : float
        Correlation coefficient.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    cov = np.mean((x - x_mean) * (y - y_mean))
    std_x = np.std(x)
    std_y = np.std(y)
    
    if std_x < 1e-10 or std_y < 1e-10:
        return 0.0
    
    return cov / (std_x * std_y)


def shape_similarity(
    f1: np.ndarray,
    f2: np.ndarray,
    normalize: bool = True
) -> float:
    """
    Compute shape similarity between two functions.
    
    Uses correlation after optional normalization.
    
    Parameters
    ----------
    f1, f2 : ndarray
        Function values on same grid.
    normalize : bool, default=True
        If True, normalize each to [0, 1] before comparing.
        
    Returns
    -------
    similarity : float
        Similarity score in [-1, 1].
        Returns 1.0 if both functions are nearly constant with similar values.
        Returns 0.0 if only one function is nearly constant.
    """
    f1 = np.asarray(f1).ravel()
    f2 = np.asarray(f2).ravel()
    
    # Check for constant functions
    std1, std2 = np.std(f1), np.std(f2)
    is_const1 = std1 < 1e-10
    is_const2 = std2 < 1e-10
    
    if is_const1 and is_const2:
        # Both constant - check if they have similar mean values
        mean_diff = abs(np.mean(f1) - np.mean(f2))
        scale = max(abs(np.mean(f1)), abs(np.mean(f2)), 1e-10)
        return 1.0 if mean_diff / scale < 0.5 else 0.5
    elif is_const1 or is_const2:
        # One constant, one not - low similarity
        return 0.0
    
    if normalize:
        f1, _, _ = normalize_to_unit(f1)
        f2, _, _ = normalize_to_unit(f2)
    
    return correlation_coefficient(f1, f2)

