"""
Likelihood computation and empirical cumulative distribution function (ECDF) / Area distance calculation.

Implements observation model comparing model CDF to empirical CDF,
supporting both cell-level and landmark-weighted computation.
"""

import numpy as np
from typing import Optional, List, Tuple
from .pde import density_to_pdf, pdf_to_cdf


def compute_ecdf(
    s_values: np.ndarray,
    s_grid: np.ndarray
) -> np.ndarray:
    """
    Compute empirical cumulative distribution function at grid points (vectorized).
    
    Parameters
    ----------
    s_values : ndarray of shape (n_samples,)
        Sample values.
    s_grid : ndarray of shape (n_grid,)
        Grid points where to evaluate ECDF.
        
    Returns
    -------
    ecdf : ndarray of shape (n_grid,)
        Empirical CDF values at grid points.
        
    Notes
    -----
    ECDF(s) = (# samples <= s) / n_samples
    Uses numpy's searchsorted for efficient vectorized computation.
    """
    n_samples = len(s_values)
    if n_samples == 0:
        return np.zeros_like(s_grid)
    
    sorted_s = np.sort(s_values)
    indices = np.searchsorted(sorted_s, s_grid, side='right')
    ecdf = indices / n_samples
    return ecdf


def compute_weighted_ecdf(
    landmark_s: np.ndarray,
    weights: np.ndarray,
    s_grid: np.ndarray
) -> np.ndarray:
    """
    Compute weighted ECDF using landmark representatives (vectorized).
    
    Parameters
    ----------
    landmark_s : ndarray of shape (n_landmarks,)
        Representative s value for each landmark cluster.
    weights : ndarray of shape (n_landmarks,)
        Weight (count) of cells in each cluster for this time point.
    s_grid : ndarray of shape (n_grid,)
        Grid points where to evaluate ECDF.
        
    Returns
    -------
    ecdf : ndarray of shape (n_grid,)
        Weighted ECDF values at grid points.
        
    Notes
    -----
    ECDF_k(s) = Σ_{c: s_c <= s} w_c / Σ_c w_c
    """
    total_weight = np.sum(weights)
    if total_weight == 0:
        return np.zeros_like(s_grid)
    
    # Sort landmarks by s
    sorted_idx = np.argsort(landmark_s)
    sorted_s = landmark_s[sorted_idx]
    sorted_w = weights[sorted_idx]
    
    # Cumulative weights
    cumsum_w = np.cumsum(sorted_w)
    
    # Vectorized: use searchsorted
    # For each s in s_grid, find how many sorted_s values are <= s
    indices = np.searchsorted(sorted_s, s_grid, side='right')
    
    # Get cumulative weight at those indices
    # indices=0 means no landmarks <= s, so ecdf=0
    ecdf = np.zeros_like(s_grid, dtype=float)
    mask = indices > 0
    ecdf[mask] = cumsum_w[indices[mask] - 1] / total_weight
    
    return ecdf


def compute_a_distance(
    cdf_model: np.ndarray,
    ecdf: np.ndarray,
    ds: float
) -> float:
    """
    Compute area distance between model CDF and empirical CDF.
    
    Parameters
    ----------
    cdf_model : ndarray of shape (n_grid,)
        Model CDF values.
    ecdf : ndarray of shape (n_grid,)
        Empirical CDF values.
    ds : float
        Grid spacing.
        
    Returns
    -------
    A : float
        Area distance: ∫|CDF_model(s) - ECDF(s)| ds
    """
    diff = np.abs(cdf_model - ecdf)
    A = np.sum(diff) * ds
    return A


def estimate_sigma_A_bootstrap(
    s_values: np.ndarray,
    s_grid: np.ndarray,
    ds: float,
    n_bootstrap: int = 100,
    random_state: Optional[int] = None
) -> float:
    """
    Estimate σ_A using bootstrap resampling.
    
    Parameters
    ----------
    s_values : ndarray of shape (n_samples,)
        Sample s values.
    s_grid : ndarray of shape (n_grid,)
        Grid points.
    ds : float
        Grid spacing.
    n_bootstrap : int, default=100
        Number of bootstrap samples.
    random_state : int, optional
        Random state for reproducibility.
        
    Returns
    -------
    sigma_A : float
        Estimated standard deviation of A-distance under resampling.
    """
    if len(s_values) == 0:
        return 1.0  # Default fallback
    
    rng = np.random.default_rng(random_state)
    n_samples = len(s_values)
    
    # Compute reference ECDF
    ecdf_ref = compute_ecdf(s_values, s_grid)
    
    # Bootstrap samples
    A_values = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        s_boot = s_values[idx]
        ecdf_boot = compute_ecdf(s_boot, s_grid)
        A = compute_a_distance(ecdf_ref, ecdf_boot, ds)
        A_values.append(A)
    
    sigma_A = np.std(A_values)
    # Ensure non-zero
    if sigma_A < 1e-10:
        sigma_A = 0.01  # Minimum sigma
    
    return sigma_A


def estimate_sigma_A_landmark_bootstrap(
    landmark_s: np.ndarray,
    weights: np.ndarray,
    s_grid: np.ndarray,
    ds: float,
    n_bootstrap: int = 100,
    random_state: Optional[int] = None
) -> float:
    """
    Estimate σ_A using multinomial resampling of landmark weights.
    
    Parameters
    ----------
    landmark_s : ndarray of shape (n_landmarks,)
        Representative s for each landmark.
    weights : ndarray of shape (n_landmarks,)
        Weights (cell counts) for each landmark.
    s_grid : ndarray of shape (n_grid,)
        Grid points.
    ds : float
        Grid spacing.
    n_bootstrap : int, default=100
        Number of bootstrap samples.
    random_state : int, optional
        Random state.
        
    Returns
    -------
    sigma_A : float
        Estimated σ_A.
        
    Notes
    -----
    Uses multinomial resampling: w' ~ Multinomial(W, p) where p_c = w_c / W
    """
    total_weight = np.sum(weights)
    if total_weight == 0:
        return 1.0
    
    rng = np.random.default_rng(random_state)
    
    # Reference ECDF
    ecdf_ref = compute_weighted_ecdf(landmark_s, weights, s_grid)
    
    # Multinomial probabilities
    p = weights / total_weight
    W = int(total_weight)
    
    A_values = []
    for _ in range(n_bootstrap):
        # Multinomial resampling
        w_boot = rng.multinomial(W, p).astype(float)
        ecdf_boot = compute_weighted_ecdf(landmark_s, w_boot, s_grid)
        A = compute_a_distance(ecdf_ref, ecdf_boot, ds)
        A_values.append(A)
    
    sigma_A = np.std(A_values)
    if sigma_A < 1e-10:
        sigma_A = 0.01
    
    return sigma_A


def estimate_sigma_A_replicate(
    s_values_list: List[np.ndarray],
    s_grid: np.ndarray,
    ds: float
) -> float:
    """
    Estimate σ_A from replicate variation.
    
    Parameters
    ----------
    s_values_list : list of ndarrays
        List of s_values arrays, one per replicate.
    s_grid : ndarray of shape (n_grid,)
        Grid points.
    ds : float
        Grid spacing.
        
    Returns
    -------
    sigma_A : float
        Estimated σ_A from replicate variation.
    """
    if len(s_values_list) < 2:
        # Can't estimate from single replicate
        return None
    
    # Compute ECDF for each replicate
    ecdfs = [compute_ecdf(s, s_grid) for s in s_values_list]
    
    # Compute pairwise A-distances
    n_rep = len(ecdfs)
    A_values = []
    for i in range(n_rep):
        for j in range(i + 1, n_rep):
            A = compute_a_distance(ecdfs[i], ecdfs[j], ds)
            A_values.append(A)
    
    sigma_A = np.std(A_values) if len(A_values) > 1 else np.mean(A_values)
    if sigma_A < 1e-10:
        sigma_A = 0.01
    
    return sigma_A


def compute_nll_cdf(
    A_values: np.ndarray,
    sigma_A_values: np.ndarray,
    mu_A: float = 0.0
) -> float:
    """
    Compute negative log-likelihood for CDF matching.
    
    Parameters
    ----------
    A_values : ndarray of shape (n_times,)
        A-distance for each time point.
    sigma_A_values : ndarray of shape (n_times,)
        Estimated σ_A for each time point.
    mu_A : float, default=0.0
        Expected A-distance (typically 0).
        
    Returns
    -------
    nll : float
        NLL_CDF = Σ_k [ (A_k - μ_A)^2 / (2 σ_A,k^2) + log σ_A,k ]
    """
    z = (A_values - mu_A) / sigma_A_values
    nll = 0.5 * np.sum(z ** 2) + np.sum(np.log(sigma_A_values))
    return nll


def compute_nll_population(
    N_model: np.ndarray,
    N_obs: np.ndarray,
    sigma_N: np.ndarray
) -> float:
    """
    Compute negative log-likelihood for population size matching.
    
    Parameters
    ----------
    N_model : ndarray of shape (n_times,)
        Model-predicted population at each time.
    N_obs : ndarray of shape (n_times,)
        Observed population at each time.
    sigma_N : ndarray of shape (n_times,)
        Uncertainty in N_obs.
        
    Returns
    -------
    nll : float
        NLL_N = Σ_k [ (N_model,k - N_obs,k)^2 / (2 σ_N,k^2) + log σ_N,k ]
    """
    z = (N_model - N_obs) / sigma_N
    nll = 0.5 * np.sum(z ** 2) + np.sum(np.log(sigma_N))
    return nll

