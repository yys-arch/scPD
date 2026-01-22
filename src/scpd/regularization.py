"""
Roughness penalty (regularization) for spline-parameterized functions.

Implements integrated squared second derivative penalty:
    Penalty = ∫(f''(s))^2 ds ≈ beta^T @ R @ beta

where R is the roughness matrix computed from spline basis functions.
"""

import numpy as np
from typing import Optional
from .spline import SplineBasis


def compute_roughness_matrix(
    spline_basis: SplineBasis,
    n_dense: int = 500
) -> np.ndarray:
    """
    Compute the roughness penalty matrix R.
    
    For a function f(s) = B(s) @ beta, the roughness penalty is:
        ∫(f''(s))^2 ds ≈ beta^T @ R @ beta
    
    Parameters
    ----------
    spline_basis : SplineBasis
        Spline basis object.
    n_dense : int, default=500
        Number of points for dense grid approximation.
        
    Returns
    -------
    R : ndarray of shape (df, df)
        Roughness penalty matrix.
        
    Notes
    -----
    R is computed as:
        R = (B'')^T @ (B'') * ds
    
    where B'' is the second derivative of the basis functions approximated
    by finite differences on a dense grid.
    """
    # Get second derivatives on dense grid
    B_dd, ds = spline_basis.second_derivative_matrix(n_dense=n_dense)
    
    # R = B''T @ B'' * ds (integral approximation)
    R = B_dd.T @ B_dd * ds
    
    return R


def roughness_penalty(
    theta: np.ndarray,
    R: np.ndarray,
    df: int,
    mode: str = "with_population"
) -> float:
    """
    Compute total roughness penalty for all parameters.
    
    Parameters
    ----------
    theta : ndarray
        Parameter vector [beta_D, beta_v, beta_g] or [beta_D, beta_v].
    R : ndarray of shape (df, df)
        Roughness penalty matrix.
    df : int
        Degrees of freedom per function.
    mode : str, default="with_population"
        Fitting mode ("with_population" or "distribution_only").
        
    Returns
    -------
    penalty : float
        Total penalty: β_D^T R β_D + β_v^T R β_v + β_g^T R β_g
        (g term omitted if mode="distribution_only")
    """
    beta_D = theta[:df]
    beta_v = theta[df:2*df]
    
    penalty_D = beta_D @ R @ beta_D
    penalty_v = beta_v @ R @ beta_v
    
    total = penalty_D + penalty_v
    
    if mode == "with_population":
        beta_g = theta[2*df:3*df]
        penalty_g = beta_g @ R @ beta_g
        total += penalty_g
    
    return total


def roughness_penalty_gradient(
    theta: np.ndarray,
    R: np.ndarray,
    df: int,
    mode: str = "with_population"
) -> np.ndarray:
    """
    Compute gradient of roughness penalty with respect to parameters.
    
    Parameters
    ----------
    theta : ndarray
        Parameter vector.
    R : ndarray of shape (df, df)
        Roughness penalty matrix.
    df : int
        Degrees of freedom per function.
    mode : str
        Fitting mode.
        
    Returns
    -------
    grad : ndarray
        Gradient of penalty w.r.t. theta.
        d/d(beta) [beta^T R beta] = 2 R beta
    """
    beta_D = theta[:df]
    beta_v = theta[df:2*df]
    
    grad_D = 2 * R @ beta_D
    grad_v = 2 * R @ beta_v
    
    if mode == "with_population":
        beta_g = theta[2*df:3*df]
        grad_g = 2 * R @ beta_g
        return np.concatenate([grad_D, grad_v, grad_g])
    else:
        return np.concatenate([grad_D, grad_v])

