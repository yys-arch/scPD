"""
PDE discretization and right-hand side computation.

Implements finite volume discretization of density dynamics PDE:
    ∂_t u = ∂_s(D ∂_s u) - ∂_s(v u) + g u

with no-flux boundary conditions: J(0,t) = J(1,t) = 0
where J = -D ∂_s u + v u is the flux.
"""

import numpy as np
from typing import Optional, Tuple
from .grid import interpolate_to_faces, upwind_values


def compute_flux_at_faces(
    u: np.ndarray,
    D: np.ndarray,
    v: np.ndarray,
    ds: float
) -> np.ndarray:
    """
    Compute flux at all faces (including boundaries).
    
    Parameters
    ----------
    u : ndarray of shape (n_grid,)
        Density at cell centers.
    D : ndarray of shape (n_grid,)
        Diffusion coefficient at cell centers.
    v : ndarray of shape (n_grid,)
        Drift velocity at cell centers.
    ds : float
        Grid spacing.
        
    Returns
    -------
    J : ndarray of shape (n_grid + 1,)
        Flux at all faces. Boundary fluxes (J[0] and J[-1]) are set to 0
        for no-flux boundary conditions.
        
    Notes
    -----
    Face flux formula:
        J_{i+1/2} = -D_{i+1/2} * (u_{i+1} - u_i) / ds + v_{i+1/2} * u_upwind
    
    where:
        - D_{i+1/2} = (D_i + D_{i+1}) / 2 (arithmetic average)
        - u_upwind = u_i if v_{i+1/2} >= 0, else u_{i+1} (first-order upwinding)
    """
    n_grid = len(u)
    J = np.zeros(n_grid + 1)
    
    # Interior faces: indices 1 to n_grid - 1
    # D and v at interior faces (averaged from cell centers)
    D_faces = interpolate_to_faces(D)  # shape (n_grid - 1,)
    v_faces = interpolate_to_faces(v)  # shape (n_grid - 1,)
    
    # Upwind u values at interior faces
    u_upwind = upwind_values(u, v_faces)  # shape (n_grid - 1,)
    
    # Diffusive flux: -D * du/ds
    du_ds = (u[1:] - u[:-1]) / ds  # shape (n_grid - 1,)
    diffusive_flux = -D_faces * du_ds
    
    # Advective flux: v * u_upwind
    advective_flux = v_faces * u_upwind
    
    # Total interior flux
    J[1:-1] = diffusive_flux + advective_flux
    
    # Boundary conditions: no-flux
    # J[0] = 0 (left boundary)
    # J[-1] = 0 (right boundary)
    # Already initialized to 0
    
    return J


def pde_rhs(
    u: np.ndarray,
    D: np.ndarray,
    v: np.ndarray,
    g: np.ndarray,
    ds: float,
    stabilize_boundary: bool = False,
    clip_to_nonnegative: bool = False
) -> np.ndarray:
    """
    Compute the right-hand side of the discretized PDE.
    
    Parameters
    ----------
    u : ndarray of shape (n_grid,)
        Current density at cell centers.
    D : ndarray of shape (n_grid,)
        Diffusion coefficient at cell centers.
    v : ndarray of shape (n_grid,)
        Drift velocity at cell centers.
    g : ndarray of shape (n_grid,)
        Net growth rate at cell centers.
    ds : float
        Grid spacing.
    stabilize_boundary : bool, default=False
        If True, smoothly reduce v(s) near s=1 to reduce boundary instabilities.
        This is an optional feature that can help with numerical stability.
    clip_to_nonnegative : bool, default=False
        If True, clip negative u values to 0 before computing RHS.
        Trade-off: can prevent oscillations but may affect mass conservation.
        
    Returns
    -------
    du_dt : ndarray of shape (n_grid,)
        Time derivative of density at each cell center.
        
    Notes
    -----
    The discretized PDE in finite volume form:
        du_i/dt = -(J_{i+1/2} - J_{i-1/2}) / ds + g_i * u_i
    
    where J is the flux computed by compute_flux_at_faces.
    """
    if clip_to_nonnegative:
        u = np.maximum(u, 0.0)
    
    v_eff = v.copy()
    if stabilize_boundary:
        # Smooth reduction of v near s=1
        # Use sigmoid-like decay for s > 0.9
        n_grid = len(u)
        s_centers = (np.arange(n_grid) + 0.5) / n_grid
        decay_factor = np.where(
            s_centers > 0.9,
            np.exp(-10 * (s_centers - 0.9)),
            1.0
        )
        v_eff = v * decay_factor
    
    # Compute flux at all faces
    J = compute_flux_at_faces(u, D, v_eff, ds)
    
    # Finite volume update: du/dt = -div(J) + g*u
    # div(J) = (J_{i+1/2} - J_{i-1/2}) / ds
    div_J = (J[1:] - J[:-1]) / ds
    
    # Growth/death term
    growth = g * u
    
    # Total RHS
    du_dt = -div_J + growth
    
    return du_dt


def compute_total_mass(u: np.ndarray, ds: float) -> float:
    """
    Compute total mass (integral of density over domain).
    
    Parameters
    ----------
    u : ndarray of shape (n_grid,)
        Density at cell centers.
    ds : float
        Grid spacing.
        
    Returns
    -------
    mass : float
        Total mass: ∫_0^1 u(s) ds ≈ Σ u_i * ds
    """
    return np.sum(u) * ds


def density_to_pdf(u: np.ndarray, ds: float) -> np.ndarray:
    """
    Normalize density to probability density function.
    
    Parameters
    ----------
    u : ndarray of shape (n_grid,)
        Density at cell centers.
    ds : float
        Grid spacing.
        
    Returns
    -------
    p : ndarray of shape (n_grid,)
        Probability density: p(s) = u(s) / ∫u ds
    """
    total_mass = compute_total_mass(u, ds)
    if total_mass <= 0:
        # Return uniform if mass is zero or negative (shouldn't happen normally)
        return np.ones_like(u) / len(u) / ds
    return u / total_mass


def pdf_to_cdf(p: np.ndarray, ds: float) -> np.ndarray:
    """
    Convert probability density to cumulative distribution function.
    
    Parameters
    ----------
    p : ndarray of shape (n_grid,)
        Probability density at cell centers.
    ds : float
        Grid spacing.
        
    Returns
    -------
    cdf : ndarray of shape (n_grid,)
        CDF values at cell centers: CDF(s_i) = ∫_0^{s_i} p(s') ds'
        
    Notes
    -----
    Uses cumulative trapezoidal integration, adjusted so that:
    - CDF[0] ≈ p[0] * ds / 2 (half first cell)
    - CDF[-1] ≈ 1 (should be close to 1)
    """
    # Cumulative sum gives integral from 0 to s_i + ds/2
    cumsum = np.cumsum(p) * ds
    # Adjust to get value at cell center (subtract half of current cell)
    cdf = cumsum - p * ds / 2
    # Ensure starts near 0 and ends near 1
    cdf = np.clip(cdf, 0.0, 1.0)
    return cdf

