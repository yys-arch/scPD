"""
Grid utilities for spatial discretization.

Provides uniform grid generation and related utility functions
for finite volume discretization of density dynamics PDE on s ∈ [0,1].
"""

import numpy as np
from typing import Tuple


def create_grid(n_grid: int = 200) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Create uniform grid on [0, 1] for finite volume discretization.
    
    Parameters
    ----------
    n_grid : int, default=200
        Number of grid cells.
        
    Returns
    -------
    s_centers : ndarray of shape (n_grid,)
        Cell center positions.
    s_faces : ndarray of shape (n_grid + 1,)
        Cell face positions (boundaries between cells).
    ds : float
        Grid spacing (uniform).
        
    Notes
    -----
    Grid construction places cell faces at:
        s_faces = [0, ds, 2*ds, ..., 1]
    and cell centers at:
        s_centers = [ds/2, 3*ds/2, ..., 1 - ds/2]
    """
    ds = 1.0 / n_grid
    s_faces = np.linspace(0.0, 1.0, n_grid + 1)
    s_centers = 0.5 * (s_faces[:-1] + s_faces[1:])
    return s_centers, s_faces, ds


def interpolate_to_faces(values_at_centers: np.ndarray) -> np.ndarray:
    """
    Interpolate cell center values to face values using arithmetic average.
    
    Parameters
    ----------
    values_at_centers : ndarray of shape (n_grid,)
        Values at cell centers.
        
    Returns
    -------
    values_at_faces : ndarray of shape (n_grid - 1,)
        Values at interior faces (between cells).
        
    Notes
    -----
    Returns only interior face values. Boundary face values are handled
    separately in PDE solver (no-flux boundary conditions).
    """
    return 0.5 * (values_at_centers[:-1] + values_at_centers[1:])


def upwind_values(
    u: np.ndarray, 
    v_faces: np.ndarray
) -> np.ndarray:
    """
    Compute upwind values at interior faces for advection discretization.
    
    Parameters
    ----------
    u : ndarray of shape (n_grid,)
        Density values at cell centers.
    v_faces : ndarray of shape (n_grid - 1,)
        Velocity values at interior faces.
        
    Returns
    -------
    u_upwind : ndarray of shape (n_grid - 1,)
        Upwind density values at interior faces.
        
    Notes
    -----
    Uses first-order upwinding:
        u_upwind[i] = u[i]     if v_faces[i] >= 0  (flow right, take left value)
                    = u[i+1]   if v_faces[i] < 0   (flow left, take right value)
    """
    u_upwind = np.where(v_faces >= 0, u[:-1], u[1:])
    return u_upwind

