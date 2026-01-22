"""
Synthetic data generation for testing and demonstration.

Provides tools to generate ground truth D, v, g functions and forward-simulate
the PDE to create synthetic snapshot datasets.
"""

import numpy as np
from typing import Tuple, Optional, Callable
from scipy.integrate import solve_ivp

from .grid import create_grid
from .pde import pde_rhs, compute_total_mass


def generate_true_rates(
    s_grid: np.ndarray,
    D_type: str = "constant",
    v_type: str = "linear",
    g_type: str = "zero",
    D_scale: float = 0.01,
    v_scale: float = 0.5,
    g_scale: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate ground-truth rate functions.
    
    Parameters
    ----------
    s_grid : ndarray of shape (n_grid,)
        Grid points.
    D_type : str, default="constant"
        Type of diffusion: "constant", "linear", "quadratic".
    v_type : str, default="linear"
        Type of drift: "constant", "linear", "sigmoidal".
    g_type : str, default="zero"
        Type of growth: "zero", "constant", "linear", "bell".
    D_scale, v_scale, g_scale : float
        Scaling factors.
        
    Returns
    -------
    D_true, v_true, g_true : ndarray of shape (n_grid,)
        Ground-truth rate functions.
    """
    n_grid = len(s_grid)
    
    # Diffusion D(s)
    if D_type == "constant":
        D_true = np.ones(n_grid) * D_scale
    elif D_type == "linear":
        D_true = D_scale * (0.5 + 0.5 * s_grid)
    elif D_type == "quadratic":
        D_true = D_scale * (0.5 + 0.5 * (s_grid - 0.5) ** 2)
    else:
        D_true = np.ones(n_grid) * D_scale
    
    # Drift v(s)
    if v_type == "constant":
        v_true = np.ones(n_grid) * v_scale
    elif v_type == "linear":
        v_true = v_scale * (1 - 0.5 * s_grid)
    elif v_type == "sigmoidal":
        v_true = v_scale * (1 / (1 + np.exp(-10 * (0.5 - s_grid))))
    else:
        v_true = np.ones(n_grid) * v_scale
    
    # Growth g(s)
    if g_type == "zero":
        g_true = np.zeros(n_grid)
    elif g_type == "constant":
        g_true = np.ones(n_grid) * g_scale
    elif g_type == "linear":
        g_true = g_scale * (0.5 - s_grid)
    elif g_type == "bell":
        g_true = g_scale * np.exp(-((s_grid - 0.3) ** 2) / 0.05)
    else:
        g_true = np.zeros(n_grid)
    
    return D_true, v_true, g_true


def simulate_forward(
    D: np.ndarray,
    v: np.ndarray,
    g: np.ndarray,
    s_grid: np.ndarray,
    ds: float,
    times: np.ndarray,
    u0: Optional[np.ndarray] = None,
    stabilize_boundary: bool = False
) -> np.ndarray:
    """
    Simulate PDE forward in time with given rates.
    
    Parameters
    ----------
    D, v, g : ndarray of shape (n_grid,)
        Rate functions.
    s_grid : ndarray of shape (n_grid,)
        Grid points.
    ds : float
        Grid spacing.
    times : ndarray
        Output time points.
    u0 : ndarray, optional
        Initial density. If None, uses narrow Gaussian near s=0.1.
    stabilize_boundary : bool, default=False
        Stabilize v near boundary.
        
    Returns
    -------
    u : ndarray of shape (n_grid, n_times)
        Density at each grid point and time.
    """
    n_grid = len(s_grid)
    
    if u0 is None:
        # Initial condition: narrow Gaussian near s=0.1
        u0 = np.exp(-((s_grid - 0.1) ** 2) / (2 * 0.02 ** 2))
        u0 /= np.sum(u0) * ds  # Normalize to integrate to 1
    
    def rhs(t, u):
        return pde_rhs(u, D, v, g, ds, stabilize_boundary=stabilize_boundary)
    
    sol = solve_ivp(
        rhs, (times[0], times[-1]), u0,
        method='BDF',
        t_eval=times,
        dense_output=False
    )
    
    if not sol.success:
        raise RuntimeError(f"PDE simulation failed: {sol.message}")
    
    return sol.y


def sample_from_density(
    u: np.ndarray,
    s_grid: np.ndarray,
    ds: float,
    n_samples: int,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Sample s values from a density distribution.
    
    Parameters
    ----------
    u : ndarray of shape (n_grid,)
        Density (unnormalized ok).
    s_grid : ndarray of shape (n_grid,)
        Grid points.
    ds : float
        Grid spacing.
    n_samples : int
        Number of samples to draw.
    random_state : int, optional
        Random seed.
        
    Returns
    -------
    s_samples : ndarray of shape (n_samples,)
        Sampled s values.
    """
    rng = np.random.default_rng(random_state)
    
    # Normalize to probability
    p = np.maximum(u, 0)
    p_sum = np.sum(p)
    if p_sum <= 0:
        p = np.ones_like(p)
        p_sum = len(p)
    p = p / p_sum
    
    # Sample cell indices
    indices = rng.choice(len(s_grid), size=n_samples, p=p)
    
    # Add uniform jitter within each cell
    s_samples = s_grid[indices] + rng.uniform(-ds/2, ds/2, size=n_samples)
    
    # Clip to [0, 1]
    s_samples = np.clip(s_samples, 0, 1)
    
    return s_samples


def generate_synthetic_dataset(
    n_times: int = 6,
    n_cells_per_time: int = 500,
    D_type: str = "constant",
    v_type: str = "linear",
    g_type: str = "zero",
    D_scale: float = 0.01,
    v_scale: float = 0.5,
    g_scale: float = 0.1,
    n_grid: int = 200,
    random_state: int = 0,
    time_span: Tuple[float, float] = (0.0, 1.0),
    include_population_noise: bool = True,
    population_noise_frac: float = 0.05
) -> dict:
    """
    Generate a complete synthetic dataset.
    
    Parameters
    ----------
    n_times : int, default=6
        Number of time points.
    n_cells_per_time : int, default=500
        Number of cells per time point.
    D_type, v_type, g_type : str
        Rate function types.
    D_scale, v_scale, g_scale : float
        Rate scaling factors.
    n_grid : int, default=200
        Grid resolution for simulation.
    random_state : int, default=0
        Random seed.
    time_span : tuple, default=(0.0, 1.0)
        Time range.
    include_population_noise : bool, default=True
        Add noise to observed population.
    population_noise_frac : float, default=0.05
        Fraction of population as noise std.
        
    Returns
    -------
    dataset : dict with keys:
        - 's': ndarray of shape (n_cells,) - all cell s values
        - 'time_labels': ndarray of shape (n_cells,) - time labels
        - 'time_values': ndarray of shape (n_times,) - time values
        - 'N_obs': ndarray of shape (n_times,) - observed population
        - 'N_true': ndarray of shape (n_times,) - true population (no noise)
        - 's_grid': ndarray - grid points
        - 'ds': float - grid spacing
        - 'D_true', 'v_true', 'g_true': ndarray - true rates
        - 'u_true': ndarray of shape (n_grid, n_times) - true density
    """
    rng = np.random.default_rng(random_state)
    
    # Create grid
    s_grid, s_faces, ds = create_grid(n_grid)
    
    # Generate true rates
    D_true, v_true, g_true = generate_true_rates(
        s_grid, D_type, v_type, g_type, D_scale, v_scale, g_scale
    )
    
    # Time points
    time_values = np.linspace(time_span[0], time_span[1], n_times)
    
    # Simulate forward
    u_true = simulate_forward(D_true, v_true, g_true, s_grid, ds, time_values)
    
    # Sample cells and compute population
    all_s = []
    all_times = []
    N_true = []
    
    for k in range(n_times):
        # True population at this time
        N_k = compute_total_mass(u_true[:, k], ds)
        N_true.append(N_k)
        
        # Sample cells
        s_k = sample_from_density(
            u_true[:, k], s_grid, ds, n_cells_per_time,
            random_state=random_state + k
        )
        
        all_s.append(s_k)
        all_times.append(np.full(n_cells_per_time, k))
    
    s = np.concatenate(all_s)
    time_labels = np.concatenate(all_times)
    N_true = np.array(N_true)
    
    # Add noise to population
    if include_population_noise:
        noise = rng.normal(0, population_noise_frac * N_true)
        N_obs = N_true + noise
    else:
        N_obs = N_true.copy()
    
    return {
        's': s,
        'time_labels': time_labels,
        'time_values': time_values,
        'N_obs': N_obs,
        'N_true': N_true,
        's_grid': s_grid,
        'ds': ds,
        'D_true': D_true,
        'v_true': v_true,
        'g_true': g_true,
        'u_true': u_true
    }

