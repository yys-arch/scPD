"""
Result containers and diagnostic information.

Provides structured output from model fitting with save/load functionality.
"""

import numpy as np
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from pathlib import Path

# Re-export from preprocess for convenience
from .preprocess import PreparedData, LandmarkInfo


@dataclass
class Diagnostics:
    """
    Fitting diagnostics for each time point.
    
    Attributes
    ----------
    A_values : ndarray of shape (n_times,)
        A-distance for each time point.
    sigma_A_values : ndarray of shape (n_times,)
        Estimated σ_A for each time point.
    nll_cdf_per_time : ndarray of shape (n_times,)
        CDF NLL contribution per time point.
    N_model : ndarray of shape (n_times,), optional
        Model population at each time.
    nll_population : float, optional
        Population NLL if mode="with_population".
    penalty : float
        Roughness penalty value.
    total_nll : float
        Total objective value.
    n_iterations : int
        Number of optimizer iterations.
    success : bool
        Whether optimization converged.
    message : str
        Optimizer message.
    """
    A_values: np.ndarray
    sigma_A_values: np.ndarray
    nll_cdf_per_time: np.ndarray
    N_model: Optional[np.ndarray] = None
    nll_population: Optional[float] = None
    penalty: float = 0.0
    total_nll: float = 0.0
    n_iterations: int = 0
    success: bool = False
    message: str = ""


@dataclass
class PseudodynamicsResult:
    """
    Results from pseudodynamics model fitting.
    
    Attributes
    ----------
    s_grid : ndarray of shape (n_grid,)
        Grid points where rates are evaluated.
    D : ndarray of shape (n_grid,)
        Fitted diffusion coefficient D(s).
    v : ndarray of shape (n_grid,)
        Fitted drift velocity v(s).
    g : ndarray of shape (n_grid,)
        Fitted net growth rate g(s). Zero if mode="distribution_only".
    W : ndarray of shape (n_grid,)
        Developmental potential W(s) = ∫_0^s -v(s') ds'.
    u : ndarray of shape (n_grid, n_times)
        Density at each grid point and time.
    time_values : ndarray of shape (n_times,)
        Time values used in simulation.
    theta : ndarray
        Fitted parameter vector.
    beta_D : ndarray of shape (df,)
        Spline coefficients for log D.
    beta_v : ndarray of shape (df,)
        Spline coefficients for v.
    beta_g : ndarray of shape (df,), optional
        Spline coefficients for g (None if mode="distribution_only").
    mode : str
        Fitting mode used.
    rho : float
        Regularization strength used.
    diagnostics : Diagnostics
        Fitting diagnostics.
    config : dict
        Configuration used for fitting.
    """
    s_grid: np.ndarray
    D: np.ndarray
    v: np.ndarray
    g: np.ndarray
    W: np.ndarray
    u: np.ndarray
    time_values: np.ndarray
    theta: np.ndarray
    beta_D: np.ndarray
    beta_v: np.ndarray
    beta_g: Optional[np.ndarray]
    mode: str
    rho: float
    diagnostics: Diagnostics
    config: Dict[str, Any] = field(default_factory=dict)
    
    def rates(self, s: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Get rates at specified s values.
        
        Parameters
        ----------
        s : ndarray, optional
            State coordinates. If None, uses s_grid.
            
        Returns
        -------
        dict with keys 'D', 'v', 'g', 's'
        """
        if s is None:
            return {'s': self.s_grid, 'D': self.D, 'v': self.v, 'g': self.g}
        
        # Interpolate to new s values
        from scipy.interpolate import interp1d
        s = np.asarray(s)
        D_interp = interp1d(self.s_grid, self.D, kind='linear', fill_value='extrapolate')
        v_interp = interp1d(self.s_grid, self.v, kind='linear', fill_value='extrapolate')
        g_interp = interp1d(self.s_grid, self.g, kind='linear', fill_value='extrapolate')
        
        return {
            's': s,
            'D': D_interp(s),
            'v': v_interp(s),
            'g': g_interp(s)
        }
    
    def developmental_potential(self, s: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get developmental potential at specified s values.
        
        Parameters
        ----------
        s : ndarray, optional
            State coordinates. If None, uses s_grid.
            
        Returns
        -------
        W : ndarray
            Developmental potential W(s) = ∫_0^s -v(s') ds'
        """
        if s is None:
            return self.W
        
        from scipy.interpolate import interp1d
        s = np.asarray(s)
        W_interp = interp1d(self.s_grid, self.W, kind='linear', fill_value='extrapolate')
        return W_interp(s)
    
    def to_cell_level(self, s_vector: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Evaluate all quantities at cell-level s values.
        
        Parameters
        ----------
        s_vector : ndarray of shape (n_cells,)
            State coordinate for each cell.
            
        Returns
        -------
        dict with keys 'D', 'v', 'g', 'W', each ndarray of shape (n_cells,)
        """
        rates = self.rates(s_vector)
        W = self.developmental_potential(s_vector)
        return {
            'D': rates['D'],
            'v': rates['v'],
            'g': rates['g'],
            'W': W
        }
    
    def save(self, path: str):
        """
        Save results to file.
        
        Parameters
        ----------
        path : str
            Output file path. Uses .npz format.
        """
        path = Path(path)
        
        # Prepare diagnostics for saving
        diag_dict = {
            'diag_A_values': self.diagnostics.A_values,
            'diag_sigma_A_values': self.diagnostics.sigma_A_values,
            'diag_nll_cdf_per_time': self.diagnostics.nll_cdf_per_time,
            'diag_penalty': self.diagnostics.penalty,
            'diag_total_nll': self.diagnostics.total_nll,
            'diag_n_iterations': self.diagnostics.n_iterations,
            'diag_success': self.diagnostics.success,
            'diag_message': self.diagnostics.message,
        }
        if self.diagnostics.N_model is not None:
            diag_dict['diag_N_model'] = self.diagnostics.N_model
        if self.diagnostics.nll_population is not None:
            diag_dict['diag_nll_population'] = self.diagnostics.nll_population
        
        # Save arrays
        np.savez(
            path,
            s_grid=self.s_grid,
            D=self.D,
            v=self.v,
            g=self.g,
            W=self.W,
            u=self.u,
            time_values=self.time_values,
            theta=self.theta,
            beta_D=self.beta_D,
            beta_v=self.beta_v,
            beta_g=self.beta_g if self.beta_g is not None else np.array([]),
            mode=self.mode,
            rho=self.rho,
            config=json.dumps(self.config),
            **diag_dict
        )
    
    @classmethod
    def load(cls, path: str) -> "PseudodynamicsResult":
        """
        Load results from file.
        
        Parameters
        ----------
        path : str
            Input file path (.npz format).
            
        Returns
        -------
        result : PseudodynamicsResult
        """
        data = np.load(path, allow_pickle=True)
        
        # Reconstruct diagnostics
        diagnostics = Diagnostics(
            A_values=data['diag_A_values'],
            sigma_A_values=data['diag_sigma_A_values'],
            nll_cdf_per_time=data['diag_nll_cdf_per_time'],
            N_model=data.get('diag_N_model'),
            nll_population=float(data['diag_nll_population']) if 'diag_nll_population' in data else None,
            penalty=float(data['diag_penalty']),
            total_nll=float(data['diag_total_nll']),
            n_iterations=int(data['diag_n_iterations']),
            success=bool(data['diag_success']),
            message=str(data['diag_message'])
        )
        
        beta_g = data['beta_g']
        if len(beta_g) == 0:
            beta_g = None
        
        return cls(
            s_grid=data['s_grid'],
            D=data['D'],
            v=data['v'],
            g=data['g'],
            W=data['W'],
            u=data['u'],
            time_values=data['time_values'],
            theta=data['theta'],
            beta_D=data['beta_D'],
            beta_v=data['beta_v'],
            beta_g=beta_g,
            mode=str(data['mode']),
            rho=float(data['rho']),
            diagnostics=diagnostics,
            config=json.loads(str(data['config']))
        )

