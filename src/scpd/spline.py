"""
Natural cubic spline basis functions.

Uses patsy's cr() (natural cubic regression splines) to create basis functions
for parameterizing D(s), v(s), and g(s) on state coordinate s ∈ [0,1].
"""

import numpy as np
from patsy import dmatrix
from typing import Optional


class SplineBasis:
    """
    Natural cubic spline basis for function parameterization.
    
    Creates basis functions B(s) such that any function can be represented as:
        f(s) = B(s) @ beta
    
    Parameters
    ----------
    df : int, default=6
        Degrees of freedom (number of basis functions).
    s_min : float, default=0.0
        Minimum value of s domain.
    s_max : float, default=1.0
        Maximum value of s domain.
        
    Attributes
    ----------
    df : int
        Degrees of freedom.
    knots : ndarray
        Internal knot positions.
        
    Notes
    -----
    Uses patsy's cr() function with "-1" to exclude intercept, giving
    exactly df basis functions. The design is created on a reference grid
    and new values are obtained by evaluating at new s values.
    """
    
    def __init__(
        self, 
        df: int = 6,
        s_min: float = 0.0,
        s_max: float = 1.0
    ):
        self.df = df
        self.s_min = s_min
        self.s_max = s_max
        
        # Create reference grid for building design info
        self._ref_s = np.linspace(s_min, s_max, 100)
        self._design_info = None
        self._build_design_info()
        
    def _build_design_info(self):
        """Build the design matrix info from reference grid."""
        # Create design matrix formula
        # Note: cr() gives natural cubic spline with df degrees of freedom
        formula = f"cr(s, df={self.df}) - 1"
        
        # Build on reference grid to capture design info
        dm = dmatrix(formula, {"s": self._ref_s})
        self._design_info = dm.design_info
        
    def evaluate(self, s: np.ndarray) -> np.ndarray:
        """
        Evaluate basis functions at given s values.
        
        Parameters
        ----------
        s : ndarray of shape (n,)
            State coordinate values where to evaluate basis.
            
        Returns
        -------
        B : ndarray of shape (n, df)
            Basis function matrix.
        """
        s = np.asarray(s).ravel()
        
        # Use stored design info for prediction
        dm = dmatrix(self._design_info, {"s": s})
        return np.asarray(dm)
    
    def evaluate_function(
        self, 
        s: np.ndarray, 
        beta: np.ndarray,
        log_transform: bool = False
    ) -> np.ndarray:
        """
        Evaluate a spline-parameterized function at given s values.
        
        Parameters
        ----------
        s : ndarray of shape (n,)
            State coordinate values.
        beta : ndarray of shape (df,)
            Spline coefficients.
        log_transform : bool, default=False
            If True, returns exp(B @ beta) instead of B @ beta.
            Used for D(s) which is parameterized as log D(s) = B @ beta_D.
            
        Returns
        -------
        f : ndarray of shape (n,)
            Function values f(s) = B(s) @ beta or exp(B(s) @ beta).
        """
        B = self.evaluate(s)
        f = B @ beta
        if log_transform:
            f = np.exp(f)
        return f
    
    def second_derivative_matrix(
        self, 
        s_dense: Optional[np.ndarray] = None,
        n_dense: int = 500
    ) -> np.ndarray:
        """
        Compute second derivative approximation of basis functions.
        
        Used for roughness penalty: ∫(f''(s))^2 ds ≈ beta^T @ R @ beta
        
        Parameters
        ----------
        s_dense : ndarray, optional
            Dense grid for numerical differentiation. If None, creates
            uniform grid with n_dense points.
        n_dense : int, default=500
            Number of points in dense grid if s_dense not provided.
            
        Returns
        -------
        B_dd : ndarray of shape (n_dense - 2, df)
            Second derivative of basis functions at interior points.
        ds : float
            Grid spacing.
        """
        if s_dense is None:
            s_dense = np.linspace(self.s_min, self.s_max, n_dense)
        
        ds = s_dense[1] - s_dense[0]
        B = self.evaluate(s_dense)
        
        # Second derivative via central difference: B''[i] ≈ (B[i+1] - 2*B[i] + B[i-1]) / ds^2
        B_dd = (B[2:] - 2 * B[1:-1] + B[:-2]) / (ds ** 2)
        
        return B_dd, ds


def create_parameter_mapping(
    spline_basis: SplineBasis,
    s_grid: np.ndarray,
    mode: str = "with_population"
):
    """
    Create mapping from parameter vector to D, v, g functions.
    
    Parameters
    ----------
    spline_basis : SplineBasis
        Spline basis for function parameterization.
    s_grid : ndarray of shape (n_grid,)
        Grid points for evaluation.
    mode : str, default="with_population"
        Either "with_population" (fit D, v, g) or "distribution_only" (fit D, v; g=0).
        
    Returns
    -------
    dict with keys:
        - 'n_params': total number of parameters
        - 'n_D': number of D parameters (= df)
        - 'n_v': number of v parameters (= df)
        - 'n_g': number of g parameters (= df or 0)
        - 'B_grid': basis evaluated at s_grid
        
    Notes
    -----
    Parameter vector layout:
        theta = [beta_D, beta_v, beta_g] for mode="with_population"
        theta = [beta_D, beta_v]         for mode="distribution_only"
    """
    df = spline_basis.df
    B_grid = spline_basis.evaluate(s_grid)
    
    if mode == "with_population":
        n_params = 3 * df
        n_g = df
    elif mode == "distribution_only":
        n_params = 2 * df
        n_g = 0
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'with_population' or 'distribution_only'.")
    
    return {
        'n_params': n_params,
        'n_D': df,
        'n_v': df,
        'n_g': n_g,
        'df': df,
        'B_grid': B_grid,
        'mode': mode,
    }


def unpack_parameters(
    theta: np.ndarray,
    param_info: dict,
    B: np.ndarray
) -> tuple:
    """
    Unpack parameter vector to D, v, g values.
    
    Parameters
    ----------
    theta : ndarray
        Parameter vector.
    param_info : dict
        Parameter mapping info from create_parameter_mapping.
    B : ndarray of shape (n, df)
        Basis matrix evaluated at desired s points.
        
    Returns
    -------
    D : ndarray of shape (n,)
        Diffusion coefficient D(s) = exp(B @ beta_D).
    v : ndarray of shape (n,)
        Drift velocity v(s) = B @ beta_v.
    g : ndarray of shape (n,)
        Net growth rate g(s) = B @ beta_g (or zeros if mode="distribution_only").
    """
    df = param_info['df']
    mode = param_info['mode']
    
    beta_D = theta[:df]
    beta_v = theta[df:2*df]
    
    D = np.exp(B @ beta_D)
    v = B @ beta_v
    
    if mode == "with_population" and param_info['n_g'] > 0:
        beta_g = theta[2*df:3*df]
        g = B @ beta_g
    else:
        g = np.zeros(B.shape[0])
    
    return D, v, g

