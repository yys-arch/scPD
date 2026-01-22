"""
Main PseudodynamicsModel class for fitting density dynamics.

Integrates PDE solver, spline parameterization, likelihood computation, and optimization
to estimate D(s), v(s), g(s) from snapshot data.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Optional, Dict, Any, List, Tuple

from .grid import create_grid
from .spline import SplineBasis, create_parameter_mapping, unpack_parameters
from .pde import pde_rhs, compute_total_mass, density_to_pdf, pdf_to_cdf
from .regularization import compute_roughness_matrix, roughness_penalty
from .likelihood import (
    compute_ecdf, compute_weighted_ecdf, compute_a_distance,
    estimate_sigma_A_bootstrap, estimate_sigma_A_landmark_bootstrap,
    compute_nll_cdf, compute_nll_population
)
from .optimize import multi_start_optimize, cross_validate_rho
from .results import PreparedData, PseudodynamicsResult, Diagnostics


class PseudodynamicsModel:
    """
    Pseudodynamics model for single-cell density dynamics.
    
    Fits diffusion D(s), drift v(s), and net growth g(s) to snapshot
    distributions on a 1D state coordinate s ∈ [0,1].
    
    Parameters
    ----------
    n_grid : int, default=200
        Number of grid points for PDE discretization.
    spline_df : int, default=6
        Degrees of freedom for natural cubic spline basis.
    stabilize_boundary : bool, default=False
        If True, smoothly reduce drift near s=1 for stability.
    clip_to_nonnegative : bool, default=False
        If True, clip negative densities to 0 during PDE integration.
        
    Attributes
    ----------
    s_grid : ndarray
        Grid points for discretization.
    ds : float
        Grid spacing.
    spline_basis : SplineBasis
        Spline basis for rate parameterization.
    R : ndarray
        Roughness penalty matrix.
    """
    
    def __init__(
        self,
        n_grid: int = 200,
        spline_df: int = 6,
        stabilize_boundary: bool = False,
        clip_to_nonnegative: bool = False
    ):
        self.n_grid = n_grid
        self.spline_df = spline_df
        self.stabilize_boundary = stabilize_boundary
        self.clip_to_nonnegative = clip_to_nonnegative
        
        # Create grid
        self.s_grid, self.s_faces, self.ds = create_grid(n_grid)
        
        # Create spline basis
        self.spline_basis = SplineBasis(df=spline_df)
        self.B_grid = self.spline_basis.evaluate(self.s_grid)
        
        # Compute roughness matrix
        self.R = compute_roughness_matrix(self.spline_basis)
        
        # Storage for current fit
        self._prepared_data: Optional[PreparedData] = None
        self._param_info: Optional[Dict] = None
        self._sigma_A: Optional[np.ndarray] = None
        
    def _create_initial_density(
        self,
        s_values: np.ndarray,
        bandwidth: Optional[float] = None
    ) -> np.ndarray:
        """Create initial density from samples using KDE."""
        if bandwidth is None:
            # Silverman's rule
            bandwidth = 1.06 * np.std(s_values) * len(s_values) ** (-0.2)
            bandwidth = max(bandwidth, 0.01)  # Minimum bandwidth
        
        # Simple Gaussian KDE
        u0 = np.zeros(self.n_grid)
        for s_i in s_values:
            u0 += np.exp(-0.5 * ((self.s_grid - s_i) / bandwidth) ** 2)
        
        u0 /= (bandwidth * np.sqrt(2 * np.pi) * len(s_values))
        
        # Ensure positive and normalized to integrate to 1
        u0 = np.maximum(u0, 1e-10)
        u0 /= compute_total_mass(u0, self.ds)
        
        return u0
    
    def _simulate_pde(
        self,
        u0: np.ndarray,
        D: np.ndarray,
        v: np.ndarray,
        g: np.ndarray,
        times: np.ndarray
    ) -> np.ndarray:
        """
        Simulate PDE forward in time.
        
        Parameters
        ----------
        u0 : ndarray of shape (n_grid,)
            Initial density.
        D, v, g : ndarray of shape (n_grid,)
            Rate functions at grid points.
        times : ndarray of shape (n_times,)
            Time points for output.
            
        Returns
        -------
        u : ndarray of shape (n_grid, n_times)
            Density at each grid point and time.
        """
        def rhs(t, u):
            return pde_rhs(
                u, D, v, g, self.ds,
                stabilize_boundary=self.stabilize_boundary,
                clip_to_nonnegative=self.clip_to_nonnegative
            )
        
        # Solve ODE system
        t_span = (times[0], times[-1])
        sol = solve_ivp(
            rhs, t_span, u0,
            method='BDF',
            t_eval=times,
            dense_output=False
        )
        
        if not sol.success:
            # Return initial condition repeated if failed
            return np.tile(u0.reshape(-1, 1), (1, len(times)))
        
        return sol.y
    
    def _estimate_sigma_A(
        self,
        prepared: PreparedData,
        n_bootstrap: int = 100,
        random_state: int = 0
    ) -> np.ndarray:
        """Estimate σ_A for each time point."""
        sigma_A = np.zeros(prepared.n_times)
        
        for k, t in enumerate(prepared.unique_times):
            if prepared.landmark_info.enabled:
                # Use landmark-based bootstrap
                sigma_A[k] = estimate_sigma_A_landmark_bootstrap(
                    prepared.landmark_info.landmark_s,
                    prepared.weights_per_time[k],
                    self.s_grid,
                    self.ds,
                    n_bootstrap=n_bootstrap,
                    random_state=random_state + k
                )
            else:
                # Use cell-level bootstrap
                sigma_A[k] = estimate_sigma_A_bootstrap(
                    prepared.s_per_time[k],
                    self.s_grid,
                    self.ds,
                    n_bootstrap=n_bootstrap,
                    random_state=random_state + k
                )
        
        return sigma_A
    
    def _compute_ecdfs(self, prepared: PreparedData) -> List[np.ndarray]:
        """Compute ECDFs for all time points.
        
        Note: Always uses original cell s values for accurate ECDF.
        Landmark mode only affects sigma_A estimation (bootstrap speedup),
        not the ECDF itself.
        """
        ecdfs = []
        for k in range(prepared.n_times):
            # Always use original cell s values for ECDF
            ecdf = compute_ecdf(prepared.s_per_time[k], self.s_grid)
            ecdfs.append(ecdf)
        return ecdfs
    
    def _build_objective(
        self,
        prepared: PreparedData,
        param_info: Dict,
        sigma_A: np.ndarray,
        ecdfs: List[np.ndarray],
        u0: np.ndarray,
        rho: float,
        lambda_N: float,
        time_indices: Optional[List[int]] = None
    ):
        """Build objective function for optimization."""
        
        if time_indices is None:
            time_indices = list(range(prepared.n_times))
        
        times = prepared.time_values[time_indices]
        n_times_subset = len(time_indices)
        
        def objective(theta: np.ndarray) -> float:
            # Unpack parameters
            D, v, g = unpack_parameters(theta, param_info, self.B_grid)
            
            # Simulate PDE
            u = self._simulate_pde(u0, D, v, g, times)
            
            # Compute A-distances
            A_values = np.zeros(n_times_subset)
            for i, k in enumerate(time_indices):
                p = density_to_pdf(u[:, i], self.ds)
                cdf_model = pdf_to_cdf(p, self.ds)
                A_values[i] = compute_a_distance(cdf_model, ecdfs[k], self.ds)
            
            # CDF NLL
            nll_cdf = compute_nll_cdf(A_values, sigma_A[time_indices])
            
            # Population NLL (if applicable)
            nll_pop = 0.0
            if param_info['mode'] == 'with_population' and prepared.N_obs is not None:
                N_model = np.array([compute_total_mass(u[:, i], self.ds) for i in range(n_times_subset)])
                N_obs_subset = prepared.N_obs[time_indices]
                sigma_N_subset = prepared.sigma_N[time_indices]
                nll_pop = compute_nll_population(N_model, N_obs_subset, sigma_N_subset)
            
            # Roughness penalty
            penalty = roughness_penalty(theta, self.R, param_info['df'], param_info['mode'])
            
            total = nll_cdf + lambda_N * nll_pop + rho * penalty
            
            return total
        
        return objective
    
    def fit(
        self,
        prepared: PreparedData,
        mode: str = "distribution_only",
        rho: Optional[float] = None,
        cv_rho: bool = False,
        rho_grid: Optional[np.ndarray] = None,
        lambda_N: float = 1.0,
        n_starts: int = 10,
        n_bootstrap: int = 100,
        random_state: int = 0,
        verbose: bool = False
    ) -> PseudodynamicsResult:
        """
        Fit pseudodynamics model to prepared data.
        
        Parameters
        ----------
        prepared : PreparedData
            Prepared input data from prepare_inputs().
        mode : str, default="distribution_only"
            Fitting mode:
            - "distribution_only": Fit D and v only, g=0. Use when N_obs unavailable.
            - "with_population": Fit D, v, and g. Requires N_obs in prepared data.
        rho : float, optional
            Regularization strength. If None and cv_rho=False, uses 0.1.
        cv_rho : bool, default=False
            If True, use leave-one-time-out cross-validation to select rho.
        rho_grid : ndarray, optional
            Grid of rho values for cross-validation. Default: logspace(-4, 2, 10).
        lambda_N : float, default=1.0
            Weight for population NLL term.
        n_starts : int, default=10
            Number of random starts for optimization.
        n_bootstrap : int, default=100
            Number of bootstrap samples for σ_A estimation.
        random_state : int, default=0
            Random seed.
        verbose : bool, default=False
            Print progress information.
            
        Returns
        -------
        result : PseudodynamicsResult
            Fitted model results.
            
        Notes
        -----
        **Identifiability of g(s):**
        
        Without observed population sizes N_obs(t), net growth rate g(s)
        is not identifiable from distribution data alone. Any g(s) can be
        absorbed by rescaling density without changing normalized distribution shape.
        
        Therefore:
        - mode="distribution_only" (default): Fix g(s)=0 and fit D, v only.
        - mode="with_population": Use N_obs(t) data to fit g(s) and anchor
          absolute scale of density dynamics.
        """
        self._prepared_data = prepared
        
        # Validate mode
        if mode == "with_population" and prepared.N_obs is None:
            raise ValueError(
                "mode='with_population' requires N_obs in prepared data. "
                "Either provide N_obs or use mode='distribution_only'."
            )
        
        # Create parameter mapping
        self._param_info = create_parameter_mapping(
            self.spline_basis, self.s_grid, mode
        )
        n_params = self._param_info['n_params']
        
        if verbose:
            print(f"Fitting with mode='{mode}', {n_params} parameters")
            if prepared.landmark_info.enabled:
                print(f"  Using {prepared.landmark_info.n_landmarks} landmarks")
        
        # Estimate σ_A
        if verbose:
            print("Estimating σ_A via bootstrap...")
        self._sigma_A = self._estimate_sigma_A(prepared, n_bootstrap, random_state)
        
        # Compute ECDFs
        ecdfs = self._compute_ecdfs(prepared)
        
        # Create initial density from first time point
        u0 = self._create_initial_density(prepared.s_per_time[0])
        
        # Handle rho
        if cv_rho:
            if verbose:
                print("Performing cross-validation for rho...")
            if rho_grid is None:
                rho_grid = np.logspace(-4, 2, 10)
            
            # Build objective functions for CV
            def build_obj(indices, rho_val):
                return self._build_objective(
                    prepared, self._param_info, self._sigma_A, ecdfs, u0,
                    rho_val, lambda_N, indices
                )
            
            def eval_held_out(held_idx):
                def eval_fn(theta):
                    D, v, g = unpack_parameters(theta, self._param_info, self.B_grid)
                    times = prepared.time_values[[held_idx]]
                    u = self._simulate_pde(u0, D, v, g, times)
                    p = density_to_pdf(u[:, 0], self.ds)
                    cdf_model = pdf_to_cdf(p, self.ds)
                    A = compute_a_distance(cdf_model, ecdfs[held_idx], self.ds)
                    return (A / self._sigma_A[held_idx]) ** 2
                return eval_fn
            
            rho, cv_info = cross_validate_rho(
                build_obj, eval_held_out,
                prepared.n_times, rho_grid, n_params,
                n_starts=max(3, n_starts // 2),
                random_state=random_state,
                verbose=verbose
            )
            if verbose:
                print(f"Selected rho = {rho:.2e}")
        else:
            if rho is None:
                rho = 0.1
        
        # Build final objective
        objective = self._build_objective(
            prepared, self._param_info, self._sigma_A, ecdfs, u0,
            rho, lambda_N
        )
        
        # Optimize
        if verbose:
            print(f"Optimizing with {n_starts} starts...")
        
        theta_opt, f_opt, opt_info = multi_start_optimize(
            objective, n_params, n_starts=n_starts,
            random_state=random_state, verbose=verbose
        )
        
        # Extract fitted parameters
        D, v, g = unpack_parameters(theta_opt, self._param_info, self.B_grid)
        
        # Simulate final solution
        u = self._simulate_pde(u0, D, v, g, prepared.time_values)
        
        # Compute developmental potential W(s) = ∫_0^s -v(s') ds'
        W = -np.cumsum(v) * self.ds
        
        # Compute diagnostics
        A_values = np.zeros(prepared.n_times)
        nll_cdf_per_time = np.zeros(prepared.n_times)
        for k in range(prepared.n_times):
            p = density_to_pdf(u[:, k], self.ds)
            cdf_model = pdf_to_cdf(p, self.ds)
            A_values[k] = compute_a_distance(cdf_model, ecdfs[k], self.ds)
            z = A_values[k] / self._sigma_A[k]
            nll_cdf_per_time[k] = 0.5 * z ** 2 + np.log(self._sigma_A[k])
        
        N_model = None
        nll_population = None
        if mode == "with_population" and prepared.N_obs is not None:
            N_model = np.array([compute_total_mass(u[:, k], self.ds) for k in range(prepared.n_times)])
            nll_population = compute_nll_population(N_model, prepared.N_obs, prepared.sigma_N)
        
        penalty_val = roughness_penalty(theta_opt, self.R, self._param_info['df'], mode)
        
        diagnostics = Diagnostics(
            A_values=A_values,
            sigma_A_values=self._sigma_A,
            nll_cdf_per_time=nll_cdf_per_time,
            N_model=N_model,
            nll_population=nll_population,
            penalty=penalty_val,
            total_nll=f_opt,
            n_iterations=opt_info['n_iterations'],
            success=opt_info['success'],
            message=str(opt_info.get('message', ''))
        )
        
        # Extract beta coefficients
        df = self._param_info['df']
        beta_D = theta_opt[:df]
        beta_v = theta_opt[df:2*df]
        beta_g = theta_opt[2*df:3*df] if mode == "with_population" else None
        
        result = PseudodynamicsResult(
            s_grid=self.s_grid,
            D=D,
            v=v,
            g=g,
            W=W,
            u=u,
            time_values=prepared.time_values,
            theta=theta_opt,
            beta_D=beta_D,
            beta_v=beta_v,
            beta_g=beta_g,
            mode=mode,
            rho=rho,
            diagnostics=diagnostics,
            config={
                'n_grid': self.n_grid,
                'spline_df': self.spline_df,
                'n_starts': n_starts,
                'n_bootstrap': n_bootstrap,
                'lambda_N': lambda_N,
                'landmarks_enabled': prepared.landmark_info.enabled,
                'n_landmarks': prepared.landmark_info.n_landmarks if prepared.landmark_info.enabled else 0
            }
        )
        
        return result
    
    def simulate(
        self,
        result: PseudodynamicsResult,
        times: np.ndarray,
        u0: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Simulate the fitted model at arbitrary times.
        
        Parameters
        ----------
        result : PseudodynamicsResult
            Fitted result.
        times : ndarray
            Time points for simulation.
        u0 : ndarray, optional
            Initial condition. If None, uses first time point from result.
            
        Returns
        -------
        u : ndarray of shape (n_grid, len(times))
            Simulated density.
        """
        if u0 is None:
            u0 = result.u[:, 0]
        
        return self._simulate_pde(u0, result.D, result.v, result.g, times)

