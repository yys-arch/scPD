"""
Optimization routines for parameter estimation.

Implements multi-start L-BFGS-B optimization and cross-validation for rho selection.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Callable, Optional, Tuple, List, Dict, Any
import warnings


def multi_start_optimize(
    objective_fn: Callable[[np.ndarray], float],
    n_params: int,
    n_starts: int = 10,
    bounds: Optional[List[Tuple[float, float]]] = None,
    x0_base: Optional[np.ndarray] = None,
    random_state: int = 0,
    maxiter: int = 500,
    verbose: bool = False
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Multi-start L-BFGS-B optimization.
    
    Parameters
    ----------
    objective_fn : callable
        Objective function f(theta) -> float to minimize.
    n_params : int
        Number of parameters.
    n_starts : int, default=10
        Number of random starting points.
    bounds : list of tuples, optional
        Parameter bounds. If None, uses (-5, 5) for all.
    x0_base : ndarray, optional
        Base initial point. Random perturbations added for multiple starts.
    random_state : int, default=0
        Random seed.
    maxiter : int, default=500
        Maximum iterations per start.
    verbose : bool, default=False
        Print progress.
        
    Returns
    -------
    x_best : ndarray
        Best parameter vector found.
    f_best : float
        Best objective value.
    info : dict
        Optimization info with keys 'n_iterations', 'success', 'message', 'all_results'.
    """
    rng = np.random.default_rng(random_state)
    
    if bounds is None:
        bounds = [(-5.0, 5.0)] * n_params
    
    if x0_base is None:
        x0_base = np.zeros(n_params)
    
    best_result = None
    best_f = np.inf
    all_results = []
    
    for i in range(n_starts):
        # Generate starting point
        if i == 0:
            x0 = x0_base.copy()
        else:
            # Random perturbation
            x0 = x0_base + rng.normal(0, 1, n_params)
            # Clip to bounds
            for j, (lb, ub) in enumerate(bounds):
                x0[j] = np.clip(x0[j], lb, ub)
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = minimize(
                    objective_fn,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': maxiter, 'disp': False}
                )
            
            all_results.append({
                'x': result.x,
                'fun': result.fun,
                'success': result.success,
                'nit': result.nit
            })
            
            if result.fun < best_f:
                best_f = result.fun
                best_result = result
                
            if verbose:
                print(f"  Start {i+1}/{n_starts}: f = {result.fun:.4f}, success = {result.success}")
                
        except Exception as e:
            if verbose:
                print(f"  Start {i+1}/{n_starts}: failed with {e}")
            all_results.append({
                'x': x0,
                'fun': np.inf,
                'success': False,
                'nit': 0,
                'error': str(e)
            })
    
    if best_result is None:
        raise RuntimeError("All optimization starts failed")
    
    info = {
        'n_iterations': best_result.nit,
        'success': best_result.success,
        'message': best_result.message if hasattr(best_result, 'message') else "",
        'all_results': all_results
    }
    
    return best_result.x, best_f, info


def cross_validate_rho(
    build_objective_fn: Callable[[List[int]], Callable],
    prepare_held_out_fn: Callable[[int], Callable],
    n_times: int,
    rho_grid: np.ndarray,
    n_params: int,
    n_starts: int = 5,
    random_state: int = 0,
    verbose: bool = False
) -> Tuple[float, Dict[str, Any]]:
    """
    Leave-one-time-point-out cross-validation for rho selection.
    
    Parameters
    ----------
    build_objective_fn : callable
        Function(time_indices) -> objective_fn that builds objective for given times.
    prepare_held_out_fn : callable
        Function(held_out_idx) -> eval_fn that returns held-out NLL evaluation function.
    n_times : int
        Number of time points.
    rho_grid : ndarray
        Grid of rho values to try.
    n_params : int
        Number of parameters.
    n_starts : int, default=5
        Number of starts per fold.
    random_state : int, default=0
        Random seed.
    verbose : bool, default=False
        Print progress.
        
    Returns
    -------
    best_rho : float
        Rho with lowest average held-out NLL.
    cv_info : dict
        Cross-validation results.
    """
    cv_scores = {rho: [] for rho in rho_grid}
    
    for fold_idx in range(n_times):
        if verbose:
            print(f"CV fold {fold_idx + 1}/{n_times}")
        
        # Indices for training (all except fold_idx)
        train_indices = [i for i in range(n_times) if i != fold_idx]
        
        # Held-out evaluation function
        eval_held_out = prepare_held_out_fn(fold_idx)
        
        for rho in rho_grid:
            # Build objective for training times with this rho
            obj_fn = build_objective_fn(train_indices, rho)
            
            # Optimize
            try:
                x_opt, _, _ = multi_start_optimize(
                    obj_fn, n_params, n_starts=n_starts,
                    random_state=random_state + fold_idx
                )
                
                # Evaluate on held-out
                held_out_nll = eval_held_out(x_opt)
                cv_scores[rho].append(held_out_nll)
                
            except Exception as e:
                if verbose:
                    print(f"  rho={rho:.2e} failed: {e}")
                cv_scores[rho].append(np.inf)
    
    # Compute mean scores
    mean_scores = {rho: np.mean(scores) for rho, scores in cv_scores.items()}
    best_rho = min(mean_scores, key=mean_scores.get)
    
    cv_info = {
        'rho_grid': rho_grid,
        'cv_scores': cv_scores,
        'mean_scores': mean_scores,
        'best_rho': best_rho
    }
    
    if verbose:
        print(f"Best rho: {best_rho:.2e} (mean held-out NLL: {mean_scores[best_rho]:.4f})")
    
    return best_rho, cv_info

