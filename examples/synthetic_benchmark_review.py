"""
Generate the synthetic benchmark used in the scPD manuscript revision.

The script creates pseudo single-cell snapshots from known one-dimensional
Fokker-Planck dynamics and fits scPD in population-aware mode. It reproduces
the two biologically motivated scenarios used for the reviewer-response
benchmark:

    1. Progressive differentiation toward a stable terminal state.
    2. Transient plasticity window during fate specification.

Key changes relative to the previous version:
    1. N_obs is generated on a relative scale: N_obs_true = mass(t) / mass(0).
       This matches scPD's internal convention, where the initial KDE density is
       normalized to total mass 1.
    2. Density mass uses the same convention as scPD: mass = sum(u) * ds.
    3. The simulation grid is cell-centered, matching the finite-volume view used
       by scPD more closely.
    4. mode="with_population" is explicitly passed to model.fit().
    5. Parameter recovery reports Pearson, Spearman, nRMSE, and occupancy-weighted nRMSE.
    6. Population-scale diagnostics are printed and saved when available.

Default setting:
    - scenarios: directed_differentiation and fate_choice_plasticity
    - n_cells_per_time: 300
    - mode: with_population
    - density_noise_sigma: 0.0
    - n_grid_generate: 400
    - n_grid_fit: 80
    - measurement_sd: 0.0
    - population_noise_cv: 0.005
    - n_starts: 5
    - n_bootstrap: 20

Outputs:
    synthetic_test_results/reviewer_synthetic_benchmark/
        cells.csv
        N_obs.csv
        ground_truth.npz
        recovery_metrics.csv
        density_heatmap.png
        true_vs_inferred_curves.png
        population_fit.csv, if diagnostics contain N_model
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr, spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import scifont

    scifont.use("nature")
except ImportError:
    pass


# ============================================================
# 0. Import scPD
# ============================================================

# Try local development paths first. Adjust if your local structure differs.
for _p in ("scPD/src", "src", "."):
    p = Path(_p)
    if p.exists():
        sys.path.insert(0, str(p.resolve()))

from scpd import PseudodynamicsModel, prepare_inputs
from scpd.plotting import (
    plot_density_heatmap as plot_scpd_density_heatmap,
    plot_ecdf_comparison,
)


# ============================================================
# 1. Utility functions
# ============================================================

def cell_centered_grid(n_grid: int) -> Tuple[np.ndarray, float]:
    """
    Return cell-centered grid on [0, 1] and grid spacing.

    scPD internally uses cell centers such as 0.0025, 0.0075, ... for n_grid=200.
    """
    ds = 1.0 / n_grid
    s = (np.arange(n_grid, dtype=float) + 0.5) * ds
    return s, ds


def mass_sum(u: np.ndarray, ds: float, axis: int = -1) -> np.ndarray:
    """
    Total density mass using the same convention as scPD:
        mass = sum(u) * ds
    """
    return np.sum(u, axis=axis) * ds


def normalize_density(u: np.ndarray, ds: float, axis: int = -1) -> np.ndarray:
    """Normalize density so that sum(u) * ds = 1."""
    m = mass_sum(u, ds=ds, axis=axis)
    if np.any(m <= 0):
        raise ValueError("Density mass must be positive.")
    return u / np.expand_dims(m, axis=axis)


def safe_corr(y_true: np.ndarray, y_pred: np.ndarray, method: str = "pearson") -> float:
    """
    Compute correlation robustly.

    Returns nan if either vector is nearly constant.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return np.nan

    if method == "pearson":
        return float(pearsonr(y_true, y_pred)[0])
    if method == "spearman":
        return float(spearmanr(y_true, y_pred)[0])
    raise ValueError("method must be 'pearson' or 'spearman'")


def normalized_rmse(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    """
    Normalized root mean squared error.

    Normalization uses the dynamic range of the ground-truth function.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    denom = np.max(y_true) - np.min(y_true)

    return float(rmse / (denom + eps))


def weighted_normalized_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Occupancy-weighted nRMSE.

    This is useful because kinetic parameters are only identifiable in regions
    actually covered by snapshot densities.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = np.maximum(weights, 0.0)

    if np.sum(weights) <= 0:
        return np.nan

    weights = weights / np.sum(weights)
    wrmse = np.sqrt(np.sum(weights * (y_true - y_pred) ** 2))
    denom = np.max(y_true) - np.min(y_true)

    return float(wrmse / (denom + eps))


def get_field(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from dict-like or attribute-like objects."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


# ============================================================
# 2. Ground-truth kinetic functions
# ============================================================

def kinetic_functions(
    s: np.ndarray,
    scenario: str = "expansion_contraction",
    growth_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return D_true(s), v_true(s), g_true(s).

    Parameters
    ----------
    s
        State coordinate in [0, 1].
    scenario
        One of:
            "directed_differentiation"
            "proliferative_expansion"
            "fate_choice_plasticity"
            "expansion_contraction"
            "smooth_progression"
            "bottleneck"
    growth_scale
        Scale factor for g(s).
        Use growth_scale=1.0 for population-aware validation.
        Use growth_scale=0.0 for clean distribution-only validation.
    """

    s = np.asarray(s, dtype=float)

    if scenario == "directed_differentiation":
        # Biologically interpretable unidirectional differentiation.
        # Early progenitor states have higher variability and forward drift;
        # terminal states become progressively more stable.
        D = 0.006 + 0.020 * (1.0 - s) ** 2
        v = 0.020 + 0.25 * (1.0 - s) ** 2
        g = (
            0.080 * np.exp(-((s - 0.35) ** 2) / (2 * 0.18 ** 2))
            - 0.020 * s
        )

    elif scenario == "directed_differentiation_x10":
        # Same shape as directed_differentiation, but with 10x larger rates
        # to test whether stronger signal improves unconstrained recovery.
        D = 0.080 + 0.350 * (1.0 - s) ** 2
        v = 0.200 + 2.50 * (1.0 - s) ** 2
        g = (
            1.50 * np.exp(-((s - 0.35) ** 2) / (2 * 0.18 ** 2))
            - 0.40 * s
        )

    elif scenario == "proliferative_expansion":
        # A proliferative intermediate state expands strongly while cells
        # continue smooth forward progression along the developmental axis.
        # Diffusion is kept as a constant background term so this scenario
        # specifically tests recovery of state-dependent net growth.
        D = np.full_like(s, 0.015)
        v = 0.040 + 0.120 * (1.0 - 0.5 * s)
        g = 0.180 * np.exp(-((s - 0.45) ** 2) / (2 * 0.15 ** 2))

    elif scenario == "fate_choice_plasticity":
        # A transient fate-choice window increases state variability while
        # forward progression slows slightly near the plasticity window.
        # Mild expansion near the plastic state and late depletion make this
        # scenario more biologically realistic without making growth dominant.
        D = 0.008 + 0.035 * np.exp(-((s - 0.50) ** 2) / (2 * 0.13 ** 2))
        v = (
            0.050
            + 0.120 * (1.0 - s)
            - 0.040 * np.exp(-((s - 0.50) ** 2) / (2 * 0.15 ** 2))
        )
        g = (
            0.040 * np.exp(-((s - 0.45) ** 2) / (2 * 0.18 ** 2))
            - 0.020 * np.exp(-((s - 0.75) ** 2) / (2 * 0.16 ** 2))
        )

    elif scenario == "smooth_progression":
        # Simple smooth progression.
        # This scenario has weak D/v variation, so correlation can be unstable.
        D = 0.008 + 0.012 * (1.0 + 0.4 * np.sin(np.pi * s))
        v = 0.045 + 0.12 * (1.0 - 0.35 * s)
        g = 0.06 * np.exp(-((s - 0.50) ** 2) / (2 * 0.20 ** 2)) - 0.015

    elif scenario == "expansion_contraction":
        # Recommended main scenario.
        D = (
            0.006
            + 0.022 * np.exp(-((s - 0.25) ** 2) / (2 * 0.16 ** 2))
            + 0.006 * np.exp(-((s - 0.65) ** 2) / (2 * 0.20 ** 2))
        )

        v = 0.035 + 0.13 * (1.0 - s) * (1.0 + 0.20 * np.sin(2 * np.pi * s))

        g = (
            0.28 * np.exp(-((s - 0.42) ** 2) / (2 * 0.12 ** 2))
            - 0.22 * np.exp(-((s - 0.82) ** 2) / (2 * 0.10 ** 2))
            - 0.020
        )

    elif scenario == "bottleneck":
        # Harder scenario with a local bottleneck.
        bottleneck = np.exp(-((s - 0.52) ** 2) / (2 * 0.08 ** 2))

        D = 0.007 + 0.020 * (1.0 - 0.75 * bottleneck)
        D = np.maximum(D, 0.003)

        v = 0.040 + 0.12 * (1.0 - s) - 0.055 * bottleneck
        v = np.maximum(v, 0.010)

        g = (
            0.10 * np.exp(-((s - 0.30) ** 2) / (2 * 0.13 ** 2))
            - 0.10 * np.exp(-((s - 0.55) ** 2) / (2 * 0.10 ** 2))
            + 0.05 * np.exp(-((s - 0.75) ** 2) / (2 * 0.13 ** 2))
            - 0.015
        )

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    g = growth_scale * g

    return D, v, g


def initial_density(s: np.ndarray, ds: float) -> np.ndarray:
    """
    Initial density concentrated near early states.

    It is normalized using scPD's convention:
        sum(u0) * ds = 1.
    """
    s = np.asarray(s, dtype=float)

    u0 = np.exp(-((s - 0.08) ** 2) / (2 * 0.035 ** 2))
    u0 += 0.30 * np.exp(-((s - 0.18) ** 2) / (2 * 0.055 ** 2))

    u0 = np.maximum(u0, 1e-12)
    u0 = u0 / mass_sum(u0, ds=ds)

    return u0


# ============================================================
# 3. Forward Fokker-Planck simulator
# ============================================================

def fokker_planck_rhs_finite_volume(
    t: float,
    u: np.ndarray,
    s: np.ndarray,
    ds: float,
    D: np.ndarray,
    v: np.ndarray,
    g: np.ndarray,
) -> np.ndarray:
    """
    Finite-volume discretization of:

        du/dt = d/ds(D du/ds) - d/ds(vu) + g u

    Conservative form:

        du/dt = - dJ/ds + g u
        J = v u - D du/ds

    No-flux boundary condition:
        J(0) = J(1) = 0.
    """

    n = len(s)

    # Avoid negative density inside ODE solver.
    u = np.maximum(u, 0.0)

    # Face fluxes, length n + 1.
    J = np.zeros(n + 1, dtype=float)

    for i in range(1, n):
        D_face = 0.5 * (D[i - 1] + D[i])
        v_face = 0.5 * (v[i - 1] + v[i])

        # Upwind advection.
        u_upwind = u[i - 1] if v_face >= 0 else u[i]

        # Central diffusion gradient between adjacent cell centers.
        grad_u = (u[i] - u[i - 1]) / ds

        # Total flux.
        J[i] = v_face * u_upwind - D_face * grad_u

    # No-flux boundaries.
    J[0] = 0.0
    J[-1] = 0.0

    dudt = -(J[1:] - J[:-1]) / ds + g * u

    return dudt


def simulate_density(
    scenario: str = "expansion_contraction",
    growth_scale: float = 1.0,
    n_grid: int = 800,
    t_eval: np.ndarray = np.arange(10),
    density_noise_sigma: float = 0.0,
    random_state: int = 2026,
) -> Dict[str, Any]:
    """
    Generate synthetic density trajectories from known D, v, g.

    Crucial convention:
        N_obs_true is relative population size, mass(t) / mass(0), not absolute
        population count. This matches scPD's normalized initial density.
    """

    rng = np.random.default_rng(random_state)

    s, ds = cell_centered_grid(n_grid)
    D, v, g = kinetic_functions(s, scenario=scenario, growth_scale=growth_scale)
    u0 = initial_density(s, ds=ds)

    sol = solve_ivp(
        fun=lambda t, u: fokker_planck_rhs_finite_volume(t, u, s, ds, D, v, g),
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        y0=u0,
        t_eval=np.asarray(t_eval, dtype=float),
        method="BDF",
        rtol=1e-6,
        atol=1e-9,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    # Clean unnormalized density.
    U_clean = np.maximum(sol.y.T, 1e-12)

    # Relative population size using the same mass convention as scPD.
    mass_clean = mass_sum(U_clean, ds=ds, axis=1)
    N_obs_true = mass_clean / mass_clean[0]

    # Observed density for sampling cells.
    # Default: no density noise for main ground-truth recovery.
    U_obs = U_clean.copy()

    if density_noise_sigma is not None and density_noise_sigma > 0:
        U_noisy = []
        for u in U_obs:
            eps = rng.normal(0.0, density_noise_sigma, size=u.shape)
            u2 = u * np.exp(eps)
            u2 = gaussian_filter1d(u2, sigma=2.0)
            u2 = np.maximum(u2, 1e-12)
            U_noisy.append(u2)
        U_obs = np.vstack(U_noisy)

    # Normalized density for sampling snapshot cells.
    P_obs = normalize_density(U_obs, ds=ds, axis=1)

    return {
        "s_grid": s,
        "ds": ds,
        "t_eval": np.asarray(t_eval, dtype=float),
        "D_true": D,
        "v_true": v,
        "g_true": g,
        "density_unnormalized_clean": U_clean,
        "density_unnormalized_observed": U_obs,
        "density_normalized_observed": P_obs,
        "N_obs_true": N_obs_true,
        "mass_clean": mass_clean,
        "scenario": scenario,
        "growth_scale": growth_scale,
    }


# ============================================================
# 4. Sample pseudo single-cell snapshot observations
# ============================================================

def sample_snapshot_cells(
    sim: Dict[str, Any],
    repeat: int = 0,
    n_cells_per_time: int = 300,
    measurement_sd: float = 0.002,
    population_noise_cv: float = 0.02,
    random_state: int = 2026,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sample pseudo single-cell observations from normalized densities.

    Important:
        n_cells_per_time is sequencing/sample size.
        N_obs_true/N_obs_observed is relative population size, with N(0) near 1.
    """

    rng = np.random.default_rng(random_state)

    s_grid = sim["s_grid"]
    t_eval = sim["t_eval"]
    P = sim["density_normalized_observed"]
    scenario = sim["scenario"]

    all_rows = []
    cell_counter = 0

    for i, t in enumerate(t_eval):
        prob = P[i].copy()
        prob = prob / prob.sum()

        sampled_s = rng.choice(
            s_grid,
            size=n_cells_per_time,
            replace=True,
            p=prob,
        )

        # Measurement noise in observed state coordinate.
        sampled_s = sampled_s + rng.normal(0.0, measurement_sd, size=n_cells_per_time)
        sampled_s = np.clip(sampled_s, 0.0, 1.0)

        for x in sampled_s:
            all_rows.append({
                "cell_id": f"{scenario}_rep{repeat}_cell{cell_counter}",
                "s": float(x),
                "time": float(t),
                "scenario": scenario,
                "repeat": int(repeat),
            })
            cell_counter += 1

    cells = pd.DataFrame(all_rows)

    # Observed relative population sizes with mild log-normal measurement noise.
    N_true = sim["N_obs_true"]
    N_observed = N_true * np.exp(
        rng.normal(0.0, population_noise_cv, size=len(N_true))
    )

    N_obs = pd.DataFrame({
        "time": t_eval,
        "N_obs_true": N_true,
        "N_obs_observed": N_observed,
        "scenario": scenario,
        "repeat": int(repeat),
    })

    return cells, N_obs


def generate_single_dataset(
    scenario: str = "expansion_contraction",
    repeat: int = 0,
    n_grid: int = 800,
    t_eval: np.ndarray = np.arange(10),
    n_cells_per_time: int = 300,
    density_noise_sigma: float = 0.0,
    measurement_sd: float = 0.002,
    population_noise_cv: float = 0.02,
    growth_scale: float = 1.0,
    random_state: int = 2026,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Generate one synthetic dataset.

    For population-aware validation:
        growth_scale=1.0

    For clean distribution-only validation:
        growth_scale=0.0
    """

    sim = simulate_density(
        scenario=scenario,
        growth_scale=growth_scale,
        n_grid=n_grid,
        t_eval=t_eval,
        density_noise_sigma=density_noise_sigma,
        random_state=random_state,
    )

    cells, N_obs = sample_snapshot_cells(
        sim,
        repeat=repeat,
        n_cells_per_time=n_cells_per_time,
        measurement_sd=measurement_sd,
        population_noise_cv=population_noise_cv,
        random_state=random_state + 123,
    )

    return cells, N_obs, sim


# ============================================================
# 5. scPD fitting helpers
# ============================================================

def prepare_for_scpd(
    cells: pd.DataFrame,
    N_obs: Optional[pd.DataFrame] = None,
    mode: str = "with_population",
    sigma_N_frac: float = 0.02,
    landmarks: Optional[str] = None,
) -> Any:
    """
    Prepare input for scPD.

    landmarks:
        None      -> use scPD default behavior.
        "off"     -> try to disable landmarks if the installed API supports it.
        "auto"    -> try to force auto landmarks if the installed API supports it.
    """

    kwargs = {
        "s": cells["s"].values,
        "time_labels": cells["time"].values,
    }

    if mode == "with_population":
        if N_obs is None:
            raise ValueError("N_obs must be provided for with_population mode.")

        kwargs["N_obs"] = N_obs["N_obs_observed"].values
        kwargs["sigma_N_frac"] = sigma_N_frac

    if landmarks is not None:
        kwargs["landmarks"] = landmarks

    try:
        prepared = prepare_inputs(**kwargs)
    except TypeError as e:
        # Some scPD versions may not accept 'landmarks'.
        if "landmarks" in kwargs:
            print("prepare_inputs does not accept landmarks; retrying without landmarks.")
            kwargs.pop("landmarks")
            prepared = prepare_inputs(**kwargs)
        else:
            raise e

    # Population-scale sanity check.
    if mode == "with_population":
        print("  Prepared N_obs:", np.array2string(prepared.N_obs, precision=4))
        print(
            f"  Prepared N_obs range: "
            f"{np.min(prepared.N_obs):.4f} - {np.max(prepared.N_obs):.4f}"
        )
        if np.max(prepared.N_obs) > 100:
            print(
                "  WARNING: N_obs is >100. For the current scPD implementation, "
                "synthetic N_obs should usually be relative with N(0) ≈ 1."
            )

    return prepared


def fit_scpd_model(
    prepared: Any,
    mode: str = "with_population",
    n_grid_fit: int = 200,
    spline_df: int = 6,
    rho: float = 0.1,
    n_starts: int = 10,
    n_bootstrap: int = 100,
    lambda_N: float = 1.0,
    verbose: bool = True,
) -> Any:
    """
    Fit scPD model.

    Crucial:
        mode is explicitly passed to model.fit().
    """

    model = PseudodynamicsModel(
        n_grid=n_grid_fit,
        spline_df=spline_df,
    )

    result = model.fit(
        prepared,
        mode=mode,
        verbose=verbose,
        rho=rho,
        lambda_N=lambda_N,
        n_starts=n_starts,
        n_bootstrap=n_bootstrap,
    )

    return result


# ============================================================
# 6. Evaluation and plotting
# ============================================================

def get_mean_occupancy_weights(sim: Dict[str, Any], s_model: np.ndarray) -> np.ndarray:
    """
    Interpolate mean observed density onto the fitted model grid.

    Used for occupancy-weighted metrics.
    """
    P_mean = np.mean(sim["density_normalized_observed"], axis=0)
    weights = np.interp(s_model, sim["s_grid"], P_mean)
    weights = np.maximum(weights, 0.0)
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
    return weights


def evaluate_result_against_truth(
    result: Any,
    sim: Dict[str, Any],
    evaluate_g: bool = True,
) -> pd.DataFrame:
    """
    Compare fitted D/v/g with ground-truth functions.

    Returns a long-format DataFrame.
    """

    s_model = result.s_grid
    s_true = sim["s_grid"]
    weights = get_mean_occupancy_weights(sim, s_model)

    truth_dict = {
        "D": np.interp(s_model, s_true, sim["D_true"]),
        "v": np.interp(s_model, s_true, sim["v_true"]),
    }

    pred_dict = {
        "D": np.asarray(result.D),
        "v": np.asarray(result.v),
    }

    if evaluate_g:
        truth_dict["g"] = np.interp(s_model, s_true, sim["g_true"])
        pred_dict["g"] = np.asarray(result.g)

    rows = []

    for param in truth_dict:
        y_true = truth_dict[param]
        y_pred = pred_dict[param]

        rows.append({
            "parameter": param,
            "pearson": safe_corr(y_true, y_pred, method="pearson"),
            "spearman": safe_corr(y_true, y_pred, method="spearman"),
            "nrmse": normalized_rmse(y_true, y_pred),
            "weighted_nrmse": weighted_normalized_rmse(y_true, y_pred, weights),
            "true_min": float(np.min(y_true)),
            "true_max": float(np.max(y_true)),
            "pred_min": float(np.min(y_pred)),
            "pred_max": float(np.max(y_pred)),
        })

    return pd.DataFrame(rows)


def extract_N_model(result: Any) -> Optional[np.ndarray]:
    """Try to extract N_model from result.diagnostics across possible scPD versions."""
    diagnostics = get_field(result, "diagnostics", None)
    N_model = get_field(diagnostics, "N_model", None)
    if N_model is None:
        N_model = get_field(result, "N_model", None)
    if N_model is None:
        return None
    return np.asarray(N_model, dtype=float)


def extract_diag_value(result: Any, name: str) -> Any:
    """Try to extract diagnostic field."""
    diagnostics = get_field(result, "diagnostics", None)
    value = get_field(diagnostics, name, None)
    if value is None:
        value = get_field(result, name, None)
    return value


def debug_population_fit(prepared: Any, result: Any, out_dir: Path) -> None:
    """Print and save population fit diagnostics when available."""

    if not hasattr(prepared, "N_obs") or prepared.N_obs is None:
        return

    print("\nPopulation-scale check")
    print("-" * 60)
    print("N_obs:", np.array2string(np.asarray(prepared.N_obs), precision=4))

    N_model = extract_N_model(result)

    if N_model is not None:
        print("N_model:", np.array2string(N_model, precision=4))
        ratio = N_model / np.asarray(prepared.N_obs)
        print("N_model / N_obs:", np.array2string(ratio, precision=4))

        pop_df = pd.DataFrame({
            "time_index": np.arange(len(prepared.N_obs)),
            "N_obs": np.asarray(prepared.N_obs),
            "N_model": N_model,
            "N_model_over_N_obs": ratio,
        })
        pop_df.to_csv(out_dir / "population_fit.csv", index=False)

        plt.figure(figsize=(4.2, 3.2))
        plt.plot(pop_df["time_index"], pop_df["N_obs"], "o-", label="Observed N")
        plt.plot(pop_df["time_index"], pop_df["N_model"], "s--", label="Model N")
        plt.xlabel("Time index")
        plt.ylabel("Relative population size")
        plt.title("Population-size fit")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(out_dir / "population_fit.png", dpi=300)
        plt.close()
    else:
        print("N_model was not found in result diagnostics.")

    for name in ("nll_population", "nll_cdf", "penalty", "total_nll"):
        val = extract_diag_value(result, name)
        if val is not None:
            print(f"{name}: {val}")


def plot_density_heatmap(sim: Dict[str, Any], out_path: Path) -> None:
    """Save simulated density heatmap."""

    s = sim["s_grid"]
    t = sim["t_eval"]
    P = sim["density_normalized_observed"]

    plt.figure(figsize=(5.2, 3.4))
    plt.imshow(
        P,
        aspect="auto",
        origin="lower",
        extent=[s.min(), s.max(), t.min(), t.max()],
    )
    plt.xlabel("State coordinate s")
    plt.ylabel("Time")
    plt.title("Simulated snapshot densities")
    plt.colorbar(label="Normalized density")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_scpd_distribution_fit(
    result: Any,
    prepared: Any,
    out_dir: Path,
    sim: Optional[Dict[str, Any]] = None,
) -> None:
    """Save scPD fitted distribution diagnostics."""

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    plot_scpd_density_heatmap(result, ax=ax, title="scPD fitted density u(s,t)")
    fig.tight_layout()
    fig.savefig(out_dir / "fitted_density_heatmap.png", dpi=300)
    plt.close(fig)

    if sim is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), sharey=True)

        true_density = np.vstack([
            np.interp(result.s_grid, sim["s_grid"], u_t)
            for u_t in sim["density_unnormalized_clean"]
        ]).T
        fitted_density = np.asarray(result.u)
        result_ds = float(result.s_grid[1] - result.s_grid[0])
        true_density = normalize_density(true_density, result_ds, axis=0)
        fitted_density = normalize_density(fitted_density, result_ds, axis=0)

        vmax = float(max(np.max(true_density), np.max(fitted_density)))
        extent = [
            result.time_values[0],
            result.time_values[-1],
            result.s_grid[0],
            result.s_grid[-1],
        ]

        im0 = axes[0].imshow(
            true_density,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
        )
        axes[0].set_title("Ground-truth density")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("State s")
        plt.colorbar(im0, ax=axes[0], label="Density")

        im1 = axes[1].imshow(
            fitted_density,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
        )
        axes[1].set_title("scPD fitted density")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("State s")
        plt.colorbar(im1, ax=axes[1], label="Density")

        fig.tight_layout()
        fig.savefig(out_dir / "true_vs_fitted_density_heatmap.png", dpi=300)
        plt.close(fig)

    s_per_time = [np.asarray(x) for x in prepared.s_per_time]
    landmark_info = prepared.landmark_info if prepared.landmark_info.enabled else None
    weights_per_time = prepared.weights_per_time if prepared.landmark_info.enabled else None
    fig = plot_ecdf_comparison(
        result,
        s_per_time,
        landmark_info=landmark_info,
        weights_per_time=weights_per_time,
        n_cols=5,
    )
    fig.savefig(out_dir / "ecdf_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ecdf_error_bars(result: Any, out_dir: Path) -> None:
    """Save A-distance per time point with bootstrap sigma_A error bars."""

    diagnostics = get_field(result, "diagnostics", None)
    A_values = get_field(diagnostics, "A_values", None)
    sigma_A = get_field(diagnostics, "sigma_A_values", None)

    if A_values is None or sigma_A is None:
        return

    A_values = np.asarray(A_values, dtype=float)
    sigma_A = np.asarray(sigma_A, dtype=float)
    time_values = np.asarray(get_field(result, "time_values", np.arange(len(A_values))), dtype=float)
    z_scores = A_values / np.maximum(sigma_A, 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))

    axes[0].bar(np.arange(len(A_values)), A_values, color="#4C78A8", alpha=0.85)
    axes[0].set_xticks(np.arange(len(A_values)))
    axes[0].set_xticklabels([f"{t:g}" for t in time_values], rotation=0)
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("A-distance")
    axes[0].set_title("ECDF fitting error")

    axes[1].bar(np.arange(len(z_scores)), z_scores, color="#F58518", alpha=0.85)
    axes[1].axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    axes[1].set_xticks(np.arange(len(z_scores)))
    axes[1].set_xticklabels([f"{t:g}" for t in time_values], rotation=0)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("A-distance / sigma_A")
    axes[1].set_title("Bootstrap-normalized error")

    fig.tight_layout()
    fig.savefig(out_dir / "ecdf_error_bars.png", dpi=300)
    plt.close(fig)

    pd.DataFrame({
        "time": time_values,
        "A_distance": A_values,
        "sigma_A": sigma_A,
        "A_over_sigma_A": z_scores,
    }).to_csv(out_dir / "ecdf_error_metrics.csv", index=False)


def plot_true_vs_inferred(
    result: Any,
    sim: Dict[str, Any],
    out_path: Path,
    evaluate_g: bool = True,
) -> None:
    """Save true vs inferred kinetic curves."""

    s_model = result.s_grid
    s_true = sim["s_grid"]

    D_true = np.interp(s_model, s_true, sim["D_true"])
    v_true = np.interp(s_model, s_true, sim["v_true"])

    params = [
        ("D", D_true, np.asarray(result.D), "Diffusion D(s)"),
        ("v", v_true, np.asarray(result.v), "Drift v(s)"),
    ]

    if evaluate_g:
        g_true = np.interp(s_model, s_true, sim["g_true"])
        params.append(("g", g_true, np.asarray(result.g), "Net growth g(s)"))

    ncols = len(params)
    _, axes = plt.subplots(1, ncols, figsize=(4.0 * ncols, 3.2), sharex=True)

    if ncols == 1:
        axes = [axes]

    weights = get_mean_occupancy_weights(sim, s_model)

    for ax, (_, y_true, y_pred, title) in zip(axes, params):
        ax.plot(s_model, y_true, linewidth=2, label="Ground truth")
        ax.plot(s_model, y_pred, linewidth=2, linestyle="--", label="scPD inferred")
        ax.set_xlabel("State coordinate s")
        ax.set_title(title)

        pearson = safe_corr(y_true, y_pred, method="pearson")
        nrmse = normalized_rmse(y_true, y_pred)
        wnrmse = weighted_normalized_rmse(y_true, y_pred, weights)

        ax.text(
            0.04,
            0.96,
            f"r={pearson:.3f}\nnRMSE={nrmse:.3f}\nw-nRMSE={wnrmse:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
        )

    axes[0].legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_ground_truth_npz(sim: Dict[str, Any], out_path: Path) -> None:
    """Save ground-truth simulation data."""
    np.savez(
        out_path,
        s_grid=sim["s_grid"],
        ds=sim["ds"],
        t_eval=sim["t_eval"],
        D_true=sim["D_true"],
        v_true=sim["v_true"],
        g_true=sim["g_true"],
        density_unnormalized_clean=sim["density_unnormalized_clean"],
        density_unnormalized_observed=sim["density_unnormalized_observed"],
        density_normalized_observed=sim["density_normalized_observed"],
        N_obs_true=sim["N_obs_true"],
        mass_clean=sim["mass_clean"],
        scenario=sim["scenario"],
        growth_scale=sim["growth_scale"],
    )


# ============================================================
# 7. Main test function
# ============================================================

def run_single_test(
    scenario: str = "expansion_contraction",
    repeat: int = 0,
    mode: str = "with_population",
    n_cells_per_time: int = 300,
    n_grid_generate: int = 800,
    n_grid_fit: int = 200,
    spline_df: int = 6,
    rho: float = 0.1,
    n_starts: int = 10,
    n_bootstrap: int = 100,
    lambda_N: float = 1.0,
    density_noise_sigma: float = 0.0,
    measurement_sd: float = 0.002,
    population_noise_cv: float = 0.02,
    landmarks: Optional[str] = None,
    random_state: int = 2026,
    out_dir: str = "synthetic_test_results",
) -> Dict[str, Any]:
    """
    Run one synthetic validation experiment.

    mode:
        "with_population":
            Generate nonzero g and evaluate D, v, g.

        "distribution_only":
            Generate g=0 synthetic data and evaluate D, v only.
            This is cleaner because distribution-only cannot identify g.
    """

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"Scenario: {scenario}")
    print(f"Repeat: {repeat}")
    print(f"Mode: {mode}")
    print(f"Cells per time point: {n_cells_per_time}")
    print(f"rho: {rho}, n_starts: {n_starts}, lambda_N: {lambda_N}")
    print(f"n_bootstrap: {n_bootstrap}")
    print("=" * 70)

    if mode == "with_population":
        growth_scale = 1.0
        evaluate_g = True
    elif mode == "distribution_only":
        growth_scale = 0.0
        evaluate_g = False
    else:
        raise ValueError("mode must be 'with_population' or 'distribution_only'.")

    # --------------------------------------------------------
    # Generate synthetic data
    # --------------------------------------------------------

    print("Generating synthetic data...")

    cells, N_obs, sim = generate_single_dataset(
        scenario=scenario,
        repeat=repeat,
        n_grid=n_grid_generate,
        t_eval=np.arange(10),
        n_cells_per_time=n_cells_per_time,
        density_noise_sigma=density_noise_sigma,
        measurement_sd=measurement_sd,
        population_noise_cv=population_noise_cv,
        growth_scale=growth_scale,
        random_state=random_state,
    )

    print(f"  Total cells: {len(cells)}")
    print(f"  Time points: {N_obs.shape[0]}")
    print(
        f"  Observed relative population range: "
        f"{N_obs['N_obs_observed'].min():.4f} - {N_obs['N_obs_observed'].max():.4f}"
    )
    print(
        f"  True relative population range: "
        f"{N_obs['N_obs_true'].min():.4f} - {N_obs['N_obs_true'].max():.4f}"
    )

    if N_obs["N_obs_observed"].max() > 100:
        raise RuntimeError(
            "N_obs is still on an absolute scale. It should be relative, "
            "with N(0) approximately 1 for the current scPD implementation."
        )

    # Save generated data.
    cells.to_csv(out_dir_path / "cells.csv", index=False)
    N_obs.to_csv(out_dir_path / "N_obs.csv", index=False)
    save_ground_truth_npz(sim, out_dir_path / "ground_truth.npz")
    plot_density_heatmap(sim, out_dir_path / "density_heatmap.png")

    # --------------------------------------------------------
    # Prepare scPD input
    # --------------------------------------------------------

    print("\nPreparing data for scPD...")

    if mode == "with_population":
        prepared = prepare_for_scpd(
            cells=cells,
            N_obs=N_obs,
            mode=mode,
            sigma_N_frac=population_noise_cv,
            landmarks=landmarks,
        )
    else:
        prepared = prepare_for_scpd(
            cells=cells,
            N_obs=None,
            mode=mode,
            landmarks=landmarks,
        )

    # --------------------------------------------------------
    # Fit scPD
    # --------------------------------------------------------

    print("\nFitting scPD model...")

    result = fit_scpd_model(
        prepared=prepared,
        mode=mode,
        n_grid_fit=n_grid_fit,
        spline_df=spline_df,
        rho=rho,
        n_starts=n_starts,
        n_bootstrap=n_bootstrap,
        lambda_N=lambda_N,
        verbose=True,
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    debug_population_fit(prepared, result, out_dir_path)
    plot_scpd_distribution_fit(result, prepared, out_dir_path, sim=sim)
    plot_ecdf_error_bars(result, out_dir_path)
    if hasattr(result, "save"):
        result.save(out_dir_path / "scpd_result.npz")

    # --------------------------------------------------------
    # Evaluate parameter recovery
    # --------------------------------------------------------

    print("\nEvaluating parameter recovery...")

    metrics = evaluate_result_against_truth(
        result=result,
        sim=sim,
        evaluate_g=evaluate_g,
    )

    metrics.insert(0, "scenario", scenario)
    metrics.insert(1, "repeat", repeat)
    metrics.insert(2, "mode", mode)
    metrics.insert(3, "n_cells_per_time", n_cells_per_time)
    metrics.insert(4, "rho", rho)
    metrics.insert(5, "lambda_N", lambda_N)
    metrics.insert(6, "spline_df", spline_df)

    metrics.to_csv(out_dir_path / "recovery_metrics.csv", index=False)

    plot_true_vs_inferred(
        result=result,
        sim=sim,
        out_path=out_dir_path / "true_vs_inferred_curves.png",
        evaluate_g=evaluate_g,
    )

    print("\nRecovery metrics:")
    cols = [
        "parameter",
        "pearson",
        "spearman",
        "nrmse",
        "weighted_nrmse",
        "true_min",
        "true_max",
        "pred_min",
        "pred_max",
    ]
    print(metrics[cols])

    return {
        "cells": cells,
        "N_obs": N_obs,
        "sim": sim,
        "result": result,
        "metrics": metrics,
        "out_dir": out_dir_path,
    }


# ============================================================
# 8. Optional batch runner
# ============================================================

def run_small_batch(
    scenarios: Iterable[str] = ("expansion_contraction",),
    repeats: Iterable[int] = (0,),
    mode: str = "with_population",
    n_cells_per_time: int = 300,
    base_out_dir: str = "synthetic_batch_results",
    rho: float = 0.1,
    n_starts: int = 10,
    lambda_N: float = 1.0,
) -> pd.DataFrame:
    """
    Run a small batch and collect metrics.

    Default is intentionally small for fast debugging.
    For manuscript-level analysis, use more repeats and larger n_cells_per_time
    after confirming the single test works.
    """

    all_metrics = []

    for scenario in scenarios:
        for repeat in repeats:
            out_dir = Path(base_out_dir) / f"{scenario}_{mode}_rep{repeat:02d}"

            res = run_single_test(
                scenario=scenario,
                repeat=repeat,
                mode=mode,
                n_cells_per_time=n_cells_per_time,
                rho=rho,
                n_starts=n_starts,
                lambda_N=lambda_N,
                out_dir=str(out_dir),
                random_state=2026 + repeat,
            )

            all_metrics.append(res["metrics"])

    all_metrics_df = pd.concat(all_metrics, axis=0, ignore_index=True)
    Path(base_out_dir).mkdir(parents=True, exist_ok=True)
    all_metrics_df.to_csv(Path(base_out_dir) / "all_recovery_metrics.csv", index=False)

    print("\n" + "=" * 70)
    print("Batch summary")
    print("=" * 70)
    print(
        all_metrics_df[
            [
                "scenario",
                "repeat",
                "mode",
                "parameter",
                "pearson",
                "spearman",
                "nrmse",
                "weighted_nrmse",
            ]
        ]
    )

    return all_metrics_df


# ============================================================
# 9. Main
# ============================================================

if __name__ == "__main__":

    # Scenario 1: progressive differentiation toward a stable terminal state.
    run_single_test(
        scenario="directed_differentiation",
        repeat=0,
        mode="with_population",
        n_cells_per_time=300,
        n_grid_generate=400,
        n_grid_fit=80,
        spline_df=6,
        rho=10.0,
        n_starts=5,
        n_bootstrap=20,
        lambda_N=1.0,
        density_noise_sigma=0.0,
        measurement_sd=0.0,
        population_noise_cv=0.005,
        landmarks=None,
        random_state=2026,
        out_dir="synthetic_test_results/reviewer_synthetic_benchmark/scenario1_progressive_differentiation",
    )

    # Scenario 2: transient plasticity window during fate specification.
    run_single_test(
        scenario="fate_choice_plasticity",
        repeat=0,
        mode="with_population",
        n_cells_per_time=300,
        n_grid_generate=400,
        n_grid_fit=80,
        spline_df=6,
        rho=0.1,
        n_starts=5,
        n_bootstrap=20,
        lambda_N=0.1,
        density_noise_sigma=0.0,
        measurement_sd=0.0,
        population_noise_cv=0.005,
        landmarks=None,
        random_state=2026,
        out_dir="synthetic_test_results/reviewer_synthetic_benchmark/scenario2_transient_plasticity_window",
    )
