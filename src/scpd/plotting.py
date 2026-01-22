"""
Visualization tools for pseudodynamics results.

Provides functions for plotting density heatmaps, rate curves, ECDF comparisons,
and developmental potential.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, List, Tuple, Union
from pathlib import Path

from .results import PseudodynamicsResult
from .likelihood import compute_ecdf, compute_weighted_ecdf
from .pde import density_to_pdf, pdf_to_cdf


def plot_density_heatmap(
    result: PseudodynamicsResult,
    ax: Optional[plt.Axes] = None,
    cmap: str = "viridis",
    title: str = "Density u(s,t)"
) -> plt.Axes:
    """
    Plot density as a heatmap over s and t.
    
    Parameters
    ----------
    result : PseudodynamicsResult
        Fitted results.
    ax : Axes, optional
        Matplotlib axes. If None, creates new figure.
    cmap : str, default="viridis"
        Colormap.
    title : str
        Plot title.
        
    Returns
    -------
    ax : Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    extent = [
        result.time_values[0], result.time_values[-1],
        result.s_grid[0], result.s_grid[-1]
    ]
    
    im = ax.imshow(
        result.u, aspect='auto', origin='lower',
        extent=extent, cmap=cmap
    )
    ax.set_xlabel('Time')
    ax.set_ylabel('State s')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Density')
    
    return ax


def plot_ecdf_comparison(
    result: PseudodynamicsResult,
    s_samples_per_time: List[np.ndarray],
    landmark_info=None,
    weights_per_time: Optional[List[np.ndarray]] = None,
    n_cols: int = 3,
    figsize: Optional[Tuple[float, float]] = None
) -> Figure:
    """
    Plot ECDF vs model CDF for each time point.
    
    Parameters
    ----------
    result : PseudodynamicsResult
        Fitted results.
    s_samples_per_time : list of ndarrays
        Sample s values for each time point.
    landmark_info : LandmarkInfo, optional
        Landmark information if using landmarks.
    weights_per_time : list of ndarrays, optional
        Weights per time point (for landmark mode).
    n_cols : int, default=3
        Number of columns in subplot grid.
    figsize : tuple, optional
        Figure size.
        
    Returns
    -------
    fig : Figure
    """
    n_times = len(result.time_values)
    n_rows = int(np.ceil(n_times / n_cols))
    
    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()
    
    ds = result.s_grid[1] - result.s_grid[0]
    
    for k in range(n_times):
        ax = axes[k]
        
        # Model CDF
        p = density_to_pdf(result.u[:, k], ds)
        cdf_model = pdf_to_cdf(p, ds)
        
        # Empirical CDF
        if landmark_info is not None and landmark_info.enabled and weights_per_time is not None:
            ecdf = compute_weighted_ecdf(
                landmark_info.landmark_s,
                weights_per_time[k],
                result.s_grid
            )
        else:
            ecdf = compute_ecdf(s_samples_per_time[k], result.s_grid)
        
        ax.plot(result.s_grid, cdf_model, 'b-', lw=2, label='Model CDF')
        ax.plot(result.s_grid, ecdf, 'r--', lw=1.5, label='ECDF')
        ax.set_xlabel('s')
        ax.set_ylabel('CDF')
        ax.set_title(f't = {result.time_values[k]:.2f}')
        ax.legend(loc='lower right', fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    # Hide empty subplots
    for k in range(n_times, len(axes)):
        axes[k].set_visible(False)
    
    fig.tight_layout()
    return fig


def plot_rates(
    result: PseudodynamicsResult,
    true_rates: Optional[dict] = None,
    figsize: Tuple[float, float] = (12, 4)
) -> Figure:
    """
    Plot fitted rate functions D(s), v(s), g(s).
    
    Parameters
    ----------
    result : PseudodynamicsResult
        Fitted results.
    true_rates : dict, optional
        Ground truth rates with keys 'D', 'v', 'g', 's'.
    figsize : tuple
        Figure size.
        
    Returns
    -------
    fig : Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # D(s)
    ax = axes[0]
    ax.plot(result.s_grid, result.D, 'b-', lw=2, label='Fitted')
    if true_rates is not None and 'D' in true_rates:
        ax.plot(true_rates['s'], true_rates['D'], 'r--', lw=1.5, label='True')
    ax.set_xlabel('s')
    ax.set_ylabel('D(s)')
    ax.set_title('Diffusion')
    ax.legend()
    
    # v(s)
    ax = axes[1]
    ax.plot(result.s_grid, result.v, 'b-', lw=2, label='Fitted')
    if true_rates is not None and 'v' in true_rates:
        ax.plot(true_rates['s'], true_rates['v'], 'r--', lw=1.5, label='True')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel('s')
    ax.set_ylabel('v(s)')
    ax.set_title('Drift')
    ax.legend()
    
    # g(s)
    ax = axes[2]
    ax.plot(result.s_grid, result.g, 'b-', lw=2, label='Fitted')
    if true_rates is not None and 'g' in true_rates:
        ax.plot(true_rates['s'], true_rates['g'], 'r--', lw=1.5, label='True')
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel('s')
    ax.set_ylabel('g(s)')
    if result.mode == "distribution_only":
        ax.set_title('Net Growth (fixed to 0)')
    else:
        ax.set_title('Net Growth')
    ax.legend()
    
    fig.tight_layout()
    return fig


def plot_developmental_potential(
    result: PseudodynamicsResult,
    ax: Optional[plt.Axes] = None,
    title: str = "Developmental Potential"
) -> plt.Axes:
    """
    Plot developmental potential W(s).
    
    Parameters
    ----------
    result : PseudodynamicsResult
        Fitted results.
    ax : Axes, optional
        Matplotlib axes.
    title : str
        Plot title.
        
    Returns
    -------
    ax : Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    
    ax.plot(result.s_grid, result.W, 'b-', lw=2)
    ax.set_xlabel('State s')
    ax.set_ylabel('W(s)')
    ax.set_title(title)
    ax.axhline(0, color='gray', lw=0.5)
    
    return ax


def plot_diagnostics(
    result: PseudodynamicsResult,
    figsize: Tuple[float, float] = (10, 4)
) -> Figure:
    """
    Plot diagnostic information.
    
    Parameters
    ----------
    result : PseudodynamicsResult
        Fitted results.
    figsize : tuple
        Figure size.
        
    Returns
    -------
    fig : Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    diag = result.diagnostics
    
    # A-distances per time
    ax = axes[0]
    ax.bar(range(len(diag.A_values)), diag.A_values, color='steelblue')
    ax.set_xlabel('Time point')
    ax.set_ylabel('A-distance')
    ax.set_title('A-distance per time point')
    
    # σ_A per time
    ax = axes[1]
    ax.bar(range(len(diag.sigma_A_values)), diag.sigma_A_values, color='coral')
    ax.set_xlabel('Time point')
    ax.set_ylabel('σ_A')
    ax.set_title('Estimated σ_A per time point')
    
    fig.tight_layout()
    return fig


def save_all_plots(
    result: PseudodynamicsResult,
    s_samples_per_time: List[np.ndarray],
    output_dir: Union[str, Path],
    true_rates: Optional[dict] = None,
    landmark_info=None,
    weights_per_time: Optional[List[np.ndarray]] = None,
    prefix: str = ""
):
    """
    Save all standard plots to output directory.
    
    Parameters
    ----------
    result : PseudodynamicsResult
        Fitted results.
    s_samples_per_time : list of ndarrays
        Sample s values per time.
    output_dir : str or Path
        Output directory.
    true_rates : dict, optional
        Ground truth rates.
    landmark_info : LandmarkInfo, optional
        Landmark info.
    weights_per_time : list, optional
        Weights per time.
    prefix : str
        Filename prefix.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Density heatmap
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_density_heatmap(result, ax=ax)
    fig.savefig(output_dir / f"{prefix}density_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # ECDF comparison
    fig = plot_ecdf_comparison(
        result, s_samples_per_time,
        landmark_info=landmark_info,
        weights_per_time=weights_per_time
    )
    fig.savefig(output_dir / f"{prefix}ecdf_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Rates
    fig = plot_rates(result, true_rates=true_rates)
    fig.savefig(output_dir / f"{prefix}rates.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Developmental potential
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_developmental_potential(result, ax=ax)
    fig.savefig(output_dir / f"{prefix}developmental_potential.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Diagnostics
    fig = plot_diagnostics(result)
    fig.savefig(output_dir / f"{prefix}diagnostics.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_vector_field(
    adata,
    result: PseudodynamicsResult,
    basis: str = 'X_umap',
    color_by: Optional[str] = None,
    cmap: Optional[str] = None,
    palette: Optional[str] = None,
    grid_res: int = 50,
    smooth_sigma: float = 2.0,
    distance_cutoff: float = 0.02,
    ax: Optional[plt.Axes] = None,
    title: str = "Inferred Dynamics: Vector Field",
    **scatter_kwargs
) -> plt.Axes:
    """
    Plot vector field based on scPD fitted results.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with cell coordinates and pseudotime.
    result : PseudodynamicsResult
        scPD fitting results.
    basis : str, default='X_umap'
        Coordinate system, e.g. 'X_umap' or 'X_pca'.
    color_by : str, optional
        Variable for background scatter coloring. Options:
        - None: uses 'dpt_pseudotime'
        - 'cell_type': color by cell type
        - 's': color by normalized pseudotime
        - 'cell_W': color by developmental potential
        - 'cell_g': color by growth rate
        - or any column name in adata.obs
    cmap : str, optional
        Colormap for continuous variables (e.g. 'viridis', 'inferno', 'plasma').
        Default: 'viridis'
    palette : str, optional
        Palette for categorical variables (e.g. 'tab20', 'Set1', 'husl').
        Default: 'tab20'
    grid_res : int, default=50
        Grid resolution.
    smooth_sigma : float, default=2.0
        Gaussian smoothing parameter.
    distance_cutoff : float, default=0.02
        Distance threshold coefficient (relative to coordinate range) for masking empty regions.
    ax : Axes, optional
        Matplotlib axes. If None, creates new figure.
    title : str
        Plot title.
    **scatter_kwargs
        Additional arguments passed to scatter (e.g. s, alpha).
        
    Returns
    -------
    ax : Axes
        
    Examples
    --------
    >>> plot_vector_field(adata, result, color_by='s', cmap='plasma')
    >>> plot_vector_field(adata, result, color_by='cell_type', palette='Set1')
    """
    from scipy.interpolate import griddata, NearestNDInterpolator
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import cKDTree
    
    coords = adata.obsm[basis]
    x, y = coords[:, 0], coords[:, 1]
    
    if 'cell_W' in adata.obs:
        Z_values = adata.obs['cell_W'].values
        use_potential = True
    else:
        Z_values = adata.obs['s'].values
        use_potential = False
    
    rates = result.rates(adata.obs['s'].values)
    v_values = rates['v']
    
    pad = (x.max() - x.min()) * 0.05
    xi = np.linspace(x.min() - pad, x.max() + pad, grid_res)
    yi = np.linspace(y.min() - pad, y.max() + pad, grid_res)
    Xi, Yi = np.meshgrid(xi, yi)
    
    Wi = griddata((x, y), Z_values, (Xi, Yi), method='linear')
    vi = griddata((x, y), v_values, (Xi, Yi), method='linear', fill_value=0)
    
    mask_nan = np.isnan(Wi)
    if np.any(mask_nan):
        interp_nn = NearestNDInterpolator(coords, Z_values)
        Wi[mask_nan] = interp_nn(Xi[mask_nan], Yi[mask_nan])
    
    Wi_smooth = gaussian_filter(Wi, sigma=smooth_sigma)
    vi_smooth = gaussian_filter(vi, sigma=smooth_sigma)
    
    Gy, Gx = np.gradient(Wi_smooth)
    if use_potential:
        Dx, Dy = -Gx, -Gy
    else:
        Dx, Dy = Gx, Gy
    
    norm = np.sqrt(Dx**2 + Dy**2)
    norm[norm == 0] = 1
    
    U = (Dx / norm) * vi_smooth
    V = (Dy / norm) * vi_smooth
    
    tree = cKDTree(coords)
    grid_points = np.vstack([Xi.ravel(), Yi.ravel()]).T
    dists, _ = tree.query(grid_points)
    Dists_grid = dists.reshape(Xi.shape)
    
    cutoff = (x.max() - x.min()) * distance_cutoff
    mask = Dists_grid > cutoff
    U[mask] = np.nan
    V[mask] = np.nan
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    if color_by is None:
        color_by = 'dpt_pseudotime'
    
    default_scatter = {'s': 20, 'alpha': 0.5, 'edgecolors': 'none'}
    default_scatter.update(scatter_kwargs)
    
    if color_by in adata.obs.columns:
        color_data = adata.obs[color_by]
        
        if color_data.dtype.name == 'category' or color_data.dtype == 'object':
            categories = color_data.cat.categories if hasattr(color_data, 'cat') else color_data.unique()
            
            color_key = f'{color_by}_colors'
            if color_key in adata.uns:
                colors = adata.uns[color_key]
                color_map = dict(zip(categories, colors))
            else:
                if palette is None:
                    palette = 'tab20'
                
                if hasattr(plt.cm, palette):
                    colors = getattr(plt.cm, palette)(np.linspace(0, 1, len(categories)))
                else:
                    try:
                        import seaborn as sns
                        colors = sns.color_palette(palette, len(categories))
                    except:
                        colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))
                
                color_map = dict(zip(categories, colors))
            
            c = [color_map[cat] for cat in color_data]
            default_scatter.pop('cmap', None)
        else:
            c = color_data.values
            if cmap is not None:
                default_scatter['cmap'] = cmap
            elif 'cmap' not in default_scatter:
                default_scatter['cmap'] = 'viridis'
    else:
        c = adata.obs['dpt_pseudotime'].values
        if cmap is not None:
            default_scatter['cmap'] = cmap
        elif 'cmap' not in default_scatter:
            default_scatter['cmap'] = 'viridis'
    
    sc = ax.scatter(x, y, c=c, **default_scatter)
    
    if color_by in adata.obs.columns:
        color_data = adata.obs[color_by]
        if not (color_data.dtype.name == 'category' or color_data.dtype == 'object'):
            plt.colorbar(sc, ax=ax, label=color_by)
    
    ax.streamplot(
        Xi, Yi, U, V,
        color='k',
        linewidth=3,
        density=2,
        arrowsize=2.5,
        arrowstyle='-|>',
        minlength=0.1
    )
    
    ax.set_title(title)
    ax.axis('off')
    
    return ax

