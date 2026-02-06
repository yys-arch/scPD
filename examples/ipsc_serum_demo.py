#!/usr/bin/env python
"""
iPSC Serum Differentiation Demo - SCPD on Real Single-Cell Data

This example demonstrates how to apply scPD to real single-cell RNA-seq data
from iPSC differentiation in serum conditions. The analysis includes:

1. Data preprocessing and time point alignment
2. Pseudotime calculation using diffusion pseudotime
3. scPD model fitting with population dynamics
4. Results visualization and interpretation

Requirements:
- scanpy for single-cell analysis
- seaborn for visualization
- A dataset with time-series single-cell data

Usage:
    python ipsc_serum_demo.py --data-path /path/to/data.h5ad --output-dir outputs/
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import time
import pandas as pd
import scanpy as sc
from pathlib import Path
import re
from sklearn.metrics import pairwise_distances

import scpd
from scpd.plotting import plot_rates, plot_ecdf_comparison, plot_vector_field


def parse_day(x):
    """Parse day values from various string formats."""
    try:
        # Handle 'D0', 'day_1', '1.0', '1' etc.
        if isinstance(x, str):
            match = re.search(r'(\d+(\.\d+)?)', x)
            if match:
                return float(match.group(1))
        return float(x)
    except Exception:
        return np.nan


def round_to_nearest_day(day_val):
    """Round fractional days to nearest integer (0.5->0, 1.5->1, etc.)."""
    return int(np.round(day_val))


def align_population_data(adata, present_times):
    """
    Align population size data with filtered time points.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object
    present_times : array
        Time points present in filtered data
        
    Returns
    -------
    N_obs : array
        Population sizes for each time point
    """
    N_obs = []
    source_found = False

    # Try to get N_obs from stored history
    if 'N_obs_full_history' in adata.uns:
        history_dict = adata.uns['N_obs_full_history']
        temp_n_obs = []
        missing_key = False
        
        print("  - Found 'N_obs_full_history' dictionary, extracting matching keys...")
        
        for t in present_times:
            count = None
            keys_to_try = [int(t), str(int(t)), t, str(t)]
            
            for k in keys_to_try:
                if k in history_dict:
                    count = history_dict[k]
                    break
            
            if count is not None:
                temp_n_obs.append(count)
            else:
                print(f"    Warning: Time point {t} missing in 'N_obs_full_history'.")
                missing_key = True
                break
                
        if not missing_key:
            N_obs = np.array(temp_n_obs)
            source_found = True
            print(f"  - Successfully matched N_obs from history: {N_obs}")

    # Fallback: calculate from current data
    if not source_found:
        print("  - Calculating N_obs from current filtered data (fallback)...")
        calculated_n_obs = []
        for t in present_times:
            c = np.sum(adata.obs['day'] == t)
            calculated_n_obs.append(c)
        N_obs = np.array(calculated_n_obs)
        print(f"  - Calculated N_obs: {N_obs}")

    return N_obs


def main():
    parser = argparse.ArgumentParser(description="iPSC Serum Differentiation Demo")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to iPSC serum dataset (.h5ad)")
    parser.add_argument("--output-dir", type=str, default="outputs/ipsc_serum",
                        help="Output directory")
    parser.add_argument("--max-days", type=int, default=9,
                        help="Maximum day to include in analysis")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("SCPD Demo - iPSC Serum Differentiation Data")
    print("=" * 60)
    
    # Set up scanpy
    sc.settings.figdir = output_dir / "figures"
    sc.settings.figdir.mkdir(exist_ok=True)
    
    # 1. Load and preprocess data
    print(f"Loading dataset from {args.data_path}...")
    adata = sc.read_h5ad(args.data_path)
    print(f"  - Loaded {adata.n_obs} cells, {adata.n_vars} genes.")

    # Parse and round time points
    adata.obs['day'] = adata.obs['day'].map(parse_day)
    adata = adata[~adata.obs['day'].isna()].copy()
    
    adata.obs['day_original'] = adata.obs['day'].copy()
    adata.obs['day'] = adata.obs['day'].map(round_to_nearest_day).astype(float)
    
    # Filter to target days
    target_days = [float(i) for i in range(args.max_days + 1)]
    mask = adata.obs['day'].isin(target_days)
    adata = adata[mask].copy()
    
    present_times = np.sort(np.unique(adata.obs['day']))
    print(f"  - Rounded fractional days to nearest integers")
    print(f"  - Filtered to days 0-{args.max_days}")
    print(f"  - Time points in data: {present_times}")
    print(f"  - Remaining cells: {adata.n_obs}")
    
    # 2. Align population data
    print("Aligning population size data...")
    N_obs = align_population_data(adata, present_times)
    adata.uns['N_obs'] = N_obs
    
    # 3. Preprocessing for pseudotime
    print("Preprocessing for pseudotime calculation...")
    sc.pp.scale(adata)
    sc.tl.pca(adata, n_comps=50)
    sc.pp.neighbors(adata, n_neighbors=30, use_rep='X_pca')
    sc.tl.diffmap(adata)
    
    # 4. Calculate pseudotime using scPD functions
    print("Computing pseudotime...")
    print("  - Finding robust root cell...")
    best_root = scpd.find_robust_root(adata, day_column='day', day_value=0.0)
    
    print("  - Computing normalized pseudotime...")
    s = scpd.compute_normalized_pseudotime(adata, n_dcs=10, percentile=99)
    
    # 5. Prepare time labels
    print("Preparing time labels...")
    day_values = adata.obs['day'].values.astype(float)
    unique_days = sorted(np.unique(day_values))
    
    if len(unique_days) != len(N_obs):
        raise ValueError(f"Mismatch! {len(unique_days)} time points but {len(N_obs)} N_obs values.")
    
    day_to_int = {d: i for i, d in enumerate(unique_days)}
    time_labels = np.array([day_to_int[d] for d in day_values])
    time_values = np.array(unique_days)
    
    # 6. Generate diagnostic plots
    print("Generating diagnostic plots...")
    
    # UMAP if not present
    if 'X_umap' not in adata.obsm:
        sc.tl.umap(adata)
    
    # Pseudotime distribution by day
    df = pd.DataFrame({
        'Pseudotime': adata.obs['dpt_pseudotime'],
        'Day': adata.obs['day'].astype(int).astype(str)
    })
    
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df, x='Day', y='Pseudotime', 
                   order=sorted(df['Day'].unique(), key=int), palette="viridis")
    plt.title("Pseudotime Distribution by Day")
    plt.ylabel("DPT Pseudotime")
    plt.xlabel("Day")
    plt.savefig(output_dir / "pseudotime_distribution.pdf")
    plt.close()
    
    # UMAP comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    sc.pl.umap(adata, color='day', ax=axes[0], title='Time (Day)', show=False)
    sc.pl.umap(adata, color='dpt_pseudotime', ax=axes[1], title='Pseudotime', show=False)
    
    if 'cell_type' in adata.obs:
        sc.pl.umap(adata, color='cell_type', ax=axes[2], title='Cell Type', show=False)
    else:
        axes[2].text(0.5, 0.5, 'No cell type\nannotation', 
                    ha='center', va='center', transform=axes[2].transAxes)
    
    plt.tight_layout()
    plt.savefig(output_dir / "umap_comparison.pdf")
    plt.close()
    
    # 7. Fit scPD model
    print("Fitting scPD model...")
    prepared = scpd.prepare_inputs(
        s=s,
        time_labels=time_labels,
        time_values=time_values,
        N_obs=N_obs,
        landmarks="auto"
    )
    
    model = scpd.PseudodynamicsModel(
        n_grid=200, 
        spline_df=6,
        stabilize_boundary=True
    )
    
    start_time = time.time()
    result = model.fit(
        prepared,
        mode="with_population",
        rho=1.0,        # Match original regularization strength
        n_starts=5,
        n_bootstrap=5,  # Match original bootstrap count
        random_state=0, # Match original random state
        verbose=True
    )
    elapsed = time.time() - start_time
    
    print(f"  - Completed in {elapsed:.1f}s")
    print(f"  - Total NLL: {result.diagnostics.total_nll:.2f}")
    print(f"  - Converged: {result.diagnostics.success}")
    
    # 8. Analyze results
    print("Analyzing results...")
    print(f"  - D(s) range: [{result.D.min():.4f}, {result.D.max():.4f}]")
    print(f"  - v(s) range: [{result.v.min():.4f}, {result.v.max():.4f}]")
    print(f"  - g(s) range: [{result.g.min():.4f}, {result.g.max():.4f}]")
    
    v_positive = np.mean(result.v > 0) * 100
    print(f"  - Forward drift in {v_positive:.1f}% of domain")
    
    # 9. Save results and plots
    print(f"Saving results to {output_dir}/...")
    
    # Save result
    result.save(output_dir / "scpd_result.npz")
    
    # Save processed data
    adata.write(output_dir / "processed_data.h5ad")
    
    # Plot rates
    fig = plot_rates(result)
    fig.suptitle(f'Fitted Dynamics ({len(unique_days)} timepoints)', y=1.02)
    fig.savefig(output_dir / "rates.pdf", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Plot ECDF comparison
    s_per_time = [s[time_labels == k] for k in range(len(unique_days))]
    fig = plot_ecdf_comparison(result, s_per_time)
    fig.suptitle('Model vs Data Comparison', y=1.02)
    fig.savefig(output_dir / "ecdf_comparison.pdf", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Detailed analysis plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Diffusion
    axes[0, 0].plot(result.s_grid, result.D, 'b-', lw=2)
    axes[0, 0].set_xlabel('Pseudotime (s)')
    axes[0, 0].set_ylabel('D(s)')
    axes[0, 0].set_title('Diffusion Coefficient')
    
    # Drift
    axes[0, 1].plot(result.s_grid, result.v, 'r-', lw=2)
    axes[0, 1].axhline(0, color='gray', ls='-', lw=0.5)
    axes[0, 1].set_xlabel('Pseudotime (s)')
    axes[0, 1].set_ylabel('v(s)')
    axes[0, 1].set_title('Drift Velocity')
    axes[0, 1].fill_between(result.s_grid, 0, result.v, 
                           where=result.v>0, alpha=0.3, color='green', label='forward')
    axes[0, 1].fill_between(result.s_grid, 0, result.v, 
                           where=result.v<0, alpha=0.3, color='red', label='backward')
    axes[0, 1].legend()
    
    # Growth
    axes[1, 0].plot(result.s_grid, result.g, 'g-', lw=2)
    axes[1, 0].axhline(0, color='gray', ls='-', lw=0.5)
    axes[1, 0].set_xlabel('Pseudotime (s)')
    axes[1, 0].set_ylabel('g(s)')
    axes[1, 0].set_title('Net Growth Rate')
    
    # Fit quality
    if hasattr(result.diagnostics, 'A_values'):
        a_vals = result.diagnostics.A_values
        sigma_vals = result.diagnostics.sigma_A_values
        x = np.arange(len(a_vals))
        axes[1, 1].bar(x, a_vals, color='steelblue', alpha=0.7)
        axes[1, 1].errorbar(x, a_vals, yerr=sigma_vals, fmt='none', color='black', capsize=3)
        axes[1, 1].set_xlabel('Time Point')
        axes[1, 1].set_ylabel('A-distance')
        axes[1, 1].set_title('Model Fit Quality')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([f'Day {int(d)}' for d in unique_days])
    
    plt.tight_layout()
    plt.savefig(output_dir / "detailed_analysis.pdf", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Vector field visualization
    print("Generating vector field plots...")
    
    # Store pseudotime in adata for vector field plotting
    adata.obs['s'] = s  # Required by plot_vector_field
    adata.obs['pseudotime'] = s
    
    # Vector field colored by pseudotime
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_vector_field(adata, result, basis='X_umap', color_by='pseudotime', 
                     ax=ax, title='Inferred Dynamics: Vector Field (colored by pseudotime)')
    plt.savefig(output_dir / "vector_field_pseudotime.pdf", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("   - Saved vector_field_pseudotime.pdf")
    
    # Vector field colored by time point
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_vector_field(adata, result, basis='X_umap', color_by='day',
                     ax=ax, title='Inferred Dynamics: Vector Field (colored by day)')
    plt.savefig(output_dir / "vector_field_day.pdf", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("   - Saved vector_field_day.pdf")
    
    # Vector field colored by cell type (if available)
    if 'cell_type' in adata.obs:
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_vector_field(adata, result, basis='X_umap', color_by='cell_type',
                         ax=ax, title='Inferred Dynamics: Vector Field (colored by cell type)')
        plt.savefig(output_dir / "vector_field_celltype.pdf", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("   - Saved vector_field_celltype.pdf")
    
    # Vector field colored by developmental potential
    cell_values = result.to_cell_level(s)
    adata.obs['cell_W'] = cell_values['W']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_vector_field(adata, result, basis='X_umap', color_by='cell_W',
                     ax=ax, title='Inferred Dynamics: Vector Field (colored by potential)')
    plt.savefig(output_dir / "vector_field_potential.pdf", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("   - Saved vector_field_potential.pdf")
    
    print("Analysis complete!")
    print(f"Results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
