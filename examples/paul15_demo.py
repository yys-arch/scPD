#!/usr/bin/env python
"""
Paul15 Demo - SCPD on Real Single-Cell Data

Demonstrates SCPD on the paul15 dataset from scanpy.

IMPORTANT NOTES:
- The "time" here is NOT physical time - it's pseudotime binned into stages
- g(s) is only identifiable with real population data
- adata.uns["iroot"] provides the root cell index for DPT computation

Data Preprocessing Considerations:
- Paul15 pseudotime is highly skewed (most cells in early stages)
- Using equal-spaced bins is more appropriate than quantile bins
- Focus on the main differentiation region (e.g., s < 0.4)

Usage:
    python paul15_demo.py [--s-max 0.4] [--n-bins 4]

Outputs saved to: outputs/paul15/
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import time
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(description="Paul15 Demo")
    parser.add_argument("--s-max", type=float, default=0.4,
                        help="Maximum pseudotime to include (default: 0.4)")
    parser.add_argument("--n-bins", type=int, default=4,
                        help="Number of equal-spaced bins (default: 4)")
    parser.add_argument("--output-dir", type=str, default="outputs/paul15",
                        help="Output directory")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("SCPD Paul15 Demo - Real Single-Cell Data")
    print("=" * 60)
    
    # Check for scanpy
    try:
        import scanpy as sc
    except ImportError:
        print("\nERROR: This demo requires scanpy. Install with:")
        print("  pip install scanpy")
        return
    
    # Load data
    print("\n1. Loading paul15 dataset...")
    adata = sc.datasets.paul15()
    print(f"   - Shape: {adata.shape[0]} cells × {adata.shape[1]} genes")
    print(f"   - Root cell index: {adata.uns.get('iroot', 'not set')}")
    
    # Preprocessing
    print("\n2. Preprocessing...")
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=1000)
    adata = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata)
    sc.tl.pca(adata, n_comps=50)
    print(f"   - PCA computed: {adata.obsm['X_pca'].shape}")
    
    sc.pp.neighbors(adata, n_neighbors=30, use_rep='X_pca')
    sc.tl.diffmap(adata)
    print("   - Diffusion map computed")
    
    if 'iroot' in adata.uns:
        sc.tl.dpt(adata)
        print(f"   - DPT computed using root cell {adata.uns['iroot']}")
    else:
        adata.uns['iroot'] = 0
        sc.tl.dpt(adata)
        print("   - DPT computed (using cell 0 as root)")
    
    # Get pseudotime
    s_raw = adata.obs['dpt_pseudotime'].values
    valid_mask = ~np.isnan(s_raw)
    if not np.all(valid_mask):
        print(f"   - Removing {np.sum(~valid_mask)} cells with NaN pseudotime")
        adata = adata[valid_mask].copy()
        s_raw = s_raw[valid_mask]
    
    s_full = (s_raw - np.min(s_raw)) / (np.max(s_raw) - np.min(s_raw))
    
    # Analyze pseudotime distribution
    print("\n3. Analyzing pseudotime distribution...")
    print(f"   - Mean: {s_full.mean():.3f}, Median: {np.median(s_full):.3f}")
    print(f"   - 75% of cells have s < {np.percentile(s_full, 75):.3f}")
    print(f"   - 95% of cells have s < {np.percentile(s_full, 95):.3f}")
    
    # Focus on main differentiation region
    print(f"\n4. Focusing on s < {args.s_max} (main differentiation region)...")
    mask = s_full < args.s_max
    s = s_full[mask] / args.s_max  # Renormalize to [0, 1]
    adata_sub = adata[mask].copy()
    n_cells = len(s)
    print(f"   - Using {n_cells} cells ({100*n_cells/len(s_full):.1f}% of total)")
    
    # Save processed data
    print("   - Saving processed data...")
    adata_sub.write(output_dir / "processed_paul15_data.h5ad")
    print(f"   - Saved processed data to {output_dir}/processed_paul15_data.h5ad")
    
    # Create equal-spaced bins
    print(f"\n5. Creating {args.n_bins} equal-spaced bins...")
    bin_edges = np.linspace(0, 1, args.n_bins + 1)
    time_labels = np.digitize(s, bin_edges[1:-1])
    
    for i in range(args.n_bins):
        count = np.sum(time_labels == i)
        if count > 0:
            s_bin = s[time_labels == i]
            print(f"   - Bin {i}: {count} cells, s ∈ [{s_bin.min():.3f}, {s_bin.max():.3f}]")
        else:
            print(f"   - Bin {i}: 0 cells (empty)")
    
    # Import scpd
    import scpd
    from scpd.plotting import (
        plot_rates,
        plot_ecdf_comparison,
    )
    
    # Prepare inputs
    print("\n6. Preparing inputs...")
    prepared = scpd.prepare_inputs(
        s=s,
        time_labels=time_labels,
        landmarks="off"
    )
    
    # Fit model
    print("\n7. Fitting model (mode=distribution_only)...")
    print("   Note: g(s) = 0 since no population data available")

    model = scpd.PseudodynamicsModel(n_grid=200, spline_df=6, stabilize_boundary=True)

    start_time = time.time()
    result = model.fit(
        prepared,
        mode="distribution_only",
        rho=1,  # Higher rho for smoother curves, less overfitting
        n_starts=5,
        n_bootstrap=5,
        random_state=0
    )
    elapsed = time.time() - start_time
    
    print(f"   - Completed in {elapsed:.1f}s")
    print(f"   - Total NLL: {result.diagnostics.total_nll:.2f}")
    print(f"   - Converged: {result.diagnostics.success}")
    
    # Analyze results
    print("\n8. Analyzing results...")
    print(f"   - D(s) range: [{result.D.min():.4f}, {result.D.max():.4f}]")
    print(f"   - v(s) range: [{result.v.min():.4f}, {result.v.max():.4f}]")
    
    v_positive = np.mean(result.v > 0) * 100
    print(f"   - v(s) > 0 in {v_positive:.1f}% of domain (forward differentiation)")
    
    # Save plots
    print(f"\n9. Saving plots to {output_dir}/...")
    
    # Rates
    fig = plot_rates(result)
    fig.suptitle(f'Fitted Rates (s < {args.s_max}, equal-spaced bins)', y=1.02)
    fig.savefig(output_dir / "rates.pdf", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("   - Saved rates.pdf")
    
    # ECDF comparison
    s_per_time = [s[time_labels == k] for k in range(args.n_bins)]
    fig = plot_ecdf_comparison(result, s_per_time)
    fig.suptitle('ECDF vs Model CDF', y=1.02)
    fig.savefig(output_dir / "ecdf_comparison.pdf", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("   - Saved ecdf_comparison.pdf")
    
    # Detailed rates figure with interpretation
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    ax = axes[0, 0]
    ax.plot(result.s_grid, result.D, 'b-', lw=2)
    ax.set_xlabel('s (pseudotime)')
    ax.set_ylabel('D(s)')
    ax.set_title('Diffusion Coefficient')
    ax.axhline(np.mean(result.D), color='gray', ls='--', lw=1, label=f'mean={np.mean(result.D):.4f}')
    ax.legend()
    
    ax = axes[0, 1]
    ax.plot(result.s_grid, result.v, 'r-', lw=2)
    ax.axhline(0, color='gray', ls='-', lw=0.5)
    ax.set_xlabel('s (pseudotime)')
    ax.set_ylabel('v(s)')
    ax.set_title('Drift Velocity')
    ax.fill_between(result.s_grid, 0, result.v, where=result.v>0, alpha=0.3, color='green', label='forward')
    ax.fill_between(result.s_grid, 0, result.v, where=result.v<0, alpha=0.3, color='red', label='backward')
    ax.legend()
    
    ax = axes[1, 0]
    ax.plot(result.s_grid, result.W, 'purple', lw=2)
    ax.set_xlabel('s (pseudotime)')
    ax.set_ylabel('W(s)')
    ax.set_title('Developmental Potential')
    ax.axhline(0, color='gray', ls='-', lw=0.5)
    
    ax = axes[1, 1]
    # A-distance diagnostic
    a_vals = result.diagnostics.A_values
    sigma_vals = result.diagnostics.sigma_A_values
    x = np.arange(len(a_vals))
    ax.bar(x, a_vals, color='steelblue', alpha=0.7, label='A-distance')
    ax.errorbar(x, a_vals, yerr=sigma_vals, fmt='none', color='black', capsize=3)
    ax.set_xlabel('Time bin')
    ax.set_ylabel('A-distance')
    ax.set_title('Model Fit Quality')
    ax.set_xticks(x)
    
    fig.tight_layout()
    fig.savefig(output_dir / "analysis.pdf", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("   - Saved analysis.pdf")

    # Embedding plot
    print("\n10. Computing visualization...")
    try:
        sc.tl.umap(adata_sub)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        sc1 = axes[0].scatter(
            adata_sub.obsm['X_umap'][:, 0],
            adata_sub.obsm['X_umap'][:, 1],
            c=s, cmap='viridis', s=3, alpha=0.7
        )
        axes[0].set_title('Colored by pseudotime s')
        plt.colorbar(sc1, ax=axes[0], label='s')
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        
        sc2 = axes[1].scatter(
            adata_sub.obsm['X_umap'][:, 0],
            adata_sub.obsm['X_umap'][:, 1],
            c=time_labels, cmap='tab10', s=3, alpha=0.7
        )
        axes[1].set_title('Colored by time bin')
        plt.colorbar(sc2, ax=axes[1], label='Time bin')
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        
        fig.tight_layout()
        fig.savefig(output_dir / "embedding.pdf", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("   - Saved embedding.pdf")
    except Exception as e:
        print(f"   - Skipping embedding: {e}")
    
    # Save result
    result.save(output_dir / "result.npz")
    print("   - Saved result.npz")
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)
    print("\n" + "=" * 60)
    print("INTERPRETATION OF RESULTS")
    print("=" * 60)
    print("""
1. D(s) - Diffusion Coefficient:
   - Measures stochasticity of state transitions
   - High D: cells can randomly switch states
   - Low D: cells are more committed to their state
   - Typically higher in early/progenitor states

2. v(s) - Drift Velocity:
   - Measures directional bias in differentiation
   - v > 0: cells tend to move toward higher s (forward differentiation)
   - v < 0: cells tend to move toward lower s (de-differentiation)
   - In normal differentiation, v > 0 throughout

3. W(s) - Developmental Potential:
   - W(s) = -∫v(s')ds' from 0 to s
   - Lower W means more mature/differentiated state
   - Higher W means more primitive/undifferentiated state
   - Cells flow from high W to low W (downhill)

4. A-distance - Model Fit Quality:
   - Measures discrepancy between model and data CDFs
   - Lower is better
   - Normalized by σ_A for statistical comparison
""")


if __name__ == "__main__":
    main()
