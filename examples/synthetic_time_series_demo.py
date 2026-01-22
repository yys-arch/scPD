#!/usr/bin/env python
"""
Synthetic Time Series Demo

Demonstrates SCPD on synthetic data with known ground truth.
Generates a time series of cell snapshots and recovers D(s), v(s), g(s).

Usage:
    python synthetic_time_series_demo.py [--landmarks auto|on|off] [--mode distribution_only|with_population]

Outputs saved to: outputs/synthetic_time_series/
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import time

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import scpd
from scpd.synthetic import generate_synthetic_dataset
from scpd.plotting import (
    plot_density_heatmap,
    plot_ecdf_comparison,
    plot_rates,
    plot_developmental_potential,
    plot_diagnostics
)


def main():
    parser = argparse.ArgumentParser(description="Synthetic Time Series Demo")
    parser.add_argument("--landmarks", choices=["auto", "on", "off"], default="auto",
                        help="Landmark mode (default: auto)")
    parser.add_argument("--mode", choices=["distribution_only", "with_population"], 
                        default="with_population", help="Fitting mode")
    parser.add_argument("--n-cells", type=int, default=500,
                        help="Cells per time point (default: 500)")
    parser.add_argument("--n-times", type=int, default=6,
                        help="Number of time points (default: 6)")
    parser.add_argument("--output-dir", type=str, default="outputs/synthetic_time_series",
                        help="Output directory")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("SCPD Synthetic Time Series Demo")
    print("=" * 60)
    
    # Generate synthetic dataset
    print("\n1. Generating synthetic dataset...")
    print(f"   - {args.n_times} time points, {args.n_cells} cells each")
    print(f"   - D(s): constant, v(s): sigmoidal decreasing, g(s): bell-shaped")
    
    dataset = generate_synthetic_dataset(
        n_times=args.n_times,
        n_cells_per_time=args.n_cells,
        D_type="constant",
        v_type="sigmoidal",
        g_type="bell" if args.mode == "with_population" else "zero",
        D_scale=0.01,
        v_scale=0.5,
        g_scale=0.15,
        random_state=42,
        time_span=(0.0, 1.0)
    )
    
    n_total_cells = len(dataset['s'])
    print(f"   - Total cells: {n_total_cells}")
    
    # Prepare inputs
    print("\n2. Preparing inputs...")
    
    N_obs = dataset['N_obs'] if args.mode == "with_population" else None
    
    prepared = scpd.prepare_inputs(
        s=dataset['s'],
        time_labels=dataset['time_labels'],
        time_values=dataset['time_values'],
        N_obs=N_obs,
        landmarks=args.landmarks
    )
    
    if prepared.landmark_info.enabled:
        print(f"   - Landmarks enabled: K = {prepared.landmark_info.n_landmarks}")
    else:
        print("   - Landmarks disabled (using all cells)")
    
    # Fit model
    print(f"\n3. Fitting model (mode={args.mode})...")
    
    model = scpd.PseudodynamicsModel(n_grid=200, spline_df=6)
    
    start_time = time.time()
    result = model.fit(
        prepared,
        mode=args.mode,
        # cv_rho=True,
        rho=0.01,
        n_starts=20,
        n_bootstrap=100,
        random_state=0,
        verbose=True
    )
    elapsed = time.time() - start_time
    
    print(f"   - Fitting completed in {elapsed:.2f}s")
    print(f"   - Total NLL: {result.diagnostics.total_nll:.4f}")
    print(f"   - Converged: {result.diagnostics.success}")
    
    # Compute shape similarities
    from scpd.utils import shape_similarity
    D_sim = shape_similarity(result.D, dataset['D_true'])
    v_sim = shape_similarity(result.v, dataset['v_true'])
    print(f"\n4. Recovery quality:")
    print(f"   - D(s) shape similarity: {D_sim:.3f}")
    print(f"   - v(s) shape similarity: {v_sim:.3f}")
    
    if args.mode == "with_population":
        g_sim = shape_similarity(result.g, dataset['g_true'])
        print(f"   - g(s) shape similarity: {g_sim:.3f}")
    
    # Save plots
    print(f"\n5. Saving plots to {output_dir}/...")
    
    true_rates = {
        's': dataset['s_grid'],
        'D': dataset['D_true'],
        'v': dataset['v_true'],
        'g': dataset['g_true']
    }
    
    # Rates comparison
    fig = plot_rates(result, true_rates=true_rates)
    fig.savefig(output_dir / "rates_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("   - Saved rates_comparison.png")
    
    # Density heatmap - fitted
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # True density
    extent = [dataset['time_values'][0], dataset['time_values'][-1], 0, 1]
    im = axes[0].imshow(dataset['u_true'], aspect='auto', origin='lower', 
                        extent=extent, cmap='viridis')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('State s')
    axes[0].set_title('True Density u(s,t)')
    plt.colorbar(im, ax=axes[0])
    
    # Fitted density
    plot_density_heatmap(result, ax=axes[1], title='Fitted Density u(s,t)')
    
    fig.tight_layout()
    fig.savefig(output_dir / "density_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("   - Saved density_heatmap.png")
    
    # ECDF comparison
    s_per_time = [dataset['s'][dataset['time_labels'] == k] for k in range(args.n_times)]
    fig = plot_ecdf_comparison(
        result, s_per_time,
        landmark_info=prepared.landmark_info if prepared.landmark_info.enabled else None,
        weights_per_time=prepared.weights_per_time if prepared.landmark_info.enabled else None
    )
    fig.savefig(output_dir / "ecdf_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("   - Saved ecdf_comparison.png")
    
    # Developmental potential
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_developmental_potential(result, ax=ax)
    fig.savefig(output_dir / "developmental_potential.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("   - Saved developmental_potential.png")
    
    # Diagnostics
    fig = plot_diagnostics(result)
    fig.savefig(output_dir / "diagnostics.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("   - Saved diagnostics.png")
    
    # Save result
    result.save(output_dir / "result.npz")
    print("   - Saved result.npz")
    
    # Landmark comparison if applicable
    if args.landmarks != "off" and n_total_cells >= 2500:
        print("\n6. Running landmark comparison...")
        
        # Run without landmarks
        prepared_no_lm = scpd.prepare_inputs(
            s=dataset['s'],
            time_labels=dataset['time_labels'],
            time_values=dataset['time_values'],
            N_obs=N_obs,
            landmarks="off"
        )
        
        start_time = time.time()
        result_no_lm = model.fit(
            prepared_no_lm,
            mode=args.mode,
            rho=0.1,
            n_starts=20,
            n_bootstrap=100,
            random_state=0
        )
        elapsed_no_lm = time.time() - start_time
        
        # Run with landmarks
        prepared_lm = scpd.prepare_inputs(
            s=dataset['s'],
            time_labels=dataset['time_labels'],
            time_values=dataset['time_values'],
            N_obs=N_obs,
            landmarks="on"
        )
        
        start_time = time.time()
        result_lm = model.fit(
            prepared_lm,
            mode=args.mode,
            rho=0.1,
            n_starts=10,
            n_bootstrap=100,
            random_state=0
        )
        elapsed_lm = time.time() - start_time
        
        print(f"   - Without landmarks: {elapsed_no_lm:.2f}s")
        print(f"   - With landmarks:    {elapsed_lm:.2f}s")
        print(f"   - Speedup: {elapsed_no_lm/elapsed_lm:.2f}x")
        
        # Compare curves
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        for ax, (name, true, no_lm, lm) in zip(axes, [
            ('D(s)', dataset['D_true'], result_no_lm.D, result_lm.D),
            ('v(s)', dataset['v_true'], result_no_lm.v, result_lm.v),
            ('g(s)', dataset['g_true'], result_no_lm.g, result_lm.g)
        ]):
            ax.plot(dataset['s_grid'], true, 'k--', lw=2, label='True')
            ax.plot(result_no_lm.s_grid, no_lm, 'b-', lw=1.5, label=f'No landmarks ({elapsed_no_lm:.1f}s)')
            ax.plot(result_lm.s_grid, lm, 'r-', lw=1.5, label=f'Landmarks ({elapsed_lm:.1f}s)')
            ax.set_xlabel('s')
            ax.set_ylabel(name)
            ax.legend(fontsize=8)
        
        fig.tight_layout()
        fig.savefig(output_dir / "landmark_comparison.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("   - Saved landmark_comparison.png")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()


