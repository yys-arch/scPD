# Synthetic Time Series Demo

This demo generates synthetic single-cell snapshot data with known ground truth and demonstrates parameter recovery using SCPD.

## Purpose

- Validate that SCPD can recover true D(s), v(s), g(s) from simulated data
- Demonstrate the fitting workflow
- Compare performance with and without landmark acceleration

## Method

### Data Generation

1. **Define ground truth rates:**
   - D(s) = 0.01 (constant diffusion)
   - v(s) = 0.5 × (1 - 0.5s) (linear decreasing drift)
   - g(s) = 0.15 × exp(-(s-0.3)²/0.05) (bell-shaped growth)

2. **Simulate density dynamics:**
   - Initial condition: Gaussian centered at s=0.1
   - Forward simulate PDE for 6-8 time points
   - Generate N_true(t) = ∫u(s,t) ds

3. **Sample cells:**
   - For each time point, sample cell positions from u(s,t)
   - Add noise to N_true to get N_obs

### Fitting

Two modes are demonstrated:

1. **distribution_only**: Fits D and v only, fixes g=0
2. **with_population**: Fits D, v, and g using N_obs

### Evaluation

- Shape similarity between fitted and true curves (correlation after normalization)
- Visual comparison of density heatmaps
- ECDF vs model CDF at each time point

## Usage

```bash
# Basic run
python examples/synthetic_time_series_demo.py

# With more cells (enables landmarking)
python examples/synthetic_time_series_demo.py --n-cells 1000

# More time points
python examples/synthetic_time_series_demo.py --n-times 8

# Force landmark mode
python examples/synthetic_time_series_demo.py --landmarks on

# Distribution-only mode (no g fitting)
python examples/synthetic_time_series_demo.py --mode distribution_only

# Custom output directory
python examples/synthetic_time_series_demo.py --output-dir results/synthetic
```

### Command-line Arguments

- `--landmarks`: Landmark mode - "auto" (default), "on", or "off"
- `--mode`: Fitting mode - "distribution_only" or "with_population" (default)
- `--n-cells`: Cells per time point (default: 500)
- `--n-times`: Number of time points (default: 6)
- `--output-dir`: Output directory (default: outputs/synthetic_time_series)

## Outputs

Saved to `outputs/synthetic_time_series/`:

- `rates_comparison.png`: D, v, g curves (fitted vs true)
- `density_heatmap.png`: True vs fitted density over s and t
- `ecdf_comparison.png`: ECDF vs model CDF for each time
- `developmental_potential.png`: W(s) curve
- `diagnostics.png`: A-distances and σ_A per time
- `landmark_comparison.png`: Timing and curve comparison (if applicable)
- `result.npz`: Saved result for later loading

## Expected Results

- D(s) shape similarity > 0.8
- v(s) shape similarity > 0.8
- g(s) shape similarity > 0.6 (when using with_population mode)

## Notes

- Landmark acceleration provides significant speedup for large datasets
- Results may vary with random seed
- Regularization strength (rho) affects smoothness vs fidelity trade-off

