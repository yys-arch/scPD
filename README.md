# scPD: a Python package for inferring continuous population dynamics from single-cell snapshot data

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/scpd.svg?v=0.1.0)](https://badge.fury.io/py/scpd)

**Density dynamics fitting for 1D state-coordinate snapshots from single-cell data.**

scPD estimates **diffusion D(s)**, **drift v(s)**, and **net growth g(s)** along a normalized state coordinate s ∈ [0,1] from discrete-time snapshot distributions of single cells.

## Key Features

- **Quantitative dynamics inference**: Separate diffusion, drift, and growth contributions
- **GPU acceleration**: Optional CUDA support for large datasets (>10K cells)
- **Landmark clustering**: Automatic acceleration for datasets >2.5K cells
- **Uncertainty quantification**: Bootstrap confidence intervals
- **Scanpy integration**: Seamless workflow with AnnData objects
- **Rich visualization**: Multiple plot types for comprehensive analysis

## Background

In snapshot single-cell data, the distribution of cells along a 1D state axis (e.g., pseudotime) evolves across discrete time points or stages. This evolution is typically driven by three factors:

1. **Drift**: Directed movement along the state axis (e.g., differentiation)
2. **Diffusion**: Stochastic spreading
3. **Net Growth**: State-dependent proliferation/death

scPD fits a continuous density dynamics model to separate and quantify these contributions:

$$\partial_t u = \partial_s(D \partial_s u) - \partial_s(v \cdot u) + g \cdot u$$

with no-flux boundary conditions.

## Installation

```bash
# Basic installation
pip install scpd

# With scanpy integration
pip install "scpd[scanpy]"

# With GPU acceleration (requires CUDA)
pip install "scpd[gpu]"

# Full installation with all features
pip install "scpd[all]"

# From source
git clone https://github.com/yys-arch/scpd.git
cd scpd
pip install -e .
```

### Optional Dependencies

- **scanpy**: `pip install "scpd[scanpy]"` - AnnData integration and scanpy workflows
- **gpu**: `pip install "scpd[gpu]"` - CUDA acceleration with CuPy (requires NVIDIA GPU)
- **dev**: Development tools (pytest, black, mypy)
- **docs**: Documentation building tools
- **all**: All optional dependencies

## Quick Start

```python
import scpd
import numpy as np

# Prepare your data
# s: state coordinate for each cell (will be normalized to [0,1])
# time_labels: time point/stage label for each cell
prepared = scpd.prepare_inputs(s, time_labels)

# Fit the model
model = scpd.PseudodynamicsModel()
result = model.fit(prepared, mode="distribution_only")

# Access results
D, v, g = result.D, result.v, result.g  # Rate functions on grid
W = result.W  # Developmental potential

# Evaluate at cell-level
cell_values = result.to_cell_level(s)
print(f"Cell-level drift: {cell_values['v']}")
print(f"Cell-level potential: {cell_values['W']}")

# Save results
result.save("my_results.npz")

# Load later
loaded_result = scpd.PseudodynamicsResult.load("my_results.npz")
```

## Scanpy Integration

For AnnData objects with scanpy preprocessing:

```python
import scanpy as sc
import scpd

# Standard preprocessing
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var.highly_variable].copy()
sc.pp.scale(adata)
sc.tl.pca(adata, n_comps=50)

# Find robust root cell (geometric centroid of Day 0 cells)
root_index = scpd.find_robust_root(adata, day_column='day', day_value=0.0)

# Compute neighbors and diffusion map (required for pseudotime)
sc.pp.neighbors(adata, n_neighbors=30, use_rep='X_pca')
sc.tl.diffmap(adata)
adata.uns['iroot'] = root_index  # Set root for DPT

# Compute normalized pseudotime
s = scpd.compute_normalized_pseudotime(adata, n_dcs=10, percentile=99)

# Prepare for fitting
prepared = scpd.prepare_inputs(s, adata.obs['day'].values)
```

## Fitting Modes

### `mode="distribution_only"` (default)
- Fits D(s) and v(s) only
- Fixes g(s) = 0
- Use when population size data is unavailable

### `mode="with_population"`
- Fits D(s), v(s), and g(s)
- Requires `N_obs` (observed population at each time)
- Enables inference of state-dependent growth/death

**Identifiability Note**: Without observed population sizes, g(s) is not identifiable from normalized distributions alone. Any growth profile can be absorbed by rescaling the density without affecting the distribution shape.

## Key Features

- **Natural cubic spline parameterization** for smooth, interpretable rate functions
- **Roughness penalty regularization** with optional cross-validation for ρ selection
- **Landmark (over-clustering) acceleration** for large datasets (>2500 cells)
- **Bootstrap uncertainty estimation** for A-distance
- **Save/load results** for reproducibility

## API Reference

### Core Functions

#### `scpd.prepare_inputs(s, time_labels, ...)`

Prepare input data for fitting.

**Parameters:**
- `s`: State coordinate array (will be normalized to [0,1])
- `time_labels`: Time/stage labels for each cell
- `N_obs`: Optional population sizes (enables `mode="with_population"`)
- `landmarks`: "auto" | "on" | "off" - landmark acceleration mode
- `landmark_threshold`: Cell count threshold for automatic landmarking (default: 2500)

**Returns:** `PreparedData` object

#### `scpd.find_robust_root(adata, ...)`

Find robust root cell for pseudotime calculation.

**Parameters:**
- `adata`: AnnData object
- `day_column`: Column name for time points (default: 'day')
- `day_value`: Time value for root selection (default: 0.0)
- `pca_key`: PCA coordinates key in obsm (default: 'X_pca')

**Returns:** Index of the cell closest to the geometric centroid of specified time point cells.

#### `scpd.compute_normalized_pseudotime(adata, ...)`

Compute normalized pseudotime using diffusion pseudotime.

**Parameters:**
- `adata`: AnnData object
- `n_dcs`: Number of diffusion components (default: 10)
- `percentile`: Percentile for robust normalization (default: 99)

**Returns:** Normalized pseudotime array s ∈ [0,1]

### Model Classes

#### `scpd.PseudodynamicsModel`

Main model class for fitting dynamics.

```python
model = scpd.PseudodynamicsModel(
    n_grid=200,       # Grid resolution
    spline_df=6,      # Spline degrees of freedom
)

result = model.fit(
    prepared,
    mode="distribution_only",  # or "with_population"
    rho=0.1,                   # Regularization strength
    cv_rho=False,              # Cross-validate rho
    n_starts=10,               # Multi-start optimization
)
```

#### `scpd.PseudodynamicsResult`

Result container with fitted dynamics and utilities.

**Key Attributes:**
- `D`, `v`, `g`: Rate functions on grid
- `W`: Developmental potential W(s) = ∫₀ˢ -v(s') ds'
- `u`: Density at each (grid, time)
- `s_grid`: Grid points
- `time_values`: Time points used

**Key Methods:**
- `rates(s)`: Evaluate D, v, g at arbitrary s values
- `developmental_potential(s)`: Evaluate W at arbitrary s values
- `to_cell_level(s_vector)`: Get all rate values for cell-level s coordinates
- `save(path)` / `load(path)`: Save/load results

### Visualization

#### `scpd.plotting.plot_vector_field(adata, result, ...)`

Plot vector field showing cell dynamics in reduced dimension space.

**Parameters:**
- `adata`: AnnData object with coordinates and pseudotime
- `result`: PseudodynamicsResult object
- `basis`: Coordinate system (default: 'X_umap')
- `color_by`: Background coloring variable:
  - `'s'`: Pseudotime
  - `'cell_type'`: Cell type labels
  - `'cell_W'`: Developmental potential
  - `'cell_g'`: Growth rate
  - Or any column in `adata.obs`
- `grid_res`: Grid resolution (default: 50)
- `smooth_sigma`: Gaussian smoothing parameter (default: 2.0)

#### Additional Plotting Functions

```python
from scpd.plotting import (
    plot_density_heatmap,      # Density evolution heatmap
    plot_rates,                # D(s), v(s), g(s) curves
    plot_ecdf_comparison,      # Model vs data ECDF comparison
    plot_developmental_potential, # W(s) potential landscape
    plot_diagnostics           # Fitting diagnostics
)

# Example usage
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
plot_density_heatmap(result, ax=axes[0,0])
plot_rates(result, ax=axes[0,1])
plot_ecdf_comparison(result, prepared, ax=axes[1,0])
plot_developmental_potential(result, ax=axes[1,1])
```

## Performance Optimization

### GPU Acceleration

For large datasets (>10K cells), enable GPU acceleration:

```python
# Install GPU dependencies
pip install "scpd[gpu]"

# GPU acceleration is automatically used when CuPy is available
# and dataset size exceeds the threshold
```

**Requirements:**
- NVIDIA GPU with CUDA support
- CuPy installation matching your CUDA version
- Sufficient GPU memory (recommended: >4GB for datasets >50K cells)

### Landmark Clustering

For datasets >2.5K cells, scPD automatically uses landmark clustering to accelerate computation:

```python
# Control landmark behavior
prepared = scpd.prepare_inputs(
    s, time_labels,
    landmarks="auto",           # "auto", "on", or "off"
    landmark_threshold=2500     # Threshold for automatic activation
)
```

**Benefits:**
- Reduces computational complexity from O(n²) to O(k²) where k << n
- Maintains statistical accuracy through weighted bootstrap
- Automatic selection of optimal cluster number

### Memory Management

For very large datasets:

```python
# Reduce grid resolution for memory efficiency
model = scpd.PseudodynamicsModel(n_grid=100)  # Default: 200

# Use fewer bootstrap samples for uncertainty estimation
result = model.fit(prepared, n_bootstrap=50)  # Default: 100
```

## Troubleshooting

### Common Issues

**Convergence Problems:**
```python
# Try multiple random starts
result = model.fit(prepared, n_starts=20)

# Adjust regularization
result = model.fit(prepared, rho=0.01)  # Lower for more flexibility

# Use cross-validation for optimal rho
result = model.fit(prepared, cv_rho=True)
```

**Memory Errors:**
```python
# Enable landmark clustering manually
prepared = scpd.prepare_inputs(s, time_labels, landmarks="on")

# Reduce grid resolution
model = scpd.PseudodynamicsModel(n_grid=100)
```

**Poor Fit Quality:**
```python
# Check data preprocessing
print(f"Pseudotime range: [{s.min():.3f}, {s.max():.3f}]")
print(f"Time points: {np.unique(time_labels)}")

# Visualize input data
plt.figure(figsize=(10, 4))
for t in np.unique(time_labels):
    mask = time_labels == t
    plt.hist(s[mask], alpha=0.6, label=f't={t}', bins=50)
plt.legend()
plt.show()
```

## Method Details

### Discretization
- Uniform grid on [0, 1] (default: 200 points)
- Finite volume flux discretization with upwinding for advection
- BDF time integration (scipy.integrate.solve_ivp)

### Observation Model
- A-distance between model CDF and empirical CDF
- Bootstrap estimation of σ_A (or replicate-based if available)
- Optional population size matching term

### Regularization
- Integrated squared second derivative penalty: ∫(f''(s))² ds
- Applied to D, v, g coefficient vectors
- Optional leave-one-time-out CV for ρ selection

### Landmark Acceleration
For large datasets (>2500 cells by default):
- MiniBatchKMeans clustering (in PCA space if available)
- Weighted ECDF using cluster representatives
- Multinomial bootstrap for uncertainty

## Examples

See `examples/` for complete demos:

- `synthetic_time_series_demo.py`: Synthetic data with known ground truth
- `paul15_demo.py`: Real data example using scanpy's paul15 dataset  
- `ipsc_serum_demo.py`: iPSC differentiation in serum conditions - demonstrates real-world application with time point alignment and population dynamics

### Running Examples

```bash
# Synthetic data demo
python examples/synthetic_time_series_demo.py

# Paul15 hematopoiesis data
python examples/paul15_demo.py

# iPSC serum differentiation (requires your own data)
python examples/ipsc_serum_demo.py --data-path /path/to/data.h5ad --output-dir outputs/
```

## Testing

scPD includes a comprehensive test suite to ensure reliability:

```bash
# Install development dependencies
pip install "scpd[dev]"

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_landmarking_rules.py

# Run with coverage report
pytest tests/ --cov=scpd --cov-report=html
```

**Test Coverage:**
- Core algorithm correctness (A-distance, ECDF)
- PDE solver (mass conservation, boundary conditions)
- Landmark clustering rules and optimization
- Spline basis functions
- Model fitting and recovery on synthetic data

All 53 tests pass successfully, ensuring the package works correctly across different scenarios.

## Dependencies

**Core**: numpy, scipy, pandas, matplotlib, patsy, scikit-learn

**Optional**: scanpy, anndata (for AnnData integration), cupy (for GPU acceleration)

## Citation
If you use scPD in your research, please cite our paper:
> Paper under review

## Contact
If you have any questions, please feel free to contact:
> yyusong526@gmail.com

## License

MIT License

