# scPD: Single-Cell Pseudodynamics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Density dynamics fitting for 1D state-coordinate snapshots from single-cell data.**

scPD estimates **diffusion D(s)**, **drift v(s)**, and **net growth g(s)** along a normalized state coordinate s ∈ [0,1] from discrete-time snapshot distributions of single cells.

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
# From PyPI
pip install scpd

# From source
git clone https://github.com/yys-arch/scpd.git
cd scpd
pip install -e .

# With optional dependencies
pip install -e ".[scanpy]"  # For scanpy/AnnData integration
pip install -e ".[all]"     # All optional dependencies
```

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

# Find robust root cell (geometric centroid of Day 0 cells)
root_index = scpd.find_robust_root(adata, day_column='day', day_value=0.0)

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

### `scpd.prepare_inputs(s, time_labels, ...)`

Prepare input data for fitting.

- `s`: State coordinate array
- `time_labels`: Time/stage labels
- `N_obs`: Optional population sizes (enables `mode="with_population"`)
- `landmarks`: "auto" | "on" | "off" - landmark acceleration mode

### `scpd.find_robust_root(adata, ...)`

Find robust root cell for pseudotime calculation.

- `adata`: AnnData object
- `day_column`: Column name for time points (default: 'day')
- `day_value`: Time value for root selection (default: 0.0)
- `pca_key`: PCA coordinates key in obsm (default: 'X_pca')

Returns the index of the cell closest to the geometric centroid of specified time point cells.

### `scpd.plotting.plot_vector_field(adata, result, ...)`

Plot vector field showing cell dynamics in reduced dimension space.

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

Visualizes inferred dynamics as streamlines, with arrow direction showing drift velocity v(s) along the developmental potential gradient.

### `scpd.PseudodynamicsModel`

Main model class.

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

### `scpd.PseudodynamicsResult`

Result container with:

- `D`, `v`, `g`: Rate functions on grid
- `W`: Developmental potential W(s) = ∫₀ˢ -v(s') ds'
- `u`: Density at each (grid, time)
- `rates(s)`: Evaluate rates at arbitrary s
- `developmental_potential(s)`: Evaluate W at arbitrary s
- `to_cell_level(s_vector)`: Get all values for cell-level s
- `save(path)` / `load(path)`: Persistence

### Visualization

```python
from scpd.plotting import plot_vector_field

# Plot vector field with different backgrounds
plot_vector_field(adata, result, color_by='s')           # Pseudotime
plot_vector_field(adata, result, color_by='cell_type')   # Cell type
plot_vector_field(adata, result, color_by='cell_W')      # Potential
```

Visualizes inferred dynamics as streamlines showing drift velocity v(s) along developmental trajectories.

## Examples

See `examples/` for complete demos:

- `synthetic_time_series_demo.py`: Synthetic data with known ground truth
- `paul15_demo.py`: Real data example using scanpy's paul15 dataset

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

## Dependencies

**Core**: numpy, scipy, pandas, matplotlib, patsy, scikit-learn

**Optional**: scanpy, anndata (for AnnData integration)

## Citation
If you use scPD in your research, please cite our paper:
> Paper under review

## Contact
If you have any questions, please feel free to contact:
> yyusong526@gmail.com

## License

MIT License

