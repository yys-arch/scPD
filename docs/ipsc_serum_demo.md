# iPSC Serum Differentiation Demo - Real Time-Series Data

This demo applies SCPD to iPSC differentiation data in serum conditions, demonstrating a complete workflow with real time-series single-cell RNA-seq data including population dynamics.

## Dataset Requirements

This demo expects an AnnData object (`.h5ad` file) with:
- Time-series single-cell RNA-seq data
- `adata.obs['day']`: Time point labels (can be strings like 'D0', 'day_1', or numeric)
- `adata.uns['N_obs_full_history']`: (Optional) Dictionary mapping time points to population sizes
- Preprocessed expression matrix (normalized, log-transformed, highly variable genes selected)

## Key Features

Unlike the paul15 demo which uses pseudotime-based stages, this demo:
- Uses **real physical time points** (days of differentiation)
- Includes **population size data** to infer g(s) (net growth rate)
- Demonstrates **time point alignment** and data preprocessing
- Shows **vector field visualization** in reduced dimension space

## Method

### 1. Data Loading and Time Alignment

- Parse time labels from various formats ('D0', 'day_1', '1.0', etc.)
- Round fractional days to nearest integers (0.5→0, 1.5→1)
- Filter to target time range (default: days 0-9)
- Align population size data with filtered time points

### 2. Population Data Handling

The demo tries to extract population sizes in order:
1. From `adata.uns['N_obs_full_history']` dictionary (preferred)
2. Fallback: Calculate from current filtered data

**Important**: For accurate g(s) inference, population sizes should reflect true biological measurements (e.g., from FACS), not just cell counts in the dataset.

### 3. Pseudotime Calculation

Uses scPD's built-in functions:
- `find_robust_root()`: Finds geometric centroid of Day 0 cells as root
- `compute_normalized_pseudotime()`: Computes DPT-based pseudotime normalized to [0,1]

### 4. Model Fitting

Fits with `mode="with_population"`:
- Estimates D(s), v(s), and g(s)
- Uses population sizes to constrain growth dynamics
- Applies regularization (rho=1.0) for smooth curves

### 5. Visualization

Generates comprehensive plots:
- Rate functions D(s), v(s), g(s)
- ECDF comparison (model vs data)
- Vector fields in UMAP space
- Diagnostic plots

## Usage

```bash
# Basic run
python examples/ipsc_serum_demo.py --data-path /path/to/iPSC_serum.h5ad

# Limit to first 5 days
python examples/ipsc_serum_demo.py --data-path /path/to/data.h5ad --max-days 5

# Custom output directory
python examples/ipsc_serum_demo.py --data-path /path/to/data.h5ad --output-dir results/ipsc_analysis
```

### Command-line Arguments

- `--data-path`: Path to iPSC serum dataset (.h5ad) - **required**
- `--max-days`: Maximum day to include in analysis (default: 9)
- `--output-dir`: Output directory (default: outputs/ipsc_serum)

## Requirements

```bash
pip install scanpy anndata seaborn pandas scikit-learn
```

## Outputs

Saved to `outputs/ipsc_serum/`:

### Data Files
- `scpd_result.npz`: Fitted model result (can be loaded with `scpd.PseudodynamicsResult.load()`)
- `processed_data.h5ad`: Processed AnnData with pseudotime and cell-level values

### Diagnostic Plots
- `pseudotime_distribution.pdf`: Violin plots of pseudotime distribution by day
- `umap_comparison.pdf`: UMAP colored by day, pseudotime, and cell type (if available)

### Model Results
- `rates.pdf`: D(s), v(s), g(s) curves
- `ecdf_comparison.pdf`: ECDF vs model CDF per time point
- `detailed_analysis.pdf`: 4-panel analysis:
  - Diffusion coefficient D(s)
  - Drift velocity v(s) with forward/backward regions
  - Net growth rate g(s)
  - A-distance diagnostics per time point

### Vector Field Visualizations
- `vector_field_pseudotime.pdf`: Vector field colored by pseudotime
- `vector_field_day.pdf`: Vector field colored by time point
- `vector_field_celltype.pdf`: Vector field colored by cell type (if available)
- `vector_field_potential.pdf`: Vector field colored by developmental potential W(s)

## Interpretation

### Diffusion D(s)
- Measures stochasticity of state transitions
- Higher D: cells can randomly switch states
- Lower D: cells are more committed
- May vary along differentiation trajectory

### Drift v(s)
- Measures directional bias in differentiation
- v > 0: forward differentiation (cells progress along pseudotime)
- v < 0: backward movement (rare, may indicate de-differentiation)
- Magnitude indicates speed of differentiation

### Net Growth g(s)
- **Only meaningful with real population data**
- g > 0: net proliferation at state s
- g < 0: net death/differentiation at state s
- Reveals state-dependent growth dynamics

### Developmental Potential W(s)
- W(s) = ∫₀ˢ -v(s') ds'
- Lower W = more differentiated state
- Higher W = more primitive state
- Cells flow from high W to low W (downhill)

### Vector Fields
- Arrows show inferred cell flow in reduced dimension space
- Direction: where cells are moving
- Color: various biological features (pseudotime, cell type, potential)
- Reveals spatial organization of dynamics

## Technical Details

### Time Point Rounding

Fractional days are rounded to nearest integers:
- Handles experimental variation in collection times
- Ensures discrete time points for model fitting
- Original values preserved in `adata.obs['day_original']`

### Root Cell Selection

Uses geometric centroid of Day 0 cells:
- More robust than single cell selection
- Reduces sensitivity to outliers
- Ensures biologically meaningful starting point

### Landmark Acceleration

Automatically enabled for large datasets (>2500 cells):
- Uses MiniBatchKMeans clustering in PCA space
- Maintains accuracy while reducing computation
- Controlled by `landmarks="auto"` in `prepare_inputs()`

### Regularization

Uses rho=1.0 (moderate regularization):
- Balances smoothness and data fidelity
- Prevents overfitting to noise
- Can be adjusted based on data quality

## Data Preparation Tips

### Required Preprocessing

Before running this demo, your data should have:
1. Quality control (filter low-quality cells/genes)
2. Normalization (e.g., `sc.pp.normalize_total()`)
3. Log transformation (`sc.pp.log1p()`)
4. Highly variable gene selection
5. Scaling (`sc.pp.scale()`)

### Population Size Data

For accurate g(s) inference:
- Store true population sizes in `adata.uns['N_obs_full_history']`
- Format: `{0: 1000, 1: 1200, 2: 1500, ...}` (day → cell count)
- Should reflect biological measurements, not sampling artifacts

### Time Point Considerations

- Need at least 3-4 time points for reliable fitting
- More time points improve temporal resolution
- Uneven spacing is acceptable (model uses actual time values)

## Caveats

1. **Population Data Quality**: g(s) inference requires accurate population measurements
2. **Pseudotime Assumptions**: Assumes cells follow a continuous trajectory
3. **1D Simplification**: Real differentiation may involve branching/multiple paths
4. **Regularization Sensitivity**: Results depend on rho choice (consider cross-validation)
5. **Boundary Effects**: Estimates near s=0 and s=1 may be less reliable

## Comparison with Other Demos

| Feature | synthetic_demo | paul15_demo | ipsc_serum_demo |
|---------|---------------|-------------|-----------------|
| Time type | Synthetic | Pseudotime bins | Real time |
| Population data | Yes (synthetic) | No | Yes (real) |
| g(s) inference | Yes | No (fixed to 0) | Yes |
| Dataset size | Configurable | ~2,700 cells | Variable |
| Use case | Validation | Exploratory | Production |

## Expected Results

- D(s) should be positive and relatively smooth
- v(s) should be mostly positive (forward differentiation)
- g(s) may show peaks at proliferative stages
- Vector fields should align with biological expectations
- A-distances should be small and consistent

## Troubleshooting

### Missing Population Data

If `N_obs_full_history` is not available:
- Demo will calculate from current data (warning issued)
- Consider using `mode="distribution_only"` instead
- g(s) will not be biologically interpretable

### Convergence Issues

If fitting fails to converge:
- Try increasing `n_starts` (more random initializations)
- Adjust `rho` (higher for smoother, lower for more flexible)
- Check for data quality issues (outliers, sparse time points)

### Memory Issues

For very large datasets:
- Landmark acceleration will activate automatically
- Reduce `n_grid` in `PseudodynamicsModel()` (default: 200)
- Process subset of cells if needed

## Citation

If you use this workflow in your research, please cite the scPD paper (under review).
