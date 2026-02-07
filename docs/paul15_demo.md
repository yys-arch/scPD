# Paul15 Demo - Real Single-Cell Data

This demo applies SCPD to the paul15 dataset, containing mouse hematopoietic progenitor cells differentiating along multiple lineages.

## Dataset

The paul15 dataset is a classic single-cell RNA-seq benchmark:
- ~2,700 mouse bone marrow cells
- Contains progenitor cells at various differentiation stages
- Multiple lineages (erythroid, myeloid, etc.)
- Includes a root cell index for pseudotime computation

## Important Notes

### Time is Pseudotime-Based

**This demo does NOT use physical time.** Instead:

1. Diffusion pseudotime (DPT) is computed from the data
2. Cells are binned into K equal-spaced "stages" along pseudotime
3. These stages serve as the discrete time points for SCPD

This is a common use case in single-cell biology where true time-series data is unavailable, but pseudotime provides a proxy ordering.

### About adata.uns["iroot"]

The root cell index (`adata.uns["iroot"]`) specifies which cell is considered the "origin" for pseudotime:
- DPT computes shortest diffusion distances from this root
- The choice of root affects the direction of pseudotime
- paul15 includes a biologically meaningful root (early progenitor)

### Identifiability of g(s)

**g(s) requires real population measurements.**

This demo uses `mode="distribution_only"` which fixes g(s) = 0. Without actual cell counts at each time point (e.g., from FACS), g(s) cannot be identified from normalized distributions alone.

### Pseudotime Distribution

Paul15 pseudotime is highly skewed - most cells are in early stages. This demo:
- Focuses on the main differentiation region (default: s < 0.4)
- Uses equal-spaced bins rather than quantile bins
- Renormalizes the selected range to [0, 1]

## Method

### Preprocessing

1. Load paul15 dataset via `sc.datasets.paul15()`
2. Filter, normalize, log-transform
3. Select highly variable genes (1000 genes)
4. Compute PCA (50 components)
5. Build neighbor graph
6. Compute diffusion map and DPT

### Stage Construction

Cells are binned by equal-spaced pseudotime intervals:
- Bin 0: earliest cells (s near 0)
- Bin K-1: most differentiated (s near 1)
- Default: 4 bins in the range s < 0.4

### Fitting

Runs `mode="distribution_only"`:
- Fits D(s) and v(s) only
- Fixes g(s) = 0 (no population data)
- Uses higher regularization (rho=1) for smoother curves

## Usage

```bash
# Basic run (s < 0.4, 4 bins)
python examples/paul15_demo.py

# Include more of the pseudotime range
python examples/paul15_demo.py --s-max 0.6

# More bins for finer resolution
python examples/paul15_demo.py --n-bins 6

# Custom output directory
python examples/paul15_demo.py --output-dir results/paul15_analysis
```

### Command-line Arguments

- `--s-max`: Maximum pseudotime to include (default: 0.4)
- `--n-bins`: Number of equal-spaced bins (default: 4)
- `--output-dir`: Output directory (default: outputs/paul15)

## Requirements

```bash
pip install scanpy anndata
```

## Outputs

Saved to `outputs/paul15/`:

- `rates.pdf`: D(s), v(s), g(s) curves
- `ecdf_comparison.pdf`: ECDF vs model CDF per stage
- `analysis.pdf`: Detailed analysis with 4 panels:
  - Diffusion coefficient D(s)
  - Drift velocity v(s) with forward/backward regions
  - Developmental potential W(s)
  - A-distance diagnostics per time bin
- `embedding.pdf`: UMAP visualization colored by pseudotime and time bins
- `processed_paul15_data.h5ad`: Processed AnnData object (subset to s < s_max)
- `result.npz`: Saved result for later loading

## Interpretation

### Diffusion D(s)
- Higher D indicates more stochastic transitions
- May increase in early stages (uncommitted cells explore more states)
- May decrease in committed/terminal states

### Drift v(s)
- Positive v: cells tend to increase s (progress along pseudotime)
- Negative v: cells tend to decrease s (rare in normal differentiation)
- Magnitude indicates speed of differentiation
- Shape reveals where differentiation is fastest

### Developmental Potential W(s)
- W(s) = ∫₀ˢ -v(s') ds'
- Lower W = more differentiated state
- Higher W = more primitive/undifferentiated state
- Cells flow from high W to low W (downhill in potential landscape)

### A-distance
- Measures model fit quality (discrepancy between model and data CDFs)
- Lower is better
- Error bars show bootstrap uncertainty

## Technical Details

### Equal-Spaced vs Quantile Bins

This demo uses equal-spaced bins because:
- Paul15 pseudotime is highly skewed (most cells at low s)
- Quantile bins would create uneven spacing in pseudotime
- Equal spacing better captures the continuous dynamics

### Focus on s < 0.4

The default focuses on early differentiation because:
- 75% of cells have s < 0.3
- Later stages have sparse data
- Main differentiation dynamics occur in early region

### Regularization

Uses rho=1 (higher than synthetic demo) because:
- Real data is noisier than synthetic
- Prevents overfitting to sparse bins
- Produces smoother, more interpretable curves

## Caveats

1. **Pseudotime ≠ Real Time**: The dynamics estimated are along a computational ordering, not physical time
2. **g(s) = 0**: Without population data, growth/death cannot be inferred
3. **Lineage Mixing**: paul15 has multiple lineages; a 1D model is a simplification
4. **Root Sensitivity**: DPT results depend on root cell choice
5. **Sparse Late Stages**: Few cells at high s may lead to unreliable estimates there

## Expected Results

- D(s) should be relatively stable or slightly decreasing
- v(s) should be mostly positive (forward differentiation)
- W(s) should decrease monotonically (loss of potential)
- A-distances should be small and consistent across bins
