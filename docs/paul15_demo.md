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
2. Cells are binned into K quantile-based "stages"
3. These stages serve as the discrete time points for SCPD

This is a common use case in single-cell biology where true time-series data is unavailable, but pseudotime provides a proxy ordering.

### About adata.uns["iroot"]

The root cell index (`adata.uns["iroot"]`) specifies which cell is considered the "origin" for pseudotime:
- DPT computes shortest diffusion distances from this root
- The choice of root affects the direction of pseudotime
- paul15 includes a biologically meaningful root (early progenitor)

### Identifiability of g(s)

**g(s) requires real population measurements.**

In this demo, we construct `N_obs` from bin sizes as a demonstration, but this does NOT reflect true biological growth/death. In practice:

- **distribution_only mode** is the honest approach when you lack population data
- **with_population mode** requires actual cell counts at each time (e.g., from FACS)

## Method

### Preprocessing

1. Load paul15 dataset via `sc.datasets.paul15()`
2. Filter, normalize, log-transform
3. Select highly variable genes
4. Compute PCA (50 components)
5. Build neighbor graph
6. Compute diffusion map and DPT

### Stage Construction

Cells are binned by pseudotime quantiles into K stages (default 6):
- Bin 0: earliest cells (s near 0)
- Bin K-1: most differentiated (s near 1)

### Fitting

Both modes are run:

1. **distribution_only**: g(s) = 0, fits D and v only
2. **with_population**: uses bin sizes as N_obs (demo only)

## Usage

```bash
# Basic run
python examples/paul15_demo.py

# More bins
python examples/paul15_demo.py --n-bins 8

# Custom output
python examples/paul15_demo.py --output-dir results/paul15_analysis
```

## Requirements

```bash
pip install scanpy anndata
```

## Outputs

Saved to `outputs/paul15/`:

- `rates_distribution_only.png`: D, v, g curves (g=0)
- `rates_with_population.png`: D, v, g curves with population fitting
- `mode_comparison.png`: Both modes overlaid
- `ecdf_comparison.png`: ECDF vs model CDF per stage
- `developmental_potential.png`: W(s) curve
- `embedding.png`: Cells in low-D embedding, colored by s and stage
- `landmark_comparison.png`: Timing comparison
- `result_*.npz`: Saved results

## Interpretation

### Diffusion D(s)
- Higher D indicates more stochastic transitions
- May increase in early stages (uncommitted cells explore more states)
- May decrease in committed/terminal states

### Drift v(s)
- Positive v: cells tend to increase s (progress along pseudotime)
- Magnitude indicates speed of differentiation
- Shape reveals where differentiation is fastest

### Net Growth g(s)
- Only meaningful with real population data
- Would indicate state-dependent proliferation/death
- In this demo, the values are NOT biologically interpretable

### Developmental Potential W(s)
- W(s) = ∫₀ˢ -v(s') ds'
- Lower W = more differentiated state
- Decreasing W along s is expected for differentiating cells

## Technical Details

### Landmark Acceleration

For this dataset (~2,700 cells):
- Close to the 2,500 cell threshold for auto-landmarking
- Demo forces both on/off for comparison
- Landmarks are computed in PCA space (50D)
- MiniBatchKMeans with fixed random state

### PCA Features for Clustering

When landmark clustering is performed:
- Uses `adata.obsm['X_pca']` (first 50 PCs)
- Captures biological variation for sensible grouping
- Falls back to 1D s-clustering if PCA unavailable

## Caveats

1. **Pseudotime ≠ Real Time**: The dynamics estimated are along a computational ordering, not physical time
2. **g(s) Validity**: Without population data, g(s) should be fixed to 0
3. **Lineage Mixing**: paul15 has multiple lineages; a 1D model is a simplification
4. **Root Sensitivity**: DPT results depend on root cell choice

