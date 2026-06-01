# Reviewer benchmark and sensitivity analyses

This document describes the scripts used for the additional analyses added during manuscript revision.

## Synthetic benchmark

The synthetic benchmark evaluates scPD against known ground-truth Fokker-Planck dynamics. Two scenarios are included:

1. Progressive differentiation toward a stable terminal state.
2. Transient plasticity window during fate specification.

Run from the repository root:

```bash
python examples/synthetic_benchmark_review.py
```

The default settings match the manuscript-revision runs:

| Setting | Scenario 1 | Scenario 2 |
| --- | ---: | ---: |
| Cells per time point | 300 | 300 |
| Generation grid | 400 | 400 |
| Fitting grid | 80 | 80 |
| Spline degrees of freedom | 6 | 6 |
| Measurement noise SD | 0.0 | 0.0 |
| Population noise CV | 0.005 | 0.005 |
| Regularization strength `rho` | 10.0 | 0.1 |
| Population loss weight `lambda_N` | 1.0 | 0.1 |
| Optimizer starts | 5 | 5 |
| Bootstrap samples | 20 | 20 |

The script saves pseudo single-cell observations, observed relative population sizes, ground-truth functions, fitted scPD results, PDF heatmaps, ECDF fitting diagnostics, and parameter-recovery metrics under:

```text
synthetic_test_results/reviewer_synthetic_benchmark/
```

The main quantitative file is `recovery_metrics.csv` in each scenario directory. It reports Pearson correlation, Spearman correlation, normalized RMSE, and occupancy-weighted normalized RMSE for the inferred kinetic profiles.

## Hyperparameter sensitivity

The hyperparameter sensitivity analysis evaluates distributional fitting error under different smoothing choices:

- Natural cubic spline degrees of freedom: `3, 4, 5, 6, 7, 8`.
- Roughness regularization strength `rho`: `0.1, 0.5, 1, 2, 5`.

The default numerical settings match the manuscript-revision analysis: `n_grid=200`, `n_starts=5`, `n_bootstrap=5`, `rho=0.5` for the spline-degree analysis, and `spline_df=6` for the rho analysis.

Run from the repository root:

```bash
python examples/hyperparameter_sensitivity_review.py \
    --data-path ../demo/data/iPSC_serum.h5ad \
    --output-dir outputs/hyperparameter_sensitivity_review
```

The script saves per-time-point A-distance values and a combined summary plot:

```text
outputs/hyperparameter_sensitivity_review/
├── spline_df_adistance_by_time.csv
├── rho_adistance_by_time.csv
├── hyperparameter_adistance_sensitivity.png
└── hyperparameter_adistance_sensitivity.pdf
```

These outputs correspond to the supplementary sensitivity analysis evaluating whether scPD fitting error is robust to moderate changes in spline basis complexity and regularization strength.
