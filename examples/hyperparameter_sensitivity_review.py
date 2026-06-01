#!/usr/bin/env python
"""
Hyperparameter sensitivity analysis used in the scPD manuscript revision.

The script evaluates distributional fitting error across:

    1. Natural cubic spline degrees of freedom: 3, 4, 5, 6, 7, 8.
    2. Roughness regularization strengths rho: 0.1, 0.5, 1, 2, 5.

For each setting, scPD is fitted to the same time-resolved iPSC dataset and
the A-distance values across time points are saved and plotted.

Example:
    python examples/hyperparameter_sensitivity_review.py \
        --data-path ../demo/data/iPSC_serum.h5ad \
        --output-dir outputs/hyperparameter_sensitivity_review
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

try:
    import scifont

    scifont.use("nature")
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import scpd
from scpd.results import PseudodynamicsResult


SPLINE_DF_VALUES = [3, 4, 5, 6, 7, 8]
RHO_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0]


def parse_day(value):
    try:
        if isinstance(value, str):
            match = re.search(r"(\d+(\.\d+)?)", value)
            if match:
                return float(match.group(1))
        return float(value)
    except Exception:
        return np.nan


def load_ipsc_prepared(data_path: Path):
    adata = sc.read_h5ad(data_path)
    adata.obs["day"] = adata.obs["day"].map(parse_day)
    adata = adata[~adata.obs["day"].isna()].copy()
    adata.obs["day_original"] = adata.obs["day"].copy()
    adata.obs["day"] = adata.obs["day"].map(lambda x: int(np.round(x))).astype(float)
    adata = adata[adata.obs["day"].isin([float(i) for i in range(10)])].copy()

    present_times = np.sort(np.unique(adata.obs["day"]))
    N_obs = np.array([np.sum(adata.obs["day"] == t) for t in present_times])

    sc.pp.scale(adata)
    sc.tl.pca(adata, n_comps=50)
    sc.pp.neighbors(adata, n_neighbors=30, use_rep="X_pca")
    sc.tl.diffmap(adata)

    scpd.find_robust_root(adata, day_column="day", day_value=0.0)
    s = scpd.compute_normalized_pseudotime(adata, n_dcs=10, percentile=99)

    day_values = adata.obs["day"].values.astype(float)
    unique_days = sorted(np.unique(day_values))
    day_to_int = {day: i for i, day in enumerate(unique_days)}
    time_labels = np.array([day_to_int[day] for day in day_values])
    time_values = np.array(unique_days)

    return scpd.prepare_inputs(
        s=s,
        time_labels=time_labels,
        time_values=time_values,
        N_obs=N_obs,
        landmarks="auto",
        use_optimized_clustering=True,
    )


def fit_or_load(prepared, result_path: Path, *, spline_df: int, rho: float, args):
    if result_path.exists() and not args.force:
        return PseudodynamicsResult.load(result_path), 0.0, True

    model = scpd.PseudodynamicsModel(
        n_grid=args.n_grid,
        spline_df=spline_df,
        stabilize_boundary=True,
    )
    start = time.time()
    result = model.fit(
        prepared,
        mode="with_population",
        rho=rho,
        n_starts=args.n_starts,
        n_bootstrap=args.n_bootstrap,
        random_state=args.random_state,
        verbose=True,
    )
    elapsed = time.time() - start
    result.save(result_path)
    return result, elapsed, False


def run_spline_df_sensitivity(prepared, output_dir: Path, args):
    rows = []
    for spline_df in SPLINE_DF_VALUES:
        subdir = output_dir / f"spline_df_{spline_df}"
        subdir.mkdir(parents=True, exist_ok=True)
        result, elapsed, reused = fit_or_load(
            prepared,
            subdir / "result.npz",
            spline_df=spline_df,
            rho=args.spline_rho,
            args=args,
        )
        for time_idx, value in enumerate(result.diagnostics.A_values):
            rows.append({
                "spline_df": spline_df,
                "rho": args.spline_rho,
                "time_index": time_idx,
                "A_distance": float(value),
                "elapsed_time": float(elapsed),
                "reused_existing_result": bool(reused),
            })
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "spline_df_adistance_by_time.csv", index=False)
    return df


def run_rho_sensitivity(prepared, output_dir: Path, args):
    rows = []
    for rho in RHO_VALUES:
        subdir = output_dir / f"rho_{rho:g}".replace(".", "p")
        subdir.mkdir(parents=True, exist_ok=True)
        result, elapsed, reused = fit_or_load(
            prepared,
            subdir / "result.npz",
            spline_df=args.rho_spline_df,
            rho=rho,
            args=args,
        )
        for time_idx, value in enumerate(result.diagnostics.A_values):
            rows.append({
                "rho": rho,
                "spline_df": args.rho_spline_df,
                "time_index": time_idx,
                "A_distance": float(value),
                "elapsed_time": float(elapsed),
                "reused_existing_result": bool(reused),
            })
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "rho_adistance_by_time.csv", index=False)
    return df


def add_boxplot(ax, groups, labels, colors, point_color, xlabel):
    positions = np.arange(len(groups))
    box = ax.boxplot(
        groups,
        positions=positions,
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.0},
        boxprops={"linewidth": 0.8},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_alpha(0.95)

    rng = np.random.default_rng(2026)
    for i, values in enumerate(groups):
        jitter = rng.normal(0, 0.035, size=len(values))
        ax.scatter(
            np.full_like(values, i, dtype=float) + jitter,
            values,
            s=9,
            color=point_color,
            alpha=0.72,
            linewidths=0,
            zorder=3,
        )

    means = [np.mean(values) for values in groups]
    ax.plot(
        positions,
        means,
        color="#D62728",
        marker="o",
        markersize=3.0,
        linewidth=1.0,
        label="Mean",
        zorder=4,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("A-distance")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=7, loc="upper right")


def plot_combined_summary(spline_df, rho_df, output_dir: Path):
    spline_groups = [
        spline_df.loc[spline_df["spline_df"] == value, "A_distance"].to_numpy()
        for value in SPLINE_DF_VALUES
    ]
    rho_groups = [
        rho_df.loc[np.isclose(rho_df["rho"], value), "A_distance"].to_numpy()
        for value in RHO_VALUES
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    add_boxplot(
        axes[0],
        spline_groups,
        [str(v) for v in SPLINE_DF_VALUES],
        ["#D9EEF8", "#BFE3F2", "#9DD5E9", "#7BC6DF", "#59B6D1", "#379EBD"],
        "#2F5D7C",
        "Spline degrees of freedom",
    )
    add_boxplot(
        axes[1],
        rho_groups,
        [f"{v:g}" for v in RHO_VALUES],
        ["#F6D8C8", "#F0B895", "#E99767", "#D9794C", "#B95D3C"],
        "#6D2E1F",
        "Regularization strength rho",
    )
    axes[1].set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_dir / "hyperparameter_adistance_sensitivity.png", dpi=300)
    fig.savefig(output_dir / "hyperparameter_adistance_sensitivity.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True, help="Path to iPSC_serum.h5ad")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/hyperparameter_sensitivity_review"))
    parser.add_argument("--n-grid", type=int, default=200)
    parser.add_argument("--n-starts", type=int, default=5)
    parser.add_argument("--n-bootstrap", type=int, default=5)
    parser.add_argument("--spline-rho", type=float, default=0.5)
    parser.add_argument("--rho-spline-df", type=int, default=6)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Refit even if result files exist")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared = load_ipsc_prepared(args.data_path)
    spline_df = run_spline_df_sensitivity(prepared, args.output_dir, args)
    rho_df = run_rho_sensitivity(prepared, args.output_dir, args)
    plot_combined_summary(spline_df, rho_df, args.output_dir)
    print(f"Saved sensitivity outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
