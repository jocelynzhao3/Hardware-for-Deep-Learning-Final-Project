"""Plot Experiment 2 retrieval-only totals at the TPU v4i GlobalBuffer size.

This script reproduces the cache simulation used by Experiment-2-Caching.ipynb
for the actual TPU v4i GlobalBuffer capacity (128 MB), then plots total
retrieval energy/latency over the whole synthetic trace rather than per-query
values.

Outputs:
  - workspace/exp2_retrieval_totals_128mb.png
  - workspace/exp2_retrieval_totals_128mb.csv
"""

from __future__ import annotations

from pathlib import Path
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cache_sim import (
    simulate_belady_opt,
    simulate_lfu_cost_aware,
    simulate_no_cache,
    synth_trace,
)


WORKSPACE = Path(__file__).resolve().parents[1]
OUT_PNG = WORKSPACE / "exp2_retrieval_totals_128mb.png"
OUT_CSV = WORKSPACE / "exp2_retrieval_totals_128mb.csv"
BASELINE_PKL = WORKSPACE / "baseline_cache.pkl"


# Matches Experiment-2-Caching.ipynb.
N_DOCS_TRACE = 2_000_000
N_QUERIES = 100_000
K_SIM = 10
DECAY_FACTOR = 0.99
SEED = 42

BEIR_DATASETS = [
    ("nq", 1.25),
    ("hotpotqa", 1.42),
    ("scidocs", 1.73),
    ("quora", 1.91),
    ("fever", 2.41),
    ("fiqa", 4.47),
]

POLICIES = ["no_cache", "lfu", "opt"]
POLICY_LABELS = {
    "no_cache": "No Cache\n(all miss)",
    "lfu": "LFU (EdgeRAG)",
    "opt": "Belady OPT",
}
POLICY_COLORS = {
    "no_cache": "#dd9c4c",
    "lfu": "#5a8cc8",
    "opt": "#5fc080",
}
POLICY_MARKERS = {
    "no_cache": "o",
    "lfu": "s",
    "opt": "^",
}
BASELINE_COLOR = "#cc4c4c"


# TPU v4i / workload constants used by the notebook.
TPU_GLOBAL_BUFFER_GB = 0.128
ENC_EMBED_DIM = 768
BITS_PER_VAL = 8
EMB_BITS = ENC_EMBED_DIM * BITS_PER_VAL
EMB_BYTES = EMB_BITS // 8
USABLE_DRAM_FRACTION = 1.0

# GlobalBuffer hit; MainMemory + GlobalBuffer fill/read miss.
DRAM_READ_ENERGY_PJ_PER_BIT = 1.88
DRAM_WRITE_ENERGY_PJ_PER_BIT = 2.36
DISK_ENERGY_PJ_PER_BIT = 7.03
DRAM_READ_BW_GBPS = 2048.0
DRAM_WRITE_BW_GBPS = 1024.0
DISK_BW_GBPS = 614.0

E_HIT_PJ = DRAM_READ_ENERGY_PJ_PER_BIT * EMB_BITS
E_MISS_PJ = (
    DISK_ENERGY_PJ_PER_BIT * EMB_BITS
    + DRAM_WRITE_ENERGY_PJ_PER_BIT * EMB_BITS
    + DRAM_READ_ENERGY_PJ_PER_BIT * EMB_BITS
)
TTFT_HIT_NS = EMB_BITS / (DRAM_READ_BW_GBPS * 1e9 * 8) * 1e9
TTFT_MISS_NS = (
    EMB_BITS / (DISK_BW_GBPS * 1e9 * 8) * 1e9
    + EMB_BITS / (DRAM_WRITE_BW_GBPS * 1e9 * 8) * 1e9
    + EMB_BITS / (DRAM_READ_BW_GBPS * 1e9 * 8) * 1e9
)


def embedding_capacity(dram_gb: float, n_docs_trace: int) -> int:
    """Convert DRAM capacity to number of cached embedding rows."""
    usable_bytes = int(dram_gb * 1024**3 * USABLE_DRAM_FRACTION)
    slots = usable_bytes // EMB_BYTES
    return min(slots, n_docs_trace)


def metrics_from_counts(hits: int, misses: int) -> dict[str, float]:
    """Return retrieval-only totals over the whole trace."""
    total_energy_pj = hits * E_HIT_PJ + misses * E_MISS_PJ
    total_latency_ns = hits * TTFT_HIT_NS + misses * TTFT_MISS_NS
    return {
        "total_energy_mJ": total_energy_pj * 1e-9,
        "total_latency_ms": total_latency_ns * 1e-6,
        "energy_nJ_per_query": total_energy_pj / N_QUERIES / 1e3,
        "latency_us_per_query": total_latency_ns / N_QUERIES / 1e3,
    }


def run_simulation() -> pd.DataFrame:
    """Generate traces and run no-cache, LFU, and Belady at 128 MB."""
    capacity = embedding_capacity(TPU_GLOBAL_BUFFER_GB, N_DOCS_TRACE)
    rows: list[dict[str, float | int | str]] = []

    print(f"TPU GlobalBuffer: {TPU_GLOBAL_BUFFER_GB * 1024:.0f} MB")
    print(f"Capacity: {capacity:,} embeddings ({100 * capacity / N_DOCS_TRACE:.1f}% of trace corpus)")
    print(f"Trace: {N_QUERIES:,} queries x {K_SIM} docs/query = {N_QUERIES * K_SIM:,} accesses")

    for dataset, reuse in BEIR_DATASETS:
        print(f"Generating trace for {dataset} (reuse target {reuse:.2f})...")
        trace = synth_trace(
            n_queries=N_QUERIES,
            n_docs=N_DOCS_TRACE,
            reuse_ratio=reuse,
            k_per_query=K_SIM,
            seed=SEED,
        )
        unique_docs = len(set(trace))
        actual_reuse = len(trace) / unique_docs

        results = {
            "no_cache": simulate_no_cache(trace),
            "lfu": simulate_lfu_cost_aware(
                trace,
                capacity,
                gen_latency={},
                decay_factor=DECAY_FACTOR,
            ),
            "opt": simulate_belady_opt(trace, capacity),
        }

        for policy, result in results.items():
            metrics = metrics_from_counts(result.hits, result.misses)
            rows.append(
                {
                    "dataset": dataset,
                    "reuse_target": reuse,
                    "actual_reuse": actual_reuse,
                    "unique_docs": unique_docs,
                    "policy": policy,
                    "policy_label": POLICY_LABELS[policy],
                    "hits": result.hits,
                    "misses": result.misses,
                    "hit_rate": result.hit_rate,
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def plot_totals(df: pd.DataFrame) -> None:
    """Plot mean total retrieval energy and latency across BEIR reuse ratios."""
    with open(BASELINE_PKL, "rb") as f:
        baseline = pickle.load(f)
    sim_einsum = baseline["sim_einsum"]
    accelforge_energy_mj = baseline["baseline_energy_J"][sim_einsum] * 1e3
    accelforge_latency_ms = baseline["baseline_latency_s"][sim_einsum] * 1e3

    means = df.groupby("policy", sort=False).agg(
        total_energy_mJ=("total_energy_mJ", "mean"),
        total_latency_ms=("total_latency_ms", "mean"),
    )
    bar_labels = ["AccelForge\nbrute-force", *[POLICY_LABELS[p] for p in POLICIES]]
    x = np.arange(len(bar_labels))
    colors = [BASELINE_COLOR, *[POLICY_COLORS[p] for p in POLICIES]]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.2), constrained_layout=True)
    fig.suptitle(
        "Mean retrieval-only total cost at TPU v4i GlobalBuffer = 128 MB",
        fontsize=10,
    )

    panels = [
        (
            axes[0],
            [accelforge_energy_mj, *[means.loc[p, "total_energy_mJ"] for p in POLICIES]],
            "Total retrieval energy [mJ]",
            "Energy",
        ),
        (
            axes[1],
            [accelforge_latency_ms, *[means.loc[p, "total_latency_ms"] for p in POLICIES]],
            "Total retrieval latency [ms]",
            "Time-to-first-token",
        ),
    ]

    for ax, values, ylabel, title in panels:
        bars = ax.bar(
            x,
            values,
            color=colors,
            edgecolor="white",
            linewidth=0.8,
        )

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.03,
                f"{value:.1f} mJ" if "energy" in ylabel else f"{value:.2f} ms",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(top=max(values) * 1.18)
        ax.grid(axis="y", alpha=0.3)

    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = run_simulation()
    df.to_csv(OUT_CSV, index=False)
    plot_totals(df)

    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_PNG}")
    print()
    print("Mean across BEIR reuse ratios:")
    means = df.groupby("policy", sort=False).agg(
        total_energy_mJ=("total_energy_mJ", "mean"),
        total_latency_ms=("total_latency_ms", "mean"),
        hit_rate=("hit_rate", "mean"),
    )
    for policy in POLICIES:
        row = means.loc[policy]
        print(
            f"  {POLICY_LABELS[policy]:<14} "
            f"E={row.total_energy_mJ:8.3f} mJ  "
            f"L={row.total_latency_ms:7.4f} ms  "
            f"hit={row.hit_rate:.3f}"
        )
    print()
    print("Summary (total retrieval energy in mJ / latency in ms):")
    for dataset in [name for name, _ in BEIR_DATASETS]:
        sub = df[df.dataset == dataset]
        print(f"  {dataset}:")
        for row in sub.itertuples():
            print(
                f"    {row.policy_label:<14} "
                f"E={row.total_energy_mJ:8.3f} mJ  "
                f"L={row.total_latency_ms:7.4f} ms  "
                f"hit={row.hit_rate:.3f}"
            )


if __name__ == "__main__":
    main()
