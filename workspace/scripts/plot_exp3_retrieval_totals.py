"""Experiment 3 plotting: SRAM/DRAM policy comparisons with shared style."""

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
    simulate_two_tier_lfu,
    simulate_two_tier_opt,
    synth_trace,
)

WORKSPACE = Path(__file__).resolve().parents[1]
OUT_BASELINE = WORKSPACE / "exp3_baseline_compare_log.png"
OUT_REUSE = WORKSPACE / "exp3_reuse_compare.png"
OUT_AVG = WORKSPACE / "exp3_average_totals.png"
OUT_HIT = WORKSPACE / "exp3_hit_breakdown.png"
OUT_CSV = WORKSPACE / "exp3_policy_compare.csv"
BASELINE_PKL = WORKSPACE / "baseline_cache.pkl"

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
DATASET_ORDER = [d for d, _ in BEIR_DATASETS]

MAIN_MEMORY_GB_VALUES = [1, 2, 4, 8, 16]
FOCUS_MAINMEM_GB = 8
GLOBAL_BUFFER_MB = 128

POLICIES = ["no_cache", "lfu_l1", "lfu_l2", "lfu_l1_l2", "opt_two_tier"]
POLICY_LABELS = {
    "no_cache": "No Cache",
    "lfu_l1": "LFU L1 (SRAM)",
    "lfu_l2": "LFU L2 (DRAM)",
    "lfu_l1_l2": "LFU L1/L2",
    "opt_two_tier": "OPT L1/L2",
}
POLICY_COLORS = {
    "no_cache": "#dd9c4c",
    "lfu_l1": "#5a8cc8",
    "lfu_l2": "#9d7fd4",
    "lfu_l1_l2": "#5fc080",
    "opt_two_tier": "#d46c9a",
}
POLICY_MARKERS = {
    "no_cache": "o",
    "lfu_l1": "s",
    "lfu_l2": "D",
    "lfu_l1_l2": "^",
    "opt_two_tier": "v",
}
BASELINE_COLOR = "#cc4c4c"

ENC_EMBED_DIM = 768
BITS_PER_VAL = 8
EMB_BITS = ENC_EMBED_DIM * BITS_PER_VAL
EMB_BYTES = EMB_BITS // 8

E_L2_PJ = 1.88 * EMB_BITS
E_DRAM_PJ = 7.03 * EMB_BITS
E_DISK_PJ = 10e-9 * 1e12
T_L2_NS = EMB_BITS / (2048.0 * 1e9 * 8) * 1e9
T_DRAM_NS = EMB_BITS / (614.0 * 1e9 * 8) * 1e9
T_DISK_NS = 100e-6 * 1e9


def embedding_capacity_bytes(capacity_bytes: int, n_docs: int) -> int:
    return min(int(capacity_bytes) // EMB_BYTES, n_docs)


def sram_capacity(n_docs: int) -> int:
    return embedding_capacity_bytes(GLOBAL_BUFFER_MB * 1024**2, n_docs)


def dram_capacity(size_gb: float, n_docs: int) -> int:
    return embedding_capacity_bytes(size_gb * 1024**3, n_docs)


def metrics_from_tiers(h2: int, h_dram: int, h_disk: int) -> dict[str, float]:
    e_pj = h2 * E_L2_PJ + h_dram * (E_L2_PJ + E_DRAM_PJ) + h_disk * (E_L2_PJ + E_DRAM_PJ + E_DISK_PJ)
    t_ns = h2 * T_L2_NS + h_dram * (T_L2_NS + T_DRAM_NS) + h_disk * (T_L2_NS + T_DRAM_NS + T_DISK_NS)
    return {
        "total_energy_mJ": e_pj * 1e-9,
        "total_latency_ms": t_ns * 1e-6,
        "energy_nJ_per_query": e_pj / N_QUERIES / 1e3,
        "latency_us_per_query": t_ns / N_QUERIES / 1e3,
    }


def run_simulation() -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    l1_cap = sram_capacity(N_DOCS_TRACE)

    for dataset, reuse in BEIR_DATASETS:
        trace = synth_trace(
            n_queries=N_QUERIES,
            n_docs=N_DOCS_TRACE,
            reuse_ratio=reuse,
            k_per_query=K_SIM,
            seed=SEED,
        )
        unique_docs = len(set(trace))
        actual_reuse = len(trace) / unique_docs

        for dram_gb in MAIN_MEMORY_GB_VALUES:
            l2_cap = dram_capacity(dram_gb, N_DOCS_TRACE)
            no_cache = simulate_no_cache(trace)
            lfu_l1 = simulate_lfu_cost_aware(trace, l1_cap, gen_latency={}, decay_factor=DECAY_FACTOR)
            lfu_l2 = simulate_lfu_cost_aware(trace, l2_cap, gen_latency={}, decay_factor=DECAY_FACTOR)
            lfu_l1_l2 = simulate_two_tier_lfu(
                trace,
                l1_capacity=l1_cap,
                l2_capacity=l2_cap,
                gen_latency={},
                decay_factor=DECAY_FACTOR,
                check_invariants=False,
            )
            opt_l1_l2 = simulate_two_tier_opt(
                trace,
                l1_capacity=l1_cap,
                l2_capacity=l2_cap,
                check_invariants=False,
            )

            policy_counts = {
                "no_cache": {"h2": 0, "h_dram": 0, "h_disk": no_cache.misses},
                "lfu_l1": {"h2": lfu_l1.hits, "h_dram": 0, "h_disk": lfu_l1.misses},
                "lfu_l2": {"h2": 0, "h_dram": lfu_l2.hits, "h_disk": lfu_l2.misses},
                "lfu_l1_l2": {"h2": lfu_l1_l2.l1_hits, "h_dram": lfu_l1_l2.l2_hits, "h_disk": lfu_l1_l2.misses},
                "opt_two_tier": {"h2": opt_l1_l2.l1_hits, "h_dram": opt_l1_l2.l2_hits, "h_disk": opt_l1_l2.misses},
            }

            for policy, c in policy_counts.items():
                total = c["h2"] + c["h_dram"] + c["h_disk"]
                rows.append(
                    {
                        "dataset": dataset,
                        "reuse_target": reuse,
                        "actual_reuse": actual_reuse,
                        "main_memory_gb": dram_gb,
                        "global_buffer_mb": GLOBAL_BUFFER_MB,
                        "policy": policy,
                        "policy_label": POLICY_LABELS[policy],
                        "h2": c["h2"],
                        "h_dram": c["h_dram"],
                        "h_disk": c["h_disk"],
                        "hit_rate": (c["h2"] + c["h_dram"]) / total if total else 0.0,
                        "h2_rate": c["h2"] / total if total else 0.0,
                        "h_dram_rate": c["h_dram"] / total if total else 0.0,
                        "h_disk_rate": c["h_disk"] / total if total else 0.0,
                        **metrics_from_tiers(c["h2"], c["h_dram"], c["h_disk"]),
                    }
                )

    return pd.DataFrame(rows)


def plot_baseline_compare_log(df: pd.DataFrame) -> None:
    with open(BASELINE_PKL, "rb") as f:
        baseline = pickle.load(f)
    sim_e = baseline["sim_einsum"]
    brute_e = baseline["baseline_energy_J"][sim_e] * 1e9
    brute_t = baseline["baseline_latency_s"][sim_e] * 1e6

    means = (
        df[df.main_memory_gb == FOCUS_MAINMEM_GB]
        .groupby("policy", sort=False)
        .agg(
            energy_nJ_per_query=("energy_nJ_per_query", "mean"),
            latency_us_per_query=("latency_us_per_query", "mean"),
        )
    )

    labels = ["AccelForge\n(brute-force)", *[POLICY_LABELS[p] for p in POLICIES]]
    x = np.arange(len(labels))
    colors = [BASELINE_COLOR, *[POLICY_COLORS[p] for p in POLICIES]]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)
    fig.suptitle("Experiment 3 baseline comparison (log scale)", fontsize=10)
    vals = [
        [brute_e, *[means.loc[p, "energy_nJ_per_query"] for p in POLICIES]],
        [brute_t, *[means.loc[p, "latency_us_per_query"] for p in POLICIES]],
    ]
    ylabels = ["Retrieval energy [nJ/query]", "Retrieval TTFT [us/query]"]
    titles = ["Energy", "Time-to-first-token"]
    for ax, values, ylabel, title in zip(axes, vals, ylabels, titles):
        ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", which="both", alpha=0.25)
    fig.savefig(OUT_BASELINE, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_reuse_compare(df: pd.DataFrame) -> None:
    sub = df[df.main_memory_gb == FOCUS_MAINMEM_GB]
    x = np.arange(len(DATASET_ORDER))
    x_labels = [f"{name}\n{reuse:.2f}x" for name, reuse in BEIR_DATASETS]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    fig.suptitle(f"Experiment 3 by reuse ratio (MainMemory={FOCUS_MAINMEM_GB} GB)", fontsize=10)

    for ax, column, ylabel, title in [
        (axes[0], "energy_nJ_per_query", "Cache energy [nJ/query]", "Cache energy"),
        (axes[1], "latency_us_per_query", "Cache TTFT [us/query]", "Time-to-first-token"),
        (axes[2], "hit_rate", "Hit rate", "Cache hit rate"),
    ]:
        for p in POLICIES:
            d = sub[sub.policy == p].set_index("dataset").loc[DATASET_ORDER]
            ax.plot(x, d[column].to_numpy(), marker=POLICY_MARKERS[p], color=POLICY_COLORS[p], label=POLICY_LABELS[p], linewidth=1.7, markersize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[2].set_ylim(-0.02, 1.05)
    fig.savefig(OUT_REUSE, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_average_totals(df: pd.DataFrame) -> None:
    means = df.groupby("policy", sort=False).agg(total_energy_mJ=("total_energy_mJ", "mean"), total_latency_ms=("total_latency_ms", "mean"))
    x = np.arange(len(POLICIES))

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), constrained_layout=True)
    fig.suptitle("Experiment 3 average totals across reuse ratios", fontsize=10)
    vals = [[means.loc[p, "total_energy_mJ"] for p in POLICIES], [means.loc[p, "total_latency_ms"] for p in POLICIES]]
    ylabels = ["Total retrieval energy [mJ]", "Total retrieval latency [ms]"]
    titles = ["Energy", "Time-to-first-token"]
    labels = [POLICY_LABELS[p] for p in POLICIES]
    colors = [POLICY_COLORS[p] for p in POLICIES]

    fmts = ["{:.1f}", "{:,.0f}"]
    for ax, values, ylabel, title, fmt in zip(axes, vals, ylabels, titles, fmts):
        bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ymin, ymax = ax.get_ylim()
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (ymax - ymin) * 0.015,
                fmt.format(v),
                ha="center", va="bottom", fontsize=7.5,
            )
    fig.savefig(OUT_AVG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hit_breakdown(df: pd.DataFrame) -> None:
    sub = df[(df.main_memory_gb == FOCUS_MAINMEM_GB)]
    means = sub.groupby("policy", sort=False).agg(h2_rate=("h2_rate", "mean"), h_dram_rate=("h_dram_rate", "mean"), h_disk_rate=("h_disk_rate", "mean"))
    x = np.arange(len(POLICIES))

    h2 = [means.loc[p, "h2_rate"] for p in POLICIES]
    hd = [means.loc[p, "h_dram_rate"] for p in POLICIES]
    hk = [means.loc[p, "h_disk_rate"] for p in POLICIES]

    fig, ax = plt.subplots(figsize=(8.8, 4.2), constrained_layout=True)
    ax.bar(x, h2, label="h2 (GlobalBuffer)")
    ax.bar(x, hd, bottom=h2, label="h_dram (MainMemory)")
    ax.bar(x, hk, bottom=np.array(h2) + np.array(hd), label="h_disk (Disk)")
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABELS[p] for p in POLICIES], fontsize=8)
    ax.set_ylabel("Access fraction")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Experiment 3 tier-hit decomposition (mean, MainMemory={FOCUS_MAINMEM_GB} GB)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(OUT_HIT, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = run_simulation()
    df.to_csv(OUT_CSV, index=False)
    plot_baseline_compare_log(df)
    plot_reuse_compare(df)
    plot_average_totals(df)
    plot_hit_breakdown(df)

    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_BASELINE}")
    print(f"Wrote {OUT_REUSE}")
    print(f"Wrote {OUT_AVG}")
    print(f"Wrote {OUT_HIT}")


if __name__ == "__main__":
    main()
