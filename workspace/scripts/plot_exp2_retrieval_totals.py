"""Experiment 2 plotting: DRAM LFU focus with shared style."""

from __future__ import annotations

from pathlib import Path
import pickle
import argparse

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
OUT_BASELINE = WORKSPACE / "exp2_baseline_compare_log.png"
OUT_REUSE = WORKSPACE / "exp2_reuse_compare.png"
OUT_AVG = WORKSPACE / "exp2_average_totals.png"
OUT_HIT = WORKSPACE / "exp2_hit_breakdown.png"
OUT_SWEEP = WORKSPACE / "exp2_dram_size_sweep_average.png"
OUT_CSV = WORKSPACE / "exp2_policy_compare.csv"
RAW_CSV = WORKSPACE / "exp2_raw_results.csv"
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

POLICIES = ["no_cache", "dram_lfu", "opt"]
POLICY_LABELS = {
    "no_cache": "No Cache",
    "dram_lfu": "DRAM LFU",
    "opt": "OPT",
}
POLICY_COLORS = {
    "no_cache": "#dd9c4c",
    "dram_lfu": "#5a8cc8",
    "opt": "#5fc080",
}
POLICY_MARKERS = {"no_cache": "o", "dram_lfu": "s", "opt": "^"}
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


def embedding_capacity_gb(size_gb: float, n_docs: int) -> int:
    return min(int(size_gb * 1024**3) // EMB_BYTES, n_docs)


def metrics_from_tiers(h2: int, h_dram: int, h_disk: int) -> dict[str, float]:
    e_pj = h2 * E_L2_PJ + h_dram * (E_L2_PJ + E_DRAM_PJ) + h_disk * (E_L2_PJ + E_DRAM_PJ + E_DISK_PJ)
    t_ns = h2 * T_L2_NS + h_dram * (T_L2_NS + T_DRAM_NS) + h_disk * (T_L2_NS + T_DRAM_NS + T_DISK_NS)
    return {
        "total_energy_mJ": e_pj * 1e-9,
        "total_latency_ms": t_ns * 1e-6,
        "energy_nJ_per_query": e_pj / N_QUERIES / 1e3,
        "latency_us_per_query": t_ns / N_QUERIES / 1e3,
    }


def _canonical_policy_id(policy: str) -> str:
    p = str(policy).strip().lower()
    if p in {"lfu", "dram_lfu"}:
        return "dram_lfu"
    return p


def _canonical_policy_label(policy: str) -> str:
    pid = _canonical_policy_id(policy)
    if pid in POLICY_LABELS:
        return POLICY_LABELS[pid]
    return str(policy)


def _format_bar_value(v: float, kind: str) -> str:
    if kind == "energy_nj":
        return f"{v:,.1f}"
    if kind == "latency_us":
        return f"{v:,.1f}"
    if kind == "energy_mj":
        return f"{v:,.2f}"
    if kind == "latency_ms":
        return f"{v:,.0f}"
    return f"{v:.3g}"


def _annotate_bars(ax: plt.Axes, bars, values: list[float], kind: str, log_scale: bool = False) -> None:
    for bar, v in zip(bars, values):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        if log_scale:
            y_text = y * 1.08
            va = "bottom"
        else:
            ymin, ymax = ax.get_ylim()
            y_text = y + (ymax - ymin) * 0.015
            va = "bottom"
        ax.text(x, y_text, _format_bar_value(v, kind), ha="center", va=va, fontsize=7.5, rotation=0)


def run_simulation() -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
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
            cap = embedding_capacity_gb(dram_gb, N_DOCS_TRACE)
            no_cache = simulate_no_cache(trace)
            dram_lfu = simulate_lfu_cost_aware(trace, cap, gen_latency={}, decay_factor=DECAY_FACTOR)
            dram_opt = simulate_belady_opt(trace, cap)

            policy_counts = {
                "no_cache": {"h2": 0, "h_dram": 0, "h_disk": no_cache.misses},
                "dram_lfu": {"h2": 0, "h_dram": dram_lfu.hits, "h_disk": dram_lfu.misses},
                "opt": {"h2": 0, "h_dram": dram_opt.hits, "h_disk": dram_opt.misses},
            }

            for policy, c in policy_counts.items():
                total = c["h2"] + c["h_dram"] + c["h_disk"]
                rows.append(
                    {
                        "dataset": dataset,
                        "reuse_target": reuse,
                        "actual_reuse": actual_reuse,
                        "main_memory_gb": dram_gb,
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


def plot_baseline_compare_log(df: pd.DataFrame, out_path: Path) -> None:
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
    fig.suptitle("Experiment 2 baseline comparison (log scale)", fontsize=10)
    vals = [
        [brute_e, *[means.loc[p, "energy_nJ_per_query"] for p in POLICIES]],
        [brute_t, *[means.loc[p, "latency_us_per_query"] for p in POLICIES]],
    ]
    ylabels = ["Retrieval energy [nJ/query]", "Retrieval TTFT [us/query]"]
    titles = ["Energy", "Time-to-first-token"]
    kinds = ["energy_nj", "latency_us"]
    for ax, values, ylabel, title, kind in zip(axes, vals, ylabels, titles, kinds):
        bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", which="both", alpha=0.25)
        _annotate_bars(ax, bars, values, kind=kind, log_scale=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_reuse_compare(df: pd.DataFrame, out_path: Path) -> None:
    sub = df[df.main_memory_gb == FOCUS_MAINMEM_GB]
    x = np.arange(len(DATASET_ORDER))
    x_labels = [f"{name}\n{reuse:.2f}x" for name, reuse in BEIR_DATASETS]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    fig.suptitle(f"Experiment 2 by reuse ratio (MainMemory={FOCUS_MAINMEM_GB} GB)", fontsize=10)

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
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_average_totals(df: pd.DataFrame, out_path: Path) -> None:
    means = df.groupby("policy", sort=False).agg(total_energy_mJ=("total_energy_mJ", "mean"), total_latency_ms=("total_latency_ms", "mean"))
    x = np.arange(len(POLICIES))

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), constrained_layout=True)
    fig.suptitle("Experiment 2 average totals across reuse ratios", fontsize=10)
    vals = [[means.loc[p, "total_energy_mJ"] for p in POLICIES], [means.loc[p, "total_latency_ms"] for p in POLICIES]]
    ylabels = ["Total retrieval energy [mJ]", "Total retrieval latency [ms]"]
    titles = ["Energy", "Time-to-first-token"]
    labels = [POLICY_LABELS[p] for p in POLICIES]
    colors = [POLICY_COLORS[p] for p in POLICIES]

    kinds = ["energy_mj", "latency_ms"]
    for ax, values, ylabel, title, kind in zip(axes, vals, ylabels, titles, kinds):
        bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        _annotate_bars(ax, bars, values, kind=kind, log_scale=False)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hit_breakdown(df: pd.DataFrame, out_path: Path) -> None:
    sub = df[(df.main_memory_gb == FOCUS_MAINMEM_GB)]
    means = sub.groupby("policy", sort=False).agg(h2_rate=("h2_rate", "mean"), h_dram_rate=("h_dram_rate", "mean"), h_disk_rate=("h_disk_rate", "mean"))
    x = np.arange(len(POLICIES))

    h2 = [means.loc[p, "h2_rate"] for p in POLICIES]
    hd = [means.loc[p, "h_dram_rate"] for p in POLICIES]
    hk = [means.loc[p, "h_disk_rate"] for p in POLICIES]

    fig, ax = plt.subplots(figsize=(8.4, 4.2), constrained_layout=True)
    ax.bar(x, h2, label="h2 (GlobalBuffer)")
    ax.bar(x, hd, bottom=h2, label="h_dram (MainMemory)")
    ax.bar(x, hk, bottom=np.array(h2) + np.array(hd), label="h_disk (Disk)")
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABELS[p] for p in POLICIES], fontsize=8)
    ax.set_ylabel("Access fraction")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Experiment 2 tier-hit decomposition (mean, MainMemory={FOCUS_MAINMEM_GB} GB)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dram_size_sweep_average(raw_df: pd.DataFrame, out_path: Path) -> None:
    tmp = raw_df.copy()
    tmp["policy"] = tmp["policy"].map(_canonical_policy_id)
    tmp = tmp[tmp["policy"].isin(POLICIES)]

    sweep = (
        tmp.groupby(["MAIN_MEMORY_GB", "policy"], as_index=False)
        .agg(
            energy_nJ_per_query=("energy_nJ_per_query", "mean"),
            ttft_us_per_query=("ttft_us_per_query", "mean"),
            hit_rate=("hit_rate", "mean"),
        )
        .sort_values(["policy", "MAIN_MEMORY_GB"])
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    fig.suptitle("Experiment 2 DRAM-size sweep (averaged across datasets/reuse ratios)", fontsize=10)

    for ax, column, ylabel, title in [
        (axes[0], "energy_nJ_per_query", "Cache energy [nJ/query]", "Energy"),
        (axes[1], "ttft_us_per_query", "Cache TTFT [us/query]", "Time-to-first-token"),
        (axes[2], "hit_rate", "Hit rate", "Hit rate"),
    ]:
        for p in POLICIES:
            d = sweep[sweep["policy"] == p]
            if d.empty:
                continue
            ax.plot(
                d["MAIN_MEMORY_GB"].to_numpy(),
                d[column].to_numpy(),
                marker=POLICY_MARKERS[p],
                color=POLICY_COLORS[p],
                label=POLICY_LABELS[p],
                linewidth=1.7,
                markersize=5,
            )
        ax.set_xticks(MAIN_MEMORY_GB_VALUES)
        ax.set_xlabel("MainMemory / DRAM size [GB]")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[2].set_ylim(-0.02, 1.05)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _resolve_data_dir(data_dir_arg: str | None) -> Path:
    if data_dir_arg:
        return Path(data_dir_arg).resolve()
    preferred = WORKSPACE / "new_exp2"
    if (preferred / "exp2_policy_compare.csv").exists() and (preferred / "exp2_raw_results.csv").exists():
        return preferred
    return WORKSPACE


def load_from_csvs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    policy_csv = data_dir / "exp2_policy_compare.csv"
    raw_csv = data_dir / "exp2_raw_results.csv"
    if not policy_csv.exists() or not raw_csv.exists():
        raise FileNotFoundError(f"Missing required CSVs in {data_dir}. Need exp2_policy_compare.csv and exp2_raw_results.csv")

    policy_df = pd.read_csv(policy_csv)
    raw_df = pd.read_csv(raw_csv)

    policy_df["policy"] = policy_df["policy"].map(_canonical_policy_id)
    policy_df["policy_label"] = policy_df["policy"].map(POLICY_LABELS)
    policy_df = policy_df[policy_df["policy"].isin(POLICIES)].copy()

    return policy_df, raw_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Experiment 2 artifacts.")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing exp2_policy_compare.csv and exp2_raw_results.csv")
    parser.add_argument("--recompute", action="store_true", help="Re-run simulation to regenerate exp2_policy_compare.csv before plotting")
    args = parser.parse_args()

    out_dir = _resolve_data_dir(args.data_dir)
    out_baseline = out_dir / OUT_BASELINE.name
    out_reuse = out_dir / OUT_REUSE.name
    out_avg = out_dir / OUT_AVG.name
    out_hit = out_dir / OUT_HIT.name
    out_sweep = out_dir / OUT_SWEEP.name
    out_csv = out_dir / OUT_CSV.name

    if args.recompute:
        df = run_simulation()
        df["policy"] = df["policy"].map(_canonical_policy_id)
        df["policy_label"] = df["policy"].map(POLICY_LABELS)
        df.to_csv(out_csv, index=False)
        raw_df = pd.read_csv(out_dir / RAW_CSV.name)
    else:
        df, raw_df = load_from_csvs(out_dir)

    plot_baseline_compare_log(df, out_baseline)
    plot_reuse_compare(df, out_reuse)
    plot_average_totals(df, out_avg)
    plot_hit_breakdown(df, out_hit)
    plot_dram_size_sweep_average(raw_df, out_sweep)

    print(f"Wrote {out_baseline}")
    print(f"Wrote {out_reuse}")
    print(f"Wrote {out_avg}")
    print(f"Wrote {out_hit}")
    print(f"Wrote {out_sweep}")


if __name__ == "__main__":
    main()
