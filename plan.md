# Experiment 2 Execution Plan

This plan lists the concrete executable work needed to finish the relevance-aware caching experiment after the final decisions:

- Use flat dense retrieval for simplicity.
- Focus latency claims on TTFT / retrieval-path latency, not full generation latency.
- Use `basic8.yaml` accelerator: LPDDR5 at 68 GB/s and NVMe disk at 7 GB/s (disk energy = 0).
- Use INT8 values: one 768-dimensional embedding row is 768 bytes.
- Treat one cache access as one document embedding row lookup.
- Avoid double-counting by replacing the AccelForge `SIM` / `doc_scores` term with the cache simulator term.

## 0. Start The Environment

From the repository root in PowerShell:

```powershell
$env:DOCKER_ARCH = "amd64"
docker compose up
```

Open Jupyter:

```text
http://localhost:8888
```

Run notebooks from inside the container paths:

```text
/home/workspace/workspace/M3-Baseline.ipynb
/home/workspace/workspace/Experiment-2-Caching.ipynb
```

## 1. Finalize Baseline Inputs

Deliverable: `workspace/baseline_cache.pkl`

Notebook: `workspace/M3-Baseline.ipynb`

Required baseline settings (matches `M3-Baseline.ipynb` after the d7c6d44 sync):

```python
WORKLOAD = "full_EdgeRAG.yaml"
ARCH = "basic8.yaml"
N_DOCS = 5_000_000
N_TOKENS = 512
K = 10
L = 512
```

The AccelForge baseline now runs at the same `N_DOCS = N_DOCS_TRACE = 5_000_000` corpus
size as the cache trace, so the SIM term and the eviction trace describe the same scale:

```text
AccelForge baseline: maps the doc_scores SIM einsum at N_DOCS = 5_000_000.
Cache trace:         simulates evictions with N_DOCS_TRACE = 5_000_000.
Final total:         non-SIM baseline + cache-modeled SIM/retrieval term.
```

Generate the pickle by running `workspace/scripts/export_baseline.py` inside the lab
container (or by re-running the new export cell at the end of `M3-Baseline.ipynb`):

```powershell
docker compose exec labs python3 /home/workspace/workspace/scripts/export_baseline.py
```

Before moving on, confirm the exported pickle contains:

```python
baseline_energy_J      # dict: einsum_name -> J
baseline_latency_s     # dict: einsum_name -> s
sim_einsum             # expected: "doc_scores" or equivalent SIM einsum
```

## 2. Update Cache Notebook Constants

Deliverable: updated `workspace/Experiment-2-Caching.ipynb`

Use this parameter block in the notebook:

```python
# AccelForge workload parameters (must match M3-Baseline)
N_DOCS    = 5_000_000
N_TOKENS  = 512
K         = 10
L         = 512

# Cache simulation parameters
N_DOCS_TRACE = 5_000_000
N_QUERIES    = 100_000
K_SIM        = 10
DECAY_FACTOR = 0.99
SEED         = 42

# DRAM sweep
DRAM_GB_VALUES = [1, 2, 4, 8, 16]

# BEIR reuse ratios from EdgeRAG
BEIR_DATASETS = [
    ("nq",       1.25),
    ("hotpotqa", 1.42),
    ("scidocs",  1.73),
    ("quora",    1.91),
    ("fever",    2.41),
    ("fiqa",     4.47),
]

DATASET_NAMES = [d for d, _ in BEIR_DATASETS]
REUSE_RATIOS  = [r for _, r in BEIR_DATASETS]
```

Use this hardware block:

```python
# EdgeRAG-aligned hardware constants
ENC_EMBED_DIM = 768
BITS_PER_VAL  = 8
EMB_BITS      = ENC_EMBED_DIM * BITS_PER_VAL
EMB_BYTES     = EMB_BITS // 8

DRAM_ENERGY_pJ_per_bit = 7.03e-12 * 1e12
DRAM_BW_GBps           = 68.0              # basic8.yaml LPDDR5 bandwidth
DISK_ENERGY_pJ_per_bit = 0.0               # disk energy = 0 in basic8.yaml
DISK_BW_GBps           = 7.0               # NVMe SSD bandwidth in GB/s

# Optional. Keep at 1.0 for ideal embedding-only capacity.
USABLE_DRAM_FRACTION = 1.0
```

Use this capacity function:

```python
def embedding_capacity(dram_gb: float, n_docs_trace: int) -> int:
    usable_bytes = int(dram_gb * 1024**3 * USABLE_DRAM_FRACTION)
    slots = usable_bytes // EMB_BYTES
    return min(slots, n_docs_trace)
```

Use this access-cost block:

```python
# One cache access = one 768-dimensional document embedding row.
# Hit: DRAM read.
# Miss: SD-card read + DRAM write/fill + DRAM read for compute.
E_HIT_pJ = DRAM_ENERGY_pJ_per_bit * EMB_BITS
E_MISS_pJ = (
    DISK_ENERGY_pJ_per_bit * EMB_BITS
    + DRAM_ENERGY_pJ_per_bit * EMB_BITS
    + DRAM_ENERGY_pJ_per_bit * EMB_BITS
)

L_HIT_ns = EMB_BITS / (DRAM_BW_GBps * 1e9 * 8) * 1e9
L_MISS_ns = (
    EMB_BITS / (DISK_BW_GBps * 1e9 * 8) * 1e9
    + EMB_BITS / (DRAM_BW_GBps * 1e9 * 8) * 1e9
    + EMB_BITS / (DRAM_BW_GBps * 1e9 * 8) * 1e9
)

print(f"Embedding size: {EMB_BYTES} bytes")
print(f"E_HIT  = {E_HIT_pJ:.1f} pJ   L_HIT  = {L_HIT_ns:.1f} ns")
print(f"E_MISS = {E_MISS_pJ:.1f} pJ   L_MISS = {L_MISS_ns:.1f} ns")
```

## 3. Load Baseline Without Double-Counting SIM

Deliverable: `E_nonsim_pJ`, `L_nonsim_ns`

Use:

```python
BASELINE_PATH = "/home/workspace/workspace/baseline_cache.pkl"

with open(BASELINE_PATH, "rb") as f:
    bl = pickle.load(f)

baseline_energy_J  = bl["baseline_energy_J"]
baseline_latency_s = bl["baseline_latency_s"]
SIM_EINSUM         = bl["sim_einsum"]

non_sim_einsums = [e for e in baseline_energy_J if e != SIM_EINSUM]

E_nonsim_pJ = sum(baseline_energy_J[e] for e in non_sim_einsums) * 1e12
L_nonsim_ns = sum(baseline_latency_s[e] for e in non_sim_einsums) * 1e9

print(f"SIM einsum replaced by cache model: {SIM_EINSUM}")
print(f"Non-SIM energy:  {E_nonsim_pJ:.3e} pJ")
print(f"Non-SIM latency: {L_nonsim_ns:.3e} ns")
```

## 4. Generate BEIR-Reuse Traces

Deliverable: `traces`

Use:

```python
traces = {}
trace_stats = {}

for name, reuse in BEIR_DATASETS:
    trace = synth_trace(
        n_queries=N_QUERIES,
        n_docs=N_DOCS_TRACE,
        reuse_ratio=reuse,
        k_per_query=K_SIM,
        seed=SEED,
    )
    unique = len(set(trace))
    actual_reuse = len(trace) / unique if unique else float("inf")
    traces[name] = trace
    trace_stats[name] = {
        "target_reuse": reuse,
        "actual_reuse": actual_reuse,
        "unique": unique,
        "total": len(trace),
    }
    print(
        f"{name:<10} target={reuse:.2f} actual={actual_reuse:.2f} "
        f"unique={unique:,} total={len(trace):,}"
    )
```

## 5. Run Cache Policies

Deliverable: `sim_results`

Use:

```python
POLICIES = ["no_cache", "lfu", "opt"]
sim_results = {gb: {name: {} for name, _ in BEIR_DATASETS} for gb in DRAM_GB_VALUES}

# Uniform cost means EdgeRAG cost-aware LFU reduces to decayed LFU.
ttft_cost_uniform = {}

total = len(DRAM_GB_VALUES) * len(BEIR_DATASETS)
done = 0

for gb in DRAM_GB_VALUES:
    capacity = embedding_capacity(gb, N_DOCS_TRACE)
    for name, _reuse in BEIR_DATASETS:
        trace = traces[name]
        sim_results[gb][name]["no_cache"] = simulate_no_cache(trace)
        sim_results[gb][name]["lfu"] = simulate_lfu_cost_aware(
            trace,
            capacity,
            ttft_cost_uniform,
            decay_factor=DECAY_FACTOR,
        )
        sim_results[gb][name]["opt"] = simulate_belady_opt(trace, capacity)
        done += 1
        print(f"[{done}/{total}] DRAM={gb} GB capacity={capacity:,} dataset={name}")
```

## 6. Convert Hit/Miss Counts To Per-Query Metrics

Deliverable: `energy`, `ttft_latency`, `edp`, `hitrate`

Use:

```python
def per_query_metrics(result: CacheResult, n_queries: int):
    hits_per_query = result.hits / n_queries
    misses_per_query = result.misses / n_queries

    energy_pJ = (
        E_nonsim_pJ
        + hits_per_query * E_HIT_pJ
        + misses_per_query * E_MISS_pJ
    )
    ttft_ns = (
        L_nonsim_ns
        + hits_per_query * L_HIT_ns
        + misses_per_query * L_MISS_ns
    )
    edp = energy_pJ * ttft_ns
    return energy_pJ, ttft_ns, edp

ndram = len(DRAM_GB_VALUES)
ndata = len(BEIR_DATASETS)

energy = {p: np.zeros((ndram, ndata)) for p in POLICIES}
ttft_latency = {p: np.zeros((ndram, ndata)) for p in POLICIES}
edp = {p: np.zeros((ndram, ndata)) for p in POLICIES}
hitrate = {p: np.zeros((ndram, ndata)) for p in POLICIES}

for i, gb in enumerate(DRAM_GB_VALUES):
    for j, (name, _reuse) in enumerate(BEIR_DATASETS):
        for policy in POLICIES:
            res = sim_results[gb][name][policy]
            e, l, d = per_query_metrics(res, N_QUERIES)
            energy[policy][i, j] = e
            ttft_latency[policy][i, j] = l
            edp[policy][i, j] = d
            hitrate[policy][i, j] = res.hit_rate
```

## 7. Run Sanity Checks

Deliverable: printed pass/fail sanity report

Use:

```python
def check(condition, message):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {message}")
    return bool(condition)

all_ok = True
total_accesses = N_QUERIES * K_SIM

for name, _reuse in BEIR_DATASETS:
    stats = trace_stats[name]
    all_ok &= check(
        abs(stats["actual_reuse"] - stats["target_reuse"]) / stats["target_reuse"] < 0.05,
        f"{name}: actual reuse close to target",
    )
    all_ok &= check(
        stats["total"] == total_accesses,
        f"{name}: trace length equals N_QUERIES * K_SIM",
    )

for gb in DRAM_GB_VALUES:
    cap = embedding_capacity(gb, N_DOCS_TRACE)
    all_ok &= check(cap <= N_DOCS_TRACE, f"{gb} GB: capacity capped at N_DOCS_TRACE")

    for name, _reuse in BEIR_DATASETS:
        no_cache = sim_results[gb][name]["no_cache"]
        lfu = sim_results[gb][name]["lfu"]
        opt = sim_results[gb][name]["opt"]

        all_ok &= check(no_cache.hits == 0, f"{gb} GB {name}: no-cache hits are zero")
        all_ok &= check(no_cache.misses == total_accesses, f"{gb} GB {name}: no-cache misses equal total accesses")
        all_ok &= check(lfu.hits + lfu.misses == total_accesses, f"{gb} GB {name}: LFU hit/miss total is valid")
        all_ok &= check(opt.hits + opt.misses == total_accesses, f"{gb} GB {name}: OPT hit/miss total is valid")
        all_ok &= check(opt.misses <= lfu.misses <= no_cache.misses, f"{gb} GB {name}: OPT <= LFU <= no-cache misses")
        all_ok &= check(L_MISS_ns > L_HIT_ns, f"{gb} GB {name}: miss latency exceeds hit latency")
        all_ok &= check(E_MISS_pJ >= E_HIT_pJ, f"{gb} GB {name}: miss energy is at least hit energy")

for j, (name, _reuse) in enumerate(BEIR_DATASETS):
    opt_rates = [hitrate["opt"][i, j] for i in range(ndram)]
    all_ok &= check(
        all(opt_rates[i] <= opt_rates[i + 1] + 1e-12 for i in range(len(opt_rates) - 1)),
        f"{name}: OPT hit rate non-decreasing with capacity",
    )

for policy in POLICIES:
    all_ok &= check(np.all(np.isfinite(energy[policy])), f"{policy}: finite energy")
    all_ok &= check(np.all(np.isfinite(ttft_latency[policy])), f"{policy}: finite TTFT latency")
    all_ok &= check(np.all(np.isfinite(edp[policy])), f"{policy}: finite EDP")
    all_ok &= check(np.all(energy[policy] >= 0), f"{policy}: non-negative energy")
    all_ok &= check(np.all(ttft_latency[policy] >= 0), f"{policy}: non-negative TTFT latency")
    all_ok &= check(np.all(edp[policy] >= 0), f"{policy}: non-negative EDP")

for j, (name, _reuse) in enumerate(BEIR_DATASETS):
    all_ok &= check(
        np.allclose(energy["no_cache"][:, j], energy["no_cache"][0, j]),
        f"{name}: no-cache energy invariant to DRAM size",
    )
    all_ok &= check(
        np.allclose(ttft_latency["no_cache"][:, j], ttft_latency["no_cache"][0, j]),
        f"{name}: no-cache TTFT invariant to DRAM size",
    )

print()
print("SANITY CHECK SUMMARY:", "PASS" if all_ok else "FAIL")
```

## 8. Produce Raw Results Table

Deliverable: raw table for final report checking

Use:

```python
rows = []

for i, gb in enumerate(DRAM_GB_VALUES):
    capacity = embedding_capacity(gb, N_DOCS_TRACE)
    for j, (name, reuse) in enumerate(BEIR_DATASETS):
        for policy in POLICIES:
            res = sim_results[gb][name][policy]
            rows.append({
                "DRAM_GB": gb,
                "capacity_embeddings": capacity,
                "dataset": name,
                "reuse": reuse,
                "policy": policy,
                "hits": res.hits,
                "misses": res.misses,
                "hit_rate": res.hit_rate,
                "energy_nJ_per_query": energy[policy][i, j] / 1e3,
                "ttft_us_per_query": ttft_latency[policy][i, j] / 1e3,
                "edp_pJ_ns": edp[policy][i, j],
            })

df = pd.DataFrame(rows)
display(df)
df.to_csv("/home/workspace/workspace/exp2_raw_results.csv", index=False)
```

## 9. Produce Plots

Deliverables:

- `workspace/exp2_heatmaps.png`
- `workspace/exp2_gap_hitrate.png`
- Optional: `workspace/exp2_edp_heatmap.png`

Required labeling changes:

```text
Use "TTFT latency" or "retrieval-path latency".
Do not use "generation latency".
Include N_DOCS_TRACE, N_QUERIES, K_SIM, DRAM_BW_GBps, and DISK_BW_GBps in titles or captions.
```

LFU-vs-OPT gap calculation:

```python
denominator = energy["no_cache"] - energy["opt"]
safe_denom = np.where(np.abs(denominator) < 1e-6, np.nan, denominator)
gap = np.clip((energy["lfu"] - energy["opt"]) / safe_denom, 0, 1)
```

Optional EDP heatmap:

```python
for policy in POLICIES:
    plt.figure(figsize=(7, 4))
    plt.imshow(edp[policy], aspect="auto")
    plt.xticks(range(ndata), DATASET_NAMES, rotation=30, ha="right")
    plt.yticks(range(ndram), [f"{g} GB" for g in DRAM_GB_VALUES])
    plt.colorbar(label="pJ * ns per query")
    plt.title(f"EDP per Query - {policy}")
    plt.tight_layout()
    plt.savefig(f"/home/workspace/workspace/exp2_edp_{policy}.png", dpi=150, bbox_inches="tight")
    plt.show()
```

## 10. Update `cache_sim.py` Comments

Deliverable: clearer documentation in `workspace/scripts/cache_sim.py`

Make sure the docstring says:

```text
One trace element is one document embedding row access.
The trace represents K_SIM retrieved-document accesses per query.
The LFU policy follows EdgeRAG's cost-aware form.
When per-document costs are uniform, cost-aware LFU reduces to decayed LFU.
The simulator models replacement behavior only; hardware energy and latency are applied in the notebook.
```

No algorithm change is required unless you decide to add nonuniform TTFT/load costs.

## 11. Report Text To Write After Results Are Final

Deliverable: final report method/result paragraphs

Method paragraph checklist:

```text
We use EdgeRAG-inspired hardware: 8 GiB LPDDR5-4250 unified memory and UHS-I SD-card storage.
We use EdgeRAG-inspired software: gte-base-en-v1.5 embeddings and a Sheared-LLaMA-2.7B-scale decoder.
We simplify retrieval from IVF to flat dense search to isolate hardware memory hierarchy behavior.
One cache access is one 768-byte INT8 document embedding row lookup.
The cache model replaces the AccelForge SIM/doc_scores term to avoid double-counting.
We report energy, TTFT latency, and EDP per query.
```

Results paragraph checklist:

```text
State whether LFU improves TTFT versus no-cache.
State whether LFU improves energy under the disk-energy-zero model.
State whether EDP improves and in which reuse/capacity regimes.
Compare LFU to OPT as headroom, not as a deployable policy.
Mention when the cache is too small or reuse is too low to help.
Mention that additive miss latency is conservative.
```

## 12. Final Deliverable Checklist

The experiment is complete when these files exist and have been checked:

- `workspace/baseline_cache.pkl`
- `workspace/Experiment-2-Caching.ipynb`
- `workspace/scripts/cache_sim.py`
- `workspace/exp2_raw_results.csv`
- `workspace/exp2_heatmaps.png`
- `workspace/exp2_gap_hitrate.png`
- Optional EDP heatmaps: `workspace/exp2_edp_no_cache.png`, `workspace/exp2_edp_lfu.png`, `workspace/exp2_edp_opt.png`
- Final report method paragraph
- Final report results paragraph

## 13. Personal Understanding Checklist

Before presenting the experiment, be able to answer:

- Why are we using flat retrieval instead of IVF?
- What exactly is one cache access?
- Why is the latency metric TTFT rather than full generation latency?
- Why does uniform cost-aware LFU reduce to decayed LFU?
- Which AccelForge term is removed before adding the cache model?
- Why does disk energy stay zero?
- Why does SD-card bandwidth make misses much more expensive than hits?
- How is DRAM capacity converted into embedding slots?
- Why does `N_DOCS_TRACE` need to be large enough to expose evictions?
- What does OPT represent, and why is it only an offline lower bound?

