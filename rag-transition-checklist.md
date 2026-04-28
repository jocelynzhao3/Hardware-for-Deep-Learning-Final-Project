# RAG Caching Transition Checklist

This is the markdown version of `rag-transition-checklist.canvas.tsx`. It captures the transition from the original caching experiment framing to the final EdgeRAG-aligned, TTFT-focused experiment.

## Transition Table

| Area | Before | After | Do To Transition Cleanly |
|---|---|---|---|
| Indexing | Ambiguous EdgeRAG replication; possible expectation of IVF/FAISS details. | Use flat dense retrieval for simplicity. | State that flat search is a hardware-friendly simplification and upper-bound scan model. Mark IVF details as out of scope. |
| Latency metric | Mixed language around generation latency. | Report TTFT-oriented latency for retrieval and prompt readiness. | Rename plots/text from generation latency to TTFT or retrieval-path latency. Avoid claims about first-to-last-token generation. |
| EdgeRAG cache policy | Cost-aware LFU described using `gen_latency * counter`. | Interpret the cost term as per-document load/availability cost contributing to TTFT. | If costs are uniform, explicitly say the policy becomes decayed LFU and isolates reuse rather than heterogeneous document cost. |
| Hardware platform | Some constants come from `basic8`: 68 GB/s DRAM and NVMe-like 7 GB/s disk. | Match EdgeRAG hardware: 8 GiB LPDDR5-4250 at 34 GB/s and UHS-I SD card at 104 MB/s. | Use `EdgeRAG_arch.yaml` constants or copy those constants into the notebook. Re-run cache latency because miss penalty will increase a lot. |
| Software stack | Could sound like full EdgeRAG reproduction. | Borrow the stack: `gte-base-en-v1.5` embeddings, FAISS-style retrieval, `Sheared-LLaMA-2.7B` decoder. | Keep complex items out of scope: IVF tuning, centroid training, dynamic embedding generation internals, and FAISS implementation details. |
| Quantization | Question of whether INT8 is needed for 8 GiB. | Already using `bits_per_value = 8`, so INT8 assumption is consistent. | Say INT8 makes the edge setup plausible. Check all workload tensors and memory-capacity calculations use 1 byte/value. |
| Cache access definition | Could be document row, token-doc interaction, or query event. | One access = one 768-dimensional document embedding row lookup. | Document that one query contributes `K_SIM` accesses. Keep token gather and decoder work separate from embedding-cache accesses. |
| Energy accounting | Risk of adding cache traffic on top of SIM baseline and double-counting. | `total = non-SIM AccelForge baseline + cache-modeled SIM/retrieval term`. | Remove `doc_scores` / `SIM` energy and latency before adding hit/miss costs. Keep encoder, top-k, gather, and decoder baseline terms. |
| Disk energy | Hit/miss ratio might be expected to save disk energy. | Disk energy remains zero unless an explicit SD-card energy model is added. | Phrase results as latency and EDP benefits, plus DRAM traffic tradeoffs. Do not claim disk-energy savings under the current model. |
| Miss latency | Unclear whether disk read and DRAM fill overlap. | Use additive miss latency as a conservative model. | Define miss = SD read + DRAM write/fill + DRAM read for compute. Add a note that overlap would reduce absolute miss penalty. |
| Capacity conversion | `DRAM_GB` to embedding slots may ignore rounding/metadata/corpus cap. | Use `floor((usable DRAM bytes) / 768 B)`, capped at `N_DOCS_TRACE`. | Choose ideal or metadata-adjusted capacity. If adjusted, reserve 5-10% for metadata and state it consistently. |
| Scale consistency | Proposal, workload, and cache sim use different `K` / `N` values. | Use one final set of parameters or clearly separate AccelForge baseline scale from trace scale. | Align `K=20` if that is the report claim. Explain any use of larger `N_DOCS_TRACE` as necessary to expose eviction behavior. |

## Recommended Final Framing

We do not reproduce EdgeRAG end to end. We borrow its edge hardware, model scale, reuse-ratio motivation, and cache-policy idea, while using a flat retrieval model to study memory hierarchy effects on TTFT, energy, and EDP.

Out of scope:

- IVF tuning
- Centroid training
- FAISS implementation internals
- Full first-token-to-last-token generation latency
- Dynamic embedding generation internals

## Final Deliverables

At the end of the transition, the experiment should produce:

1. **Updated caching notebook**
   - File: `workspace/Experiment-2-Caching.ipynb`
   - Should use EdgeRAG-aligned hardware constants, TTFT naming, flat retrieval framing, and final sanity checks.
   - You should understand: how the trace is generated, what one cache access means, and how hit/miss counts become per-query energy, TTFT latency, and EDP.
   - You should personally sanity check: constants, units, `K_SIM`, `N_DOCS_TRACE`, cache capacity, no-cache invariance, and OPT/LFU ordering.

2. **Updated cache simulator documentation**
   - File: `workspace/scripts/cache_sim.py`
   - Should describe flat retrieval, one document-row access, decayed LFU, OPT, and uniform-cost behavior.
   - You should understand: why cost-aware LFU reduces to decayed LFU when all per-document costs are equal.
   - You should personally sanity check: hit + miss accounting, capacity behavior, and OPT miss-count lower bound.

3. **Final energy/TTFT/EDP result tables**
   - Produced by `Experiment-2-Caching.ipynb`.
   - Should include one row per `(DRAM_GB, dataset/reuse_ratio, policy)`.
   - You should understand: which terms come from AccelForge and which terms come from the cache simulator.
   - You should personally sanity check: no double-counting of `SIM`, no negative or impossible values, and no-cache values staying flat across DRAM size.

4. **Final plots**
   - Expected outputs: `workspace/exp2_heatmaps.png`, `workspace/exp2_gap_hitrate.png`, and any final report-ready variants.
   - Should show energy, TTFT latency, EDP or LFU-vs-OPT gap, and hit rate.
   - You should understand: whether each plot is showing absolute values, normalized values, or gap-to-OPT values.
   - You should personally sanity check: axis labels, units, hardware constants, and whether the trend matches the raw numbers.

5. **Sanity-check output block**
   - Produced inside the notebook after the sweep.
   - Should print pass/fail status for every sanity check listed below.
   - You should understand: which failures are true bugs versus expected consequences of a parameter choice.
   - You should personally sanity check: every warning before using the plots in the report.

6. **Report-ready method paragraph**
   - Goes into the final writeup.
   - Should say that the experiment uses EdgeRAG-inspired hardware/software assumptions but simplifies retrieval to flat dense search.
   - You should understand: exactly what is and is not being claimed relative to EdgeRAG.
   - You should personally sanity check: no overclaiming about IVF, FAISS internals, disk energy, or full generation latency.

7. **Report-ready results paragraph**
   - Goes into the final writeup after plots are final.
   - Should explain where caching helps or hurts in energy, TTFT, and EDP.
   - You should understand: whether the main story is latency-only, energy-only, or EDP.
   - You should personally sanity check: every qualitative claim against the raw tables.

## Full Sanity Checks

### Trace And Policy Checks

- No-cache hit rate is exactly `0`.
- No-cache misses equal total accesses.
- For every trace, `hits + misses = N_QUERIES * K_SIM`.
- Actual reuse, `len(trace) / unique_docs`, is close to the target BEIR reuse ratio.
- OPT misses are always `<=` LFU misses.
- LFU misses are always `<=` no-cache misses.
- OPT hit rate is non-decreasing as DRAM capacity increases.
- LFU hit rate generally improves with capacity; any large drop should be investigated.
- If cache capacity is greater than or equal to the number of unique accessed docs, LFU and OPT converge to cold misses only.

### Capacity And Unit Checks

- Embedding size is `768 bytes`, because `enc_embed_dim = 768` and `bits_per_value = 8`.
- Cache capacity is `floor(usable_DRAM_bytes / 768)`.
- Cache capacity is capped at `N_DOCS_TRACE`.
- If metadata overhead is included, the usable fraction is documented and applied consistently.
- Energy units are consistently converted between J, pJ, and nJ.
- Latency units are consistently converted between s, ns, and us.
- EDP is computed from the displayed energy and latency units consistently.

### Hardware Checks

- Final results use EdgeRAG-aligned DRAM bandwidth: `34 GB/s`.
- Final results use EdgeRAG-aligned SD-card bandwidth: `0.104 GB/s`.
- Disk energy remains `0` unless an explicit SD-card energy model is added.
- Hit latency is lower than miss latency.
- Miss latency reflects the chosen model: additive SD read + DRAM write + DRAM read.
- If overlap is later modeled, the overlap formula is written down and used everywhere.

### Energy And Latency Accounting Checks

- `SIM` / `doc_scores` baseline energy is excluded before adding cache-modeled retrieval energy.
- `SIM` / `doc_scores` baseline latency is excluded before adding cache-modeled retrieval latency.
- Encoder, top-k, document gather, and decoder baseline terms remain included.
- No-cache energy does not change with DRAM capacity.
- No-cache latency does not change with DRAM capacity.
- With current disk-energy-zero model, hit/miss ratio should mostly affect TTFT and EDP, not disk energy.
- Hit energy is one DRAM read.
- Miss energy is DRAM fill/write plus DRAM read, with disk energy equal to zero.

### Parameter Consistency Checks

- The report, workload, and trace generator agree on `K`, or any mismatch is explicitly explained.
- `K_SIM = 20` if the final report claims 20 retrieved documents per query.
- `N_DOCS_TRACE` is large enough to expose eviction behavior across the 1-16 GB sweep.
- If the AccelForge baseline uses smaller `N_DOCS` than the trace, the report explains this as a split between baseline compute scale and cache-trace scale.
- BEIR reuse ratios are listed with dataset names.
- Random seed is fixed for reproducibility.

### Plot And Report Checks

- Plot titles say TTFT or retrieval-path latency, not generation latency.
- Plot axes include units.
- Plot captions state the hardware constants.
- Plot captions state `N_DOCS_TRACE`, `N_QUERIES`, and `K_SIM`.
- Heatmap colors are comparable within the intended group.
- LFU-vs-OPT gap handles zero denominators safely.
- No NaN, infinity, or negative EDP appears in final tables.
- Every qualitative claim in the report can be traced back to a raw value or plotted trend.

