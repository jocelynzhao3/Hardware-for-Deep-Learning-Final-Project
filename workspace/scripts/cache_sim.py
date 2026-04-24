"""Cache simulation for Experiment 2: Relevance-Aware DRAM Caching.

Three cache replacement policies over a synthetic Zipf-distributed access trace:
  - No cache:      every access is served from disk.
  - LFU cost-aware: EdgeRAG Algorithm 2 — evict argmin(gen_latency * counter)
                    with per-step exponential counter decay.
  - Bélády OPT:    offline oracle — evict the item whose next use is farthest
                   in the future (minimum miss-count lower bound).

Trace generator:
  synth_trace() draws K document IDs per query from a Zipf distribution whose
  exponent is tuned (via binary search on the expected-unique-docs formula) so
  that total_accesses / unique_docs ≈ the requested reuse_ratio.

Note on parameter scale
-----------------------
With the plan's default n_docs=100 000 and n_queries=5 000:
  total accesses = 100 000,  unique docs ≤ 100 000 (80 000 at reuse 1.25).
  DRAM capacity at 1 GB = 1 398 101 embedding slots >> 80 000 unique docs.
So the cache never fills and LFU ≈ OPT for all DRAM sizes in that regime
(the DRAM-sweep axis becomes flat; the interesting axis is reuse_ratio).

To see evictions and a non-trivial LFU-vs-OPT gap across the 1–16 GB sweep,
use n_docs_trace ≥ 2 000 000 and n_queries ≥ 100 000 in the notebook.
The functions below work correctly for any parameter combination.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CacheResult:
    hits: int
    misses: int
    hit_rate: float
    miss_positions: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Trace generator
# ---------------------------------------------------------------------------

def synth_trace(
    n_queries: int,
    n_docs: int,
    reuse_ratio: float,
    k_per_query: int = 20,
    seed: int = 0,
) -> list[int]:
    """Generate a synthetic Zipf-distributed access trace.

    The Zipf exponent is tuned by binary search so that the resulting trace
    satisfies  len(trace) / len(set(trace))  ≈  reuse_ratio.

    Each query draws k_per_query *distinct* document IDs (no repeated doc
    within a single query).  The same document may appear across queries,
    which is the source of cross-query reuse.

    Parameters
    ----------
    n_queries   : number of queries to simulate
    n_docs      : corpus size (total number of distinct document embeddings)
    reuse_ratio : target total_accesses / unique_docs (≥ 1.0)
    k_per_query : retrieved documents per query
    seed        : RNG seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    total_accesses = n_queries * k_per_query

    # Target number of unique docs given the reuse ratio; clamp to valid range.
    target_unique = int(round(total_accesses / max(reuse_ratio, 1.0)))
    target_unique = max(k_per_query, min(n_docs, target_unique))

    doc_ranks = np.arange(1, n_docs + 1, dtype=np.float64)

    # Binary-search the Zipf exponent α using the expected-unique formula:
    #   E[unique] = sum_i (1 - (1 - p_i)^total_accesses)
    lo, hi = 1e-4, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        weights = 1.0 / (doc_ranks ** mid)
        weights /= weights.sum()
        p_never = (1.0 - weights) ** total_accesses
        expected_unique = int(np.sum(1.0 - p_never))
        if expected_unique > target_unique:
            lo = mid   # need more skew (larger α) → fewer unique docs
        else:
            hi = mid   # need less skew → more unique docs

    alpha = (lo + hi) / 2.0
    weights = 1.0 / (doc_ranks ** alpha)
    weights /= weights.sum()

    trace: list[int] = []
    for _ in range(n_queries):
        draws = rng.choice(n_docs, size=k_per_query, replace=False, p=weights)
        trace.extend(int(d) for d in draws)

    return trace


# ---------------------------------------------------------------------------
# Policy 1: No cache
# ---------------------------------------------------------------------------

def simulate_no_cache(trace: list[int]) -> CacheResult:
    """Baseline: every access is a cache miss (nothing is ever cached)."""
    n = len(trace)
    return CacheResult(
        hits=0,
        misses=n,
        hit_rate=0.0,
        miss_positions=list(range(n)),
    )


# ---------------------------------------------------------------------------
# Policy 2: EdgeRAG cost-aware LFU  (Algorithm 2)
# ---------------------------------------------------------------------------

def simulate_lfu_cost_aware(
    trace: list[int],
    capacity: int,
    gen_latency: dict[int, float],
    decay_factor: float = 0.99,
) -> CacheResult:
    """EdgeRAG Algorithm 2: cost-aware LFU with exponential counter decay.

    On each access to document i:
      Hit:  counter[i] += 1
      Miss: evict argmin_j{ gen_latency[j] * counter[j] }; insert i
    After each access: counter[j] *= decay_factor  for all j.

    Decay is applied *lazily*: each cached item stores (base_count, last_step)
    so that  effective_count = base_count * decay_factor^(step - last_step).
    This avoids the O(|cache|) decay sweep per step.

    Eviction uses a min-heap with lazy deletion so each eviction is
    O(log |cache|) amortised rather than O(|cache|) linear scan.

    Parameters
    ----------
    trace        : flat list of document IDs accessed in sequence
    capacity     : maximum number of embeddings the cache can hold
    gen_latency  : per-document generation/loading latency (uniform = all 1.0)
    decay_factor : multiplicative decay applied to counters after each access
    """
    n = len(trace)
    if capacity <= 0:
        return simulate_no_cache(trace)

    in_cache: set[int] = set()

    # Lazy-decay counters: doc → (base_count, step_last_updated)
    base_count: dict[int, float] = {}
    last_step:  dict[int, int]   = {}

    # Min-heap entries: (priority, generation_id, doc_id)
    # generation_id is incremented on each push; stale entries have outdated ids.
    heap: list[tuple[float, int, int]] = []
    heap_gen: dict[int, int] = {}   # doc → current generation in heap
    _gen: list[int] = [0]           # mutable counter (avoids `nonlocal`)

    def _effective_count(doc: int, step: int) -> float:
        elapsed = step - last_step.get(doc, step)
        return base_count.get(doc, 0.0) * (decay_factor ** elapsed)

    def _push(doc: int, step: int) -> None:
        _gen[0] += 1
        gid = _gen[0]
        heap_gen[doc] = gid
        priority = gen_latency.get(doc, 1.0) * _effective_count(doc, step)
        heapq.heappush(heap, (priority, gid, doc))

    def _evict_one(step: int) -> None:
        """Pop the min-priority item still in cache (skip stale heap entries)."""
        while heap:
            _, gid, doc = heapq.heappop(heap)
            if doc in in_cache and heap_gen.get(doc) == gid:
                in_cache.discard(doc)
                return

    def _update_counter(doc: int, step: int) -> None:
        elapsed = step - last_step.get(doc, step)
        decayed = base_count.get(doc, 0.0) * (decay_factor ** elapsed)
        base_count[doc] = decayed + 1.0
        last_step[doc] = step

    hits = 0
    misses = 0
    miss_positions: list[int] = []

    for i, doc in enumerate(trace):
        if doc in in_cache:
            hits += 1
            _update_counter(doc, i)
            _push(doc, i)          # re-push with updated priority; old entry becomes stale
        else:
            misses += 1
            miss_positions.append(i)
            if len(in_cache) >= capacity:
                _evict_one(i)
            in_cache.add(doc)
            _update_counter(doc, i)
            _push(doc, i)

    return CacheResult(
        hits=hits,
        misses=misses,
        hit_rate=hits / n if n > 0 else 0.0,
        miss_positions=miss_positions,
    )


# ---------------------------------------------------------------------------
# Policy 3: Bélády OPT
# ---------------------------------------------------------------------------

def simulate_belady_opt(
    trace: list[int],
    capacity: int,
) -> CacheResult:
    """Bélády's optimal offline replacement algorithm.

    On each cache miss, evicts the cached item whose *next* access is farthest
    in the future (treating items with no future access as infinitely far away).
    This minimises the total number of cache misses for any given capacity.

    Implementation:
      1. Right-to-left pass to precompute next_use[i] = the next index j > i
         where trace[j] == trace[i]  (or len(trace) if never accessed again).
      2. Forward simulation with a max-heap (negated for Python's min-heap)
         keyed by next_use, with lazy deletion for O(n log C) total time.

    Parameters
    ----------
    trace    : flat list of document IDs accessed in sequence
    capacity : maximum number of embeddings the cache can hold
    """
    n = len(trace)
    if capacity <= 0:
        return simulate_no_cache(trace)

    # --- Step 1: precompute next_use ---
    next_use: list[int] = [n] * n          # default: no future use
    last_seen: dict[int, int] = {}
    for i in range(n - 1, -1, -1):
        doc = trace[i]
        if doc in last_seen:
            next_use[i] = last_seen[doc]
        last_seen[doc] = i

    # --- Step 2: forward simulation with max-heap ---
    # Heap entry: (-next_use_value, version_id, doc_id)
    cache: set[int] = set()
    heap_ver: dict[int, int] = {}          # doc → version id when last pushed
    heap: list[tuple[int, int, int]] = []
    _ver: list[int] = [0]

    def _push_opt(doc: int, nu: int) -> None:
        _ver[0] += 1
        vid = _ver[0]
        heap_ver[doc] = vid
        heapq.heappush(heap, (-nu, vid, doc))

    def _evict_farthest() -> None:
        """Evict the cached item with the largest next_use (farthest future use)."""
        while heap:
            _, vid, doc = heapq.heappop(heap)
            if doc in cache and heap_ver.get(doc) == vid:
                cache.discard(doc)
                return

    hits = 0
    misses = 0
    miss_positions: list[int] = []

    for i, doc in enumerate(trace):
        nu = next_use[i]
        if doc in cache:
            hits += 1
            _push_opt(doc, nu)   # update next_use; old heap entry becomes stale
        else:
            misses += 1
            miss_positions.append(i)
            if len(cache) >= capacity:
                _evict_farthest()
            cache.add(doc)
            _push_opt(doc, nu)

    return CacheResult(
        hits=hits,
        misses=misses,
        hit_rate=hits / n if n > 0 else 0.0,
        miss_positions=miss_positions,
    )
