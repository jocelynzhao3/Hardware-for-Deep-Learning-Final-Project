"""Cache simulation for Experiment 2: Relevance-Aware DRAM Caching.

Scope
-----
This module models cache *replacement behavior* only.  Hardware energy and
TTFT latency are computed in the notebook by multiplying hit/miss counts by
per-access costs derived from EdgeRAG-aligned hardware constants (LPDDR5-4250
DRAM at 34 GB/s, UHS-I SD-card at 0.104 GB/s).

One trace element = one document embedding row access
-----------------------------------------------------
A single cache access is one lookup of a 768-dimensional INT8 document
embedding row (768 bytes).  Each query in the trace contributes k_per_query
such accesses, so  len(trace) = n_queries × k_per_query.

Three cache replacement policies
---------------------------------
  - No cache:       every access is a miss; nothing is ever cached.
  - LFU cost-aware: EdgeRAG Algorithm 2 — evict argmin(ttft_cost[j] × counter[j])
                    with per-step exponential counter decay.
                    When all per-document TTFT costs are equal (uniform), this
                    reduces to plain decayed LFU, isolating reuse locality.
  - Bélády OPT:    offline oracle — evict the item whose next use is farthest
                   in the future (minimum-miss-count lower bound, not deployable).

Trace generator
---------------
synth_trace() draws k_per_query document IDs per query from a Zipf distribution
whose exponent is tuned (binary search on the expected-unique-docs formula) so
that  total_accesses / unique_docs ≈ the requested reuse_ratio.

Note on parameter scale
-----------------------
With n_docs=100 000 and n_queries=5 000:
  total accesses = 100 000,  unique docs ≤ 100 000 (80 000 at reuse 1.25).
  DRAM capacity at 1 GB = 1 398 101 embedding slots >> 80 000 unique docs.
The cache never fills and LFU ≈ OPT for all DRAM sizes in that regime.

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


@dataclass
class TwoTierResult:
    """Hit/miss accounting for an inclusive L1/L2 embedding cache."""

    l1_hits: int
    l2_hits: int
    misses: int
    total: int
    l1_hit_rate: float
    l2_hit_rate: float
    overall_hit_rate: float
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

    # numpy's `rng.choice(p=...)` rebuilds the CDF on every call, so it is
    # O(n_docs) per query regardless of replace=True/False.  For n_docs in the
    # millions × 100k+ queries that is intractable.  Precompute the CDF once
    # and sample via vectorised `searchsorted` (O(k log n) per query).
    #
    # We sample with replacement and dedupe per query.  Since k_per_query is
    # tiny relative to n_docs, the duplicate rate is negligible
    # (P(dup) ~ k * max(weights)), but we oversample by 3x and fall back to
    # a top-up loop in the rare case the dedupe set is short.
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0  # guard against floating-point round-off

    trace: list[int] = []
    oversample = max(k_per_query * 3, k_per_query + 16)

    for _ in range(n_queries):
        raw = np.searchsorted(cdf, rng.random(size=oversample), side="right")
        seen: set[int] = set()
        uniq: list[int] = []
        for d in raw:
            di = int(d)
            if di < n_docs and di not in seen:
                seen.add(di)
                uniq.append(di)
                if len(uniq) == k_per_query:
                    break

        # Top-up loop for the (rare) case the oversample didn't yield enough
        # uniques.  Each top-up draws another batch using the same fast path.
        while len(uniq) < k_per_query:
            extra = np.searchsorted(
                cdf, rng.random(size=k_per_query), side="right",
            )
            for d in extra:
                di = int(d)
                if di < n_docs and di not in seen:
                    seen.add(di)
                    uniq.append(di)
                    if len(uniq) == k_per_query:
                        break

        trace.extend(uniq)

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
# Policy 2b: Inclusive two-tier EdgeRAG LFU
# ---------------------------------------------------------------------------

def simulate_two_tier_lfu(
    trace: list[int],
    l1_capacity: int,
    l2_capacity: int,
    gen_latency: dict[int, float] | None = None,
    decay_factor: float = 0.99,
    check_invariants: bool = False,
) -> TwoTierResult:
    """Inclusive L1/L2 cache with EdgeRAG-style decayed LFU replacement.

    This helper is generic and used by Experiment 2 for:
      L1 = GlobalBuffer (SRAM cache), L2 = MainMemory (DRAM cache),
      backing tier = Disk.

    (In Experiment 3 it may also be used for LocalBuffer/GlobalBuffer.)

    Inclusion invariant
    -------------------
    L1 is always a subset of L2.  On an L2 hit, the document is eagerly
    promoted into L1.  On an L2 miss, the document is filled into both tiers.
    L2 eviction avoids entries currently resident in L1; if L2 is full and
    every L2 entry is also in L1, the simulator evicts one L1 entry first.

    LFU policy
    ----------
    Both tiers share one lazily decayed access counter per document.  Eviction
    priority is  gen_latency[doc] * effective_counter[doc], matching the
    EdgeRAG cost-aware LFU form.  If all costs are uniform, this reduces to
    decayed LFU.

    Parameters
    ----------
    trace            : flat list of document IDs accessed in sequence
    l1_capacity      : maximum number of embeddings in L1
    l2_capacity      : maximum number of embeddings in L2
    gen_latency      : per-document generation/loading latency; defaults to 1.0
    decay_factor     : multiplicative decay applied lazily between accesses
    check_invariants : if True, assert L1 <= L2 after every access
    """
    n = len(trace)
    if n == 0:
        return TwoTierResult(0, 0, 0, 0, 0.0, 0.0, 0.0, [])

    if l2_capacity <= 0:
        no_cache = simulate_no_cache(trace)
        return TwoTierResult(
            l1_hits=0,
            l2_hits=0,
            misses=no_cache.misses,
            total=n,
            l1_hit_rate=0.0,
            l2_hit_rate=0.0,
            overall_hit_rate=0.0,
            miss_positions=no_cache.miss_positions,
        )

    l1_capacity = max(0, min(l1_capacity, l2_capacity))
    gen_latency = gen_latency or {}

    l1_cache: set[int] = set()
    l2_cache: set[int] = set()

    # Lazy-decay counters: doc -> (base_count, step_last_updated)
    base_count: dict[int, float] = {}
    last_step: dict[int, int] = {}

    # Per-tier min-heaps with lazy deletion.
    # Heap entry: (priority, generation_id, doc_id)
    l1_heap: list[tuple[float, int, int]] = []
    l2_heap: list[tuple[float, int, int]] = []
    l1_heap_gen: dict[int, int] = {}
    l2_heap_gen: dict[int, int] = {}
    generation = [0]

    def _effective_count(doc: int, step: int) -> float:
        elapsed = step - last_step.get(doc, step)
        return base_count.get(doc, 0.0) * (decay_factor ** elapsed)

    def _priority(doc: int, step: int) -> float:
        return gen_latency.get(doc, 1.0) * _effective_count(doc, step)

    def _update_counter(doc: int, step: int) -> None:
        base_count[doc] = _effective_count(doc, step) + 1.0
        last_step[doc] = step

    def _push(
        heap: list[tuple[float, int, int]],
        heap_gen: dict[int, int],
        doc: int,
        step: int,
    ) -> None:
        generation[0] += 1
        gid = generation[0]
        heap_gen[doc] = gid
        heapq.heappush(heap, (_priority(doc, step), gid, doc))

    def _evict_from_tier(
        heap: list[tuple[float, int, int]],
        heap_gen: dict[int, int],
        cache: set[int],
        forbidden: set[int],
        step: int,
    ) -> int | None:
        """Evict the lowest-priority valid item not in `forbidden`."""
        skipped_forbidden: list[int] = []
        while heap:
            priority, gid, doc = heapq.heappop(heap)
            if doc not in cache or heap_gen.get(doc) != gid:
                continue
            if doc in forbidden:
                skipped_forbidden.append(doc)
                continue

            cache.discard(doc)
            heap_gen.pop(doc, None)
            for skipped_doc in skipped_forbidden:
                if skipped_doc in cache and heap_gen.get(skipped_doc) is not None:
                    _push(heap, heap_gen, skipped_doc, step)
            return doc
        for skipped_doc in skipped_forbidden:
            if skipped_doc in cache and heap_gen.get(skipped_doc) is not None:
                _push(heap, heap_gen, skipped_doc, step)
        return None

    def _ensure_l1_room(step: int) -> None:
        if l1_capacity <= 0:
            return
        if len(l1_cache) >= l1_capacity:
            _evict_from_tier(l1_heap, l1_heap_gen, l1_cache, set(), step)

    def _ensure_l2_room(step: int) -> None:
        if len(l2_cache) < l2_capacity:
            return

        victim = _evict_from_tier(l2_heap, l2_heap_gen, l2_cache, l1_cache, step)
        if victim is not None:
            return

        # Degenerate inclusive case: L1 and L2 are equally full, so there is no
        # L2-only victim.  Evict from L1 first, then retry L2 eviction.
        _evict_from_tier(l1_heap, l1_heap_gen, l1_cache, set(), step)
        _evict_from_tier(l2_heap, l2_heap_gen, l2_cache, l1_cache, step)

    def _promote_to_l1(doc: int, step: int) -> None:
        if l1_capacity <= 0 or doc in l1_cache:
            return
        _ensure_l1_room(step)
        l1_cache.add(doc)
        _push(l1_heap, l1_heap_gen, doc, step)

    l1_hits = 0
    l2_hits = 0
    misses = 0
    miss_positions: list[int] = []

    for step, doc in enumerate(trace):
        if doc in l1_cache:
            l1_hits += 1
            _update_counter(doc, step)
            _push(l1_heap, l1_heap_gen, doc, step)
            _push(l2_heap, l2_heap_gen, doc, step)
        elif doc in l2_cache:
            l2_hits += 1
            _update_counter(doc, step)
            _push(l2_heap, l2_heap_gen, doc, step)
            _promote_to_l1(doc, step)
        else:
            misses += 1
            miss_positions.append(step)
            _update_counter(doc, step)
            _ensure_l2_room(step)
            l2_cache.add(doc)
            _push(l2_heap, l2_heap_gen, doc, step)
            _promote_to_l1(doc, step)

        if check_invariants:
            assert l1_cache <= l2_cache
            assert len(l1_cache) <= l1_capacity
            assert len(l2_cache) <= l2_capacity

    return TwoTierResult(
        l1_hits=l1_hits,
        l2_hits=l2_hits,
        misses=misses,
        total=n,
        l1_hit_rate=l1_hits / n,
        l2_hit_rate=l2_hits / n,
        overall_hit_rate=(l1_hits + l2_hits) / n,
        miss_positions=miss_positions,
    )


# ---------------------------------------------------------------------------
# Policy 2c: Inclusive two-tier Bélády OPT
# ---------------------------------------------------------------------------

def simulate_two_tier_opt(
    trace: list[int],
    l1_capacity: int,
    l2_capacity: int,
    check_invariants: bool = False,
) -> TwoTierResult:
    """Inclusive L1/L2 Bélády OPT with farthest-next-use eviction.

    This is an offline upper bound for a two-tier inclusive hierarchy:
      - L1 cache is a subset of L2 cache.
      - On L2 hit, the item is promoted to L1.
      - On miss, the item is inserted into L2 and then promoted to L1.
      - L2 eviction prefers L2-only victims; if all L2 entries are also in L1,
        one L1 victim is evicted first and then L2 eviction is retried.
    """
    n = len(trace)
    if n == 0:
        return TwoTierResult(0, 0, 0, 0, 0.0, 0.0, 0.0, [])

    if l2_capacity <= 0:
        no_cache = simulate_no_cache(trace)
        return TwoTierResult(
            l1_hits=0,
            l2_hits=0,
            misses=no_cache.misses,
            total=n,
            l1_hit_rate=0.0,
            l2_hit_rate=0.0,
            overall_hit_rate=0.0,
            miss_positions=no_cache.miss_positions,
        )

    l1_capacity = max(0, min(l1_capacity, l2_capacity))

    # Precompute access positions per document for O(1) next-use lookup.
    pos_by_doc: dict[int, list[int]] = {}
    for i, d in enumerate(trace):
        pos_by_doc.setdefault(d, []).append(i)
    pos_ptr: dict[int, int] = {d: 0 for d in pos_by_doc}

    def _next_use(doc: int) -> int:
        ptr = pos_ptr.get(doc, 0)
        positions = pos_by_doc.get(doc, [])
        nxt = ptr + 1
        return positions[nxt] if nxt < len(positions) else n

    l1_cache: set[int] = set()
    l2_cache: set[int] = set()

    # Max-heap emulation via negative next_use, with lazy deletion via versions.
    l1_heap: list[tuple[int, int, int]] = []  # (-next_use, version, doc)
    l2_heap: list[tuple[int, int, int]] = []
    l1_ver: dict[int, int] = {}
    l2_ver: dict[int, int] = {}
    ver = [0]

    def _push(
        heap: list[tuple[int, int, int]],
        versions: dict[int, int],
        doc: int,
    ) -> None:
        ver[0] += 1
        v = ver[0]
        versions[doc] = v
        heapq.heappush(heap, (-_next_use(doc), v, doc))

    def _evict_farthest(
        heap: list[tuple[int, int, int]],
        versions: dict[int, int],
        cache: set[int],
        forbidden: set[int],
    ) -> int | None:
        skipped: list[int] = []
        while heap:
            _, v, doc = heapq.heappop(heap)
            if doc not in cache or versions.get(doc) != v:
                continue
            if doc in forbidden:
                skipped.append(doc)
                continue
            cache.discard(doc)
            versions.pop(doc, None)
            for d in skipped:
                if d in cache and d in versions:
                    _push(heap, versions, d)
            return doc
        for d in skipped:
            if d in cache and d in versions:
                _push(heap, versions, d)
        return None

    def _ensure_l1_room() -> None:
        if l1_capacity <= 0:
            return
        if len(l1_cache) >= l1_capacity:
            _evict_farthest(l1_heap, l1_ver, l1_cache, set())

    def _ensure_l2_room() -> None:
        if len(l2_cache) < l2_capacity:
            return
        victim = _evict_farthest(l2_heap, l2_ver, l2_cache, l1_cache)
        if victim is not None:
            return
        # Degenerate inclusive case: all L2 entries are currently in L1.
        _evict_farthest(l1_heap, l1_ver, l1_cache, set())
        _evict_farthest(l2_heap, l2_ver, l2_cache, l1_cache)

    def _promote_to_l1(doc: int) -> None:
        if l1_capacity <= 0 or doc in l1_cache:
            return
        _ensure_l1_room()
        l1_cache.add(doc)
        _push(l1_heap, l1_ver, doc)

    l1_hits = 0
    l2_hits = 0
    misses = 0
    miss_positions: list[int] = []

    for step, doc in enumerate(trace):
        if doc in pos_ptr:
            pos_ptr[doc] += 1

        if doc in l1_cache:
            l1_hits += 1
            _push(l1_heap, l1_ver, doc)
            _push(l2_heap, l2_ver, doc)
        elif doc in l2_cache:
            l2_hits += 1
            _push(l2_heap, l2_ver, doc)
            _promote_to_l1(doc)
        else:
            misses += 1
            miss_positions.append(step)
            _ensure_l2_room()
            l2_cache.add(doc)
            _push(l2_heap, l2_ver, doc)
            _promote_to_l1(doc)

        if check_invariants:
            assert l1_cache <= l2_cache
            assert len(l1_cache) <= l1_capacity
            assert len(l2_cache) <= l2_capacity

    return TwoTierResult(
        l1_hits=l1_hits,
        l2_hits=l2_hits,
        misses=misses,
        total=n,
        l1_hit_rate=l1_hits / n,
        l2_hit_rate=l2_hits / n,
        overall_hit_rate=(l1_hits + l2_hits) / n,
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
