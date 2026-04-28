"""Generate workspace/baseline_cache.pkl for Experiment-2-Caching.ipynb.

Runs the per-einsum baseline mapping for `full_EdgeRAG.yaml` on `basic8.yaml`
with the same parameters as `M3-Baseline.ipynb` and dumps:

    baseline_energy_J   einsum_name -> total energy in Joules
    baseline_latency_s  einsum_name -> total latency in seconds
    sim_einsum          'doc_scores' (the term replaced by cache_sim)
    rag_params          workload jinja parameters used
    arch_overrides      basic8 jinja parameters used

Run inside the lab container:
    docker compose exec labs python /home/workspace/workspace/scripts/export_baseline.py
"""

import os
import pickle
import sys

WORKSPACE = "/home/workspace/workspace"
sys.path.insert(0, WORKSPACE)
sys.path.insert(0, os.path.join(WORKSPACE, "scripts"))

import accelforge as af
from _load_spec import get_spec
from utils import Result

WORKLOAD_PATH = os.path.join(WORKSPACE, "workload", "full_EdgeRAG.yaml")
OUT_PATH = os.path.join(WORKSPACE, "baseline_cache.pkl")
SIM_EINSUM = "doc_scores"

rag_params = {
    "N_DOCS": 5_000_000,
    "N_TOKENS": 512,
    "K": 10,
    "L": 512,
}

arch_overrides = {
    "DRAM_SIZE_GB": 8,
    "DRAM_BW_GBps": 68,
    "SRAM_SIZE_MB": 8,
    "SRAM_READ_BW_GBps": 512,
    "SRAM_WRITE_BW_GBps": 128,
    "MAC_ENERGY_pJ": 0.084,
    "MAC_CLOCK_GHz": 1,
}


def run_per_einsum_baseline():
    workload = af.Workload.from_yaml(
        WORKLOAD_PATH, top_key="workload", jinja_parse_data=rag_params,
    )

    print(f"Loaded {len(workload.einsums)} einsums from {WORKLOAD_PATH}")
    print(f"  N_DOCS={rag_params['N_DOCS']:,}  K={rag_params['K']}  "
          f"L={rag_params['L']}  N_TOKENS={rag_params['N_TOKENS']}")

    results = {}
    for einsum in workload.einsums:
        spec = get_spec(
            "basic8", add_dummy_main_memory=False, jinja_parse_data=arch_overrides,
        )
        spec.workload = af.Workload(
            einsums=[einsum],
            rank_sizes=workload.rank_sizes,
            bits_per_value=workload.bits_per_value,
            persistent_tensors=workload.persistent_tensors,
        )
        spec.mapper.max_pmapping_templates_per_einsum = 1
        spec.mapper.metrics = af.mapper.Metrics.ENERGY | af.mapper.Metrics.LATENCY

        mappings = spec.map_workload_to_arch(
            print_progress=False, print_number_of_pmappings=False,
        )
        results[einsum.name] = Result(mappings, variables=rag_params)
        e_pJ = results[einsum.name].per_compute("energy") * 1e12
        try:
            l_s = float(mappings.latency())
        except Exception:
            l_s = float("nan")
        print(f"  {einsum.name:<22}  {e_pJ:>8.2f} pJ/compute   latency={l_s:.6f} s")

    return results


def main():
    results = run_per_einsum_baseline()

    baseline_energy_J = {}
    baseline_latency_s = {}
    for name, result in results.items():
        baseline_energy_J[name] = sum(result.per_component_energy.values())
        try:
            baseline_latency_s[name] = float(result._mappings.latency())
        except Exception as exc:  # noqa: BLE001
            print(f"  warning: latency unavailable for {name}: {exc}")
            baseline_latency_s[name] = 0.0

    if SIM_EINSUM not in baseline_energy_J:
        raise RuntimeError(
            f"sim_einsum '{SIM_EINSUM}' not found. "
            f"Available: {sorted(baseline_energy_J)}",
        )

    payload = {
        "baseline_energy_J": baseline_energy_J,
        "baseline_latency_s": baseline_latency_s,
        "sim_einsum": SIM_EINSUM,
        "rag_params": rag_params,
        "arch_overrides": arch_overrides,
    }
    with open(OUT_PATH, "wb") as f:
        pickle.dump(payload, f)

    total_e_nj = sum(baseline_energy_J.values()) * 1e9
    total_l_us = sum(baseline_latency_s.values()) * 1e6
    sim_e_nj = baseline_energy_J[SIM_EINSUM] * 1e9
    sim_l_us = baseline_latency_s[SIM_EINSUM] * 1e6

    print()
    print(f"Wrote {OUT_PATH}")
    print(f"  einsums:        {len(baseline_energy_J)}")
    print(f"  sim_einsum:     {SIM_EINSUM}  "
          f"(E={sim_e_nj:,.2f} nJ, L={sim_l_us:,.2f} us)")
    print(f"  total energy:   {total_e_nj:,.2f} nJ")
    print(f"  total latency:  {total_l_us:,.2f} us")


if __name__ == "__main__":
    main()
