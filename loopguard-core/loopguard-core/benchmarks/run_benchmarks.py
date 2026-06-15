from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np

from loopguard.core import LoopGuardEngine
from loopguard.core.exceptions import LoopInterruptException


def _ascii_percentile(values_ns: List[int], percentile: float) -> int:
    if not values_ns:
        return 0
    sorted_vals = sorted(values_ns)
    n = len(sorted_vals)
    idx = int(round((percentile / 100.0) * (n - 1)))
    if idx < 0:
        idx = 0
    if idx >= n:
        idx = n - 1
    return sorted_vals[idx]


def run_stress_test() -> None:
    engine = LoopGuardEngine()

    iterations = 100_000
    rng = np.random.default_rng(1337)

    # Pre-computed random inputs: (s_hash, a_hash, t_entropy)
    inputs: List[Tuple[float, float, float]] = [
        (float(s), float(a), float(t))
        for s, a, t in zip(
            rng.random(iterations, dtype=np.float32) * 1_000_000.0,
            rng.random(iterations, dtype=np.float32) * 1_000_000.0,
            rng.random(iterations, dtype=np.float32) * 100.0,
        )
    ]

    latencies_ns: List[int] = [0] * iterations

    total_start = time.perf_counter_ns()

    for i in range(iterations):
        s_hash, a_hash, t_entropy = inputs[i]

        start_ns = time.perf_counter_ns()
        try:
            engine.step(s_hash=s_hash, a_hash=a_hash, t_entropy=t_entropy)
        except LoopInterruptException:
            # Disable kill-switch for continuous benchmark runs.
            pass
        end_ns = time.perf_counter_ns()

        latencies_ns[i] = end_ns - start_ns

    total_end = time.perf_counter_ns()
    total_ns = total_end - total_start

    total_ms = total_ns / 1_000_000.0
    avg_ms = (sum(latencies_ns) / len(latencies_ns)) / 1_000_000.0

    p50_ns = _ascii_percentile(latencies_ns, 50.0)
    p90_ns = _ascii_percentile(latencies_ns, 90.0)
    p99_ns = _ascii_percentile(latencies_ns, 99.0)

    p50_ms = p50_ns / 1_000_000.0
    p90_ms = p90_ns / 1_000_000.0
    p99_ms = p99_ns / 1_000_000.0

    print("+-------------------------------+------------------+")
    print("| Metric                        | Value            |")
    print("+-------------------------------+------------------+")
    print(f"| Total Execution Time (100k) | {total_ms:>16.6f} ms |")
    print(f"| Average Latency             | {avg_ms:>16.6f} ms |")
    print(f"| P50 Latency                  | {p50_ms:>16.6f} ms |")
    print(f"| P90 Latency                  | {p90_ms:>16.6f} ms |")
    print(f"| P99 Latency                  | {p99_ms:>16.6f} ms |")
    print("+-------------------------------+------------------+")


if __name__ == "__main__":
    run_stress_test()
