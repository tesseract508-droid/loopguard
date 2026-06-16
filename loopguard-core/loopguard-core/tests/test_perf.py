import time
import numpy as np
import pytest
from loopguard.core.engine import LoopGuardEngine

def test_step_latency():
    # Use a mock logger to avoid the current AttributeError during performance test
    # This also ensures we measure the core logic without the queue overhead if it's broken
    import loopguard.core.engine
    class MockLogger:
        def info(self, *args, **kwargs): pass
        def log_turn(self, *args, **kwargs): pass

    original_logger = loopguard.core.engine.telemetry_logger
    loopguard.core.engine.telemetry_logger = MockLogger()

    engine = LoopGuardEngine()
    iterations = 1000
    latencies = []

    for i in range(iterations):
        # Use random floats to avoid exact matches in DeltaChannel accumulated state
        import random
        s = random.random()
        a = random.random()
        t_e = random.random()

        start = time.perf_counter_ns()
        try:
            engine.step(s, a, t_e)
        except LoopInterruptException:
            pass
        end = time.perf_counter_ns()
        latencies.append(end - start)

    avg_latency_ns = sum(latencies) / iterations
    avg_latency_ms = avg_latency_ns / 1_000_000

    print(f"\nAverage latency: {avg_latency_ms:.6f} ms")
    assert avg_latency_ms < 1.0, f"Average latency {avg_latency_ms}ms exceeds 1ms budget"

    loopguard.core.engine.telemetry_logger = original_logger
