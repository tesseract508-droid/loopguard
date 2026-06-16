import pytest
import numpy as np
from loopguard.core.engine import LoopGuardEngine

def test_robustness_malformed_hashes():
    engine = LoopGuardEngine()
    # Test with string hashes that look like numbers
    engine.step("1.0", "2.0", "0.5")

    # Test with extreme values
    engine.step(1e30, -1e30, 0.0)

def test_robustness_none_input():
    engine = LoopGuardEngine()
    with pytest.raises((TypeError, ValueError)):
        engine.step(None, 1.0, 0.5)

def test_robustness_invalid_agent_id():
    engine = LoopGuardEngine()
    # agent_id should be string, but let's see if it handles other types
    engine.step(1.0, 1.0, 0.1, agent_id=123)
    assert "123" in engine._sessions or 123 in engine._sessions
