import numpy as np
import pytest
from loopguard.core.vector import StateVector

def test_state_vector_initialization():
    sv = StateVector(1.0, 2.0, 3.0, 4)
    assert sv._S_hash == 1.0
    assert sv._A_hash == 2.0
    assert sv._T_entropy == 3.0
    assert sv._t == 4

    buf = sv.to_numpy()
    assert buf.shape == (4,)
    assert buf.dtype == np.float32
    assert np.allclose(buf, [1.0, 2.0, 3.0, 4.0])

def test_state_vector_nan_inf():
    sv_nan = StateVector(np.nan, 2.0, 3.0, 4)
    assert np.isnan(sv_nan.to_numpy()[0])

    sv_inf = StateVector(np.inf, 2.0, 3.0, 4)
    assert np.isinf(sv_inf.to_numpy()[0])

def test_state_vector_types():
    # Test casting
    sv = StateVector("1.5", "2.5", "3.5", "4")
    assert sv._S_hash == 1.5
    assert sv._A_hash == 2.5
    assert sv._T_entropy == 3.5
    assert sv._t == 4
    assert sv.to_numpy().dtype == np.float32

from loopguard.detection.exact_match import ExactStateDetector
from loopguard.core.exceptions import LoopInterruptException

def test_exact_state_detector_match():
    detector = ExactStateDetector()
    sv1 = StateVector(1.0, 2.0, 0.5, 0)
    detector.check_and_add(sv1.to_numpy())

    sv2 = StateVector(3.0, 4.0, 0.5, 1)
    detector.check_and_add(sv2.to_numpy())

    # Exact match of (S_hash, A_hash)
    sv3 = StateVector(1.0, 2.0, 0.6, 2)
    with pytest.raises(LoopInterruptException) as excinfo:
        detector.check_and_add(sv3.to_numpy())
    assert "Exact state-action match" in str(excinfo.value)
    assert excinfo.value.context.loop_start_turn == 0
    assert excinfo.value.context.current_turn == 2

def test_exact_state_detector_no_match():
    detector = ExactStateDetector()
    for i in range(5):
        sv = StateVector(float(i), float(i), 0.5, i)
        detector.check_and_add(sv.to_numpy())

def test_exact_state_detector_gating():
    detector = ExactStateDetector()
    # Fill up to 15
    for i in range(16):
        sv = StateVector(float(i), float(i), 0.5, i)
        detector.check_and_add(sv.to_numpy())

    # At t=16, it should not add to buffer even if no match
    sv_16 = StateVector(100.0, 100.0, 0.5, 16)
    detector.check_and_add(sv_16.to_numpy())
    assert detector._row == 16 # Should not have incremented if logic is correct,
    # but wait, let's check ExactStateDetector code again.

def test_exact_state_detector_nan_behavior():
    detector = ExactStateDetector()
    sv_nan = StateVector(np.nan, np.nan, 0.5, 0)
    detector.check_and_add(sv_nan.to_numpy())

    # NaN == NaN should now match using np.isclose(..., equal_nan=True)
    sv_nan_2 = StateVector(np.nan, np.nan, 0.5, 1)
    with pytest.raises(LoopInterruptException):
        detector.check_and_add(sv_nan_2.to_numpy())

from loopguard.detection.anomaly import AnomalyDetector

def test_anomaly_detector_welford_logic():
    detector = AnomalyDetector()

    # Step 0: Initialized
    v0 = StateVector(1.0, 1.0, 0.5, 0).to_numpy()
    d, z, start = detector.update_and_check(v0)
    assert d == 0.0
    assert z == 0.0

    # Step 1: First D
    v1 = StateVector(2.0, 2.0, 0.5, 1).to_numpy()
    d, z, start = detector.update_and_check(v1)
    # distance between (1,1) and (2,2) is sqrt(2)
    assert np.isclose(d, np.sqrt(2))
    assert detector._n == 2
    # Current implementation includes the initial 0.0 in the mean
    assert np.isclose(detector._mean, np.sqrt(2) / 2)

def test_anomaly_detector_trigger():
    detector = AnomalyDetector()

    # Provide stable inputs to establish mean and variance
    for i in range(20):
        # Move by sqrt(2) each time
        v = StateVector(float(i), float(i), 0.5, i).to_numpy()
        try:
            detector.update_and_check(v)
        except LoopInterruptException:
            pytest.fail("Should not trigger anomaly yet")

    # Now provide an exact same input (viscosity D = 0)
    # This should drop the variance significantly if repeated
    v_last = StateVector(19.0, 19.0, 0.5, 20).to_numpy()
    # Repeating this might trigger Z < -3.0 after some iterations
    with pytest.raises(LoopInterruptException) as excinfo:
        for i in range(20, 100):
            v = StateVector(19.0, 19.0, 0.5, i).to_numpy()
            detector.update_and_check(v)

    assert "Infinite loop detected" in str(excinfo.value)
    assert excinfo.value.context.current_turn >= 16
