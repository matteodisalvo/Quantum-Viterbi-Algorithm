import numpy as np
import pytest

from quantum_viterbi import (
    DecodingResult,
    decode,
    default_iterations,
    encode,
    omega_range,
    parse_code,
    scan_omega,
    tuned_omega,
)


def test_decoding_result_properties():
    probabilities = np.zeros(16)
    probabilities[0b1011] = 0.7
    probabilities[0b0011] = 0.3
    result = DecodingResult(parse_code("11 01 00 11"), 0.5, 3, "numpy", probabilities)

    assert result.best_state == 11
    assert result.decoded_bits == (1, 0, 1, 1)
    assert result.reencoded == (1, 1, 0, 1, 0, 0, 1, 0)
    assert result.hamming_distance == 1
    assert result.top_states(2) == [("1011", 0.7), ("0011", 0.3)]
    assert "1011" in result.summary()


def test_noiseless_codeword_is_decoded():
    bits = (1, 0, 1, 1, 0, 1)
    code = encode(bits)
    scan = scan_omega(code, omega_range(0, np.pi, 0.01), target=bits)
    result = decode(code, omega=scan.best_omega, backend="numpy")
    assert result.decoded_bits == bits
    assert result.hamming_distance == 0


def test_omega_is_tuned_by_default():
    code = "11 01 00 10"
    result = decode(code, backend="numpy")
    assert result.tuned
    assert result.omega == pytest.approx(tuned_omega(code))
    assert result.iterations == default_iterations(4)
    assert result.shots is None
    assert "(tuned)" in result.summary()


def test_shots_and_seed_are_threaded_through():
    code, omega = "11 01 00 10", 0.6
    exact = decode(code, omega=omega, backend="numpy")
    sampled = decode(code, omega=omega, backend="numpy", shots=4096, seed=1)
    repeated = decode(code, omega=omega, backend="numpy", shots=4096, seed=1)
    other_seed = decode(code, omega=omega, backend="numpy", shots=4096, seed=2)

    assert sampled.shots == 4096
    assert sampled.probabilities.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(sampled.probabilities, repeated.probabilities)
    assert not np.allclose(sampled.probabilities, other_seed.probabilities)
    assert not np.allclose(sampled.probabilities, exact.probabilities)
    assert "4096 shots" in sampled.summary()


def test_explicit_iterations_are_used_and_omega_is_tuned_for_them():
    code = "11 01 00 10"
    result = decode(code, iterations=3, backend="numpy")
    assert result.iterations == 3
    assert result.omega == pytest.approx(tuned_omega(code, iterations=3))


def test_decode_uses_the_documented_default_backend():
    from quantum_viterbi.backends import DEFAULT_BACKEND

    result = decode("11 01 00 10", omega=0.6)
    assert result.backend == DEFAULT_BACKEND == "qiskit"
    reference = decode("11 01 00 10", omega=0.6, backend="numpy")
    np.testing.assert_allclose(result.probabilities, reference.probabilities, atol=1e-12)


def test_top_states_rejects_a_non_positive_count():
    result = decode("11 01 00 10", omega=0.6, backend="numpy")
    for count in (0, -1, 2.5):
        with pytest.raises(ValueError, match="k must be a positive integer"):
            result.top_states(count)


def test_a_given_omega_is_not_reported_as_tuned():
    result = decode("11 01 00 10", omega=0.5, backend="numpy")
    assert not result.tuned
    assert result.omega == 0.5
    assert "(tuned)" not in result.summary()
