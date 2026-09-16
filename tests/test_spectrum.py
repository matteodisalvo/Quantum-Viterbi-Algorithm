import math

import numpy as np
import pytest

from quantum_viterbi import (
    channel_omega,
    class_probabilities,
    decode,
    default_iterations,
    distance_spectrum,
    distance_weights,
    encode,
    get_backend,
    path_metrics,
    resonant_omega,
    scan_omega,
    scan_resonance,
    viterbi_decode,
)

README_CODE = "11 01 00 10 00 00"


def _random_word(rng, n, p):
    codeword = np.array(encode(rng.integers(0, 2, n)))
    return (codeword ^ (rng.random(2 * n) < p)).astype(int).tolist()


def test_distance_spectrum_matches_brute_force():
    rng = np.random.default_rng(1)
    for n in range(1, 11):
        for p in (0.0, 0.1, 0.3):
            code = _random_word(rng, n, p)
            expected = np.bincount(path_metrics(code), minlength=2 * n + 1)
            assert distance_spectrum(code) == tuple(expected.tolist())
            np.testing.assert_allclose(distance_weights(code), expected / 2**n, rtol=0, atol=1e-15)


def test_distance_spectrum_known_values():
    assert distance_spectrum(README_CODE) == (0, 1, 2, 4, 6, 9, 14, 15, 10, 2, 0, 1, 0)
    assert distance_spectrum("00" * 4) == (1, 0, 1, 3, 5, 4, 1, 1, 0)  # Grice & Meyer, N = 4
    noiseless_10 = (1, 0, 1, 3, 6, 20, 37, 74, 124, 149, 171, 175, 122, 70, 47, 20, 3, 1, 0, 0, 0)
    assert distance_spectrum("00" * 10) == noiseless_10


def test_distance_spectrum_is_exact_for_large_n():
    rng = np.random.default_rng(2)
    code = _random_word(rng, 100, 0.05)
    spectrum = distance_spectrum(code)
    assert sum(spectrum) == 2**100
    assert next(d for d, count in enumerate(spectrum) if count) == viterbi_decode(code).metric
    assert distance_weights(code).sum() == pytest.approx(1.0, abs=1e-12)


def test_class_probabilities_match_the_statevector():
    rng = np.random.default_rng(3)
    backend = get_backend("numpy")
    for n in range(1, 10):
        code = _random_word(rng, n, 0.1)
        omegas = rng.uniform(0, 2 * math.pi, 5)
        rounds = default_iterations(n) + 2
        classes = class_probabilities(code, omegas, rounds)
        spectrum = np.array(distance_spectrum(code), dtype=float)
        metrics = path_metrics(code)
        np.testing.assert_allclose(classes.sum(axis=1), 1.0, atol=1e-12)
        expected = backend.probabilities_batch(code, omegas, rounds)
        np.testing.assert_allclose(classes[:, metrics] / spectrum[metrics], expected, atol=1e-12)


def test_class_probabilities_reproduce_known_points():
    assert class_probabilities(README_CODE, [0.6733], 6)[0, 1] == pytest.approx(0.2971, abs=5e-5)
    assert class_probabilities(README_CODE, [0.6808], 9)[0, 1] == pytest.approx(0.4969, abs=5e-5)
    # Grice & Meyer (noiseless words): worked example and the N = 10 table entry
    assert class_probabilities("00" * 4, [0.68], 3)[0, 0] == pytest.approx(0.6736, abs=5e-5)
    assert class_probabilities("00" * 10, [0.31], 25)[0, 0] == pytest.approx(0.7289, abs=5e-5)
    assert class_probabilities("00" * 10, [0.3144], 25)[0, 0] == pytest.approx(0.8254, abs=5e-5)


def test_resonant_omega_readme_example():
    res = resonant_omega(README_CODE)
    assert (res.min_distance, res.multiplicity, res.iterations) == (1, 1, 9)
    assert res.omega == pytest.approx(0.6677, abs=5e-5)
    assert res.optimal_iterations == pytest.approx(8.608, abs=1e-3)
    assert res.probability == pytest.approx(0.4759, abs=5e-5)
    at_resonance = class_probabilities(README_CODE, [res.omega], res.iterations)[0, 1]
    assert at_resonance == pytest.approx(0.4547, abs=5e-5)
    assert class_probabilities(README_CODE, [res.omega], 6)[0, 1] == pytest.approx(0.2956, abs=5e-5)
    result = decode(README_CODE, omega=res.omega, iterations=res.iterations, backend="numpy")
    assert result.best_probability == pytest.approx(0.4547, abs=5e-5)
    assert result.decoded_bits == (1, 0, 1, 1, 0, 1)


@pytest.mark.parametrize(
    ("n", "omega", "k_star", "peak", "exact"),
    [
        (3, 0.9130, 1.80, 0.930, 0.6924),
        (4, 0.7292, 3.02, 0.795, 0.6230),
        (5, 0.6065, 4.54, 0.777, 0.7372),
        (6, 0.5152, 6.64, 0.775, 0.7618),
        (7, 0.4458, 9.54, 0.783, 0.7688),
        (8, 0.3920, 13.54, 0.802, 0.7653),
        (9, 0.3492, 19.12, 0.821, 0.8209),
        (10, 0.3146, 26.95, 0.838, 0.8366),
    ],
)
def test_resonant_omega_noiseless_words(n, omega, k_star, peak, exact):
    code = "00" * n
    res = resonant_omega(code)
    assert res.omega == pytest.approx(omega, abs=5e-5)
    assert res.optimal_iterations == pytest.approx(k_star, abs=5e-3)
    assert res.probability == pytest.approx(peak, abs=5e-4)
    assert class_probabilities(code, [res.omega], res.iterations)[0, 0] == pytest.approx(exact, abs=5e-5)


def test_resonant_omega_is_asymptotically_exact():
    code = "00" * 20
    res = resonant_omega(code)
    assert res.omega == pytest.approx(0.157133, abs=5e-7)
    assert res.iterations == 834
    assert res.width == pytest.approx(1.89e-4, rel=0.01)
    exact = class_probabilities(code, [res.omega], res.iterations)[0, 0]
    assert exact == pytest.approx(res.probability, abs=1e-3)
    assert exact == pytest.approx(0.9297, abs=5e-5)


def test_scan_resonance_matches_scan_omega():
    scan = scan_resonance(README_CODE, points=301)
    assert scan.target == int("101101", 2) and scan.iterations == 6
    reference = scan_omega(README_CODE, scan.omegas, target=scan.target)
    np.testing.assert_allclose(scan.scores, reference.scores, atol=1e-12)
    fine = scan_resonance(README_CODE)
    assert fine.best_omega == pytest.approx(0.6733, abs=5e-4)
    assert fine.best_score == pytest.approx(0.2971, abs=5e-5)


def test_scan_resonance_normalizes_by_the_number_of_maximum_likelihood_paths():
    # A word with two maximum-likelihood paths: the scores are per path, not per class
    code = "10 00 01 01"
    res = resonant_omega(code)
    assert (res.min_distance, res.multiplicity) == (2, 2)
    scan = scan_resonance(code, points=51)
    reference = scan_omega(code, scan.omegas, target=scan.target)
    np.testing.assert_allclose(scan.scores, reference.scores, atol=1e-12)


def test_resonant_omega_rejects_a_flat_spectrum():
    with pytest.raises(ValueError):
        resonant_omega("01")  # both paths are at distance 1


def test_resonant_omega_on_an_aliased_word():
    # A path at distance 2N - d_min = 21 shares the target phase next to pi/(N - d_min):
    # the smallest-S root predicts 0.50 but reaches only 0.26, so another root is chosen
    code = "11 11 10 11 00 10 11 01 01 11 01 00"
    spectrum = distance_spectrum(code)
    assert spectrum[3] == 1 and spectrum[21] == 1
    res = resonant_omega(code)
    assert res.min_distance == 3
    assert res.omega == pytest.approx(2.96668, abs=5e-5)
    assert class_probabilities(code, [res.omega], res.iterations)[0, 3] == pytest.approx(0.3603, abs=5e-4)
    assert class_probabilities(code, [0.351261], 71)[0, 3] == pytest.approx(0.2565, abs=5e-4)


def test_resonant_omega_probability_is_clamped():
    assert resonant_omega("00").probability == 1.0
    assert resonant_omega("00 11").probability == 1.0


def test_omega_grids_accept_generators():
    assert class_probabilities("11 01 00 10", (x for x in [0.1, 0.2]), 1).shape == (2, 9)
    assert class_probabilities("11 01 00 10", np.array(0.5), 1).shape == (1, 9)
    assert scan_omega("11 01 00 10", (x for x in [0.1, 0.2]), iterations=1).omegas.size == 2


def test_channel_omega():
    assert channel_omega(10) == pytest.approx(math.pi / (10 + 10 / 2**10))
    assert channel_omega(12, 0.1) == pytest.approx(math.pi / (12 + 10 / 2**12 - 2))  # j = floor(2.04)
    assert channel_omega(8, 0.15) == pytest.approx(math.pi / (8 + 10 / 2**8 - 1))  # j = floor(1.86)
    assert channel_omega(np.int64(64)) == pytest.approx(channel_omega(64))
    with pytest.raises(ValueError):
        channel_omega(6, 0.6)
    with pytest.raises(ValueError):
        channel_omega(6.5)
