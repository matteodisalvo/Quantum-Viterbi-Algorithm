import math

import numpy as np
import pytest

from quantum_viterbi import (
    decode,
    default_iterations,
    distance_spectrum,
    get_backend,
    omega_range,
    parse_code,
    path_metrics,
    scan_omega,
    tuned_omega,
)
from quantum_viterbi.tuning import _decoded_scores

README_CODE = "11 01 00 10 00 00"


def _score(code, omegas, iterations=None):
    """The score tuned_omega maximizes, at each omega."""
    pairs = parse_code(code)
    spectrum = distance_spectrum(pairs)
    d0 = next(d for d, count in enumerate(spectrum) if count)
    rounds = default_iterations(len(pairs)) if iterations is None else iterations
    return _decoded_scores(pairs, spectrum, d0, np.atleast_1d(omegas), rounds)


def _lead(code, **options):
    """Decode, then return how much more probable a maximum-likelihood path is than any other."""
    metrics = path_metrics(code)
    probabilities = decode(code, backend="numpy", **options).probabilities
    best = metrics == metrics.min()
    other = probabilities[~best].max() if (~best).any() else 0.0
    return float(probabilities[best].max() - other)


def test_default_iterations_follow_the_grover_formula():
    for n in range(1, 10):
        assert default_iterations(n) == math.floor(math.pi / 4 * math.sqrt(2**n))
    assert default_iterations(6) == 6


def test_tuned_omega_decodes_the_readme_word_with_a_clear_lead():
    omega = tuned_omega(README_CODE)
    assert omega == pytest.approx(0.6819, abs=5e-4)
    assert _lead(README_CODE) == pytest.approx(0.2007, abs=5e-4)
    dense = np.linspace(math.pi / 20_010, math.pi, 20_010)  # ten times the default grid
    assert _score(README_CODE, omega)[0] >= _score(README_CODE, dense).max() - 1e-12


def test_tuned_omega_follows_the_iteration_count():
    dense = np.linspace(math.pi / 20_010, math.pi, 20_010)
    for iterations in (1, 4, 9):
        omega = tuned_omega(README_CODE, iterations=iterations)
        best = _score(README_CODE, dense, iterations).max()
        assert _score(README_CODE, omega, iterations)[0] >= best - 1e-12
        assert _lead(README_CODE, iterations=iterations) > 0


def test_tuned_omega_grid_is_fine_enough_at_larger_n():
    # The peaks narrow like 4 * 2**(-N/2) / N, so the coarse grid grows with N
    code = "11 01 00 10 00 00 11 01 00 10 00 00 11 01 00 10"
    dense = np.linspace(math.pi / 32_770, math.pi, 32_770)  # ten times the grid at N = 16
    assert _score(code, tuned_omega(code))[0] >= _score(code, dense).max() - 1e-12


def test_scan_threshold_is_inclusive():
    code, target = "11 01 00 10 00 00", "101101"
    omegas = omega_range(0, 180, 1.0, degrees=True)
    full = scan_omega(code, omegas, target=target)
    threshold = float(full.scores[40])  # a value the scan reaches exactly
    early = scan_omega(code, omegas, target=target, threshold=threshold)
    assert early.omegas.size == int(np.argmax(full.scores >= threshold)) + 1
    assert early.threshold_reached


def test_tuned_omega_refines_a_peak_that_falls_on_the_grid_boundary():
    # On this five-point grid the best point is the first one, which is not an interior
    # local maximum; refining only the interior peaks stops at a score of 0.19
    code = "01111010"
    dense = np.linspace(1e-4, math.pi, 200_001)
    assert _score(code, tuned_omega(code, points=5))[0] >= _score(code, dense).max() - 1e-6


@pytest.mark.parametrize("code", ["01100100", "00 00 01 00 00", "010001110110", "10001111"])
def test_aliased_words_decode_to_a_maximum_likelihood_message(code):
    # Maximizing only the probability of the right answer let a path at distance
    # 2N - d_min win on these words, or tie with the right one
    assert _lead(code) > 0.001


def test_some_short_aliased_words_need_fewer_iterations():
    # At N = 4 and the default K = 3 no omega decodes this word correctly; one round does
    code = "11111110"
    assert _score(code, np.linspace(1e-4, math.pi, 20_001)).max() <= 0
    assert _lead(code, iterations=1) > 0


def test_tuned_omega_handles_degenerate_words():
    # For N = 1 both codewords, 00 and 11, are one bit away from 01: omega cannot help
    assert 0 < tuned_omega("01") <= math.pi
    for points in (1, 2.5):
        with pytest.raises(ValueError, match="points must be an integer"):
            tuned_omega("11 01 00 10", points=points)


def test_omega_range_includes_both_endpoints():
    assert omega_range(0.5, 50, 0.01).size == 4951
    assert omega_range(0, 360, 0.0005, degrees=True).size == 720_001
    np.testing.assert_allclose(omega_range(0, 180, 90, degrees=True), [0, math.pi / 2, math.pi])


def test_scan_without_target_uses_the_peak_probability():
    code, omegas = "11 01 00 10", np.linspace(0.1, 3.0, 30)
    scan = scan_omega(code, omegas, iterations=3)
    backend = get_backend("numpy")
    expected = [backend.probabilities(code, omega, 3).max() for omega in omegas]
    np.testing.assert_allclose(scan.scores, expected)
    assert scan.best_omega == pytest.approx(omegas[int(np.argmax(expected))])
    assert scan.top(1) == [(scan.best_omega, scan.best_score)]


def test_scan_with_threshold_stops_at_the_first_hit():
    code, target = "11 01 00 10 00 00", "101100"
    omegas = omega_range(0, 360, 0.5, degrees=True)
    full = scan_omega(code, omegas, target=target, chunk_size=7)
    threshold = 0.5 * full.best_score
    early = scan_omega(code, omegas, target=target, threshold=threshold, chunk_size=7)

    first_hit = int(np.argmax(full.scores >= threshold))
    assert early.omegas.size == first_hit + 1
    assert early.threshold_reached
    assert early.best_omega == pytest.approx(omegas[first_hit])
