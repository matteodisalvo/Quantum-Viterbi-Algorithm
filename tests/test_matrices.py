from pathlib import Path

import numpy as np
import pytest

from quantum_viterbi.convolutional import path_metrics
from quantum_viterbi.matrices import build_g, diffusion_matrix, qva_statevector

# Reference data exported from an independent implementation of the same algorithm
REFERENCE = Path(__file__).parent / "data" / "reference_run.npz"


@pytest.mark.parametrize("num_transitions", range(1, 7))
def test_oracle_is_a_phase_proportional_to_the_hamming_distance(num_transitions):
    rng = np.random.default_rng(num_transitions)
    for _ in range(5):
        code = rng.integers(0, 2, 2 * num_transitions)
        omega = rng.uniform(0, 2 * np.pi)
        expected = np.diag(np.exp(1j * omega * path_metrics(code)))
        np.testing.assert_allclose(build_g(omega, code), expected, atol=1e-12)


def test_matrices_reproduce_the_reference_run():
    reference = np.load(REFERENCE)
    code = reference["code"]
    omega = float(reference["omega"])
    iterations = int(reference["iterations"])

    np.testing.assert_allclose(build_g(omega, code), reference["oracle"], atol=1e-10)
    np.testing.assert_allclose(diffusion_matrix(code.size // 2), reference["diffusion"], atol=1e-10)
    state = qva_statevector(code, omega, iterations)
    np.testing.assert_allclose(state, reference["statevector"], atol=1e-10)
    np.testing.assert_allclose(np.abs(state) ** 2, reference["probabilities"], atol=1e-10)
