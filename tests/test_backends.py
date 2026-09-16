import itertools

import numpy as np
import pytest

from quantum_viterbi.backends import BACKEND_NAMES, available_backends, get_backend
from quantum_viterbi.grover import diffusion_gates
from quantum_viterbi.matrices import build_g, diffusion_matrix, qva_statevector
from quantum_viterbi.oracle import oracle_gates

INSTALLED = available_backends()
BACKENDS = [
    pytest.param(name, marks=pytest.mark.skipif(name not in INSTALLED, reason=f"{name} is not installed"))
    for name in BACKEND_NAMES
]

# Every pair in every position of a three-transition code: covers all the
# V_i0, V_i1 and V_k phase blocks for every received pair.
ALL_THREE_STEP_CODES = [" ".join(p) for p in itertools.product(["00", "01", "10", "11"], repeat=3)]
OTHER_CODES = ["11", "10 01", "11 01 00 10 00", "00 11 01 11 01 00"]


@pytest.mark.parametrize("backend_name", BACKENDS)
@pytest.mark.parametrize("code", ALL_THREE_STEP_CODES + OTHER_CODES)
def test_statevector_matches_the_matrix_reference(backend_name, code):
    omega, iterations = 0.6733, 2
    expected = qva_statevector(code, omega, iterations)
    actual = get_backend(backend_name).statevector(code, omega, iterations)
    np.testing.assert_allclose(actual, expected, atol=1e-10)


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_batch_matches_single_runs(backend_name):
    backend = get_backend(backend_name)
    code, omegas = "11 01 00 10", [0.1, 0.8, 2.5]
    batch = backend.probabilities_batch(code, omegas)
    single = np.array([backend.probabilities(code, omega) for omega in omegas])
    np.testing.assert_allclose(batch, single, atol=1e-10)


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_sampling_converges_to_exact_probabilities(backend_name):
    backend = get_backend(backend_name)
    code, omega = "11 01 00 10", 0.8
    exact = backend.probabilities(code, omega)
    sampled = backend.probabilities(code, omega, shots=20_000, seed=1234)
    assert sampled.sum() == pytest.approx(1.0)
    assert np.max(np.abs(sampled - exact)) < 0.02


def test_qiskit_unitaries_match_the_reference_matrices():
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Operator

    from quantum_viterbi.backends.qiskit_backend import gates_to_circuit

    omega = 1.1
    for code in ALL_THREE_STEP_CODES:
        circuit = gates_to_circuit(oracle_gates(code), 3, omega)
        np.testing.assert_allclose(Operator(circuit).data, build_g(omega, code), atol=1e-10)
    for num_qubits in (1, 2, 4):
        circuit = gates_to_circuit(diffusion_gates(num_qubits), num_qubits, omega)
        np.testing.assert_allclose(Operator(circuit).data, diffusion_matrix(num_qubits), atol=1e-10)


def test_pennylane_unitaries_match_the_reference_matrices():
    qml = pytest.importorskip("pennylane")
    from quantum_viterbi.backends.pennylane_backend import apply_gates

    omega = 1.1
    for code in ALL_THREE_STEP_CODES:
        gates = oracle_gates(code)
        matrix = qml.matrix(lambda gates=gates: apply_gates(gates, omega), wire_order=[0, 1, 2])()
        np.testing.assert_allclose(matrix, build_g(omega, code), atol=1e-10)
    for num_qubits in (1, 2, 4):
        gates = diffusion_gates(num_qubits)
        wires = list(range(num_qubits))
        matrix = qml.matrix(lambda gates=gates: apply_gates(gates, omega), wire_order=wires)()
        np.testing.assert_allclose(matrix, diffusion_matrix(num_qubits), atol=1e-10)


def test_qiskit_circuit_can_be_rebound_and_exported():
    pytest.importorskip("qiskit")
    from qiskit import qasm3
    from qiskit.quantum_info import Statevector

    backend = get_backend("qiskit")
    circuit = backend.build_circuit("11 01 00", iterations=1)
    assert circuit.parameters
    bound = circuit.assign_parameters({backend.omega: 0.4})
    np.testing.assert_allclose(Statevector(bound).data, qva_statevector("11 01 00", 0.4, 1), atol=1e-10)
    assert "OPENQASM 3" in qasm3.dumps(circuit)


def test_base_probabilities_batch_is_usable_by_a_minimal_subclass():
    from quantum_viterbi.backends.base import Backend

    class _MinimalBackend(Backend):
        """Implements only the two abstract methods, so it inherits the generic batch."""

        name = "minimal"

        def _statevector(self, pairs, omega, iterations):
            return get_backend("numpy").statevector(pairs, omega, iterations)

        def _sample(self, pairs, omega, iterations, shots, seed):
            raise NotImplementedError

    backend, code, omegas = _MinimalBackend(), "11 01 00 10", [0.1, 0.8]
    batch = backend.probabilities_batch(code, omegas, 2)
    single = np.array([backend.probabilities(code, omega, 2) for omega in omegas])
    np.testing.assert_allclose(batch, single, atol=1e-12)
    from_generator = backend.probabilities_batch(code, (w for w in omegas), 2)
    np.testing.assert_allclose(from_generator, batch, atol=1e-12)


def test_qiskit_backend_uses_the_configured_sampler_and_pass_manager():
    pytest.importorskip("qiskit")
    from qiskit.primitives import StatevectorSampler

    from quantum_viterbi.backends.qiskit_backend import QiskitBackend

    calls = []

    class _RecordingPassManager:
        def run(self, circuit):
            calls.append("pass_manager")
            return circuit

    class _RecordingSampler(StatevectorSampler):
        def run(self, pubs, **kwargs):
            calls.append("sampler")
            return super().run(pubs, **kwargs)

    backend = QiskitBackend(sampler=_RecordingSampler(seed=3), pass_manager=_RecordingPassManager())
    probabilities = backend.probabilities("11 01 00", 0.5, shots=512)
    assert calls == ["pass_manager", "sampler"]
    assert probabilities.sum() == pytest.approx(1.0)


def test_pennylane_backend_uses_the_configured_device_and_seed():
    pytest.importorskip("pennylane")
    from quantum_viterbi.backends.pennylane_backend import PennyLaneBackend

    backend = PennyLaneBackend(device="default.qubit")
    assert backend.device_name == "default.qubit"
    with pytest.raises(Exception, match="not.a.device"):
        PennyLaneBackend(device="not.a.device").build_qnode("11 01", iterations=1)

    sampled = backend.probabilities("11 01 00", 0.5, shots=1024, seed=5)
    repeated = backend.probabilities("11 01 00", 0.5, shots=1024, seed=5)
    np.testing.assert_allclose(sampled, repeated)


def test_shots_must_be_a_positive_integer():
    backend = get_backend("numpy")
    for shots in (0, -1, 2.7):
        with pytest.raises(ValueError, match="shots must be a positive integer"):
            backend.probabilities("11 01", 0.5, shots=shots)


def test_get_backend_validation():
    backend = get_backend("numpy")
    assert get_backend(backend) is backend
    with pytest.raises(ValueError):
        get_backend("cirq")
