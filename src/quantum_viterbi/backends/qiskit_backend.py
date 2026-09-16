"""Qiskit backend: the QVA as a :class:`qiskit.QuantumCircuit`."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector

from ..convolutional import CodeLike, Pair, parse_code
from ..gates import Gate
from ..grover import qva_gates
from ..spectrum import as_omega_grid
from .base import Backend, counts_to_probabilities


def gates_to_circuit(gates: Iterable[Gate], num_qubits: int, omega: Any) -> QuantumCircuit:
    """Translate a framework-independent gate list into a Qiskit circuit.

    Qiskit numbers qubits little-endian (qubit 0 is the least significant bit),
    whereas this package is big-endian (wire 0 is the most significant bit).
    Wire ``w`` is therefore mapped to qubit ``num_qubits - 1 - w``: with this
    choice Qiskit statevectors, probabilities and measured bitstrings use the
    same ordering as the rest of the package, without any reindexing.

    Args:
        gates: gate list, e.g. from :func:`quantum_viterbi.grover.qva_gates`.
        num_qubits: circuit width.
        omega: float value or Qiskit :class:`~qiskit.circuit.Parameter`.
    """
    circuit = QuantumCircuit(num_qubits)
    for gate in gates:
        qubits = [num_qubits - 1 - wire for wire in gate.wires]
        match gate.name:
            case "h":
                circuit.h(qubits[0])
            case "x":
                circuit.x(qubits[0])
            case "p":
                circuit.p(gate.angle(omega), qubits[0])
            case "cx":
                circuit.cx(qubits[0], qubits[1], ctrl_state=gate.ctrl_state)
            case "cp":
                circuit.cp(gate.angle(omega), qubits[0], qubits[1], ctrl_state=gate.ctrl_state)
            case "mcz":
                if len(qubits) == 1:
                    circuit.z(qubits[0])
                else:
                    circuit.mcp(math.pi, qubits[:-1], qubits[-1])
            case "gphase":
                circuit.global_phase += gate.angle(omega)
            case _:
                raise ValueError(f"Unsupported gate: {gate.name!r}")
    return circuit


class QiskitBackend(Backend):
    """Run the QVA with Qiskit.

    Exact results use :class:`qiskit.quantum_info.Statevector`. Shot-based
    results use a Sampler V2 primitive: the local
    :class:`qiskit.primitives.StatevectorSampler` by default, or any compatible
    sampler, e.g. ``qiskit_aer.primitives.SamplerV2`` for noisy simulations or
    ``qiskit_ibm_runtime.SamplerV2`` for IBM Quantum hardware.

    Args:
        sampler: Sampler V2 primitive used when ``shots`` is given (``seed`` is then
            ignored: seed the sampler itself).
        pass_manager: transpiler pass manager applied before sampling; required
            by hardware samplers, e.g.
            ``generate_preset_pass_manager(backend=device, optimization_level=1)``.
    """

    name = "qiskit"

    def __init__(self, sampler: Any = None, pass_manager: Any = None) -> None:
        self.sampler = sampler
        self.pass_manager = pass_manager
        #: Symbolic parameter used by circuits built without a value of omega.
        self.omega = Parameter("omega")

    def build_circuit(
        self,
        code: CodeLike,
        omega: float | None = None,
        iterations: int | None = None,
        measure: bool = False,
    ) -> QuantumCircuit:
        """Build the full QVA circuit.

        Args:
            code: received word.
            omega: value of omega; if ``None`` the circuit keeps the symbolic
                parameter :attr:`omega`, to be bound later with
                ``circuit.assign_parameters``.
            iterations: number of Grover iterations (default ``floor(pi/4 * sqrt(2**N))``).
            measure: append a measurement of all qubits.
        """
        pairs = parse_code(code)
        value = self.omega if omega is None else float(omega)
        circuit = gates_to_circuit(qva_gates(pairs, iterations), len(pairs), value)
        circuit.name = "QVA"
        if measure:
            circuit.measure_all()
        return circuit

    def _statevector(self, pairs: tuple[Pair, ...], omega: float, iterations: int) -> np.ndarray:
        return Statevector(self.build_circuit(pairs, omega, iterations)).data

    def _sample(
        self,
        pairs: tuple[Pair, ...],
        omega: float,
        iterations: int,
        shots: int,
        seed: int | None,
    ) -> np.ndarray:
        circuit = self.build_circuit(pairs, omega, iterations, measure=True)
        if self.pass_manager is not None:
            circuit = self.pass_manager.run(circuit)
        sampler = self.sampler if self.sampler is not None else StatevectorSampler(seed=seed)
        result = sampler.run([circuit], shots=shots).result()[0]
        return counts_to_probabilities(result.data.meas.get_counts(), len(pairs))

    def probabilities_batch(
        self, code: CodeLike, omegas: Iterable[float], iterations: int | None = None
    ) -> np.ndarray:
        # Build the parametric circuit once and only rebind omega
        pairs = parse_code(code)
        template = self.build_circuit(pairs, None, iterations)
        rows = [
            Statevector(template.assign_parameters({self.omega: float(w)}, strict=False)).probabilities()
            for w in as_omega_grid(omegas)
        ]
        return np.reshape(rows, (len(rows), 2 ** len(pairs)))
