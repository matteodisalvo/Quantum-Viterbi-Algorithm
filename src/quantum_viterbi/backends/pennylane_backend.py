"""PennyLane backend: the QVA as a PennyLane QNode."""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
import pennylane as qml

from ..convolutional import CodeLike, Pair, parse_code
from ..gates import Gate
from ..grover import qva_gates
from ..spectrum import as_omega_grid
from .base import Backend

# Recent PennyLane releases take ``shots`` in the QNode (device-level shots are
# deprecated), while older releases only accept them in the device.
_QNODE_ACCEPTS_SHOTS = "shots" in inspect.signature(qml.QNode.__init__).parameters


def apply_gates(gates: Iterable[Gate], omega: Any) -> None:
    """Queue a framework-independent gate list inside a PennyLane quantum function.

    PennyLane orders wires big-endian (wire 0 is the most significant bit),
    which already matches this package, so wires are used unchanged.
    """
    for gate in gates:
        wires = gate.wires
        match gate.name:
            case "h":
                qml.Hadamard(wires=wires[0])
            case "x":
                qml.PauliX(wires=wires[0])
            case "p":
                qml.PhaseShift(gate.angle(omega), wires=wires[0])
            case "cx":
                qml.ctrl(
                    qml.PauliX(wires=wires[1]),
                    control=wires[0],
                    control_values=[bool(gate.ctrl_state)],
                )
            case "cp":
                qml.ctrl(
                    qml.PhaseShift(gate.angle(omega), wires=wires[1]),
                    control=wires[0],
                    control_values=[bool(gate.ctrl_state)],
                )
            case "mcz":
                if len(wires) == 1:
                    qml.PauliZ(wires=wires[0])
                else:
                    qml.ctrl(qml.PauliZ(wires=wires[-1]), control=list(wires[:-1]))
            case "gphase":
                # PennyLane convention: GlobalPhase(phi) = exp(-1j * phi)
                qml.GlobalPhase(-gate.angle(omega))
            case _:
                raise ValueError(f"Unsupported gate: {gate.name!r}")


class PennyLaneBackend(Backend):
    """Run the QVA with PennyLane.

    Args:
        device: PennyLane device name, e.g. ``"default.qubit"`` (default),
            ``"lightning.qubit"`` or a hardware plugin device.
        **device_kwargs: extra keyword arguments forwarded to :func:`pennylane.device`.
    """

    name = "pennylane"

    def __init__(self, device: str = "default.qubit", **device_kwargs: Any) -> None:
        self.device_name = device
        self.device_kwargs = device_kwargs

    def build_qnode(
        self,
        code: CodeLike,
        iterations: int | None = None,
        measurement: Literal["probs", "state"] = "probs",
        shots: int | None = None,
        seed: int | None = None,
    ) -> qml.QNode:
        """Build a QNode ``circuit(omega)`` returning ``qml.probs`` or ``qml.state``.

        Example::

            qnode = PennyLaneBackend().build_qnode("11 01 00 10")
            print(qml.draw(qnode)(0.6))
        """
        if measurement not in ("probs", "state"):
            raise ValueError(f"measurement must be 'probs' or 'state', got {measurement!r}.")
        pairs = parse_code(code)
        gates = qva_gates(pairs, iterations)
        wires = list(range(len(pairs)))

        options = dict(self.device_kwargs)
        if seed is not None:
            options["seed"] = seed
        if shots is not None and not _QNODE_ACCEPTS_SHOTS:
            options["shots"] = shots
        device = qml.device(self.device_name, wires=wires, **options)

        def circuit(omega):
            apply_gates(gates, omega)
            return qml.state() if measurement == "state" else qml.probs(wires=wires)

        if shots is None or not _QNODE_ACCEPTS_SHOTS:
            return qml.QNode(circuit, device)
        return qml.QNode(circuit, device, shots=shots)

    def _statevector(self, pairs: tuple[Pair, ...], omega: float, iterations: int) -> np.ndarray:
        return np.asarray(self.build_qnode(pairs, iterations, "state")(omega))

    def _sample(
        self,
        pairs: tuple[Pair, ...],
        omega: float,
        iterations: int,
        shots: int,
        seed: int | None,
    ) -> np.ndarray:
        return np.asarray(self.build_qnode(pairs, iterations, "probs", shots, seed)(omega))

    def probabilities_batch(
        self, code: CodeLike, omegas: Iterable[float], iterations: int | None = None
    ) -> np.ndarray:
        # Build the QNode once and evaluate it for every omega
        pairs = parse_code(code)
        qnode = self.build_qnode(pairs, iterations, "probs")
        rows = [np.asarray(qnode(float(w))) for w in as_omega_grid(omegas)]
        return np.reshape(rows, (len(rows), 2 ** len(pairs)))
