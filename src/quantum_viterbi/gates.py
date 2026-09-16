"""Framework-independent description of the quantum gates used by the QVA.

The whole algorithm is expressed as a flat list of :class:`Gate` objects. Each
circuit backend (Qiskit, PennyLane) only has to translate these few primitives, so
the circuit decomposition lives in a single place and every framework runs
exactly the same circuit.

Wire convention
---------------
Wire ``k`` carries the information bit ``u_{k+1}`` of a trellis path. Wire 0 is
the *most significant* bit of a basis-state index (big-endian ordering).

Angles
------
Every rotation angle is an affine function of the metric parameter omega::

    angle = omega_factor * omega + offset

so the same gate list can be evaluated with a float, a Qiskit ``Parameter`` or a
PennyLane argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

GateName = Literal["h", "x", "p", "cx", "cp", "mcz", "gphase"]


@dataclass(frozen=True)
class Gate:
    """A single gate of the QVA circuit.

    Attributes:
        name: gate identifier

            * ``"h"``: Hadamard
            * ``"x"``: Pauli-X
            * ``"p"``: phase gate ``diag(1, exp(1j*angle))``
            * ``"cx"``: controlled NOT, ``wires = (control, target)``
            * ``"cp"``: controlled phase gate, ``wires = (control, target)``
            * ``"mcz"``: Z on the last wire controlled by all the others
            * ``"gphase"``: global phase ``exp(1j*angle)`` (no wires)
        wires: wires the gate acts on.
        omega_factor: coefficient of omega in the rotation angle.
        offset: constant part of the rotation angle (radians).
        ctrl_state: control state that activates a controlled gate
            (1 = standard control, 0 = anti-control).
    """

    name: GateName
    wires: tuple[int, ...] = ()
    omega_factor: float = 0.0
    offset: float = 0.0
    ctrl_state: int = 1

    def __post_init__(self) -> None:
        if self.ctrl_state not in (0, 1):
            raise ValueError(f"ctrl_state must be 0 or 1, got {self.ctrl_state!r}.")

    def angle(self, omega: Any) -> Any:
        """Evaluate the rotation angle for a numeric or symbolic ``omega``."""
        if self.omega_factor == 0:
            return self.offset
        value = self.omega_factor * omega
        return value + self.offset if self.offset else value


def h(wire: int) -> Gate:
    """Hadamard gate."""
    return Gate("h", (wire,))


def x(wire: int) -> Gate:
    """Pauli-X (NOT) gate."""
    return Gate("x", (wire,))


def phase(wire: int, omega_factor: float = 0.0, offset: float = 0.0) -> Gate:
    """Phase gate ``diag(1, exp(1j*angle))``.

    ``phase(w, omega_factor=2)`` is the gate ``diag(1, exp(2j*omega))``
    that marks a double mismatch.
    """
    return Gate("p", (wire,), omega_factor, offset)


def cx(control: int, target: int, ctrl_state: int = 1) -> Gate:
    """(Anti-)controlled NOT gate."""
    return Gate("cx", (control, target), ctrl_state=ctrl_state)


def cphase(
    control: int,
    target: int,
    omega_factor: float = 0.0,
    offset: float = 0.0,
    ctrl_state: int = 1,
) -> Gate:
    """(Anti-)controlled phase gate."""
    return Gate("cp", (control, target), omega_factor, offset, ctrl_state)


def mcz(wires: tuple[int, ...]) -> Gate:
    """Multi-controlled Z: flips the sign of the all-ones state of ``wires``."""
    return Gate("mcz", tuple(wires))


def global_phase(omega_factor: float = 0.0, offset: float = 0.0) -> Gate:
    """Global phase ``exp(1j*angle)``."""
    return Gate("gphase", (), omega_factor, offset)
