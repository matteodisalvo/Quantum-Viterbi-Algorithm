"""Grover part of the QVA: initial superposition, diffusion operator and full circuit.

One QVA iteration applies the oracle followed by the diffusion operator::

    |phi_0> = H^N |0...0>
    |phi_k> = D G(omega) |phi_{k-1}>,      k = 1, ..., K

with ``D = 2|s><s| - I`` and ``|s>`` the uniform superposition over all
``2**N`` trellis paths.
"""

from __future__ import annotations

import math

from .convolutional import CodeLike, parse_code
from .gates import Gate, global_phase, h, mcz, x
from .oracle import oracle_gates


def default_iterations(num_qubits: int) -> int:
    """Default number of Grover iterations: ``floor(pi/4 * sqrt(2**N))``."""
    if num_qubits < 1:
        raise ValueError("At least one qubit is required.")
    return math.floor(math.pi / 4 * math.sqrt(2**num_qubits))


def resolve_iterations(num_qubits: int, iterations: int | None) -> int:
    """Validate ``iterations``, falling back to :func:`default_iterations` when ``None``."""
    if iterations is None:
        return default_iterations(num_qubits)
    if int(iterations) != iterations or iterations < 0:
        raise ValueError(f"iterations must be a non-negative integer, got {iterations!r}.")
    return int(iterations)


def superposition_gates(num_qubits: int) -> list[Gate]:
    """``H^N``: uniform superposition over all trellis paths."""
    return [h(wire) for wire in range(num_qubits)]


def diffusion_gates(num_qubits: int) -> list[Gate]:
    """Diffusion operator ``D = 2|s><s| - I``.

    ``H^N X^N (multi-controlled Z) X^N H^N`` equals ``I - 2|s><s|``; the extra
    global phase ``pi`` fixes the sign, so the unitary is exactly
    ``D = 2|s><s| - I``.
    """
    wires = tuple(range(num_qubits))
    return [
        *superposition_gates(num_qubits),
        *(x(wire) for wire in wires),
        mcz(wires),
        *(x(wire) for wire in wires),
        *superposition_gates(num_qubits),
        global_phase(offset=math.pi),
    ]


def qva_gates(code: CodeLike, iterations: int | None = None) -> list[Gate]:
    """Complete QVA circuit: ``H^N`` followed by ``iterations`` rounds of ``D G(omega)``."""
    pairs = parse_code(code)
    num_qubits = len(pairs)
    rounds = resolve_iterations(num_qubits, iterations)
    grover_iteration = oracle_gates(pairs) + diffusion_gates(num_qubits)
    return superposition_gates(num_qubits) + grover_iteration * rounds
