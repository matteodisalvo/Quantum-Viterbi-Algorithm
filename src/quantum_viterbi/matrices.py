"""Reference implementation with explicit matrices.

Every operator of the QVA is built as an explicit matrix with Kronecker products
(phase blocks on one to three qubits, complete operators of size ``2**N x 2**N``):

=========================================  ========================
Operator                                   Function
=========================================  ========================
``V_i0`` phase block (first transition)    :func:`build_vi0`
``V_i1`` phase block (second transition)   :func:`build_vi1`
``V_k`` phase block (further transitions)  :func:`build_v`
phase oracle ``G(omega)``                  :func:`build_g`
Walsh-Hadamard transform ``H^N``           :func:`hadamard_n`
diffusion operator ``D``                   :func:`diffusion_matrix`
complete Grover loop                       :func:`qva_statevector`
=========================================  ========================

Memory grows as ``4**N``, so this module is meant as a readable reference and as
ground truth for the test suite (``N`` up to about 10). The backends in
:mod:`quantum_viterbi.backends` give identical results much faster.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import reduce

import numpy as np

from .convolutional import CodeLike, as_pair, parse_code
from .grover import resolve_iterations

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
P0 = np.array([[1, 0], [0, 0]], dtype=complex)  # |0><0|
P1 = np.array([[0, 0], [0, 1]], dtype=complex)  # |1><1|


def kron(*matrices: np.ndarray) -> np.ndarray:
    """Kronecker product of any number of matrices."""
    return reduce(np.kron, matrices, np.eye(1, dtype=complex))


def _projectors(state: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(|state><state|, complementary projector)``."""
    return (P1, P0) if state == 1 else (P0, P1)


def build_vi0(omega: float, pair: Sequence[int]) -> np.ndarray:
    """Phase block of ``V_i0``: first initialization step, one qubit."""
    p = np.exp(1j * omega)  # phase shift parameter
    rz = np.diag([1, p**2])  # phase rotation gate
    re = p * I2  # global phase
    pair = as_pair(pair)
    if pair == (0, 0):
        return rz
    if pair in ((0, 1), (1, 0)):
        return re
    return X @ rz @ X


def build_vi1(omega: float, pair: Sequence[int]) -> np.ndarray:
    """Phase block of ``V_i1``: second initialization step, two qubits."""
    p = np.exp(1j * omega)
    rz = np.diag([1, p**2])
    re = p * I2
    pair = as_pair(pair)
    if pair == (0, 0):
        rzc = kron(P0, rz) + kron(P1, I2)
        rec = kron(P0, I2) + kron(P1, re)
        return rec @ rzc
    if pair == (0, 1):
        rzc = kron(P0, I2) + kron(P1, rz)
        rec = kron(P0, re) + kron(P1, I2)
        return rec @ rzc
    if pair == (1, 0):
        rzc = kron(P0, I2) + kron(P1, rz)
        rec = kron(P0, re) + kron(P1, I2)
        xc = kron(P0, I2) + kron(P1, X)
        return xc @ rec @ rzc @ xc
    rzc = kron(P0, rz) + kron(P1, I2)
    rec = kron(P0, I2) + kron(P1, re)
    xc = kron(P0, X) + kron(P1, I2)
    return xc @ rec @ rzc @ xc


def build_v(omega: float, pair: Sequence[int]) -> np.ndarray:
    """Phase block of ``V_k``: generic trellis step, three qubits.

    The four received pairs (00, 01, 10, 11) only differ in the control states
    that activate each gate, so a single function covers all of them.
    """
    p = np.exp(1j * omega)
    rz = np.diag([1, p**2])
    ge = p * I2
    r1, r2 = as_pair(pair)

    # Steps 1 and 4: CNOT qubit 1 -> qubit 3, active on |1> (received 0x) or |0> (received 1x)
    active, idle = _projectors(1 - r1)
    cnot = kron(idle, I2, I2) + kron(active, I2, X)

    # Step 2: phase gate on qubit 3 controlled by qubit 2, active on |0> (received 00, 11) or |1> (01, 10)
    active, idle = _projectors(r1 ^ r2)
    gs3 = kron(I2, kron(active, rz) + kron(idle, I2))

    # Step 3: global phase exp(1j*omega) controlled by the complementary state of qubit 2
    gs4 = kron(I2, kron(active, I2) + kron(idle, ge))

    return cnot @ gs4 @ gs3 @ cnot


def build_g(omega: float, code: CodeLike) -> np.ndarray:
    """Full phase oracle ``G(omega)`` for codes of any length ``N >= 1``."""
    pairs = parse_code(code)
    n = len(pairs)

    # Initialization steps on the first one and two qubits
    g = kron(build_vi0(omega, pairs[0]), *[I2] * (n - 1))
    if n > 1:
        g = g @ kron(build_vi1(omega, pairs[1]), *[I2] * (n - 2))

    # Trellis: iteration k applies block V_{k+2} (transition t = k+3) to qubits (k, k+1, k+2)
    trellis = np.eye(2**n, dtype=complex)
    for k in range(n - 2):
        step = kron(*[I2] * k, build_v(omega, pairs[k + 2]), *[I2] * (n - k - 3))
        trellis = step @ trellis
    return g @ trellis


def hadamard_n(num_qubits: int) -> np.ndarray:
    """Walsh-Hadamard transform ``H^N``."""
    return kron(*[H] * num_qubits)


def diffusion_matrix(num_qubits: int) -> np.ndarray:
    """Diffusion operator ``D = 2|s><s| - I``."""
    dim = 2**num_qubits
    u = np.full((dim, 1), 1 / np.sqrt(dim))
    return 2 * (u @ u.T) - np.eye(dim)


def qva_statevector(code: CodeLike, omega: float, iterations: int | None = None) -> np.ndarray:
    """Final QVA state ``(D G)^K H^N |0...0>`` after ``iterations`` Grover rounds."""
    pairs = parse_code(code)
    n = len(pairs)
    rounds = resolve_iterations(n, iterations)
    g = build_g(omega, pairs)
    d = diffusion_matrix(n)
    phi = hadamard_n(n)[:, 0]  # H^N |0...0>
    for _ in range(rounds):
        phi = d @ (g @ phi)
    return phi
