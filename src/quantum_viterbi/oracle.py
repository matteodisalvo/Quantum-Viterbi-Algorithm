"""Phase oracle ``G(omega)`` of the Quantum Viterbi Algorithm.

The oracle marks every candidate path ``u = (u_1, ..., u_N)`` with a phase that
grows with its Hamming distance from the received word::

    G(omega) |u> = exp(1j * omega * d_H(u)) |u>

Each mismatching code bit contributes ``omega``, so the most likely path
(minimum distance) collects the smallest phase ``omega * d_min``, before the
reduction modulo ``2*pi``. The oracle is built one trellis transition at a time
with the phase blocks of the trellis gates ``V_k`` (``t`` is the 1-based
transition index):

====================  =================================  ===============
Python function       block                              wires
====================  =================================  ===============
:func:`vi0_gates`     ``V_i0``, first transition         ``0``
:func:`vi1_gates`     ``V_i1``, second transition        ``0, 1``
:func:`v_gates`       ``V_k``, every further transition  ``t-3, t-2, t-1``
:func:`oracle_gates`  complete oracle ``G(omega)``       all
====================  =================================  ===============

Every block is diagonal in the computational basis, so the blocks commute and
their order does not change the resulting unitary.
"""

from __future__ import annotations

from collections.abc import Sequence

from .convolutional import CodeLike, as_pair, parse_code
from .gates import Gate, cphase, cx, global_phase, phase, x


def _phase_on_state(wire: int, state: int) -> list[Gate]:
    """Phase ``exp(1j*omega)`` applied only when ``wire`` is in ``|state>``.

    This is the "controlled global phase"
    ``exp(1j*omega) |state><state| + |1-state><1-state|`` on ``wire``: a
    global phase controlled by a qubit is simply a phase gate on the control
    qubit, conjugated by X when the active state is ``|0>``.
    """
    if state == 1:
        return [phase(wire, omega_factor=1.0)]
    return [x(wire), phase(wire, omega_factor=1.0), x(wire)]


def vi0_gates(pair: Sequence[int], wire: int = 0) -> list[Gate]:
    """First transition, gate ``V_i0``.

    From the all-zero register the encoder emits ``00`` for ``u_1 = 0`` and
    ``11`` for ``u_1 = 1``.
    """
    r1, r2 = as_pair(pair)
    if r1 != r2:
        # Received 01 or 10: both hypotheses are at distance 1 -> global phase.
        return [global_phase(omega_factor=1.0)]
    if r1 == 0:
        # Received 00: only u_1 = 1 mismatches (two bits) -> P(2 omega) = diag(1, e^{2i omega}).
        return [phase(wire, omega_factor=2.0)]
    # Received 11: only u_1 = 0 mismatches -> X P(2 omega) X = diag(e^{2i omega}, 1).
    return [x(wire), phase(wire, omega_factor=2.0), x(wire)]


def vi1_gates(pair: Sequence[int], wires: tuple[int, int] = (0, 1)) -> list[Gate]:
    """Second transition, gate ``V_i1``, acting on ``(u_1, u_2)``.

    The emitted pair is ``(u_2, u_2 XOR u_1)``. The block is built as
    ``Vi1 = Xc * Rec * Rzc * Xc`` where

    * ``Rzc`` is the controlled phase gate ``diag(1, e^{2i omega})`` on ``u_2``, controlled by ``u_1``,
      active on ``|r1 XOR r2>``;
    * ``Rec`` is the global phase ``e^{i omega}`` controlled by the
      complementary state of ``u_1``;
    * ``Xc`` (only for received pairs ``10`` and ``11``) is a CNOT
      ``u_1 -> u_2`` active on ``|1 - r2>``.
    """
    control, target = wires
    r1, r2 = as_pair(pair)
    rz_state = r1 ^ r2
    gates = [
        cphase(control, target, omega_factor=2.0, ctrl_state=rz_state),  # Rzc
        *_phase_on_state(control, 1 - rz_state),  # Rec
    ]
    if r1 == 1:
        flip = cx(control, target, ctrl_state=1 - r2)  # Xc
        gates = [flip, *gates, flip]
    return gates


def v_gates(pair: Sequence[int], wires: tuple[int, int, int]) -> list[Gate]:
    """Generic transition ``t >= 3``, gate ``V_k`` with ``k = t - 1``.

    Acts on ``(u_{t-2}, u_{t-1}, u_t)``; the emitted pair is
    ``(u_t XOR u_{t-2}, u_t XOR u_{t-1} XOR u_{t-2})``. The steps are:

    1. CNOT ``u_{t-2} -> u_t``, active on ``|1>`` for received ``0x`` and on
       ``|0>`` for ``1x``: afterwards qubit ``t`` flags a mismatch of the first
       output bit;
    2. ``diag(1, e^{2i omega})`` on qubit ``t`` controlled by ``u_{t-1}``, active
       on ``|r1 XOR r2>``: in this branch the two output bits are either both
       correct or both wrong, so a flagged mismatch costs ``2 omega``;
    3. global phase ``e^{i omega}`` controlled by the complementary state of
       ``u_{t-1}``: in this branch exactly one output bit mismatches;
    4. the CNOT of step 1 again, restoring qubit ``t``.
    """
    a, b, c = wires
    r1, r2 = as_pair(pair)
    flip = cx(a, c, ctrl_state=1 - r1)  # steps 1 and 4
    rz_state = r1 ^ r2
    return [
        flip,
        cphase(b, c, omega_factor=2.0, ctrl_state=rz_state),  # step 2
        *_phase_on_state(b, 1 - rz_state),  # step 3
        flip,
    ]


def oracle_gates(code: CodeLike) -> list[Gate]:
    """Complete phase oracle ``G(omega)`` for a received word."""
    pairs = parse_code(code)
    gates = vi0_gates(pairs[0], wire=0)
    if len(pairs) > 1:
        gates += vi1_gates(pairs[1], wires=(0, 1))
    for t in range(2, len(pairs)):
        gates += v_gates(pairs[t], wires=(t - 2, t - 1, t))
    return gates
