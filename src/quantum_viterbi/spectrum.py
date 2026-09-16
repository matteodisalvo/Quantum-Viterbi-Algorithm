"""Distance spectrum of a received word and exact QVA simulation on distance classes.

The initial state gives every path the same amplitude, the oracle multiplies the
amplitude of path ``u`` by ``exp(1j * omega * d(u))`` and the diffusion
``D = 2|s><s| - I`` only involves the mean amplitude. Hence, after any number of
Grover iterations, all paths at the same Hamming distance ``d`` share one amplitude
and the QVA reduces exactly to the ``2N + 1`` distance classes::

    b_d <- exp(1j * omega * d) * b_d                  (oracle G)
    b_d <- 2 * sum_d' w_d' * b_d' - b_d               (diffusion D)

with ``b_d = 1`` in ``|s>``, ``w_d = A_d / 2**N`` and ``A_d`` the number of paths at
distance ``d`` (the distance spectrum). A path at distance ``d`` has amplitude
``b_d / sqrt(2**N)``, and the probability of measuring some path at distance ``d``
is ``w_d * |b_d|**2``.

The spectrum is computed by a forward pass over the four trellis states in
``O(N**2)`` operations, so one omega costs ``O(K * N)`` instead of ``O(K * 2**N)``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from .convolutional import CodeLike, Pair, _branch_output, parse_code
from .grover import resolve_iterations


def _trellis() -> tuple[tuple[int, int, tuple[int, ...]], ...]:
    """Branches ``(state, next_state, output_pair)`` with ``state = 2 * u_{t-1} + u_{t-2}``."""
    branches = []
    for state in range(4):
        history = (state >> 1, state & 1)  # (u_{t-1}, u_{t-2})
        for bit in (0, 1):
            branches.append((state, (bit << 1) | history[0], _branch_output(bit, history)))
    return tuple(branches)


_TRELLIS = _trellis()


def _spectrum_row(pairs: tuple[Pair, ...], exact: bool) -> np.ndarray:
    """Forward pass keeping, per trellis state, the number of paths at each accumulated metric.

    With ``exact=True`` the counts are Python integers; otherwise every step is scaled
    by 1/2, so the float64 entries are the fractions ``A_d / 2**N`` and never overflow.
    """
    n = len(pairs)
    table = np.zeros((4, 2 * n + 1), dtype=object if exact else float)
    table[0, 0] = 1  # the register starts from 00
    for t, received in enumerate(pairs):
        reach = 2 * t + 1  # metrics 0..2t can occur before step t
        new = np.zeros_like(table)
        for state, next_state, output in _TRELLIS:
            delta = int(output[0] != received[0]) + int(output[1] != received[1])
            new[next_state, delta : delta + reach] += table[state, :reach]
        table = new if exact else 0.5 * new
    return table.sum(axis=0)


def distance_spectrum(code: CodeLike) -> tuple[int, ...]:
    """Number of paths ``A_d`` at Hamming distance ``d = 0, ..., 2N`` from the received word.

    Equals ``np.bincount(path_metrics(code), minlength=2*N + 1)`` but costs ``O(N**2)``
    instead of ``O(N * 2**N)`` and is exact for any ``N`` (Python integers). The first
    nonzero entry sits at the Viterbi metric ``d_min``.

    Example:
        >>> distance_spectrum("11 01 00 10 00 00")
        (0, 1, 2, 4, 6, 9, 14, 15, 10, 2, 0, 1, 0)
    """
    return tuple(int(count) for count in _spectrum_row(parse_code(code), exact=True))


def distance_weights(code: CodeLike) -> np.ndarray:
    """Fraction of paths at each distance, ``w_d = A_d / 2**N``, as a float64 array of size ``2N + 1``."""
    return np.asarray(_spectrum_row(parse_code(code), exact=False), dtype=float)


def as_omega_grid(omegas: float | Iterable[float]) -> np.ndarray:
    """Omega values as a float array of at least one dimension (generators are materialized)."""
    if isinstance(omegas, Iterable) and not isinstance(omegas, (np.ndarray, Sequence)):
        omegas = list(omegas)
    return np.atleast_1d(np.asarray(omegas, dtype=float))


def class_probabilities(
    code: CodeLike, omegas: float | Iterable[float], iterations: int | None = None
) -> np.ndarray:
    """Exact probability of measuring some path at each Hamming distance.

    Args:
        code: received word.
        omegas: omega values in radians.
        iterations: number of Grover iterations (default ``floor(pi/4 * sqrt(2**N))``).

    Returns:
        Array of shape ``(len(omegas), 2N + 1)`` whose rows sum to 1. Column ``d_min``
        is the probability that the QVA returns a maximum-likelihood path; one given
        path ``u`` is measured with probability ``P[:, d(u)] / A_{d(u)}``.
    """
    pairs = parse_code(code)
    rounds = resolve_iterations(len(pairs), iterations)
    weights = distance_weights(pairs)
    grid = as_omega_grid(omegas)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("omegas must be a non-empty one-dimensional sequence.")
    phases = np.exp(1j * np.outer(grid, np.arange(weights.size)))
    amplitudes = np.ones(phases.shape, dtype=complex)  # sqrt(2**N) times the amplitude in |s>
    for _ in range(rounds):
        amplitudes *= phases  # G(omega)
        amplitudes = 2 * (amplitudes @ weights)[:, None] - amplitudes  # D = 2|s><s| - I
    return weights * np.abs(amplitudes) ** 2
