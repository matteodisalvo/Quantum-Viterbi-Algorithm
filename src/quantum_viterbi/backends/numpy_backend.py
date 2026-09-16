"""Fast exact simulator written in pure NumPy."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..convolutional import CodeLike, Pair, parse_code, path_metrics
from ..grover import resolve_iterations
from ..spectrum import as_omega_grid
from .base import Backend


class NumpyBackend(Backend):
    """Exact QVA simulation without building any circuit or matrix.

    The phase oracle is diagonal, ``G = diag(exp(1j*omega*d_H(u)))``, and the
    diffusion operator is ``D = 2|s><s| - I``, so one Grover iteration costs
    ``O(2**N)`` operations instead of the ``O(4**N)`` of a dense matrix-vector
    product. Results are numerically identical to
    :func:`quantum_viterbi.matrices.qva_statevector` (checked by the tests).

    Many omega values can be evaluated at once, which makes omega scans fast.
    """

    name = "numpy"

    #: Target size ``len(omegas) * 2**N`` of the arrays in one batch (~64 MB per complex array).
    max_batch_elements = 2**22

    def _statevector(self, pairs: tuple[Pair, ...], omega: float, iterations: int) -> np.ndarray:
        return self._evolve(path_metrics(pairs), np.array([omega]), iterations)[0]

    def _sample(
        self,
        pairs: tuple[Pair, ...],
        omega: float,
        iterations: int,
        shots: int,
        seed: int | None,
    ) -> np.ndarray:
        probabilities = np.abs(self._statevector(pairs, omega, iterations)) ** 2
        counts = np.random.default_rng(seed).multinomial(shots, probabilities / probabilities.sum())
        return counts / shots

    def probabilities_batch(
        self, code: CodeLike, omegas: Iterable[float], iterations: int | None = None
    ) -> np.ndarray:
        pairs = parse_code(code)
        rounds = resolve_iterations(len(pairs), iterations)
        metrics = path_metrics(pairs)
        grid = as_omega_grid(omegas)
        chunk = max(1, self.max_batch_elements // metrics.size)
        rows = [
            np.abs(self._evolve(metrics, grid[start : start + chunk], rounds)) ** 2
            for start in range(0, grid.size, chunk)
        ]
        return np.concatenate(rows) if rows else np.empty((0, metrics.size))

    @staticmethod
    def _evolve(metrics: np.ndarray, omegas: np.ndarray, iterations: int) -> np.ndarray:
        """Final states for every omega, shape ``(len(omegas), 2**N)``."""
        dim = metrics.size
        oracle = np.exp(1j * np.outer(omegas, metrics))  # row i: diagonal of G(omegas[i])
        state = np.full(oracle.shape, 1 / np.sqrt(dim), dtype=complex)  # H^N |0...0>
        for _ in range(iterations):
            state = state * oracle  # G
            state = 2 * state.mean(axis=1, keepdims=True) - state  # D = 2|s><s| - I
        return state
