"""Abstract interface shared by all QVA backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping

import numpy as np

from ..convolutional import CodeLike, Pair, parse_code
from ..grover import resolve_iterations
from ..spectrum import as_omega_grid


class Backend(ABC):
    """Run the QVA circuit for a received word and return output distributions.

    Subclasses implement :meth:`_statevector` (exact simulation) and
    :meth:`_sample` (finite number of shots). Public methods accept any code
    format understood by :func:`~quantum_viterbi.convolutional.parse_code` and
    use ``floor(pi/4 * sqrt(2**N))`` Grover iterations when ``iterations`` is
    ``None``.

    State indices are big-endian: the first information bit is the most
    significant bit.
    """

    #: Identifier used by :func:`~quantum_viterbi.backends.get_backend`.
    name: str = "base"

    def statevector(self, code: CodeLike, omega: float, iterations: int | None = None) -> np.ndarray:
        """Exact final state ``(D G)^K H^N |0...0>``, a complex vector of length ``2**N``."""
        pairs = parse_code(code)
        rounds = resolve_iterations(len(pairs), iterations)
        return self._statevector(pairs, float(omega), rounds)

    def probabilities(
        self,
        code: CodeLike,
        omega: float,
        iterations: int | None = None,
        shots: int | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """Output distribution over the ``2**N`` trellis paths.

        Args:
            code: received word.
            omega: metric parameter of the phase oracle (radians).
            iterations: number of Grover iterations.
            shots: ``None`` for exact probabilities, otherwise the number of
                measurement shots used to estimate them.
            seed: random seed used when sampling.
        """
        pairs = parse_code(code)
        rounds = resolve_iterations(len(pairs), iterations)
        if shots is None:
            return np.abs(self._statevector(pairs, float(omega), rounds)) ** 2
        if int(shots) != shots or shots < 1:  # not int(shots) < 1: that accepts 2.7 as 2 shots
            raise ValueError(f"shots must be a positive integer, got {shots!r}.")
        return self._sample(pairs, float(omega), rounds, int(shots), seed)

    def probabilities_batch(
        self, code: CodeLike, omegas: Iterable[float], iterations: int | None = None
    ) -> np.ndarray:
        """Exact distributions for several omega values, shape ``(len(omegas), 2**N)``."""
        pairs = parse_code(code)
        rounds = resolve_iterations(len(pairs), iterations)
        rows = [np.abs(self._statevector(pairs, float(w), rounds)) ** 2 for w in as_omega_grid(omegas)]
        return np.reshape(rows, (len(rows), 2 ** len(pairs)))

    @abstractmethod
    def _statevector(self, pairs: tuple[Pair, ...], omega: float, iterations: int) -> np.ndarray:
        """Exact simulation; inputs are already validated."""

    @abstractmethod
    def _sample(
        self,
        pairs: tuple[Pair, ...],
        omega: float,
        iterations: int,
        shots: int,
        seed: int | None,
    ) -> np.ndarray:
        """Shot-based estimate of the output distribution; inputs are already validated."""


def counts_to_probabilities(counts: Mapping[str, int], num_qubits: int) -> np.ndarray:
    """Convert measurement counts such as ``{"0110": 12, ...}`` (MSB first) into frequencies."""
    probabilities = np.zeros(2**num_qubits)
    for bitstring, count in counts.items():
        probabilities[int(bitstring.replace(" ", ""), 2)] += count
    return probabilities / probabilities.sum()
