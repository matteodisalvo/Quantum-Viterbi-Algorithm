"""High-level decoding interface."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .backends import DEFAULT_BACKEND, Backend, get_backend
from .convolutional import (
    CodeLike,
    Pair,
    StateLike,
    ViterbiResult,
    bits_to_string,
    encode,
    flatten_code,
    hamming_distance,
    index_to_bits,
    parse_code,
    report_lines,
    viterbi_decode,
)
from .grover import resolve_iterations
from .tuning import tuned_omega


@dataclass(frozen=True)
class DecodingResult:
    """Outcome of :func:`decode`.

    Attributes:
        code: received word as ``(bit, bit)`` pairs.
        omega: metric parameter used by the phase oracle.
        iterations: number of Grover iterations.
        backend: name of the backend that produced the result.
        probabilities: probability (or, with ``shots``, relative frequency) of each of
            the ``2**N`` paths, indexed MSB first.
        shots: number of measurement shots, or ``None`` for exact probabilities.
        tuned: whether omega was chosen by :func:`~quantum_viterbi.tuning.tuned_omega`
            instead of being supplied by the caller.
    """

    code: tuple[Pair, ...]
    omega: float
    iterations: int
    backend: str
    probabilities: np.ndarray = field(repr=False)
    shots: int | None = None
    tuned: bool = False

    @property
    def num_qubits(self) -> int:
        return len(self.code)

    @property
    def best_state(self) -> int:
        """Index of the most probable path."""
        return int(np.argmax(self.probabilities))

    @property
    def best_probability(self) -> float:
        return float(self.probabilities[self.best_state])

    @property
    def decoded_bits(self) -> tuple[int, ...]:
        """Information bits of the most probable path."""
        return index_to_bits(self.best_state, self.num_qubits)

    @property
    def reencoded(self) -> tuple[int, ...]:
        """Codeword emitted by the encoder for :attr:`decoded_bits`."""
        return encode(self.decoded_bits)

    @property
    def hamming_distance(self) -> int:
        """Hamming distance between :attr:`reencoded` and the received word."""
        return hamming_distance(self.reencoded, flatten_code(self.code))

    @property
    def classical(self) -> ViterbiResult:
        """Classical Viterbi decoding of the same received word, for comparison."""
        return viterbi_decode(self.code)

    def top_states(self, k: int = 10) -> list[tuple[str, float]]:
        """The ``k`` most probable paths as ``(bitstring, probability)``.

        Raises:
            ValueError: if ``k`` is not a positive integer. A negative ``k`` would not
                mean "``k`` paths" but "all paths except the last ``|k|``".
        """
        if int(k) != k or k < 1:
            raise ValueError(f"k must be a positive integer, got {k!r}.")
        order = np.argsort(-self.probabilities, kind="stable")[:k]
        return [
            (bits_to_string(index_to_bits(int(i), self.num_qubits)), float(self.probabilities[i]))
            for i in order
        ]

    def summary(self, original: StateLike | None = None) -> str:
        """Multi-line human-readable report.

        Args:
            original: transmitted message (bitstring, bits or index), if known. It adds
                the original codeword, the number of channel errors and the outcome.
        """
        mode = "exact" if self.shots is None else f"{self.shots} shots"
        classical = self.classical
        agreement = "same as QVA" if classical.bits == self.decoded_bits else "differs from QVA"
        lines = report_lines(
            self.code,
            self.decoded_bits,
            original,
            decoder_lines=[
                f"  qubits (N)        : {self.num_qubits}",
                f"  omega             : {self.omega:.4f} rad{' (tuned)' if self.tuned else ''}",
                f"  Grover iterations : {self.iterations}",
                f"  backend           : {self.backend} ({mode})",
            ],
            decoded_note=f"  (P = {self.best_probability:.4f})",
            comparison_lines=[f"  classical Viterbi : {bits_to_string(classical.bits)}  ({agreement})"],
        )
        return "\n".join(["Quantum Viterbi decoding", *lines])


def decode(
    code: CodeLike,
    omega: float | None = None,
    iterations: int | None = None,
    backend: str | Backend = DEFAULT_BACKEND,
    shots: int | None = None,
    seed: int | None = None,
) -> DecodingResult:
    """Decode a received word with the Quantum Viterbi Algorithm.

    Args:
        code: received word (see :func:`~quantum_viterbi.convolutional.parse_code`).
        omega: metric parameter in radians. By default it is tuned for this received
            word and this number of iterations with
            :func:`~quantum_viterbi.tuning.tuned_omega`.
        iterations: Grover iterations; defaults to ``floor(pi/4 * sqrt(2**N))``.
        backend: ``"qiskit"`` (default), ``"pennylane"``, ``"numpy"`` or a
            configured :class:`~quantum_viterbi.backends.Backend`.
        shots: ``None`` for exact probabilities, otherwise number of measurement shots.
        seed: random seed used when sampling.

    Example::

        result = decode("11 01 00 10 00 00", backend="pennylane")
        print(result.summary(original="101101"))
    """
    pairs = parse_code(code)
    rounds = resolve_iterations(len(pairs), iterations)
    tuned = omega is None
    omega = tuned_omega(pairs, rounds) if tuned else float(omega)
    engine = get_backend(backend)
    probabilities = engine.probabilities(pairs, omega, rounds, shots=shots, seed=seed)
    return DecodingResult(pairs, omega, rounds, engine.name, probabilities, shots, tuned)
