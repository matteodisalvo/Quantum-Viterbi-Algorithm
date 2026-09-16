"""Classical side of the decoder: the rate-1/2 (5, 7) convolutional code.

The phase oracle of the decoder (:mod:`quantum_viterbi.oracle`) is built for the
standard constraint-length-3 convolutional code with octal generators (5, 7).
At trellis step ``t`` the information bit ``u_t`` and the register contents
``(u_{t-1}, u_{t-2})`` produce the output pair::

    c_t = (u_t XOR u_{t-2},  u_t XOR u_{t-1} XOR u_{t-2})

The register starts in the all-zero state and no tail bits are appended, so
``N`` information bits produce ``2N`` code bits and the decoder needs ``N``
qubits (one per trellis transition).

Bit-ordering convention (big-endian, as in ``numpy.kron``): the first information bit is
the most significant bit of a basis-state index, i.e. the path
``(u_1, ..., u_N)`` has index ``sum_k u_k * 2**(N - k)``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

#: Generator taps applied to ``(u_t, u_{t-1}, u_{t-2})``: octal 5 = 101, octal 7 = 111.
GENERATORS: tuple[tuple[int, int, int], ...] = ((1, 0, 1), (1, 1, 1))

#: One received (or emitted) pair of code bits.
Pair = tuple[int, int]

#: Any code format understood by :func:`parse_code`.
CodeLike = str | Sequence[int] | Sequence[Sequence[int]] | np.ndarray

#: Any basis-state format understood by :func:`resolve_state`.
StateLike = int | str | Sequence[int]

# Characters ignored when a code is given as a string
_SEPARATORS = set(",;[]()_")


def parse_code(code: CodeLike) -> tuple[Pair, ...]:
    """Normalize a received word into a tuple of ``(bit, bit)`` pairs.

    Accepted (equivalent) formats::

        [1, 1, 0, 1, 0, 0]          # flat list of bits
        [(1, 1), (0, 1), (0, 0)]    # explicit pairs
        "11 01 00"                  # string; separators such as spaces, commas or brackets are ignored

    Raises:
        ValueError: if the code is empty, has an odd length or is not binary.
    """
    if isinstance(code, str):
        symbols = [ch for ch in code if not ch.isspace() and ch not in _SEPARATORS]
        if any(ch not in "01" for ch in symbols):
            raise ValueError(f"Not a binary code: {code!r} (only 0 and 1 are allowed).")
        bits = [int(ch) for ch in symbols]
    else:
        array = np.asarray(code)
        if not np.all((array == 0) | (array == 1)):
            raise ValueError("Not a binary code: every element must be 0 or 1.")
        bits = [int(bit) for bit in array.ravel()]

    if not bits:
        raise ValueError("The received word is empty.")
    if len(bits) % 2:
        raise ValueError(f"A rate-1/2 code needs an even number of bits, got {len(bits)}.")
    return tuple((bits[i], bits[i + 1]) for i in range(0, len(bits), 2))


def as_pair(pair: Sequence[int]) -> Pair:
    """Validate a single pair of code bits and return it as a tuple."""
    bits = tuple(pair)  # check the values before coercing: int(0.5) would pass as a 0
    if len(bits) != 2 or any(bit not in (0, 1) for bit in bits):
        raise ValueError(f"Not a binary pair of code bits: {pair!r}.")
    return int(bits[0]), int(bits[1])


def flatten_code(pairs: Iterable[Sequence[int]]) -> tuple[int, ...]:
    """Flatten ``((a, b), (c, d), ...)`` into ``(a, b, c, d, ...)``."""
    return tuple(int(bit) for pair in pairs for bit in pair)


def format_code(code: CodeLike) -> str:
    """Human-readable representation of a code, e.g. ``"11 01 00"``."""
    return " ".join(f"{a}{b}" for a, b in parse_code(code))


def _branch_output(bit: int, history: tuple[int, int]) -> tuple[int, ...]:
    """Pair emitted for ``bit`` when the register holds ``history = (u_{t-1}, u_{t-2})``."""
    taps = (bit, *history)
    return tuple(sum(g * t for g, t in zip(generator, taps, strict=True)) % 2 for generator in GENERATORS)


def encode(bits: Iterable[int]) -> tuple[int, ...]:
    """Encode information bits with the (5, 7) code (zero initial state, no tail).

    Example:
        >>> encode([1, 0, 1, 1])
        (1, 1, 0, 1, 0, 0, 1, 0)
    """
    history = (0, 0)  # register contents (u_{t-1}, u_{t-2})
    output: list[int] = []
    for bit in bits:
        if bit not in (0, 1):
            raise ValueError(f"Information bits must be 0 or 1, got {bit!r}.")
        output.extend(_branch_output(int(bit), history))
        history = (int(bit), history[0])
    return tuple(output)


def hamming_distance(a: Sequence[int], b: Sequence[int]) -> int:
    """Number of positions at which two bit sequences differ."""
    if len(a) != len(b):
        raise ValueError(f"Sequences have different lengths ({len(a)} and {len(b)}).")
    return sum(int(x != y) for x, y in zip(a, b, strict=True))


def index_to_bits(index: int, num_bits: int) -> tuple[int, ...]:
    """Basis-state index -> information bits ``(u_1, ..., u_N)``, most significant first."""
    if not 0 <= index < 2**num_bits:
        raise ValueError(f"State index {index} is out of range for {num_bits} qubits.")
    return tuple((index >> (num_bits - 1 - k)) & 1 for k in range(num_bits))


def bits_to_index(bits: Sequence[int]) -> int:
    """Information bits ``(u_1, ..., u_N)`` -> basis-state index."""
    index = 0
    for bit in bits:
        index = (index << 1) | int(bit)
    return index


def bits_to_string(bits: Iterable[int]) -> str:
    """``(1, 0, 1)`` -> ``"101"``."""
    return "".join(str(int(bit)) for bit in bits)


def resolve_state(state: StateLike, num_bits: int) -> int:
    """Convert a basis state into its index.

    Args:
        state: index (``int``), bitstring such as ``"0110"`` or bit sequence,
            always with the first information bit first.
        num_bits: number of qubits ``N``.
    """
    if isinstance(state, str):
        if len(state) != num_bits or any(ch not in "01" for ch in state):
            raise ValueError(f"Expected a bitstring of length {num_bits}, got {state!r}.")
        return int(state, 2)
    if isinstance(state, (int, np.integer)):
        index = int(state)
    else:
        bits = [int(bit) for bit in state]
        if len(bits) != num_bits or any(bit not in (0, 1) for bit in bits):
            raise ValueError(f"Expected {num_bits} bits, got {state!r}.")
        index = bits_to_index(bits)
    if not 0 <= index < 2**num_bits:
        raise ValueError(f"State index {index} is out of range for {num_bits} qubits.")
    return index


def path_metrics(code: CodeLike) -> np.ndarray:
    """Hamming distance between the received word and every candidate path.

    Entry ``k`` equals ``hamming_distance(encode(index_to_bits(k, N)), flatten_code(parse_code(code)))``.
    This is exactly the phase exponent of the phase oracle:
    ``G(omega) = diag(exp(1j * omega * path_metrics(code)))``.

    The computation is vectorized over all ``2**N`` paths, so memory grows as
    ``O(N * 2**N)``.
    """
    pairs = parse_code(code)
    n = len(pairs)
    received = np.array(pairs, dtype=np.uint8)  # shape (N, 2)

    # Bits of every path, most significant first: shape (2**N, N)
    shifts = np.arange(n - 1, -1, -1)
    paths = ((np.arange(2**n)[:, None] >> shifts) & 1).astype(np.uint8)
    # Prepend the two zeros of the initial register: column 2 + t holds u_{t+1}
    padded = np.pad(paths, ((0, 0), (2, 0)))

    metrics = np.zeros(2**n, dtype=np.int64)
    for j, generator in enumerate(GENERATORS):
        emitted = np.zeros_like(paths)
        for delay, tap in enumerate(generator):
            if tap:
                emitted ^= padded[:, 2 - delay : 2 - delay + n]
        metrics += np.count_nonzero(emitted != received[:, j], axis=1)
    return metrics


def maximum_likelihood_states(code: CodeLike) -> np.ndarray:
    """Indices of the paths at minimum Hamming distance (brute-force classical reference)."""
    metrics = path_metrics(code)
    return np.flatnonzero(metrics == metrics.min())


def _count(number: int, noun: str) -> str:
    """``(3, "bit error")`` -> ``"3 bit errors"``."""
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def report_lines(
    code: Sequence[Pair],
    bits: Sequence[int],
    original: StateLike | None = None,
    decoder_lines: Sequence[str] = (),
    decoded_note: str = "",
    comparison_lines: Sequence[str] = (),
) -> list[str]:
    """Lines shared by the classical and quantum decoding reports.

    Args:
        code: received word as ``(bit, bit)`` pairs.
        bits: decoded message ``(u_1, ..., u_N)``.
        original: transmitted message (bitstring, bits or index), if known. It adds
            the original codeword, the number of channel errors and the outcome.
        decoder_lines: decoder-specific lines placed after the received word.
        decoded_note: text appended to the decoded message line.
        comparison_lines: lines placed after the number of corrected bits.
    """
    num_bits = len(code)
    received = flatten_code(code)
    decoded_codeword = encode(bits)
    lines: list[str] = []
    received_note = ""
    original_bits: tuple[int, ...] | None = None
    if original is not None:
        original_bits = index_to_bits(resolve_state(original, num_bits), num_bits)
        sent = encode(original_bits)
        lines += [
            f"  original message  : {bits_to_string(original_bits)}",
            f"  original codeword : {format_code(sent)}",
        ]
        received_note = f"  ({_count(hamming_distance(sent, received), 'bit error')})"
    lines.append(f"  received word     : {format_code(code)}{received_note}")
    lines += decoder_lines
    lines += [
        f"  decoded message   : {bits_to_string(bits)}{decoded_note}",
        f"  decoded codeword  : {format_code(decoded_codeword)}",
        # Bits the decoder changed: they are corrections only when the decoding is right
        f"  bits flipped      : {hamming_distance(decoded_codeword, received)}",
    ]
    lines += comparison_lines
    if original_bits is not None:
        wrong = hamming_distance(tuple(bits), original_bits)
        outcome = "correct" if wrong == 0 else f"wrong ({wrong} of {num_bits} message bits differ)"
        lines.append(f"  outcome           : {outcome}")
    return lines


@dataclass(frozen=True)
class ViterbiResult:
    """Outcome of :func:`viterbi_decode`.

    Attributes:
        code: received word as ``(bit, bit)`` pairs.
        bits: decoded information bits ``(u_1, ..., u_N)``.
        metric: Hamming distance between their codeword and the received word.
    """

    code: tuple[Pair, ...]
    bits: tuple[int, ...]
    metric: int

    @property
    def reencoded(self) -> tuple[int, ...]:
        """Codeword emitted by the encoder for :attr:`bits`."""
        return encode(self.bits)

    def summary(self, original: StateLike | None = None) -> str:
        """Multi-line human-readable report.

        Args:
            original: transmitted message (bitstring, bits or index), if known. It adds
                the original codeword, the number of channel errors and the outcome.
        """
        return "\n".join(["Classical Viterbi decoding", *report_lines(self.code, self.bits, original)])


def viterbi_decode(code: CodeLike) -> ViterbiResult:
    """Classical hard-decision Viterbi decoding of the (5, 7) code.

    The trellis has four states ``sigma = (u_{t-1}, u_t)`` and starts from ``00``.
    At every step the decoder keeps, for each state, only the path with the
    smallest accumulated Hamming distance (the survivor)::

        m_t(sigma') = min over sigma -> sigma' of [ m_{t-1}(sigma) + delta_t(sigma -> sigma') ]

    and at the end traces the best survivor back. The result is a
    maximum-likelihood path, found in ``O(N)`` operations instead of the
    ``O(N * 2**N)`` of :func:`path_metrics`.
    """
    pairs = parse_code(code)
    num_states = 4
    metrics = [0.0] + [math.inf] * (num_states - 1)  # the register starts from 00
    survivors: list[list[int]] = []  # survivors[t][state] = best predecessor state

    for received in pairs:
        new_metrics = [math.inf] * num_states
        predecessors = [0] * num_states
        for state, metric in enumerate(metrics):
            if metric == math.inf:
                continue
            older, newer = state >> 1, state & 1  # (u_{t-2}, u_{t-1})
            for bit in (0, 1):
                candidate = metric + hamming_distance(_branch_output(bit, (newer, older)), received)
                next_state = (newer << 1) | bit
                if candidate < new_metrics[next_state]:
                    new_metrics[next_state] = candidate
                    predecessors[next_state] = state
        metrics = new_metrics
        survivors.append(predecessors)

    # Trace back from the best final state
    state = min(range(num_states), key=metrics.__getitem__)
    best_metric = int(metrics[state])
    bits: list[int] = []
    for predecessors in reversed(survivors):
        bits.append(state & 1)
        state = predecessors[state]
    return ViterbiResult(pairs, tuple(reversed(bits)), best_metric)
