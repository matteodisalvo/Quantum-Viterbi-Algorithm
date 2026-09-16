"""Choice of the metric parameter omega and of the number of iterations.

The best omega depends on the received word, so the usual workflow is to tune it
first and then decode with that value. Available strategies:

* :func:`tuned_omega`: grid search over ``(0, pi]``, refined around the highest
  peaks, for the omega at which the decoder most reliably returns a
  maximum-likelihood message. It is what :func:`~quantum_viterbi.decode` uses
  when omega is not given;
* :func:`scan_omega`: grid search that maximizes the peak probability or the
  probability of a target path, optionally stopping at a threshold;
* :func:`resonant_omega`: omega and number of iterations from the two-level
  (detuned Grover) theory of the QVA on distance classes. It needs the distance
  spectrum of the received word, which also reveals the Viterbi metric;
* :func:`scan_resonance`: exact scan of omega around the resonance, computed on
  distance classes;
* :func:`channel_omega`: decoding-free estimate from ``N`` and the crossover
  probability of the channel.

Since every phase is ``exp(1j * omega * d)`` with integer ``d``, probabilities
are ``2*pi``-periodic in omega and symmetric under ``omega -> 2*pi - omega``
(the final state is only complex-conjugated), so ``[0, pi]`` already contains
every distinct result.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .backends import Backend, get_backend
from .convolutional import (
    CodeLike,
    Pair,
    StateLike,
    bits_to_index,
    parse_code,
    resolve_state,
    viterbi_decode,
)
from .grover import resolve_iterations
from .spectrum import as_omega_grid, class_probabilities, distance_spectrum, distance_weights

#: Coarse grid used by :func:`tuned_omega`, and the size above which it is not grown.
_COARSE_POINTS = 2001
_MAX_COARSE_POINTS = 50_001
#: Peaks of the coarse grid that are refined, and how the refinement proceeds.
_REFINED_PEAKS = 5
_REFINEMENT_ROUNDS = 4
_REFINEMENT_POINTS = 65


def _decoded_scores(
    pairs: tuple[Pair, ...],
    spectrum: tuple[int, ...],
    d0: int,
    omegas: np.ndarray,
    rounds: int,
) -> np.ndarray:
    """Score of each omega for the decoder, from the per-path probabilities.

    Paths at the same distance share one amplitude, so a class of ``A_d`` paths splits its
    probability ``A_d`` ways. :func:`~quantum_viterbi.decode` returns the most probable
    single path, not the most probable class, so the classes have to be compared per path:
    a class with fewer paths can win even with less probability in total. That is what
    happens for an aliased word, whose class at ``2N - d_min`` rides at nearly the target
    phase.

    With ``p`` the probability of one maximum-likelihood path and ``margin`` its lead over
    the most probable path of any other class, the score is ``sqrt(p * margin)`` when the
    lead is positive and ``margin`` otherwise. Omegas that decode to a wrong message thus
    rank below every omega that decodes correctly, and the score vanishes at a tie:
    maximizing ``p`` alone pushes the search onto the boundary where the right and the
    wrong path are equally likely, and the decoded message then depends on rounding.
    """
    counts = np.array(spectrum, dtype=float)
    populated = counts > 0
    per_path = np.zeros((len(np.atleast_1d(omegas)), counts.size))
    classes = class_probabilities(pairs, omegas, rounds)
    per_path[:, populated] = classes[:, populated] / counts[populated]
    target = per_path[:, d0].copy()
    per_path[:, d0] = -np.inf  # every class except the maximum-likelihood one
    margin = target - np.maximum(per_path.max(axis=1), 0.0)  # no rival at all: lead is p
    return np.where(margin > 0, np.sqrt(target * np.maximum(margin, 0.0)), margin)


def tuned_omega(code: CodeLike, iterations: int | None = None, points: int | None = None) -> float:
    """Omega at which the decoder most reliably returns a maximum-likelihood message.

    A coarse grid over ``(0, pi]`` locates the peaks of a score, defined below, and the
    highest five are refined by repeatedly re-gridding the bracket around the best point.
    All paths at the same distance share one amplitude, so the score is evaluated on the
    distance classes (see :mod:`quantum_viterbi.spectrum`) rather than on the ``2**N``
    statevector: the whole search costs ``O(points * K * N)``. This is what
    :func:`~quantum_viterbi.decode` uses when omega is not given.

    The score balances the two things :func:`~quantum_viterbi.decode` needs: the
    probability ``p`` of one maximum-likelihood path, and its lead ``margin`` over the most
    probable path at any other distance. It is ``sqrt(p * margin)``, or the margin itself
    when negative, so an omega that decodes to a wrong message never beats one that decodes
    correctly and a tie scores zero. Maximizing ``p`` alone is not enough: for an aliased
    word, whose class at ``2N - d_min`` rides at nearly the target phase, the most probable
    path can then be a wrong one, or tie with it. For a few short aliased words (16 of the
    256 words with ``N = 4``) no omega decodes correctly at the default number of
    iterations; the value returned is then the least bad one.

    The peaks get narrower as ``N`` grows -- their width shrinks roughly like
    ``4 * 2**(-N/2) / N`` -- so the coarse grid grows with ``N`` up to 50001 points. Up to
    ``N = 16`` the score reached matches the best score of a grid ten times finer, on
    noiseless and noisy words. Above that a grid can step over a peak entirely, and
    :func:`resonant_omega`, which locates the resonance analytically, is a better starting
    point.

    Warning:
        Tuning uses the distance spectrum of the received word, which already reveals the
        Viterbi metric ``d_min``. Like :func:`resonant_omega`, it is a tool for analysis
        and benchmarking, not information that a decoder gets for free.

    Args:
        code: received word.
        iterations: number of Grover iterations to tune for
            (default ``floor(pi/4 * sqrt(2**N))``). Omega and the iteration count
            have to be chosen together, so the value returned only applies to these.
        points: size of the coarse grid; by default it is derived from ``N``.

    Raises:
        ValueError: if ``points`` is not an integer of at least 2.
    """
    pairs = parse_code(code)
    num_qubits = len(pairs)
    rounds = resolve_iterations(num_qubits, iterations)
    if points is None:
        needed = math.ceil(0.8 * num_qubits * 2 ** (num_qubits / 2))
        points = min(_MAX_COARSE_POINTS, max(_COARSE_POINTS, needed))
    elif int(points) != points or points < 2:
        raise ValueError(f"points must be an integer of at least 2, got {points!r}.")
    points = int(points)
    spectrum = distance_spectrum(pairs)
    d0 = next(d for d, count in enumerate(spectrum) if count)

    grid = np.linspace(math.pi / points, math.pi, points)
    scores = _decoded_scores(pairs, spectrum, d0, grid, rounds)
    best_omega = float(grid[int(scores.argmax())])
    best_score = float(scores.max())

    # Interior local maxima of the coarse grid, plus the best grid point itself: the
    # argmax can sit on an endpoint, which is never a local maximum and would then be
    # returned unrefined. It also covers a flat curve, which has no interior maximum.
    peaks = np.flatnonzero((scores[1:-1] >= scores[:-2]) & (scores[1:-1] >= scores[2:])) + 1
    peaks = np.union1d(peaks, [int(scores.argmax())])
    step = float(grid[1] - grid[0])
    for peak in peaks[np.argsort(-scores[peaks])[:_REFINED_PEAKS]]:
        lower, upper = float(grid[peak]) - step, float(grid[peak]) + step
        for _ in range(_REFINEMENT_ROUNDS):
            window = np.linspace(max(lower, 0.0), min(upper, math.pi), _REFINEMENT_POINTS)
            values = _decoded_scores(pairs, spectrum, d0, window, rounds)
            top = int(values.argmax())
            if values[top] > best_score:
                best_omega, best_score = float(window[top]), float(values[top])
            half = float(window[1] - window[0])
            lower, upper = float(window[top]) - half, float(window[top]) + half
    return best_omega


def channel_omega(num_transitions: int, error_prob: float = 0.0) -> float:
    """Decoding-free omega from the channel only: ``pi / (N + 10 / 2**N - j)``.

    ``2 N p (1 - 1.5 p)`` approximates the expected minimum distance ``E[d_min]`` of a
    received word on a binary symmetric channel with crossover probability ``p``, and
    ``j`` is its floor (it can be almost 1 below ``E[d_min]``). Averaged over the
    channel, the probability of measuring a maximum-likelihood path is a comb of narrow
    peaks at ``omega = pi / (N - j)`` (every received word has mean path distance
    exactly ``N``); ``10 / 2**N`` is a small-``N`` correction fitted on noiseless words.
    Fitted for the default number of iterations, ``4 <= N <= 12`` and ``p <= 0.15``;
    even there the channel-averaged probability can be about 20% below that of the best
    single omega (e.g. ``N = 6, p = 0.1``).

    Note:
        One omega per ``(N, p)`` only serves the received words whose ``d_min``
        equals ``j``. When the word itself is available, :func:`resonant_omega`
        is far more accurate.

    Args:
        num_transitions: number of trellis transitions ``N``.
        error_prob: crossover probability ``p`` of the binary symmetric channel, in ``[0, 0.5]``.
    """
    if int(num_transitions) != num_transitions or num_transitions < 1:
        raise ValueError("num_transitions must be a positive integer.")
    if not 0 <= error_prob <= 0.5:
        raise ValueError("error_prob must be in [0, 0.5] for a binary symmetric channel.")
    n = int(num_transitions)
    j = math.floor(2 * n * error_prob * (1 - 1.5 * error_prob))
    return math.pi / (n + 10 / 2**n - j)


def omega_range(start: float, stop: float, step: float, degrees: bool = False) -> np.ndarray:
    """Grid from ``start`` to ``stop`` with spacing ``step``, returned in radians.

    ``stop`` is included when ``stop - start`` is a multiple of ``step`` (up to rounding).

    Args:
        start, stop, step: grid definition.
        degrees: interpret the three values as degrees.

    Example: ``omega_range(0, 360, 0.0005, degrees=True)`` contains 720,001 values.
    """
    if step <= 0:
        raise ValueError("step must be positive.")
    if stop < start:
        raise ValueError("stop must not be smaller than start.")
    intervals = (stop - start) / step
    count = math.floor(intervals + 1e-9 * max(1.0, intervals)) + 1  # tolerate rounding errors
    grid = start + step * np.arange(count)
    return np.deg2rad(grid) if degrees else grid


@dataclass(frozen=True)
class OmegaScan:
    """Result of :func:`scan_omega`.

    Attributes:
        omegas: evaluated omega values (truncated when the threshold stops the scan).
        scores: probability of ``target`` for each omega, or the peak probability
            over all states when no target is given.
        target: index of the target state, or ``None``.
        threshold: probability that stops the scan, or ``None``.
        iterations: number of Grover iterations used.
    """

    omegas: np.ndarray
    scores: np.ndarray
    target: int | None
    threshold: float | None
    iterations: int

    @property
    def best_index(self) -> int:
        """Position of the best score (first one in case of ties)."""
        return int(np.argmax(self.scores))

    @property
    def best_omega(self) -> float:
        return float(self.omegas[self.best_index])

    @property
    def best_score(self) -> float:
        return float(self.scores[self.best_index])

    @property
    def threshold_reached(self) -> bool:
        return self.threshold is not None and self.best_score >= self.threshold

    def top(self, k: int = 1) -> list[tuple[float, float]]:
        """The ``k`` best ``(omega, score)`` pairs, sorted by decreasing score."""
        order = np.argsort(-self.scores, kind="stable")[:k]
        return [(float(self.omegas[i]), float(self.scores[i])) for i in order]


def scan_omega(
    code: CodeLike,
    omegas: Iterable[float],
    iterations: int | None = None,
    target: StateLike | None = None,
    threshold: float | None = None,
    backend: str | Backend = "numpy",
    chunk_size: int = 1024,
) -> OmegaScan:
    """Evaluate the QVA on a grid of omega values.

    Args:
        code: received word.
        omegas: omega values to test, in radians (see :func:`omega_range`).
        iterations: number of Grover iterations (default ``floor(pi/4 * sqrt(2**N))``).
        target: state whose probability is maximized (index, bitstring or bits).
            If ``None`` the score is the peak probability over all states.
        threshold: stop at the first omega whose score reaches this value.
        backend: backend name or instance. The default ``"numpy"`` is exact and,
            for small ``N``, evaluates thousands of omegas per second; ``"qiskit"`` and
            ``"pennylane"`` give the same numbers but are much slower.
        chunk_size: number of omegas evaluated per batch.
    """
    pairs = parse_code(code)
    num_qubits = len(pairs)
    rounds = resolve_iterations(num_qubits, iterations)
    grid = as_omega_grid(omegas)
    if grid.size == 0:
        raise ValueError("At least one omega value is required.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    target_index = None if target is None else resolve_state(target, num_qubits)
    engine = get_backend(backend)

    scores: list[np.ndarray] = []
    for start in range(0, grid.size, chunk_size):
        probabilities = engine.probabilities_batch(pairs, grid[start : start + chunk_size], rounds)
        chunk = probabilities.max(axis=1) if target_index is None else probabilities[:, target_index]
        if threshold is not None:
            hits = np.flatnonzero(chunk >= threshold)
            if hits.size:
                scores.append(chunk[: hits[0] + 1])
                break
        scores.append(chunk)

    values = np.concatenate(scores)
    return OmegaScan(grid[: values.size], values, target_index, threshold, rounds)


@dataclass(frozen=True)
class Resonance:
    """Result of :func:`resonant_omega`.

    Attributes:
        omega: resonant omega in radians, the root of ``C(omega) = 0``.
        iterations: recommended number of Grover iterations, ``optimal_iterations``
            rounded to the nearest integer (at least 1).
        optimal_iterations: real-valued ``K* = pi/4 * sqrt(S * 2**N / A_dmin) - 1/2``.
        min_distance: distance ``d_min`` of the maximum-likelihood paths (the Viterbi metric).
        multiplicity: number ``A_dmin`` of maximum-likelihood paths.
        probability: predicted probability ``min(1, 1/S)`` of measuring a maximum-likelihood
            path at ``(omega, optimal_iterations)``; divide by ``multiplicity`` for one path.
        width: half-width in omega of the resonance envelope, ``2 * sqrt(S * w0) / |C'(omega)|``.
            It shrinks roughly like ``4 * 2**(-N/2) / N``, so omega must be set this precisely.
    """

    omega: float
    iterations: int
    optimal_iterations: float
    min_distance: int
    multiplicity: int
    probability: float
    width: float


#: Roots of ``C`` compared by exact probability when the received word is aliased.
_ALIAS_CANDIDATES = 5
#: Above this number of iterations the exact comparison is skipped (it costs ``O(K * N)``).
_MAX_CHECKED_ITERATIONS = 20_000


def _detuning_sums(
    omegas: np.ndarray, gaps: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``C``, ``S`` and ``dC/domega`` at each omega; ``gaps = d - d0`` and ``weights = w_d`` for d != d0."""
    half = np.outer(omegas, gaps) / 2
    sine = np.sin(half)
    csc2 = 1.0 / sine**2
    return (np.cos(half) / sine) @ weights, csc2 @ weights, -(csc2 * (gaps / 2)) @ weights


def resonant_omega(code: CodeLike, omega_max: float = math.pi) -> Resonance:
    """Resonant omega and number of iterations for a received word, from its distance spectrum.

    On the distance classes the QVA is a Grover walk with weights ``w_d = A_d / 2**N``
    (see :mod:`quantum_viterbi.spectrum`). For the maximum-likelihood class
    ``d0 = d_min``, with weight ``w0``, define::

        C(omega) = sum_{d != d0} w_d * cot(omega * (d - d0) / 2)
        S(omega) = sum_{d != d0} w_d / sin(omega * (d - d0) / 2)**2

    When ``w0`` is small the target behaves like a detuned two-level Grover search::

        P_ML(omega, K) ~= 4 w0 / (C**2 + 4 S w0) * sin((K + 1/2) * sqrt(C**2 + 4 S w0) / S)**2

    so a good omega solves ``C(omega) = 0``, the peak probability is ``1/S`` and it is
    reached after ``K* = pi/4 * sqrt(S / w0) - 1/2`` iterations. ``C`` decreases strictly
    between its poles ``omega = 2 pi k / (d - d0)``, hence it has one root per pole
    interval; the root with the smallest ``S`` (largest predicted peak) is returned.

    For noiseless words the prediction is asymptotically exact: from ``N = 12`` the exact
    probability at ``(omega, iterations)`` agrees with :attr:`Resonance.probability` to
    about ``1e-3``. On noisy words the two-level model breaks down when another class is
    almost in phase with ``d0``, i.e. a pole of ``C`` lies within a few
    :attr:`Resonance.width` of the root, and the error can exceed 0.2 even at ``N = 20``.
    This always happens for aliased words (``A_{2N - d0} > 0``), whose alias has the same
    phase as the target at ``omega = pi / (N - d0)``: for them the ``5`` roots with the
    smallest ``S`` are compared by exact probability at their own ``K*`` (unless ``K*``
    exceeds ``20000``). For ``N <= 8`` the model is rough as well (for noiseless words the
    error is 1-4% at ``N = 5..8`` and 17% at ``N = 4``). Check the result with
    :func:`class_probabilities` or refine it with :func:`scan_resonance`.

    Warning:
        The spectrum reveals ``d_min``, and the trellis pass that computes it could
        also trace back a maximum-likelihood path. This is an analysis and
        benchmarking tool, not information that a decoder gets for free.

    Args:
        code: received word.
        omega_max: roots of ``C`` are searched in ``(0, omega_max]``.

    Raises:
        ValueError: if every path has the same metric (omega has no effect) or ``C`` has no root.
    """
    pairs = parse_code(code)
    if not 0 < omega_max <= math.pi:
        raise ValueError("omega_max must be in (0, pi].")
    spectrum = distance_spectrum(pairs)
    weights = distance_weights(pairs)
    populated = [d for d, count in enumerate(spectrum) if count]
    if len(populated) == 1:
        raise ValueError("Every path has the same metric: omega has no effect.")
    d0 = populated[0]
    gaps = np.array(populated[1:], dtype=float) - d0
    rest = weights[populated[1:]]
    w0 = float(weights[d0])

    # Poles of C in (0, omega_max]: omega = 2 pi k / g, deduplicated as reduced fractions k / g
    fractions = {
        (k // math.gcd(k, g), g // math.gcd(k, g))
        for g in (int(gap) for gap in gaps)
        for k in range(1, math.floor(g * omega_max / (2 * math.pi)) + 1)
    }
    poles = sorted(2 * math.pi * k / g for k, g in fractions)
    lower = np.array([0.0, *poles])
    upper = np.array([*poles, omega_max])
    keep = upper > lower
    lower, upper = lower[keep], upper[keep]
    if (not poles or poles[-1] < omega_max) and _detuning_sums(np.array([omega_max]), gaps, rest)[0][0] > 0:
        # The last interval ends at omega_max instead of a pole and C stays positive there
        lower, upper = lower[:-1], upper[:-1]
    if lower.size == 0:
        raise ValueError("C(omega) has no root in (0, omega_max].")

    # Vectorized bisection: C(lower+) = +inf and C(upper-) <= 0 in every interval
    for _ in range(200):
        middle = 0.5 * (lower + upper)
        if np.all((middle == lower) | (middle == upper)):
            break
        positive = _detuning_sums(middle, gaps, rest)[0] > 0
        lower = np.where(positive, middle, lower)
        upper = np.where(positive, upper, middle)
    roots = 0.5 * (lower + upper)

    s_roots = _detuning_sums(roots, gaps, rest)[1]
    omega = float(roots[int(np.argmin(s_roots))])
    alias = 2 * len(pairs) - d0
    if alias != d0 and spectrum[alias]:
        # The alias shares the target phase next to the resonance, so the two-level
        # prediction is unreliable: rank the best candidates by exact probability
        candidates = [
            (float(roots[i]), max(1, math.floor(math.pi / 4 * math.sqrt(s_roots[i] / w0))))
            for i in np.argsort(s_roots)[:_ALIAS_CANDIDATES]
        ]
        if max(rounds for _, rounds in candidates) <= _MAX_CHECKED_ITERATIONS:
            exact = [class_probabilities(pairs, [root], rounds)[0, d0] for root, rounds in candidates]
            omega = candidates[int(np.argmax(exact))][0]

    _, s_value, slope = (float(value[0]) for value in _detuning_sums(np.array([omega]), gaps, rest))
    k_star = math.pi / 4 * math.sqrt(s_value / w0) - 0.5
    return Resonance(
        omega=omega,
        iterations=max(1, math.floor(k_star + 0.5)),
        optimal_iterations=k_star,
        min_distance=d0,
        multiplicity=spectrum[d0],
        probability=min(1.0, 1.0 / s_value),
        width=2 * math.sqrt(s_value * w0) / abs(slope),
    )


def scan_resonance(
    code: CodeLike,
    iterations: int | None = None,
    span: float = 3.0,
    points: int = 2001,
) -> OmegaScan:
    """Exact scan of omega in ``resonance.omega +/- span * resonance.width``.

    The target of the scan is the maximum-likelihood path returned by
    :func:`~quantum_viterbi.convolutional.viterbi_decode`. The scores equal those of
    ``scan_omega(code, omegas, iterations, target=...)`` on the same grid, but they are
    computed on the distance classes in ``O(points * K * N)`` operations, without
    building the ``2**N`` statevector.

    Args:
        code: received word.
        iterations: number of Grover iterations (default ``floor(pi/4 * sqrt(2**N))``).
        span: half-width of the window in units of :attr:`Resonance.width`.
        points: number of omega values in the window.
    """
    pairs = parse_code(code)
    rounds = resolve_iterations(len(pairs), iterations)
    if span <= 0:
        raise ValueError("span must be positive.")
    if points < 2:
        raise ValueError("points must be at least 2.")
    resonance = resonant_omega(pairs)
    lower = max(0.0, resonance.omega - span * resonance.width)
    upper = min(math.pi, resonance.omega + span * resonance.width)
    grid = np.linspace(lower, upper, points)
    classes = class_probabilities(pairs, grid, rounds)
    scores = classes[:, resonance.min_distance] / float(resonance.multiplicity)
    target = bits_to_index(viterbi_decode(pairs).bits)
    return OmegaScan(grid, scores, target, None, rounds)
