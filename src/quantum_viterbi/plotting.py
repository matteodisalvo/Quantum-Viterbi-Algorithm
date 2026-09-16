"""Matplotlib figures for QVA output distributions and omega scans."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .convolutional import StateLike, resolve_state
from .tuning import OmegaScan

#: Default bar color.
BASE_COLOR = (0.2, 0.6, 0.8)
#: Color of the highlighted (most probable or target) state.
HIGHLIGHT_COLOR = "tab:red"


def _num_qubits(probabilities: np.ndarray) -> int:
    """Number of qubits of a probability vector of length ``2**N``."""
    size = probabilities.size
    num_qubits = size.bit_length() - 1
    if probabilities.ndim != 1 or size < 2 or 2**num_qubits != size:
        raise ValueError("Expected a one-dimensional probability vector of length 2**N.")
    return num_qubits


def _label(index: int, num_qubits: int) -> str:
    """Binary label of a basis state with ``N`` digits, most significant bit first."""
    return format(index, f"0{num_qubits}b")


def _axes(ax: Axes | None, figsize: tuple[float, float]) -> Axes:
    """Return ``ax`` or create a new figure with a single axes."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def plot_distribution(
    probabilities: np.ndarray,
    highlight: StateLike | None = None,
    tick_step: int | None = None,
    title: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Bar chart of the whole output distribution.

    Args:
        probabilities: vector of length ``2**N``.
        highlight: state drawn in red and annotated (default: most probable state).
        tick_step: label one state every ``tick_step`` (default: every state up
            to 32 states, otherwise one every 8); the highlighted state is
            always labeled.
        title: axes title.
        ax: existing axes to draw on.
    """
    probs = np.asarray(probabilities, dtype=float)
    num_qubits = _num_qubits(probs)
    marked = int(np.argmax(probs)) if highlight is None else resolve_state(highlight, num_qubits)
    ax = _axes(ax, figsize=(10, 4))

    colors = [BASE_COLOR] * probs.size
    colors[marked] = HIGHLIGHT_COLOR
    ax.bar(np.arange(probs.size), probs, color=colors, edgecolor="none")

    step = tick_step or (1 if probs.size <= 32 else 8)
    ticks = sorted(set(range(0, probs.size, step)) | {marked})
    ax.set_xticks(ticks, [_label(t, num_qubits) for t in ticks], rotation=45, ha="right")
    ax.annotate(
        f" P = {probs[marked]:.3f}",
        xy=(marked, probs[marked]),
        ha="left",
        va="bottom",
        color="tab:blue",
        fontweight="bold",
    )

    ax.set_ylim(0, 1.12 * max(probs.max(), 1e-12))  # headroom for the annotation
    ax.set_xlabel(r"Path $|u_1 \ldots u_N\rangle$")
    ax.set_ylabel("Probability")
    ax.set_title(title or "Output distribution")
    ax.grid(axis="y", alpha=0.3)
    return ax


def plot_top_states(
    probabilities: np.ndarray,
    k: int = 10,
    highlight: StateLike | None = None,
    ax: Axes | None = None,
) -> Axes:
    """The ``k`` most probable states in decreasing order.

    Args:
        probabilities: vector of length ``2**N``.
        k: number of states shown.
        highlight: state drawn in red if it is among the top ``k``
            (default: most probable state).
        ax: existing axes to draw on.

    Raises:
        ValueError: if ``k`` is not a positive integer.
    """
    probs = np.asarray(probabilities, dtype=float)
    num_qubits = _num_qubits(probs)
    if int(k) != k or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}.")
    order = np.argsort(-probs, kind="stable")[:k]
    marked = int(order[0]) if highlight is None else resolve_state(highlight, num_qubits)
    ax = _axes(ax, figsize=(6, 4))

    positions = np.arange(order.size)
    colors = [HIGHLIGHT_COLOR if index == marked else BASE_COLOR for index in order]
    ax.bar(positions, probs[order], color=colors, edgecolor="none")
    ax.set_xticks(positions, [_label(int(i), num_qubits) for i in order], rotation=45, ha="right")

    ax.set_xlabel("Paths (decreasing probability)")
    ax.set_ylabel("Probability")
    ax.set_title(f"Top {order.size} paths")
    ax.grid(axis="y", alpha=0.3)
    return ax


def plot_omega_scan(scan: OmegaScan, degrees: bool = False, ax: Axes | None = None) -> Axes:
    """Score as a function of omega, with the best value (and threshold) marked.

    Args:
        scan: result of :func:`quantum_viterbi.tuning.scan_omega`.
        degrees: use degrees on the horizontal axis.
        ax: existing axes to draw on.
    """
    ax = _axes(ax, figsize=(8, 4))
    omegas = np.rad2deg(scan.omegas) if degrees else scan.omegas
    marker = "o" if scan.omegas.size <= 50 else None
    ax.plot(omegas, scan.scores, marker=marker, markersize=4, color=BASE_COLOR)

    unit = "deg" if degrees else "rad"
    ax.axvline(
        omegas[scan.best_index],
        color=HIGHLIGHT_COLOR,
        linestyle="--",
        linewidth=1,
        label=rf"best $\omega$ = {omegas[scan.best_index]:.4f} {unit}",
    )
    if scan.threshold is not None:
        ax.axhline(scan.threshold, color="gray", linestyle=":", label=f"threshold = {scan.threshold:.2f}")

    ax.set_xlabel(rf"$\omega$ ({unit})")
    ax.set_ylabel("Peak probability" if scan.target is None else "Target probability")
    ax.set_title(r"QVA score vs $\omega$")
    ax.grid(alpha=0.3)
    ax.legend()
    return ax


def show_or_save(figure: Figure, path: str | None = None) -> None:
    """Show ``figure`` interactively, or save it to ``path`` when one is given."""
    figure.tight_layout()
    if path:
        figure.savefig(path, dpi=150)
        print(f"Figure saved to {path}")
    else:
        plt.show()
