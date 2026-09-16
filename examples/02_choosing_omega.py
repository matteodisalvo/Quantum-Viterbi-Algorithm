"""Tune omega for a received word: grid search, resonance and the channel rule.

The probability of a maximum-likelihood path is scanned over omega in [0, pi] and
compared with three choices: the value that ``decode`` picks by itself, the resonance
predicted from the distance spectrum, and the rule that only knows the length of the
message and the channel. The path at distance 2N - d_min, which shares the target phase
at pi/(N - d_min), is plotted as well.

    python examples/02_choosing_omega.py
    python examples/02_choosing_omega.py --code "11 01 01 10 11 00" --step 0.05
"""

from __future__ import annotations

import argparse
import math

import matplotlib.pyplot as plt

from quantum_viterbi import (
    channel_omega,
    class_probabilities,
    distance_spectrum,
    omega_range,
    parse_code,
    resonant_omega,
    tuned_omega,
    viterbi_decode,
)
from quantum_viterbi.convolutional import bits_to_string
from quantum_viterbi.plotting import BASE_COLOR, HIGHLIGHT_COLOR, show_or_save

RECEIVED_CODE = "11 01 00 10 00 00"
STEP_DEGREES = 0.02
ERROR_PROB = 0.1


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--code", default=RECEIVED_CODE, help="received word")
    parser.add_argument("--step", type=float, default=STEP_DEGREES, help="grid step in degrees")
    parser.add_argument(
        "--error-prob", type=float, default=ERROR_PROB, help="crossover probability of the channel"
    )
    parser.add_argument("--save", metavar="FILE", help="save the figure instead of showing it")
    args = parser.parse_args()

    num_qubits = len(parse_code(args.code))
    classical = viterbi_decode(args.code)
    spectrum = distance_spectrum(args.code)
    resonance = resonant_omega(args.code)
    tuned = tuned_omega(args.code)
    rule = channel_omega(num_qubits, args.error_prob)

    # All maximum-likelihood paths share one amplitude, so their class probability
    # divided by their number is the probability of measuring one of them
    per_path = resonance.multiplicity

    def probability(omega: float) -> float:
        return float(class_probabilities(args.code, [omega])[0, resonance.min_distance]) / per_path

    grid = omega_range(0, 180, args.step, degrees=True)
    classes = class_probabilities(args.code, grid)
    likelihood = classes[:, resonance.min_distance] / per_path
    best = int(likelihood.argmax())

    print(f"Received word     : {args.code}  ({num_qubits} qubits)")
    print(f"Viterbi path      : {bits_to_string(classical.bits)}  (distance {classical.metric})")
    print(f"tuned_omega       : omega = {tuned:.4f} rad   P = {probability(tuned):.4f}   (used by decode)")
    print(f"grid search       : omega = {grid[best]:.4f} rad   P = {likelihood[best]:.4f}")
    print(f"resonance         : omega = {resonance.omega:.4f} rad   P = {probability(resonance.omega):.4f}")
    print(f"  predicted at K = {resonance.iterations}: {resonance.probability / per_path:.4f}")
    print(f"channel rule      : omega = {rule:.4f} rad   P = {probability(rule):.4f}")

    ax = plt.subplots(figsize=(11, 4.5))[1]
    ml_label = f"maximum-likelihood path (d = {resonance.min_distance})"
    ax.plot(grid, likelihood, color=BASE_COLOR, label=ml_label)
    alias = 2 * num_qubits - resonance.min_distance
    if alias != resonance.min_distance and spectrum[alias]:
        ax.plot(
            grid,
            classes[:, alias] / spectrum[alias],
            color="tab:gray",
            ls="--",
            label=f"aliased path (d = {alias})",
        )
        ax.axvline(
            math.pi / (num_qubits - resonance.min_distance),
            color="tab:gray",
            ls=":",
            label=r"$\pi/(N - d_{\min})$",
        )
    ax.axvline(tuned, color=HIGHLIGHT_COLOR, ls="-", lw=1, label="tuned_omega")
    ax.axvline(resonance.omega, color="tab:purple", ls="--", lw=1, label="resonance")
    ax.axvline(rule, color="tab:green", ls="--", lw=1, label="channel rule")
    ax.set_xlabel(r"$\omega$ (rad)")
    ax.set_ylabel("Probability of one path")
    ax.set_title(f"Output probability vs $\\omega$  ({args.code})")
    ax.grid(alpha=0.3)
    ax.legend()
    show_or_save(ax.figure, args.save)


if __name__ == "__main__":
    main()
