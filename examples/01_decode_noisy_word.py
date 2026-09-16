"""Decode a noisy received word with the classical and the quantum decoder.

A message is encoded, the channel flips one code bit, and the received word is decoded
twice: with the classical Viterbi algorithm and with the QVA. The output distribution of
the QVA is shown as an overview plus a zoom on the most probable paths.

    python examples/01_decode_noisy_word.py --backend qiskit
    python examples/01_decode_noisy_word.py --backend pennylane --shots 4096
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from quantum_viterbi import decode, viterbi_decode
from quantum_viterbi.backends import BACKEND_NAMES
from quantum_viterbi.plotting import plot_distribution, plot_top_states, show_or_save

# The message 101101 is encoded as 11 01 00 10 10 00; the channel flips the ninth bit
ORIGINAL_MESSAGE = "101101"
RECEIVED_CODE = "11 01 00 10 00 00"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--backend", choices=BACKEND_NAMES, default="qiskit", help="simulation backend")
    parser.add_argument(
        "--omega",
        type=float,
        default=None,
        help="metric parameter in radians (default: tuned for this word, see example 02)",
    )
    parser.add_argument("--shots", type=int, default=None, help="measurement shots (default: exact)")
    parser.add_argument("--top", type=int, default=10, help="number of paths in the zoom plot")
    parser.add_argument("--save", metavar="FILE", help="save the figure instead of showing it")
    args = parser.parse_args()

    print(viterbi_decode(RECEIVED_CODE).summary(original=ORIGINAL_MESSAGE), end="\n\n")
    result = decode(RECEIVED_CODE, omega=args.omega, backend=args.backend, shots=args.shots)
    print(result.summary(original=ORIGINAL_MESSAGE))

    fig, (overview, zoom) = plt.subplots(1, 2, figsize=(15, 4.5), gridspec_kw={"width_ratios": [2, 1]})
    plot_distribution(
        result.probabilities,
        highlight=ORIGINAL_MESSAGE,
        title=f"QVA output (omega = {result.omega:.4f} rad)",
        ax=overview,
    )
    plot_top_states(result.probabilities, k=args.top, highlight=ORIGINAL_MESSAGE, ax=zoom)
    show_or_save(fig, args.save)


if __name__ == "__main__":
    main()
