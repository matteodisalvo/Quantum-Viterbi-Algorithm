"""Command-line interface.

Examples::

    quantum-viterbi decode 11 01 00 10 00 00 --original 101101
    quantum-viterbi decode 11 01 00 10 --backend pennylane --shots 4096 --plot
    quantum-viterbi viterbi 11 01 00 10 00 00 --original 101101
    quantum-viterbi scan 11 01 00 10 00 00 --target 101101 --step 0.01 --degrees
    quantum-viterbi circuit 11 01 00 --omega 0.5 --iterations 1
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence

from . import __version__
from .backends import BACKEND_NAMES, DEFAULT_BACKEND
from .convolutional import bits_to_string, index_to_bits, parse_code, viterbi_decode
from .decoder import decode
from .tuning import omega_range, scan_omega


def build_parser() -> argparse.ArgumentParser:
    """Create the parser with the ``decode``, ``viterbi``, ``scan`` and ``circuit`` commands."""
    parser = argparse.ArgumentParser(
        prog="quantum-viterbi",
        description="Quantum and classical Viterbi decoding of the rate-1/2 (5, 7) convolutional code.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    # Argument shared by every command
    code_args = argparse.ArgumentParser(add_help=False)
    code_args.add_argument("code", nargs="+", help="received word, e.g. 11 01 00 10")

    # Arguments shared by the quantum commands
    common = argparse.ArgumentParser(add_help=False, parents=[code_args])
    common.add_argument(
        "-k",
        "--iterations",
        type=int,
        default=None,
        help="Grover iterations (default: floor(pi/4 * sqrt(2**N)))",
    )

    decode_cmd = commands.add_parser("decode", parents=[common], help="decode a received word")
    decode_cmd.add_argument(
        "-w",
        "--omega",
        type=float,
        default=None,
        help="metric parameter in radians (default: tuned for the received word)",
    )
    decode_cmd.add_argument("-b", "--backend", choices=BACKEND_NAMES, default=DEFAULT_BACKEND)
    decode_cmd.add_argument("--shots", type=int, default=None, help="measurement shots (default: exact)")
    decode_cmd.add_argument("--seed", type=int, default=None, help="random seed used when sampling")
    decode_cmd.add_argument("--top", type=int, default=5, help="number of most probable paths to list")
    _add_original_argument(decode_cmd)
    _add_figure_arguments(decode_cmd)

    viterbi_cmd = commands.add_parser(
        "viterbi", parents=[code_args], help="decode with the classical Viterbi algorithm"
    )
    _add_original_argument(viterbi_cmd)

    scan_cmd = commands.add_parser("scan", parents=[common], help="search for the best omega on a grid")
    scan_cmd.add_argument("--start", type=float, default=0.0, help="first omega (default: 0)")
    scan_cmd.add_argument(
        "--stop",
        type=float,
        default=None,
        # Probabilities are 2*pi-periodic and mirrored around pi, so half a period is enough
        help="last omega (default: pi, which already contains every distinct result)",
    )
    scan_cmd.add_argument("--step", type=float, default=0.001, help="grid step (default: 0.001)")
    scan_cmd.add_argument("--degrees", action="store_true", help="start, stop and step are in degrees")
    scan_cmd.add_argument(
        "--target",
        default=None,
        help="target path as bitstring or index (default: maximize the peak probability)",
    )
    scan_cmd.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="stop at the first omega whose score reaches this probability",
    )
    scan_cmd.add_argument("-b", "--backend", choices=BACKEND_NAMES, default="numpy")
    _add_figure_arguments(scan_cmd)

    circuit_cmd = commands.add_parser("circuit", parents=[common], help="print the Qiskit circuit")
    circuit_cmd.add_argument("-w", "--omega", type=float, default=None, help="bind omega (default: symbolic)")
    circuit_cmd.add_argument("--qasm", action="store_true", help="print OpenQASM 3 instead of a drawing")
    return parser


def _add_original_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--original", default=None, help="transmitted message as a bitstring, to check the decoding"
    )


def _add_figure_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plot", action="store_true", help="show a figure")
    parser.add_argument("--save", metavar="FILE", default=None, help="save the figure to FILE")


def _parse_target(value: str | None, num_qubits: int) -> str | int | None:
    """A bitstring of length N is a path; anything else is read as an integer index."""
    if value is None:
        return None
    if len(value) == num_qubits and set(value) <= {"0", "1"}:
        return value
    return int(value)


def _run_decode(args: argparse.Namespace, code: str) -> None:
    result = decode(
        code,
        omega=args.omega,
        iterations=args.iterations,
        backend=args.backend,
        shots=args.shots,
        seed=args.seed,
    )
    print(result.summary(original=args.original))
    print("\nMost probable paths:")
    for bits, probability in result.top_states(args.top):
        print(f"  {bits}  {probability:.4f}")

    if args.plot or args.save:
        from .plotting import plot_distribution, show_or_save

        ax = plot_distribution(result.probabilities, title=f"QVA output (omega = {result.omega:.4f} rad)")
        show_or_save(ax.figure, args.save)


def _run_viterbi(args: argparse.Namespace, code: str) -> None:
    print(viterbi_decode(code).summary(original=args.original))


def _run_scan(args: argparse.Namespace, code: str) -> None:
    num_qubits = len(parse_code(code))
    stop = args.stop if args.stop is not None else (180.0 if args.degrees else math.pi)
    grid = omega_range(args.start, stop, args.step, degrees=args.degrees)
    scan = scan_omega(
        code,
        grid,
        iterations=args.iterations,
        target=_parse_target(args.target, num_qubits),
        threshold=args.threshold,
        backend=args.backend,
    )

    if scan.target is None:
        score_name = "peak probability"
    else:
        score_name = f"P({bits_to_string(index_to_bits(scan.target, num_qubits))})"
    print(f"Evaluated {scan.omegas.size} of {grid.size} omega values ({scan.iterations} Grover iterations)")
    print(
        f"Best omega: {scan.best_omega:.6f} rad ({math.degrees(scan.best_omega):.4f} deg), "
        f"{score_name} = {scan.best_score:.4f}"
    )
    if scan.threshold is not None:
        status = "reached" if scan.threshold_reached else "NOT reached"
        print(f"Threshold {scan.threshold:.2f} {status}")

    if args.plot or args.save:
        from .plotting import plot_omega_scan, show_or_save

        ax = plot_omega_scan(scan, degrees=args.degrees)
        show_or_save(ax.figure, args.save)


def _run_circuit(args: argparse.Namespace, code: str) -> None:
    from .backends.qiskit_backend import QiskitBackend

    circuit = QiskitBackend().build_circuit(code, omega=args.omega, iterations=args.iterations)
    if args.qasm:
        from qiskit import qasm3

        print(qasm3.dumps(circuit))
    else:
        print(circuit.draw(output="text"))


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``quantum-viterbi`` command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    code = " ".join(args.code)
    handlers = {"decode": _run_decode, "viterbi": _run_viterbi, "scan": _run_scan, "circuit": _run_circuit}
    try:
        handlers[args.command](args, code)
    except (ValueError, ImportError) as error:
        parser.error(str(error))
    return 0
