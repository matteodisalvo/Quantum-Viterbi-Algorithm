"""Print the QVA circuit and run it on every installed framework.

The circuit is built once as a framework-independent gate list and translated to Qiskit
and PennyLane. The script prints the Qiskit drawing (or OpenQASM 3) for a small number of
Grover rounds, so that it stays readable, and then compares the probability of the most
likely path across backends, with exact probabilities and with a finite number of
measurement shots. The comparison runs the whole algorithm, at the default
floor(pi/4 * sqrt(2**N)) rounds, so it does not use the truncated circuit that is drawn.

    python examples/03_circuits.py
    python examples/03_circuits.py --code "11 01 00 10" --omega 0.6 --qasm
"""

from __future__ import annotations

import argparse

from quantum_viterbi import available_backends, decode, default_iterations, parse_code
from quantum_viterbi.backends.qiskit_backend import QiskitBackend
from quantum_viterbi.convolutional import bits_to_string

RECEIVED_CODE = "11 01 00"
OMEGA = 0.5
SHOTS = 4096


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--code", default=RECEIVED_CODE, help="received word")
    parser.add_argument("--omega", type=float, default=OMEGA, help="metric parameter in radians")
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Grover rounds in the drawing only; the comparison always runs the default number",
    )
    parser.add_argument("--shots", type=int, default=SHOTS, help="measurement shots")
    parser.add_argument("--qasm", action="store_true", help="print OpenQASM 3 instead of a drawing")
    args = parser.parse_args()

    circuit = QiskitBackend().build_circuit(args.code, omega=args.omega, iterations=args.iterations)
    if args.qasm:
        from qiskit import qasm3

        print(qasm3.dumps(circuit))
    else:
        print(circuit.draw(output="text"))

    rounds = default_iterations(len(parse_code(args.code)))
    print(
        f"\nReceived word {args.code}, omega {args.omega:.4f} rad, {rounds} Grover rounds"
        f"  (the drawing above shows {args.iterations})"
    )
    for name in available_backends():
        exact = decode(args.code, omega=args.omega, backend=name)
        sampled = decode(args.code, omega=args.omega, backend=name, shots=args.shots, seed=7)
        print(
            f"  {name:9s} exact {bits_to_string(exact.decoded_bits)} P = {exact.best_probability:.4f}"
            f"   |   {args.shots} shots {bits_to_string(sampled.decoded_bits)}"
            f" P = {sampled.best_probability:.4f}"
        )


if __name__ == "__main__":
    main()
