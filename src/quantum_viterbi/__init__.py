"""Quantum Viterbi Algorithm (QVA).

Grover-based decoding of the rate-1/2 (5, 7) convolutional code. The quantum
circuit can be executed with Qiskit, PennyLane or a fast NumPy simulator, and a
classical Viterbi decoder is included to compare the two approaches.

Quick start::

    from quantum_viterbi import decode, viterbi_decode

    code = "11 01 00 10 00 00"
    print(viterbi_decode(code).summary())  # classical Viterbi algorithm
    print(decode(code).summary())  # Quantum Viterbi Algorithm, omega tuned for this word
"""

from .backends import Backend, available_backends, get_backend
from .convolutional import (
    ViterbiResult,
    encode,
    maximum_likelihood_states,
    parse_code,
    path_metrics,
    viterbi_decode,
)
from .decoder import DecodingResult, decode
from .grover import default_iterations, qva_gates
from .oracle import oracle_gates
from .spectrum import class_probabilities, distance_spectrum, distance_weights
from .tuning import (
    OmegaScan,
    Resonance,
    channel_omega,
    omega_range,
    resonant_omega,
    scan_omega,
    scan_resonance,
    tuned_omega,
)

__version__ = "0.1.0"

__all__ = [
    "Backend",
    "DecodingResult",
    "OmegaScan",
    "Resonance",
    "ViterbiResult",
    "available_backends",
    "channel_omega",
    "class_probabilities",
    "decode",
    "default_iterations",
    "distance_spectrum",
    "distance_weights",
    "encode",
    "get_backend",
    "maximum_likelihood_states",
    "omega_range",
    "oracle_gates",
    "parse_code",
    "path_metrics",
    "qva_gates",
    "resonant_omega",
    "scan_omega",
    "scan_resonance",
    "tuned_omega",
    "viterbi_decode",
]
