"""Simulation backends for the QVA.

=============  ===========================================================
name           description
=============  ===========================================================
``qiskit``     Qiskit circuit: exact ``Statevector`` or any Sampler V2
``pennylane``  PennyLane QNode on any PennyLane device
``numpy``      fast exact simulator exploiting the diagonal phase oracle
=============  ===========================================================

Backends are imported lazily, so PennyLane is only needed when it is used.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

from .base import Backend, counts_to_probabilities

# name -> (module, class, framework required by the backend)
_REGISTRY: dict[str, tuple[str, str, str]] = {
    "qiskit": (".qiskit_backend", "QiskitBackend", "qiskit"),
    "pennylane": (".pennylane_backend", "PennyLaneBackend", "pennylane"),
    "numpy": (".numpy_backend", "NumpyBackend", "numpy"),
}

#: Names accepted by :func:`get_backend`.
BACKEND_NAMES: tuple[str, ...] = tuple(_REGISTRY)

#: Backend used by :func:`quantum_viterbi.decode` when none is given.
DEFAULT_BACKEND = "qiskit"

__all__ = [
    "BACKEND_NAMES",
    "DEFAULT_BACKEND",
    "Backend",
    "available_backends",
    "counts_to_probabilities",
    "get_backend",
]


def available_backends() -> list[str]:
    """Names of the backends whose framework is installed."""
    return [
        name for name, (_, _, package) in _REGISTRY.items() if importlib.util.find_spec(package) is not None
    ]


def get_backend(backend: str | Backend = DEFAULT_BACKEND, **options: Any) -> Backend:
    """Return a backend instance.

    Args:
        backend: backend name (``"qiskit"``, ``"pennylane"`` or ``"numpy"``) or an
            already configured :class:`Backend`, which is returned unchanged.
        **options: keyword arguments for the backend constructor, e.g.
            ``get_backend("pennylane", device="lightning.qubit")``.
    """
    if isinstance(backend, Backend):
        if options:
            raise ValueError("Options can only be given together with a backend name.")
        return backend

    key = str(backend).lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown backend {backend!r}; choose one of {', '.join(BACKEND_NAMES)}.")

    module_name, class_name, package = _REGISTRY[key]
    try:
        module = importlib.import_module(module_name, __name__)
    except ModuleNotFoundError as error:
        if error.name and error.name.split(".")[0] == package:
            raise ImportError(
                f"The {key!r} backend requires {package}. Install it with: pip install {package}"
            ) from error
        raise
    return getattr(module, class_name)(**options)
