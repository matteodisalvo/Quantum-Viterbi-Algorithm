"""Allow running the command-line interface with ``python -m quantum_viterbi``."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
