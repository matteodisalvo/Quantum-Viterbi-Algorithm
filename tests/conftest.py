"""Shared test configuration."""

import os

# Render figures off-screen during the tests
os.environ.setdefault("MPLBACKEND", "Agg")
