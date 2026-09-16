import math

import numpy as np
import pytest

from quantum_viterbi import scan_omega
from quantum_viterbi.cli import main


def test_decode_command(capsys):
    assert main(["decode", "11", "01", "00", "10", "--omega", "0.6", "--backend", "numpy"]) == 0
    assert "decoded message" in capsys.readouterr().out


def test_decode_command_options_are_used(capsys):
    assert (
        main(
            ["decode", "11 01 00 10 00 00", "--original", "101101", "--backend", "numpy"]
            + ["--top", "3", "--shots", "4096", "--seed", "7"]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "original message  : 101101" in out
    assert "numpy (4096 shots)" in out
    assert len(out.split("Most probable paths:")[1].strip().splitlines()) == 3


def test_cli_rejects_a_non_positive_top():
    with pytest.raises(SystemExit):
        main(["decode", "11 01 00 10", "--backend", "numpy", "--top", "-1"])


def test_scan_command(capsys):
    assert main(["scan", "11 01 00 10", "--step", "0.1", "--target", "1011", "--threshold", "0.5"]) == 0
    assert "Best omega" in capsys.readouterr().out


def test_scan_defaults_to_half_a_period(capsys):
    # Without --stop the grid covers [0, 180] degrees: beyond pi the curve only mirrors
    assert main(["scan", "11 01 00 10 00 00", "--target", "101101", "--step", "0.05", "--degrees"]) == 0
    out = capsys.readouterr().out
    assert "Evaluated 3601 of 3601" in out
    best = float(out.split("Best omega: ")[1].split()[0])
    assert 0 < best <= math.pi
    assert best == pytest.approx(0.6737, abs=5e-4)


def test_circuit_command_prints_qasm(capsys):
    pytest.importorskip("qiskit")
    assert main(["circuit", "11", "01", "--iterations", "1", "--qasm"]) == 0
    assert "OPENQASM 3" in capsys.readouterr().out


def test_invalid_code_is_reported():
    with pytest.raises(SystemExit):
        main(["decode", "1", "2", "--backend", "numpy"])


def _probabilities():
    return np.random.default_rng(0).dirichlet(np.ones(16))


def test_plot_distribution_draws_the_probabilities_with_big_endian_labels():
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    from quantum_viterbi.plotting import BASE_COLOR, HIGHLIGHT_COLOR, plot_distribution

    probabilities = _probabilities()
    ax = plot_distribution(probabilities, highlight="1011")
    np.testing.assert_allclose([patch.get_height() for patch in ax.patches], probabilities)

    colors = [patch.get_facecolor() for patch in ax.patches]
    assert [i for i, c in enumerate(colors) if c == to_rgba(HIGHLIGHT_COLOR)] == [0b1011]
    assert colors[0] == to_rgba(BASE_COLOR)

    ticks = [int(t) for t in ax.get_xticks()]
    assert [t.get_text() for t in ax.get_xticklabels()] == [format(t, "04b") for t in ticks]
    assert 0b1011 in ticks  # the highlighted state is always labeled
    plt.close("all")


def test_plot_distribution_annotates_the_probability_of_the_marked_state():
    import matplotlib.pyplot as plt

    from quantum_viterbi.plotting import plot_distribution

    probabilities = _probabilities()
    ax = plot_distribution(probabilities)  # highlight defaults to the most probable state
    assert f" P = {probabilities.max():.3f}" in [text.get_text() for text in ax.texts]

    dense = plot_distribution(np.full(64, 1 / 64), highlight=0, tick_step=16)
    assert [int(t) for t in dense.get_xticks()] == [0, 16, 32, 48]
    plt.close("all")


def test_plot_top_states_ranks_by_decreasing_probability():
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    from quantum_viterbi.plotting import HIGHLIGHT_COLOR, plot_top_states

    probabilities = _probabilities()
    ax = plot_top_states(probabilities, k=5)
    order = np.argsort(-probabilities, kind="stable")[:5]
    np.testing.assert_allclose([patch.get_height() for patch in ax.patches], probabilities[order])
    assert [t.get_text() for t in ax.get_xticklabels()] == [format(int(i), "04b") for i in order]
    assert ax.get_title() == "Top 5 paths"
    colors = [patch.get_facecolor() for patch in ax.patches]
    assert [i for i, c in enumerate(colors) if c == to_rgba(HIGHLIGHT_COLOR)] == [0]
    plt.close("all")


def test_plot_omega_scan_plots_the_scores():
    import matplotlib.pyplot as plt

    from quantum_viterbi.plotting import plot_omega_scan

    scan = scan_omega("11 01 00 10", np.linspace(0.1, 3, 20))
    x, y = plot_omega_scan(scan).get_lines()[0].get_data()
    np.testing.assert_allclose(x, scan.omegas)
    np.testing.assert_allclose(y, scan.scores)

    in_degrees = plot_omega_scan(scan, degrees=True).get_lines()[0].get_data()[0]
    np.testing.assert_allclose(in_degrees, np.rad2deg(scan.omegas))
    plt.close("all")


def test_plots_reject_bad_input():
    from quantum_viterbi.plotting import plot_distribution, plot_top_states

    with pytest.raises(ValueError, match="one-dimensional probability vector"):
        plot_distribution(np.ones(10) / 10)  # not a power of two
    with pytest.raises(ValueError, match="one-dimensional probability vector"):
        plot_distribution(np.ones((2, 8)) / 16)
    with pytest.raises(ValueError, match="k must be a positive integer"):
        plot_top_states(np.full(16, 1 / 16), k=0)


def test_save_writes_a_figure(tmp_path):
    import matplotlib.pyplot as plt

    target = tmp_path / "scan.png"
    assert main(["scan", "11 01 00 10", "--step", "0.2", "--save", str(target)]) == 0
    assert target.is_file() and target.stat().st_size > 0
    plt.close("all")
