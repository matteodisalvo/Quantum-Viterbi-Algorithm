import numpy as np

from quantum_viterbi import decode, encode, path_metrics, viterbi_decode
from quantum_viterbi.cli import main
from quantum_viterbi.convolutional import bits_to_index


def test_viterbi_returns_a_minimum_distance_path():
    rng = np.random.default_rng(7)
    for n in range(1, 11):
        for _ in range(20):
            code = rng.integers(0, 2, 2 * n)
            result = viterbi_decode(code)
            metrics = path_metrics(code)
            assert len(result.bits) == n
            assert result.metric == metrics.min()
            assert metrics[bits_to_index(result.bits)] == result.metric


def test_viterbi_corrects_isolated_errors_on_a_long_message():
    rng = np.random.default_rng(3)
    bits = tuple(int(bit) for bit in rng.integers(0, 2, 200))
    codeword = np.array(encode(bits))
    assert viterbi_decode(codeword).bits == bits

    corrupted = codeword.copy()
    corrupted[[10, 57, 120, 181]] ^= 1  # errors far from each other and from the end
    result = viterbi_decode(corrupted)
    assert result.bits == bits
    assert result.metric == 4


def test_decoding_summary_reports_the_classical_result():
    result = decode("11 01 00 10 00 00", omega=0.6733, backend="numpy")
    assert result.classical.metric == 1
    assert result.hamming_distance == result.classical.metric
    assert "classical Viterbi" in result.summary()


def test_viterbi_summary_and_command_line(capsys):
    result = viterbi_decode("11 01 00 10 00 00")
    assert result.reencoded == (1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0)
    assert "decoded message   : 101101" in result.summary()

    assert main(["viterbi", "11", "01", "00", "10", "00", "00"]) == 0
    output = capsys.readouterr().out
    assert "Classical Viterbi decoding" in output
    assert "bits flipped      : 1" in output


def test_summary_with_the_original_message(capsys):
    received = "11 01 00 10 00 00"  # codeword of 101101 with one flipped bit
    report = decode(received, omega=0.6733, backend="numpy").summary(original="101101")
    assert "original codeword : 11 01 00 10 10 00" in report
    assert "received word     : 11 01 00 10 00 00  (1 bit error)" in report
    assert "decoded codeword  : 11 01 00 10 10 00" in report
    assert "classical Viterbi : 101101  (same as QVA)" in report
    assert "outcome           : correct" in report

    assert "outcome           : wrong (2 of 6 message bits differ)" in (
        viterbi_decode(received).summary(original="001001")
    )
    assert "original" not in viterbi_decode(received).summary()

    assert main(["viterbi", received, "--original", "101101"]) == 0
    assert "outcome           : correct" in capsys.readouterr().out
