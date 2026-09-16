import numpy as np
import pytest

from quantum_viterbi.convolutional import (
    bits_to_index,
    encode,
    format_code,
    hamming_distance,
    index_to_bits,
    maximum_likelihood_states,
    parse_code,
    path_metrics,
    resolve_state,
)


def test_parse_code_accepts_all_formats():
    expected = ((1, 1), (0, 1), (0, 0))
    assert parse_code([1, 1, 0, 1, 0, 0]) == expected
    assert parse_code([(1, 1), (0, 1), (0, 0)]) == expected
    assert parse_code("11 01 00") == expected
    assert parse_code("[1 1, 0 1, 0 0]") == expected
    assert parse_code(np.array([1, 1, 0, 1, 0, 0])) == expected
    assert parse_code(expected) == expected


@pytest.mark.parametrize("bad_code", ["", "110", "11 02", [1, 2], [1, 0, 1]])
def test_parse_code_rejects_invalid_input(bad_code):
    with pytest.raises(ValueError):
        parse_code(bad_code)


def test_encode_matches_hand_computed_codeword():
    # u = 1 0 1 1 -> 11 01 00 10
    assert encode([1, 0, 1, 1]) == (1, 1, 0, 1, 0, 0, 1, 0)
    assert format_code(encode([1, 0, 1, 1])) == "11 01 00 10"


def test_path_metrics_agree_with_encoder():
    rng = np.random.default_rng(1)
    for n in range(1, 8):
        code = rng.integers(0, 2, 2 * n)
        expected = [hamming_distance(encode(index_to_bits(k, n)), code) for k in range(2**n)]
        np.testing.assert_array_equal(path_metrics(code), expected)


def test_noiseless_codeword_is_the_unique_maximum_likelihood_path():
    bits = (1, 0, 1, 1, 0, 1)
    code = encode(bits)
    assert path_metrics(code)[bits_to_index(bits)] == 0
    np.testing.assert_array_equal(maximum_likelihood_states(code), [bits_to_index(bits)])


def test_state_conversions_round_trip():
    for index in range(16):
        bits = index_to_bits(index, 4)
        assert bits_to_index(bits) == index
        assert resolve_state("".join(map(str, bits)), 4) == index
        assert resolve_state(bits, 4) == index
        assert resolve_state(index, 4) == index
    # Most significant bit first
    assert index_to_bits(6, 4) == (0, 1, 1, 0)
    with pytest.raises(ValueError):
        resolve_state(16, 4)
    with pytest.raises(ValueError):
        resolve_state("101", 4)
