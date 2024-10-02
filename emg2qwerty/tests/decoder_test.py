# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
from collections import OrderedDict
from pathlib import Path

import kenlm
import numpy as np
import pytest
import scipy.special
from hypothesis import given, strategies as st

from emg2qwerty.charset import CharacterSet, charset
from emg2qwerty.data import LabelData
from emg2qwerty.decoder import BeamState, CTCBeamDecoder, CTCGreedyDecoder, logsumexp


@given(xs=st.lists(st.floats(max_value=10, allow_infinity=True), min_size=1))
def test_logsumexp(xs: list[float]):
    actual = logsumexp(*xs)
    expected = scipy.special.logsumexp(xs)
    absdiff = abs(actual - expected)
    # Equality check is for the -inf case as absdiff will be NaN
    assert actual == expected or absdiff < 1e-6


@pytest.mark.parametrize(
    "_input, expected",
    [
        # \b for blank
        ("\b", ""),
        ("\b\b\b\b", ""),
        ("hello", "helo"),
        ("hh\b\beeell\bloo", "hello"),
    ],
)
def test_greedy_decoder(_input: str, expected: str):
    """Test CTCGreedyDecoder basics."""
    _charset = charset()
    decoder = CTCGreedyDecoder(_charset=_charset)

    T = len(_input)
    emissions = np.zeros((T, _charset.num_classes))
    for t, key in enumerate(_input):
        label = _charset.null_class if key == "\b" else _charset.key_to_label(key)
        emissions[t, label] = 1

    # Test `decode` API.
    decoding = decoder.decode(emissions=emissions, timestamps=np.arange(T))
    assert decoding.text == expected
    assert len(decoding.text) == len(decoding.timestamps)  # type: ignore

    # Test `decode_batch` API with single-item batch.
    decodings = decoder.decode_batch(
        emissions=np.expand_dims(emissions, axis=1),
        emission_lengths=np.array([T]),
    )
    assert len(decodings) == 1
    assert decodings[0] == decoding


@pytest.mark.parametrize(
    "batch",
    [
        # \b for blank
        [
            # (input, expected)
            ("\b", ""),
            ("\b\b\b\b", ""),
            ("hello", "helo"),
            ("hh\b\beeell\bloo", "hello"),
        ],
    ],
)
def test_greedy_decoder_batch(batch: list[tuple[str, str]]):
    """Test CTCGreedyDecoder.decode_batch() API for batched decoding."""
    _charset = charset()
    decoder = CTCGreedyDecoder(_charset=_charset)

    N = len(batch)
    inputs = [_input for _input, _ in batch]
    expected = [_expected for _, _expected in batch]
    emission_lengths = np.array([len(_input) for _input in inputs])
    T = emission_lengths.max()
    emissions = np.zeros((T, N, _charset.num_classes))

    for n, _input in enumerate(inputs):
        for t, key in enumerate(_input):
            label = _charset.null_class if key == "\b" else _charset.key_to_label(key)
            emissions[t, n, label] = 1

    decodings = decoder.decode_batch(emissions, emission_lengths)
    assert len(decodings) == N
    assert [decoding.text for decoding in decodings] == expected


def test_beamstate_no_lm():
    """Test BeamState without LM."""
    _charset = charset()
    sent = "the quick  brown"
    labels = _charset.str_to_labels(sent)

    blank_label = _charset.null_class
    state = BeamState.init(blank_label)

    assert state.label_node.is_root
    assert state.label_node.depth == 0

    assert state.label == blank_label
    assert len(state.decoding) == 0

    with pytest.raises(RuntimeError):
        _ = state.lm_state
    with pytest.raises(RuntimeError):
        _ = state.lm_states
    with pytest.raises(RuntimeError):
        _ = state.lm_score
    with pytest.raises(RuntimeError):
        _ = state.lm_scores

    # Test state extension with new labels
    for i, label in enumerate(labels):
        prev_state = state
        state = BeamState(label_node=prev_state.label_node.child((label, i)))

        assert not state.label_node.is_root
        assert state.label_node.parent == prev_state.label_node
        assert state.label_node.depth == i + 1

        assert state.label == label
        assert state.decoding == labels[: i + 1]
        assert state.timestamps == list(range(i + 1))

        hash_ = hashlib.sha256(bytes(state.decoding))
        assert state.hash().digest() == hash_.digest()
        assert state.hash().digest() == prev_state.hash(label).digest()


def test_beamstate_lm():
    """Test BeamState with LM."""
    _charset = charset()
    sent = "the quick  brown"
    keys = _charset.str_to_keys(sent)
    labels = _charset.str_to_labels(sent)

    blank_label = _charset.null_class
    lm_path = Path(__file__).parents[0].joinpath("reuters-3-gram-char-lm.arpa")
    lm = kenlm.Model(str(lm_path))
    state = BeamState.init(blank_label, lm=lm)

    assert state.label_node.is_root
    assert state.label_node.depth == 0
    assert state.lm_node is not None
    assert state.lm_node.is_root
    assert state.lm_node.depth == 0

    assert state.label == blank_label
    assert len(state.decoding) == 0
    assert len(state.lm_states) == 1
    assert len(state.lm_scores) == 1
    assert state.lm_score == 0.0

    # Test state extension with new labels
    for i, label in enumerate(labels):
        prev_state = state
        assert prev_state.lm_node is not None

        lm_state = kenlm.State()
        lm_score = lm.BaseScore(prev_state.lm_state, keys[i], lm_state)
        state = BeamState(
            label_node=prev_state.label_node.child((label, i)),
            lm_node=prev_state.lm_node.child((lm_state, lm_score)),
        )

        assert not state.label_node.is_root
        assert state.label_node.parent == prev_state.label_node
        assert state.label_node.depth == i + 1
        assert state.lm_node is not None
        assert not state.lm_node.is_root
        assert state.lm_node.parent == prev_state.lm_node
        assert state.lm_node.depth == i + 1

        assert state.label == label
        assert state.decoding == labels[: i + 1]
        assert state.timestamps == list(range(i + 1))
        assert len(state.lm_states) == i + 2
        assert len(state.lm_scores) == i + 2

        hash_ = hashlib.sha256(bytes(state.decoding))
        assert state.hash().digest() == hash_.digest()
        assert state.hash().digest() == prev_state.hash(label).digest()

    expected_lm_score = lm.score(" ".join(keys), bos=True, eos=False)
    assert abs(sum(state.lm_scores) - expected_lm_score) < 1e-4


def test_beamstate_multiple_paths():
    """Test `BeamState` and `TrieNode` operations when several paths
    lead to same decoding.

    Two paths leading to the same decoding prefix should match in
    everything (such as LM states and scores) except the onset timestamps."""
    classes = [(c, ord(c)) for c in ["a", "b"]]
    _charset = CharacterSet(_key_to_unicode=OrderedDict(classes))

    lm_path = Path(__file__).parents[0].joinpath("reuters-3-gram-char-lm.arpa")
    decoder = CTCBeamDecoder(_charset=_charset, lm_path=str(lm_path), delete_key=None)
    blank_label = _charset.null_class
    init_state = BeamState.init(blank_label, lm=decoder.lm)

    # Two paths leading to the same decoding "ab"
    path1 = "abbbbbb"
    path2 = "aaaaaab"

    def _get_final_beam_state(path: str) -> BeamState:
        labels = _charset.str_to_labels(path)

        prev_label = blank_label
        state = init_state
        for timestamp, label in enumerate(labels):
            if label != prev_label:
                state = decoder.next_state(
                    prev_state=state, label=label, timestamp=timestamp
                )
                prev_label = label

        return state

    state1 = _get_final_beam_state(path1)
    state2 = _get_final_beam_state(path2)

    # Decoded labels, LM states and scores should match
    assert state1.decoding == state2.decoding
    assert state1.lm_states == state2.lm_states
    assert state1.lm_scores == state2.lm_scores

    # Onset timestamps should differ
    assert state1.timestamps == [0, 1]
    assert state2.timestamps == [0, 6]


def test_timestamps():
    """Test that the decoder tracks onset timestamps correctly despite
    several paths leading to the same decoding.

    Consider the following two paths leading to the same decoding 'ab':

        Path 1:     a b b
        Path 2:     a a b
        Timestamps: 0 1 2

    The decoding trie in both cases encodes the same prefix 'ab' but differ in
    their timestamps - for token 'b', path 1 has an earlier onset
    timestamp (t=1) compared to path 2 (t=2). We test the following scenarios:

        Scenario 1:
            Path 1 has a much higher probability than path 2 which gets kicked
            out of the beam mid-way (say at t=3). Path 2 doesn't get a
            chance to reach the token 'b' and we should get the correct onset
            timestamp of 'b' (t=1) corresponding to path 1.
        Scenario 2:
            Inverse of scenario 1 - path 2 has a higher probability and path 1
            gets kicked out of the beam mid-way (say at t=1). The correct
            onset timestamp for 'b' should be t=2 corresponding to path 2.
    """
    classes = [(c, ord(c)) for c in ["a", "b"]]
    _charset = CharacterSet(_key_to_unicode=OrderedDict(classes))

    lm_path = Path(__file__).parents[0].joinpath("reuters-3-gram-char-lm.arpa")
    decoder = CTCBeamDecoder(_charset=_charset, lm_path=str(lm_path), delete_key=None)

    # Two paths leading to the same decoding 'ab'
    path1 = "abb"
    path2 = "aab"

    def _decode(paths: list[str], beam_size: int) -> LabelData:
        T = max(len(path) for path in paths)

        emissions = np.full((T, _charset.num_classes), fill_value=1e-10)
        for path in paths:
            labels = _charset.str_to_labels(path)
            emissions[range(T), labels] = 1.0
        emissions = np.log(emissions)

        decoder.beam_size = beam_size
        decoder.reset()
        return decoder.decode(emissions=emissions, timestamps=np.arange(T))

    # Scenario 1: path1 has higher prob and path2 gets kicked out.
    # Token 'b' should have onset timestamp corresponding to path1.
    decoding = _decode([path1], beam_size=1)
    assert decoding.text == "ab"
    assert list(decoding.timestamps) == [0, 1]  # type: ignore

    # Scenario 2: path2 has higher prob and path1 gets kicked out.
    # Token 'b' should have onset timestamp corresponding to path2.
    decoding = _decode([path2], beam_size=1)
    assert decoding.text == "ab"
    assert list(decoding.timestamps) == [0, 2]  # type: ignore


@given(
    num_deletes=st.integers(0, 20),
    lm_weight=st.sampled_from([0.0, 0.5, 1.0]),
    insertion_bonus=st.sampled_from([0.0, 0.5, 1.0]),
)
def test_lm_score(num_deletes: int, lm_weight: float, insertion_bonus: float):
    """Test CTCBeamDecoder delete handling. Total lm score for a word
    should be the same irrespective of deletes."""
    _charset = charset()
    sent = "the quick  brown"
    labels = _charset.str_to_labels(sent)

    lm_path = Path(__file__).parents[0].joinpath("reuters-3-gram-char-lm.arpa")
    lm = kenlm.Model(str(lm_path))

    # Expected LM score:
    # lm_weight * sum(lm score for each word) + insertion_bonus * len(sentence)
    words_score = 0.0
    words = sent.split(" ")
    for i, word in enumerate(words):
        if word:
            eos = i != (len(words) - 1)  # Last word doesn't end
            words_score += lm.score(" ".join(word), bos=True, eos=eos)
        else:
            # Out-of-vocab (OOV) unigram score for duplicate spaces
            words_score += lm.score("<unk>", False, False)
    expected_lm_score = lm_weight * words_score + insertion_bonus * len(sent)

    # Randomly insert typos and deletes without modifying the original sentence
    delete_key = "Key.backspace"
    delete_label = _charset.key_to_label(delete_key)
    for _ in range(num_deletes):
        pos = np.random.randint(0, len(labels) + 1)
        typo_label = np.random.randint(0, len(_charset))
        labels.insert(pos, typo_label)
        labels.insert(pos + 1, delete_label)

    # Compute LM score from decoder
    decoder = CTCBeamDecoder(
        _charset=_charset,
        lm_path=str(lm_path),
        lm_weight=lm_weight,
        insertion_bonus=insertion_bonus,
        delete_key=delete_key,
    )
    blank_label = _charset.null_class
    state = BeamState.init(blank_label, lm=decoder.lm)
    lm_score = 0.0
    for label in labels:
        next_state = decoder.next_state(prev_state=state, label=label)
        lm_score += decoder.lm_score(state, next_state)
        state = next_state
    assert abs(lm_score - expected_lm_score) < 1e-4

    # Walk through the LM trie and sum up individual LM scores along the
    # decoding path to assert that things add up.
    assert state.lm_node is not None
    lm_score = lm_weight * sum(state.lm_scores) + insertion_bonus * state.lm_node.depth
    assert abs(lm_score - expected_lm_score) < 1e-4
