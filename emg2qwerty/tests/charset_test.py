# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import string

import pytest

from emg2qwerty.charset import charset, KeyChar


@pytest.mark.parametrize(
    "_input, expected",
    [
        ("", ""),
        ("aa", "aa"),
        ("\n\r\b\x08", "⏎⏎⌫⌫"),
        ("⏎\n⇧⌫\b", "⏎⏎⇧⌫⌫"),
        ("⏎\n⇧⌫�\b", "⏎⏎⇧⌫⌫"),
        ("’“”—", '\'""-'),
    ],
)
def test_clean_str(_input: str, expected: str):
    assert charset().clean_str(_input) == expected


@pytest.mark.parametrize(
    "_input, expected",
    [
        ("", []),
        (string.ascii_lowercase, list(string.ascii_lowercase)),
        (string.ascii_uppercase, list(string.ascii_uppercase)),
        (string.punctuation, list(string.punctuation)),
        (
            "\x08⌫⏎\n\r \x20⇧",
            [
                "Key.backspace",
                "Key.backspace",
                "Key.enter",
                "Key.enter",
                "Key.enter",
                "Key.space",
                "Key.space",
                "Key.shift",
            ],
        ),
    ],
)
def test_str_to_keys(_input: str, expected: list[KeyChar]):
    assert charset().str_to_keys(_input) == expected


@pytest.mark.parametrize(
    "_input, expected",
    [
        ([], ""),
        (list(string.ascii_lowercase), string.ascii_lowercase),
        (list(string.ascii_uppercase), string.ascii_uppercase),
        (list(string.punctuation), string.punctuation),
        (
            [
                "Key.backspace",
                "Key.backspace",
                "Key.enter",
                "Key.enter",
                "Key.enter",
                "Key.space",
                "Key.space",
                "Key.shift",
            ],
            "⌫⌫⏎⏎⏎  ⇧",
        ),
    ],
)
def test_keys_to_str(_input: list[KeyChar], expected: str):
    assert charset().keys_to_str(_input) == expected


@pytest.mark.parametrize(
    "_input",
    [
        "",
        string.ascii_lowercase,
        string.ascii_uppercase,
        string.punctuation,
        "\x08⌫⏎\n\r \x20⇧",
        "aa",
        "\n\r\b\x08",
        "⏎\n⇧⌫\b",
        "⏎\n⇧⌫�\b",
        "’“”—",
    ],
)
def test_str_to_labels(_input: str):
    labels = charset().str_to_labels(_input)
    assert charset().labels_to_str(labels) == charset().clean_str(_input)
