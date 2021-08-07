import string
from typing import List, Tuple

from emg2qwerty.charset import charset


def test_clean_str():
    _charset = charset()
    test_samples = [
        ("", ""),
        ("aa", "aa"),
        ("\n\r\b\x08", "⏎⏎⌫⌫"),
        ("⏎\n⇧⌫\b", "⏎⏎⇧⌫⌫"),
        ("⏎\n⇧⌫�\b", "⏎⏎⇧⌫⌫"),
        ("’“”—", '\'""-'),
    ]

    for _input, expected in test_samples:
        assert _charset.clean_str(_input) == expected


def test_str_to_keys():
    _charset = charset()
    test_samples = [
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
    ]

    for _input, expected in test_samples:
        assert _charset.str_to_keys(_input) == expected


def test_keys_to_str():
    _charset = charset()
    test_samples: List[Tuple[List[str], str]] = [
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
    ]

    for _input, expected in test_samples:
        assert _charset.keys_to_str(_input) == expected


def test_str_to_labels():
    _charset = charset()
    test_samples = [
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
    ]

    for input_str in test_samples:
        labels = _charset.str_to_labels(input_str)
        assert _charset.labels_to_str(labels) == _charset.clean_str(input_str)
