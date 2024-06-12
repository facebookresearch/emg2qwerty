# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import string
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import ClassVar

import unidecode


UniChar = str  # Unicode character
KeyChar = str  # pynput keyboard.Key

_charset: CharacterSet | None = None  # Global instance of CharacterSet


def charset() -> CharacterSet:
    """Lazily load and return a global instance of ``CharacterSet``."""
    global _charset
    if _charset is None:
        _charset = CharacterSet()
    return _charset


@dataclass
class CharacterSet:
    """Encapsulate the supported character set with conversion to/from
    unicode, human-readable strings, pynput Key objects, and class labels.

    Current representations supported:
    - Unicode string:
        The canonical internal string representation. This is advantageous in
        being able to maintain a 1:1 representation to keypresses
        and unicode characters. While not directly printable or human-readable,
        this enables a simple interface for our modeling code.
    - Human-readable string (with icons):
        Special characters such as backspace and enter are converted to icons
        for human readability.
    - pynput keys:
        Special characters such as shift, backspace, and enter are
        represented by their corresponding pynput Key class. See
        https://pynput.readthedocs.io/en/latest/keyboard.html#pynput.keyboard.Key  # noqa
    - Class labels:
        The string is represented via a sequence of categorical class labels
        for model training. The null class label can be obtained via ``null_class``
        (such as for the blank token label in CTC).

    Args:
        key_to_unicode (OrderedDict): OrderedDict of valid/supported characters
            and pynput keys to their corresponding unicode values. Order
            matters for class label generation. (default: ``KEY_TO_UNICODE``)
    """

    # Tuples of supported chars and corresponding unicode values.
    CHAR_TO_UNICODE: ClassVar[list[tuple[UniChar, int]]] = [
        (c, ord(c)) for c in string.ascii_letters + string.digits + string.punctuation
    ]

    # Tuples of supported modifier keys (in pynput representation) and
    # corresponding unicode values.
    # https://wincent.com/wiki/Unicode_representations_of_modifier_keys.
    MODIFIER_TO_UNICODE: ClassVar[list[tuple[KeyChar, int]]] = [
        ("Key.backspace", 9003),  # ⌫
        ("Key.enter", 9166),  # ⏎
        ("Key.space", 32),
        ("Key.shift", 8679),  # ⇧
    ]

    # Map of supported characters/keys to unicode values.
    # NOTE: The order matters for class label generation.
    KEY_TO_UNICODE: ClassVar[OrderedDict] = OrderedDict(
        [
            *CHAR_TO_UNICODE,
            *MODIFIER_TO_UNICODE,
        ]
    )

    # Map of unicode chars to pynput key representations.
    UNICHAR_TO_KEY: ClassVar[Mapping[UniChar, KeyChar]] = {
        " ": "Key.space",
        "\r": "Key.enter",
        "\u21E5": "Key.tab",  # ⇥
        "\u21E7": "Key.shift",  # ⇧
        "\u2303": "Key.ctrl",  # ⌃
        "\u2318": "Key.cmd",  # ⌘
        "\u232B": "Key.backspace",  # ⌫
        "\u23CE": "Key.enter",  # ⏎
        "\u2191": "Key.shift_l",  # ↑ -- straight
        "\u21E1": "Key.shift_r",  # ⇡ -- dotted
    }

    # Map of unsupported chars to their equivalent supported counterparts.
    CHAR_SUBSTITUTIONS: ClassVar[Mapping[UniChar, UniChar]] = {
        "\n": "⏎",
        "\r": "⏎",
        "\b": "⌫",  # same as \x08
        "’": "'",
        "“": '"',
        "”": '"',
        "—": "-",
    }

    _key_to_unicode: OrderedDict = field(
        default_factory=lambda: CharacterSet.KEY_TO_UNICODE
    )

    def __post_init__(self) -> None:
        self._unicode_to_key = {v: k for k, v in self._key_to_unicode.items()}

    def __len__(self) -> int:
        return len(self._key_to_unicode)

    def __contains__(self, item: KeyChar | int) -> bool:
        if isinstance(item, KeyChar):
            return item in self._key_to_unicode
        if isinstance(item, int):
            return item in self._unicode_to_key
        raise ValueError(f"Unexpected type: {type(item)}")

    @property
    def null_class(self) -> int:
        """Categorical label of the null-class (blank label)."""
        return len(self)

    @property
    def num_classes(self) -> int:
        """Number of training classes including null-class (blank label)."""
        return len(self) + 1

    @property
    def allowed_keys(self) -> tuple[KeyChar, ...]:
        """Sequence of allowed keys, order respected."""
        return tuple(self._key_to_unicode.keys())

    @property
    def allowed_unicodes(self) -> tuple[int, ...]:
        """Sequence of allowed unicode values, order respected."""
        return tuple(self._key_to_unicode.values())

    @property
    def allowed_chars(self) -> tuple[UniChar, ...]:
        """Sequence of allowed chars, order respected."""
        return tuple(self.unicode_to_char(key) for key in self.allowed_unicodes)

    def key_to_unicode(self, key: KeyChar) -> int:
        """Fetch the unicode value corresponding to the given key."""
        return self._key_to_unicode[key]  # type: ignore[no-any-return]

    def unicode_to_key(self, unicode_val: int) -> KeyChar:
        """Fetch the key corresponding to the given unicode value."""
        return self._unicode_to_key[unicode_val]  # type: ignore[no-any-return]

    def key_to_label(self, key: KeyChar) -> int:
        """Fetch the categorical label corresponding to the given key."""
        return self.allowed_keys.index(key)

    def label_to_key(self, label: int) -> KeyChar:
        """Fetch the key corresponding to the given categorical label."""
        return self.allowed_keys[label]

    def unicode_to_label(self, unicode_val: int) -> int:
        """Fetch the categorical label for the given unicode value."""
        return self.allowed_unicodes.index(unicode_val)

    def label_to_unicode(self, label: int) -> int:
        """Fetch the unicode value for the given categorical label."""
        return self.allowed_unicodes[label]

    def str_to_keys(self, unicode_str: str) -> list[KeyChar]:
        r"""Convert a string to the corresponding sequence of supported keys
        after cleaning up and standardizing.

        Supports conversion from both unicode chars (such as "⏎⇧⌫") and
        printable character strings (such as '\n', '\r', '\x08', ' ').

        Example:
        ``unicode_str`` = ``"the\x08⏎\n"``
        returns ``['t', 'h', 'e', 'Key.backspace', 'Key.enter', 'Key.enter']``
        """
        keys = list(self._normalize_str(unicode_str))
        return self.clean_keys(keys)

    def keys_to_str(self, keys: Sequence[KeyChar]) -> str:
        """Convert a sequence of keys to its corresponding textual
        representation after standardizing."""
        unicode_str = "".join(chr(self.key_to_unicode(key)) for key in keys)
        return self._normalize_str(unicode_str)

    def str_to_labels(self, unicode_str: str) -> list[int]:
        """Convert a string to the corresponding sequence of labels after
        cleaning up and stardardizing. Also see ``str_to_keys()``."""
        keys = self.str_to_keys(unicode_str)
        return [self.key_to_label(key) for key in keys]

    def labels_to_str(self, labels: Sequence[int]) -> str:
        """Convert a sequence of labels to its corresponding textual
        representation."""
        keys = [self.label_to_key(label) for label in labels]
        return self.keys_to_str(keys)

    def key_to_char(self, key: KeyChar) -> UniChar:
        """Convert a single key to its corresponding textual representation
        after standardizing."""
        return self.unicode_to_char(self.key_to_unicode(key))

    def unicode_to_char(self, unicode_val: int) -> UniChar:
        """Convert a unicode value to its corresponding textual representation
        after standardizing."""
        return self._normalize_str(chr(unicode_val))

    def label_to_char(self, label: int) -> UniChar:
        """Convert a single label to its corresponding textual representation
        after standardizing."""
        return self.key_to_char(self.label_to_key(label))

    def clean_keys(self, keys: Sequence[KeyChar]) -> list[KeyChar]:
        """Normalize and filter the given sequence of keys in any
        representation. Every single key returned is guaranteed to be a
        member of ``allowed_keys`` by standardizing supported ones and
        filtering out unsupported ones."""
        keys = self._normalize_keys(keys)  # Normalize
        return [key for key in keys if key in self]  # Filter

    def clean_str(self, unicode_str: str) -> str:
        """Return a normalized AND filtered canonical string corresponding
        to the ``unicode_str`` such that the characters of the returned string
        are guaranteed to be part of the character set.

        In addition to ``_normalize_str()``, this filters out
        out-of-vocabulary characters that can't be mapped into the
        character set."""
        # Normalize input str and convert to keys
        keys = list(self._normalize_str(unicode_str))
        # Normalize keys and filter out those not in charset
        keys = self.clean_keys(keys)
        # Convert back to str and return
        return self.keys_to_str(keys)

    def _normalize_keys(self, keys: Sequence[KeyChar]) -> list[KeyChar]:
        """Normalize the given sequence of keys in any representation.

        NOTE: This doesn NOT filter out out-of-vocabulary keys, directly call
        `clean_keys()` to perform both normalization and filtering."""

        def _normalize_key(key: KeyChar) -> KeyChar:
            if key in self:  # Already normalized
                return key

            if len(key) == 1:
                # Treat unsupported key chars as raw strings. The additional
                # len(key) == 1 clause is to exclude unsupported modifier keys
                # in pynput.Key format such as 'Key.tab'.
                key = self._normalize_str(key)
                key = self.UNICHAR_TO_KEY.get(key, key)

            return key

        return [_normalize_key(key) for key in keys]

    def _normalize_str(self, unicode_str: str) -> str:
        """Return a normalized string by substituting unsupported characters
        with supported ones while leaving unicode characters corresponding
        to modifier keys as is.

        NOTE: This does NOT filter out out-of-vocabulary characters that can't
        be mapped into the character set. Use ``clean_str()`` for that."""
        normalized_str = unicode_str

        # Apply known substitutions
        for k, v in self.CHAR_SUBSTITUTIONS.items():
            normalized_str = normalized_str.replace(k, v)

        def _spurious_char(c: UniChar) -> bool:
            return c not in self and c not in self.UNICHAR_TO_KEY

        # Handle any remaining spurious unicode chars
        unidecode_map = {}
        for c in normalized_str:
            if not _spurious_char(c):
                continue

            c_ = unidecode.unidecode(c)
            if c_ != c and len(c_) == 1 and not _spurious_char(c_):
                unidecode_map[c] = c_

        # Apply unidecode substitutions
        for k, v in unidecode_map.items():
            normalized_str = normalized_str.replace(k, v)

        return normalized_str

    def __str__(self) -> str:
        return self.keys_to_str(self.allowed_keys)
