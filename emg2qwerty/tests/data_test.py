# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import pytest

from emg2qwerty.data import LabelData


@pytest.mark.parametrize(
    "text, labels",
    [
        ("aa", [0, 0]),
        ("ab", [0, 1]),
        ("", []),
        ("\n\r\b\x08", [95, 95, 94, 94]),
        ("⏎\na⌫\b", [95, 95, 0, 94, 94]),
        ("’“”—", [68, 63, 63, 74]),
    ],
)
def test_label_data(text: str, labels: list[int]):
    label_data1 = LabelData.from_str(text)
    label_data2 = LabelData.from_labels(labels)

    assert label_data1 == label_data2
    assert label_data1.text == label_data2.text
    assert list(label_data1.labels) == labels
    assert list(label_data2.labels) == labels

    assert LabelData.from_str(text + text) == label_data1 + label_data2
    assert LabelData.from_labels(labels + labels) == label_data1 + label_data2
