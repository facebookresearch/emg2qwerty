# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Script to preprocess a raw text corpus before feeding to kenlm for
# building a character-level n-gram language model using lmplz.
#
# Vocabulary is limited to lower-case alphabets only, and the
# generated n-grams do not span word boundaries.

import sys

import click
import nltk

from emg2qwerty.charset import charset


LM_VOCABULARY = set(
    [
        charset().unicode_to_key(c)
        for c in charset().allowed_unicodes
        if chr(c).isalpha()
    ]
)


def word_in_vocabulary(word: str) -> bool:
    return all([c in LM_VOCABULARY for c in word])


def process_word(word: str) -> None:
    word = word.lower()
    if word_in_vocabulary(word):
        print(" ".join(word))


@click.command()
def main():
    # Read raw text corpus from stdin
    for line in sys.stdin:
        for word in nltk.word_tokenize(line):
            process_word(word)


if __name__ == "__main__":
    main()
