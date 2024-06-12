# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Script to preprocess WikiText-103 raw character-level text dataset and output
to stdout in a format suitable to feed into kenlm's lmplz binary for building a
character-level n-gram language model.

LM vocabulary is limited to lower-case alphabets only, and the
generated n-grams do not span word boundaries.
"""

import click
import datasets
import nltk

from emg2qwerty.charset import charset


LM_VOCABULARY = {c for c in charset().allowed_chars if c.isalpha()}


def word_in_vocabulary(word: str) -> bool:
    return all(c in LM_VOCABULARY for c in word)


def process_word(word: str) -> None:
    word = word.lower()
    if word_in_vocabulary(word):
        print(" ".join(word))


def process_line(line: str) -> None:
    for word in nltk.word_tokenize(line):
        process_word(word)


@click.command()
def main():
    # Download NLTK Punkt Tokenizer
    nltk.download("punkt")

    # Load WikiText-103 raw character level dataset:
    # https://huggingface.co/datasets/Salesforce/wikitext and
    # https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/.
    wikitext = datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split="all")

    # Preprocess and output to stdout one word per line for kenlm's lmplz binary
    # to consume. Each line is a single word from the WikiText-103 raw dataset
    # after applying NLTK Punkt word tokenizer, lower-casing, and filtering out words
    # containing characters not in the LM vocabulary. The output words have characters
    # delimited by space so that a character-level n-gram LM can be constructed with it.
    for line in wikitext["text"]:
        process_line(line)


if __name__ == "__main__":
    main()
