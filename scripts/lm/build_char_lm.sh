#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Script to build a character-level n-gram language model using kenlm
# from WikiText-103 raw corpus.
#
# Vocabulary is limited to lower-case alphabets only, and the
# generated n-grams do not span word boundaries.
#
# Dependencies: Download kenlm (https://github.com/kpu/kenlm) to ~/kenlm
# and build from source.
#
# Usage: ./build_char_lm.sh <NGRAM_ORDER>

set -e

if [ $# -lt 1 ]; then
  echo "Usage: ./build_char_lm.sh <NGRAM_ORDER>"
  exit 1
fi

NGRAM_ORDER=$1

SRC_DIR=$(dirname "$0")
ROOT_DIR="${SRC_DIR}/../.."
OUT_DIR="${ROOT_DIR}/models/lm"

mkdir -p "${OUT_DIR}"

PREPROCESSOR="${SRC_DIR}/preprocess_char_lm.py"
PREPROCESSED_DATA="${OUT_DIR}/wikitext-103-raw-preprocessed.txt"

LM_ARPA="${OUT_DIR}/wikitext-103-${NGRAM_ORDER}gram-charlm.arpa"
LM_BIN="${OUT_DIR}/wikitext-103-${NGRAM_ORDER}gram-charlm.bin"

export PYTHONPATH="${PYTHONPATH}:${ROOT_DIR}"
export PATH="${PATH}:~/kenlm/build/bin"

# Download and preprocess wikitext-103 raw character level dataset:
# https://huggingface.co/datasets/Salesforce/wikitext and
# https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/.
if [ ! -f "${PREPROCESSED_DATA}" ]; then
  echo -e "\n### Preprocessing WikiText-103 raw to ${PREPROCESSED_DATA} ###\n"
  python "${PREPROCESSOR}" > "${PREPROCESSED_DATA}"
else
  echo -e "\n### Reusing preprocessed WikiText-103 raw data from ${PREPROCESSED_DATA} ###\n"
fi

echo -e "\n### Building ${NGRAM_ORDER}-gram character-level LM from ${PREPROCESSED_DATA} ###\n"
lmplz -o "${NGRAM_ORDER}" --discount_fallback < "${PREPROCESSED_DATA}" > "${LM_ARPA}"
build_binary "${LM_ARPA}" "${LM_BIN}"

echo -e "\n### Generated ${NGRAM_ORDER}-gram character-level LM ###\n"
echo -e "LM ARPA file (human-readable): ${LM_ARPA}"
echo -e "LM binary file: ${LM_BIN}"
