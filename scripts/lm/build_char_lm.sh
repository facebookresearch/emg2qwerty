#!/bin/bash

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

SRC_DIR="$(dirname $0)"
ROOT_DIR="${SRC_DIR}/../.."
OUT_DIR="${ROOT_DIR}/models"

PREPROCESSOR="${SRC_DIR}/preprocess_char_lm.py"
PREPROCESSED_DATA="wikitext-103-raw-preprocessed.txt"

LM_ARPA="${OUT_DIR}/wikitext-103-${NGRAM_ORDER}-gram-char-lm.arpa"
LM_BIN="${OUT_DIR}/wikitext-103-${NGRAM_ORDER}-gram-char-lm.bin"

export PYTHONPATH="${PYTHONPATH}:${ROOT_DIR}"
export PATH="${PATH}:~/kenlm/build/bin"

# Download and preprocess wikitext-103 raw character level dataset:
# https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/
if [ ! -d wikitext-103-raw ]; then
  echo "Downloading wikitext-103 raw character level dataset"
  wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
  unzip wikitext-103-raw-v1.zip

  echo "Preprocessing to ${PREPROCESSED_DATA}"
  cat wikitext-103-raw/* | python ${PREPROCESSOR} > ${PREPROCESSED_DATA}
fi

mkdir -p ${OUT_DIR}

echo -e "\nBuilding ${NGRAM_ORDER}-gram char-LM"
lmplz -o ${NGRAM_ORDER} --discount_fallback < ${PREPROCESSED_DATA} > ${LM_ARPA}
build_binary ${LM_ARPA} ${LM_BIN}
