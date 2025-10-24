#!/bin/bash
set -e
set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR=$SCRIPT_DIR/..
DATA_DIR=$WORK_DIR/data
FIGURE_DIR=$WORK_DIR/data/figures
GEN_DIR=$WORK_DIR/gen

MODELS=(
    "llama-3.2-1B"
    "aya-23-8B"
    "llamax3-8B"

)

mkdir -p $FIGURE_DIR

python3 $WORK_DIR/plot_auc_tokens.py \
  --data_dir "${DATA_DIR}/dictionary" \
  --stats_file "tokenizer_lang_coverage.json" \
  --base_dir "${GEN_DIR}" \
  --output_dir "${FIGURE_DIR}"