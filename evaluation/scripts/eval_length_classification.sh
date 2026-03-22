#!/bin/bash

set -e
set -u

# Splits texts into short/medium/long by percentile-based token length
# and evaluates watermark detection per bin.

# Set working directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR="$SCRIPT_DIR/../.."

# Load shared configuration
source "$WORK_DIR/evaluation/common/config.sh"
source "$WORK_DIR/evaluation/common/languages.sh"

TGT_LANGS=("${NEW_SUPPORTED_LANGS[@]}")

echo "=== Length Classification (STEAM BO, percentile-based) ==="
echo ""

for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME="${MODEL_NAMES[$i]}"
    MODEL_ABBR="${MODEL_ABBRS[$i]}"

    for SEED in "${SEEDS[@]}"; do
        for WATERMARK_METHOD in "${WATERMARK_METHODS[@]}"; do
            WATERMARK_DIR="$GEN_DIR/$MODEL_ABBR/new_supported/${WATERMARK_METHOD}_seed${SEED}"

            echo "--- $MODEL_ABBR / $WATERMARK_METHOD (seed=$SEED) ---"

            for TGT_LANG in "${TGT_LANGS[@]}"; do
                WM_FILE="$WATERMARK_DIR/mc4.${TGT_LANG}.bo.z_score.jsonl"
                HUM_FILE="$WATERMARK_DIR/mc4.${TGT_LANG}.bo.hum.z_score.jsonl"

                if [ ! -f "$WM_FILE" ] || [ ! -f "$HUM_FILE" ]; then
                    echo "Skipping $TGT_LANG (BO files not found)"
                    continue
                fi

                python3 "$WORK_DIR/evaluation/eval_length_classification.py" \
                    --tgt_lang "$TGT_LANG" \
                    --base_wm_dir "$WATERMARK_DIR" \
                    --tokenizer "$MODEL_NAME"

            done
        done
    done
done

echo ""
echo "Done."
