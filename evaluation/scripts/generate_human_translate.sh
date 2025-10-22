#!/bin/bash

set -e
set -u

# Usage: ./generate_human.sh <category>
# Example: ./generate_human.sh new_supported

if [ $# -ne 1 ]; then
    echo "Usage: $0 <category>"
    echo "Categories: new_supported, original_supported, unsupported, holdout, holdin"
    exit 1
fi

CATEGORY="$1"

# Set working directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR="$SCRIPT_DIR/../.."

# Load shared configuration
source "$WORK_DIR/evaluation/common/config.sh"
source "$WORK_DIR/evaluation/common/languages.sh"
source "$WORK_DIR/evaluation/common/utils.sh"

# Get target languages based on category
case "$CATEGORY" in
    "new_supported")
        TGT_LANGS=("${NEW_SUPPORTED_LANGS[@]}")
        ;;
    "original_supported")
        TGT_LANGS=("${ORIGINAL_SUPPORTED_LANGS[@]}")
        ;;
    "unsupported")
        TGT_LANGS=("${UNSUPPORTED_LANGS[@]}")
        ;;
    "holdout")
        TGT_LANGS=("${HOLDOUT_LANGS[@]}")
        ;;
    "holdin")
        TGT_LANGS=("${HOLDIN_LANGS[@]}")
        ;;
    *)
        echo "❌ Unknown category: $CATEGORY"
        exit 1
        ;;
esac

# Main loop
for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME="${MODEL_NAMES[$i]}"
    MODEL_ABBR="${MODEL_ABBRS[$i]}"

    for SEED in "${SEEDS[@]}"; do
        for WATERMARK_METHOD in "${WATERMARK_METHODS[@]}"; do
            echo "▶️ Generating human baseline for $MODEL_NAME (seed=$SEED) - $CATEGORY"

            OUT_DIR="$GEN_DIR/$MODEL_ABBR/$CATEGORY/${WATERMARK_METHOD}_seed${SEED}"
            mkdir -p "$OUT_DIR"

            # Detect on human text (should have low scores)
            run_detection "$MODEL_NAME" "$DATA_DIR/dataset/mc4.en.jsonl" "$OUT_DIR/mc4.en.hum.z_score.jsonl" "--watermark_method kgw" "$SEED"

            # Translation & detection for each target language
            for TGT_LANG in "${TGT_LANGS[@]}"; do
                echo "🌍 Translating human text to $TGT_LANG"

                run_translation "$DATA_DIR/dataset/mc4.en.jsonl" "$OUT_DIR/mc4.en-${TGT_LANG}.hum.jsonl" "response" "en" "$TGT_LANG"
                run_detection "$MODEL_NAME" "$OUT_DIR/mc4.en-${TGT_LANG}.hum.jsonl" "$OUT_DIR/mc4.en-${TGT_LANG}.hum.z_score.jsonl" "--watermark_method kgw" "$SEED"
            done
        done
    done
done