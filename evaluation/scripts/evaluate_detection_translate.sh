#!/bin/bash

set -e
set -u

# Usage: ./evaluate_detection.sh <category>
# Example: ./evaluate_detection.sh new_supported

if [ $# -ne 1 ]; then
    echo "Usage: $0 <category>"
    echo "Categories: new_supported, original_supported, unsupported"
    exit 1
fi

CATEGORY="$1"

# Set working directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR="$SCRIPT_DIR/../.."
GEN_DIR="$WORK_DIR/evaluation/gen"

# Load shared configuration
source "$WORK_DIR/evaluation/common/config.sh"
source "$WORK_DIR/evaluation/common/languages.sh"

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
    *)
        echo "❌ Unknown category: $CATEGORY"
        exit 1
        ;;
esac

# Main evaluation loop
for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME="${MODEL_NAMES[$i]}"
    MODEL_ABBR="${MODEL_ABBRS[$i]}"

    for SEED in "${SEEDS[@]}"; do
        for WATERMARK_METHOD in "${WATERMARK_METHODS[@]}"; do
            WATERMARK_DIR="$GEN_DIR/$MODEL_ABBR/$CATEGORY/${WATERMARK_METHOD}_seed${SEED}"

            echo "$MODEL_NAME $WATERMARK_METHOD (seed=$SEED) No-attack"

            evaluate_detection "$WATERMARK_DIR/mc4.en.hum.z_score.jsonl" "$WATERMARK_DIR/mc4.en.mod.z_score.jsonl"

            echo "======================================="

            for TGT_LANG in "${TGT_LANGS[@]}"; do
                echo "$MODEL_NAME $WATERMARK_METHOD (seed=$SEED) Translation ($TGT_LANG)"
                evaluate_detection "$WATERMARK_DIR/mc4.en-${TGT_LANG}.hum.z_score.jsonl" "$WATERMARK_DIR/mc4.en-${TGT_LANG}.mod.z_score.jsonl"
            done
            echo "======================================="
        done
    done
done