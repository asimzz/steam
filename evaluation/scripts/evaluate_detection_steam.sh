#!/bin/bash

set -e
set -u

# Usage: ./evaluate_detection_steam.sh <category>
# Example: ./evaluate_detection_steam.sh new_supported

if [ $# -ne 1 ]; then
    echo "Usage: $0 <category>"
    echo "Categories: new_supported, original_supported"
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

            for TGT_LANG in "${TGT_LANGS[@]}"; do
                WM_FILE="$WATERMARK_DIR/mc4.${TGT_LANG}.bo.z_score.jsonl"
                HM_FILE="$WATERMARK_DIR/mc4.${TGT_LANG}.bo.hum.z_score.jsonl"

                if [ ! -f "$WM_FILE" ] || [ ! -f "$HM_FILE" ]; then
                    continue
                fi

                echo "$MODEL_ABBR $WATERMARK_METHOD (seed=$SEED) STEAM-BO ($TGT_LANG)"
                evaluate_detection "$HM_FILE" "$WM_FILE"
            done
            echo "======================================="
        done
    done
done
