#!/bin/bash

set -e
set -u

# Usage: ./run_steam.sh <category> [translator]
# Example: ./run_steam.sh new_supported google

if [ $# -lt 1 ]; then
    echo "Usage: $0 <category> [translator]"
    echo "Categories: new_supported, original_supported"
    echo "Translators: google (default), deepseek, gpt4o"
    exit 1
fi

CATEGORY="$1"
TRANSLATOR="${2:-google}"

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

# Main loop
for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME="${MODEL_NAMES[$i]}"
    MODEL_ABBR="${MODEL_ABBRS[$i]}"

    for SEED in "${SEEDS[@]}"; do
        for WATERMARK_METHOD in "${WATERMARK_METHODS[@]}"; do
            WATERMARK_DIR="$GEN_DIR/$MODEL_ABBR/$CATEGORY/${WATERMARK_METHOD}_seed${SEED}"
            MAPPING_FILE="$MAPPING_DIR/${WATERMARK_METHOD}_mapping.json"

            WATERMARK_FLAGS=$(get_watermark_flags "$WATERMARK_METHOD" "$MAPPING_FILE")

            for TGT_LANG in "${TGT_LANGS[@]}"; do
                MOD_FILE="$WATERMARK_DIR/mc4.en-${TGT_LANG}.mod.jsonl"
                HUM_FILE="$WATERMARK_DIR/mc4.en-${TGT_LANG}.hum.jsonl"

                if [ ! -f "$MOD_FILE" ] || [ ! -f "$HUM_FILE" ]; then
                    echo "⚠️ Missing files for $TGT_LANG — skipping"
                    continue
                fi

                echo "=== $MODEL_ABBR $WATERMARK_METHOD seed=$SEED $TGT_LANG (defense=$TRANSLATOR) ==="

                run_steam "$MODEL_NAME" "$MODEL_ABBR" "$MOD_FILE" "$HUM_FILE" \
                    "$WATERMARK_DIR" "$TGT_LANG" "$TRANSLATOR" "$SEED" "$WATERMARK_FLAGS"

                echo ""
            done
        done
    done
done
