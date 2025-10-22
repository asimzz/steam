#!/bin/bash

set -e
set -u

# Usage: ./generate_watermark.sh <category>
# Example: ./generate_watermark.sh new_supported
# Note: For holdout/holdin, use generate_watermark_holdout.sh

if [ $# -ne 1 ]; then
    echo "Usage: $0 <category>"
    echo "Categories: new_supported, original_supported, unsupported"
    echo "For holdout/holdin, use: ./generate_watermark_holdout.sh <type> <language>"
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
    *)
        echo "❌ Unknown category: $CATEGORY"
        echo "Available categories: new_supported, original_supported, unsupported"
        exit 1
        ;;
esac

ORG_LANG=("en")

# Main loop
for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME="${MODEL_NAMES[$i]}"
    MODEL_ABBR="${MODEL_ABBRS[$i]}"

    for SEED in "${SEEDS[@]}"; do
        for WATERMARK_METHOD in "${WATERMARK_METHODS[@]}"; do
            echo "▶️ Running $WATERMARK_METHOD (seed=$SEED) on $MODEL_NAME for $CATEGORY"

            MAPPING_FILE="$MAPPING_DIR/$WATERMARK_METHOD/$CATEGORY/300_mapping_${MODEL_ABBR}_seed${SEED}.json"
            OUT_DIR="$GEN_DIR/$MODEL_ABBR/$CATEGORY/${WATERMARK_METHOD}_seed${SEED}"
            mkdir -p "$OUT_DIR"

            WATERMARK_FLAGS="$(get_watermark_flags "$WATERMARK_METHOD" "$MAPPING_FILE")"

            for TGT_LANG in "${TGT_LANGS[@]}"; do

                run_translation "$DATA_DIR/dataset/mc4.$ORG_LANG.jsonl" "$OUT_DIR/mc4.$ORG_LANG-$TGT_LANG-cwra.jsonl" "prompt" "$ORG_LANG" "$TGT_LANG"

                # Generate watermarked data
                run_generation "$MODEL_NAME" "$OUT_DIR/mc4.$ORG_LANG-$TGT_LANG-cwra.jsonl" "$OUT_DIR/mc4.$ORG_LANG-$TGT_LANG-cwra.mod.jsonl" "$WATERMARK_FLAGS" "$SEED"

                # Detect watermark in English
                run_detection "$MODEL_NAME" "$OUT_DIR/mc4.$ORG_LANG-$TGT_LANG-cwra.mod.jsonl" "$OUT_DIR/mc4.$ORG_LANG-$TGT_LANG-cwra.mod.z_score.jsonl" "$WATERMARK_FLAGS" "$SEED"

                # Translation & detection for each target language
                echo "🔄 CWRA: back-translating response $TGT_LANG ➝ $ORG_LANG"

                run_translation "$OUT_DIR/mc4.$ORG_LANG-$TGT_LANG-cwra.mod.jsonl" "$OUT_DIR/mc4.$TGT_LANG-$ORG_LANG-cwra.mod.jsonl" "response" "$TGT_LANG" "$ORG_LANG"
                run_detection "$MODEL_NAME" "$OUT_DIR/mc4.$TGT_LANG-$ORG_LANG-cwra.mod.jsonl" "$OUT_DIR/mc4.$TGT_LANG-$ORG_LANG-cwra.mod.z_score.jsonl" "$WATERMARK_FLAGS" "$SEED"
            done
        done
    done
done