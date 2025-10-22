#!/bin/bash

set -e
set -u

# Usage: ./generate_mapping.sh <category> [language]
# Example: ./generate_mapping.sh new_supported
# Example: ./generate_mapping.sh holdout en

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <category> [language]"
    echo "Categories: new_supported, original_supported, holdout"
    echo "For holdout, specify target language: en, fr, de, zh, ja"
    exit 1
fi

CATEGORY="$1"
HOLDOUT_LANG="${2:-}"

# Set working directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR="$SCRIPT_DIR/../.."

# Load shared configuration
source "$WORK_DIR/evaluation/common/config.sh"

# Handle holdout specially (requires language parameter)
if [ "$CATEGORY" = "holdout" ]; then
    if [ -z "$HOLDOUT_LANG" ]; then
        echo "❌ Holdout category requires language parameter"
        echo "Usage: $0 holdout <language>"
        echo "Languages: en, fr, de, zh, ja"
        exit 1
    fi
    DICTIONARY_FILE="$DATA_DIR/dictionary/holdout_dictionary_${HOLDOUT_LANG}.txt"
    OUTPUT_DIR_SUFFIX="holdout_${HOLDOUT_LANG}"
else
    # Handle other categories
    case "$CATEGORY" in
        "new_supported")
            DICTIONARY_FILE="$DATA_DIR/dictionary/new_supported_dictionary.txt"
            OUTPUT_DIR_SUFFIX="new_supported"
            ;;
        "original_supported")
            DICTIONARY_FILE="$DATA_DIR/dictionary/original_dictionary.txt"
            OUTPUT_DIR_SUFFIX="original_supported"
            ;;
        *)
            echo "❌ Unknown category: $CATEGORY"
            echo "Available categories: new_supported, original_supported, holdout"
            exit 1
            ;;
    esac
fi

# Generate mappings
for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME="${MODEL_NAMES[$i]}"
    MODEL_ABBR="${MODEL_ABBRS[$i]}"

    for WATERMARK_METHOD in "${WATERMARK_METHODS[@]}"; do

        echo "▶️ Generating semantic mappings for $MODEL_NAME - $CATEGORY using $WATERMARK_METHOD"

        for SEED in "${SEEDS[@]}"; do
            echo "Generating semantic mappings for $MODEL_NAME with seed $SEED - $CATEGORY"

            OUT_DIR="$MAPPING_DIR/$WATERMARK_METHOD/$OUTPUT_DIR_SUFFIX"
            mkdir -p "$OUT_DIR"

            python3 "$WORK_DIR/watermarks/$WATERMARK_METHOD/generate_semantic_mappings.py" \
                --model "$MODEL_NAME" \
                --dictionary "$DICTIONARY_FILE" \
                --output_file "$OUT_DIR/300_mapping_${MODEL_ABBR}_seed${SEED}.json" \
                --seed "$SEED"
        done
    done
done