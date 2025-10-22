#!/bin/bash

set -e
set -u

# Usage: ./generate_human_holdout.sh <language>
# Example: ./generate_human_holdout.sh en

if [ $# -ne 1 ]; then
    echo "Usage: $0 <language>"
    echo "Languages: en, fr, de, zh, ja"
    exit 1
fi

TGT_LANG="$1"

# Validate type
if [ "$TYPE" != "holdout" ] && [ "$TYPE" != "holdin" ]; then
    echo "❌ Invalid type: $TYPE"
    echo "Valid types: holdout, holdin"
    exit 1
fi

# Set working directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR="$SCRIPT_DIR/../.."


# Load shared configuration
source "$WORK_DIR/evaluation/common/config.sh"
source "$WORK_DIR/evaluation/common/utils.sh"

ORG_LANGS=("en" "fr" "de" "zh")

# Main loop
for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME="${MODEL_NAMES[$i]}"
    MODEL_ABBR="${MODEL_ABBRS[$i]}"

    for SEED in "${SEEDS[@]}"; do
        for WATERMARK_METHOD in "${WATERMARK_METHODS[@]}"; do
            echo "▶️ Generating human baseline for $MODEL_NAME (seed=$SEED) - holdout_${TGT_LANG}"

            MAPPING_FILE="$MAPPING_DIR/$WATERMARK_METHOD/holdout_${TGT_LANG}/300_mapping_${MODEL_ABBR}_seed${SEED}.json"
            OUT_DIR="$GEN_DIR/$MODEL_ABBR/holdout_${TGT_LANG}/${WATERMARK_METHOD}_seed${SEED}"
            mkdir -p "$OUT_DIR"

            WATERMARK_FLAGS="$(get_watermark_flags "$WATERMARK_METHOD" "$MAPPING_FILE")"

            for ORG_LANG in "${ORG_LANGS[@]}"; do
                if [ "$ORG_LANG" == "$TGT_LANG" ]; then
                    continue
                fi

                echo "🌍 Processing human text $ORG_LANG → $TGT_LANG"

                # Detect on human text (should have low scores)
                run_detection "$MODEL_NAME" "$OUT_DIR/mc4.${ORG_LANG}.hum.jsonl" "$OUT_DIR/mc4.${ORG_LANG}.hum.z_score.jsonl" "$WATERMARK_FLAGS" "$SEED"

                # Translate human text from source to target language
                run_translation "$OUT_DIR/mc4.${ORG_LANG}.hum.jsonl" "$OUT_DIR/mc4.${ORG_LANG}-${TGT_LANG}.hum.jsonl" "response" "$ORG_LANG" "$TGT_LANG"

                # Detect on translated human text
                run_detection "$MODEL_NAME" "$OUT_DIR/mc4.${ORG_LANG}-${TGT_LANG}.hum.jsonl" "$OUT_DIR/mc4.${ORG_LANG}-${TGT_LANG}.hum.z_score.jsonl" "$WATERMARK_FLAGS" "$SEED"
            done
        done
    done
done