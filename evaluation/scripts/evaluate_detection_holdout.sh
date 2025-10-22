#!/bin/bash

set -e
set -u

# Usage: ./evaluate_detection_holdout.sh <lang>
# Example: ./evaluate_detection_holdout.sh ja

if [ $# -ne 1 ]; then
    echo "Usage: $0 <lang>"
    echo "Example: $0 ja"
    exit 1
fi

TGT_LANG="$1"
CATEGORY="holdout"

# Set working directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR="$SCRIPT_DIR/../.."
GEN_DIR="$WORK_DIR/evaluation/gen"

# Load shared configuration
source "$WORK_DIR/evaluation/common/config.sh"
source "$WORK_DIR/evaluation/common/languages.sh"

# Validate requested language is in HOLDOUT_LANGS (optional strict check)
if [[ ! " ${HOLDOUT_LANGS[@]} " =~ " ${LANGUAGE} " ]]; then
    echo "⚠️  Warning: ${LANGUAGE} not listed in HOLDOUT_LANGS; proceeding anyway."
fi

ORG_LANGS=("en" "fr" "de" "zh")

# Main evaluation loop
for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME="${MODEL_NAMES[$i]}"
    MODEL_ABBR="${MODEL_ABBRS[$i]}"

    for SEED in "${SEEDS[@]}"; do
        for WATERMARK_METHOD in "${WATERMARK_METHODS[@]}"; do
            WATERMARK_DIR="$GEN_DIR/$MODEL_ABBR/${CATEGORY}_${TGT_LANG}/${WATERMARK_METHOD}_seed${SEED}"

            for ORG_LANG in "${ORG_LANGS[@]}"; do
                if [ "$ORG_LANG" == "$TGT_LANG" ]; then
                    continue
                fi

                echo "$MODEL_NAME $WATERMARK_METHOD (seed=$SEED) No-attack"
                evaluate_detection "$WATERMARK_DIR/mc4.${ORG_LANG}.hum.z_score.jsonl" "$WATERMARK_DIR/mc4.${ORG_LANG}.mod.z_score.jsonl"

                echo "======================================="

                echo "$MODEL_NAME $WATERMARK_METHOD (seed=$SEED) Translation ($ORG_LANG → $TGT_LANG)"
                evaluate_detection "$WATERMARK_DIR/mc4.${ORG_LANG}-${TGT_LANG}.hum.z_score.jsonl" "$WATERMARK_DIR/mc4.${ORG_LANG}-${TGT_LANG}.mod.z_score.jsonl"

                echo "======================================="
            done
        done
    done
done


