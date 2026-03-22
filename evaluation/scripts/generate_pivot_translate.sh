#!/bin/bash

set -e
set -u

# Multi-step translation attack: translates already-translated files (tgt -> pvt)
# for both watermarked (.mod) and human (.hum) texts.
#
# Prerequisite: generate_watermark_translate.sh and generate_human_translate.sh
# must have already produced mc4.en-{tgt}.mod.jsonl and mc4.en-{tgt}.hum.jsonl
#
# Output:
#   mc4.{tgt}-{pvt}-pivot.mod.jsonl
#   mc4.{tgt}-{pvt}-pivot.hum.jsonl

# Set working directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR="$SCRIPT_DIR/../.."

# Load shared configuration
source "$WORK_DIR/evaluation/common/config.sh"
source "$WORK_DIR/evaluation/common/languages.sh"
source "$WORK_DIR/evaluation/common/utils.sh"

TGT_LANGS=("${NEW_SUPPORTED_LANGS[@]}")

# Main loop
for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME="${MODEL_NAMES[$i]}"
    MODEL_ABBR="${MODEL_ABBRS[$i]}"

    for SEED in "${SEEDS[@]}"; do
        for WATERMARK_METHOD in "${WATERMARK_METHODS[@]}"; do
            OUT_DIR="$GEN_DIR/$MODEL_ABBR/new_supported/${WATERMARK_METHOD}_seed${SEED}"

            for TGT_LANG in "${TGT_LANGS[@]}"; do
                for PVT_LANG in "${PIVOT_LANGS[@]}"; do
                    if [ "$PVT_LANG" == "$TGT_LANG" ]; then
                        continue
                    fi

                    MOD_INPUT="$OUT_DIR/mc4.en-${TGT_LANG}.mod.jsonl"
                    HUM_INPUT="$OUT_DIR/mc4.en-${TGT_LANG}.hum.jsonl"

                    if [ ! -f "$MOD_INPUT" ]; then
                        echo "⚠️ Missing $MOD_INPUT — skipping"
                        continue
                    fi

                    echo "🔀 Pivot $TGT_LANG -> $PVT_LANG ($MODEL_ABBR $WATERMARK_METHOD seed=$SEED)"

                    # Pivot-translate watermarked text
                    run_translation "$MOD_INPUT" "$OUT_DIR/mc4.${TGT_LANG}-${PVT_LANG}-pivot.mod.jsonl" \
                        "response" "$TGT_LANG" "$PVT_LANG"

                    # Pivot-translate human text
                    if [ -f "$HUM_INPUT" ]; then
                        run_translation "$HUM_INPUT" "$OUT_DIR/mc4.${TGT_LANG}-${PVT_LANG}-pivot.hum.jsonl" \
                            "response" "$TGT_LANG" "$PVT_LANG"
                    fi
                done
            done
        done
    done
done
