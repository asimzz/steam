#!/bin/bash

set -e
set -u

# Run STEAM detection on multi-step translation attack files.
#
# Prerequisite: generate_pivot_translate.sh must have produced
#   mc4.{tgt}-{pvt}-pivot.mod.jsonl and mc4.{tgt}-{pvt}-pivot.hum.jsonl
#
# Output:
#   mc4.{tgt}-{pvt}-pivot.bo.z_score.jsonl
#   mc4.{tgt}-{pvt}-pivot.bo.hum.z_score.jsonl

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
            WATERMARK_DIR="$GEN_DIR/$MODEL_ABBR/new_supported/${WATERMARK_METHOD}_seed${SEED}"
            MAPPING_FILE="$MAPPING_DIR/${WATERMARK_METHOD}_mapping.json"

            WATERMARK_FLAGS=$(get_watermark_flags "$WATERMARK_METHOD" "$MAPPING_FILE")

            for TGT_LANG in "${TGT_LANGS[@]}"; do
                for PVT_LANG in "${PIVOT_LANGS[@]}"; do
                    if [ "$PVT_LANG" == "$TGT_LANG" ]; then
                        continue
                    fi

                    PREFIX="${TGT_LANG}-${PVT_LANG}-pivot"
                    MOD_FILE="$WATERMARK_DIR/mc4.${PREFIX}.mod.jsonl"
                    HUM_FILE="$WATERMARK_DIR/mc4.${PREFIX}.hum.jsonl"

                    if [ ! -f "$MOD_FILE" ]; then
                        echo "Skipping $PREFIX: input not found"
                        continue
                    fi

                    echo "=== STEAM: $TGT_LANG -> $PVT_LANG ($MODEL_ABBR $WATERMARK_METHOD seed=$SEED) ==="

                    run_steam "$MODEL_NAME" "$MODEL_ABBR" "$MOD_FILE" "$HUM_FILE" \
                        "$WATERMARK_DIR" "$TGT_LANG" "google" "$SEED" "$WATERMARK_FLAGS" "$PREFIX"

                    echo ""
                done
            done
        done
    done
done
