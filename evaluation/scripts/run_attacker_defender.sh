#!/bin/bash

set -e
set -u

# Usage: ./run_attacker_defender.sh <attacker> <defender>
# Example: ./run_attacker_defender.sh deepseek google
#
# Attackers: google, deepseek, gpt4o
# Defenders: google, deepseek, gpt4o
#
# Input files per attacker:
#   google   → mc4.en-{tgt}.mod.jsonl
#   deepseek → mc4.en-{tgt}.deepseek.jsonl
#   gpt4o    → mc4.en-{tgt}.gpt4o.jsonl
#
# Output files: mc4.{tgt}.{defender}.bo.z_score.jsonl, mc4.{tgt}.{defender}.bo.hum.z_score.jsonl

if [ $# -ne 2 ]; then
    echo "Usage: $0 <attacker> <defender>"
    echo "Attackers: google, deepseek, gpt4o"
    echo "Defenders: google, deepseek, gpt4o"
    exit 1
fi

ATTACKER="$1"
DEFENDER="$2"

# Set working directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR="$SCRIPT_DIR/../.."

# Load shared configuration
source "$WORK_DIR/evaluation/common/config.sh"
source "$WORK_DIR/evaluation/common/languages.sh"
source "$WORK_DIR/evaluation/common/utils.sh"

TARGET_LANGS=("${ATTACKER_DEFENDER_LANGS[@]}")

# Validate attacker and defender
validate_translator "$ATTACKER"
validate_translator "$DEFENDER"

# Main loop
for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME="${MODEL_NAMES[$i]}"
    MODEL_ABBR="${MODEL_ABBRS[$i]}"

    for SEED in "${SEEDS[@]}"; do
        for WATERMARK_METHOD in "${WATERMARK_METHODS[@]}"; do
            WATERMARK_DIR="$GEN_DIR/$MODEL_ABBR/$ATTACKER/${WATERMARK_METHOD}_seed${SEED}"
            MAPPING_FILE="$MAPPING_DIR/${WATERMARK_METHOD}_mapping.json"

            WATERMARK_FLAGS=$(get_watermark_flags "$WATERMARK_METHOD" "$MAPPING_FILE")

            for TGT_LANG in "${TARGET_LANGS[@]}"; do
                MOD_FILE=$(get_mod_file "$WATERMARK_DIR" "$TGT_LANG" "$ATTACKER")

                if [ ! -f "$MOD_FILE" ]; then
                    echo "⚠️ Missing attack file: $MOD_FILE — skipping"
                    continue
                fi
                echo "=== $ATTACKER attack / $DEFENDER defense: $TGT_LANG ($MODEL_ABBR $WATERMARK_METHOD seed=$SEED) ==="

                HUM_FILE="$WATERMARK_DIR/mc4.en-${TGT_LANG}.hum.jsonl"

                run_steam "$MODEL_NAME" "$MODEL_ABBR" "$MOD_FILE" "$HUM_FILE" \
                    "$WATERMARK_DIR" "$TGT_LANG" "$DEFENDER" "$SEED" "$WATERMARK_FLAGS" "${TGT_LANG}.${DEFENDER}"

                echo ""
            done
        done
    done
done
