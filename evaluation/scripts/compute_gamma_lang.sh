#!/bin/bash
# Compute γ_lang for all models and seeds
# Output: data/gamma_lang/{model}/kgw_seed{seed}/gamma_lang.json

set -e
set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR="$SCRIPT_DIR/../.."

source "$WORK_DIR/evaluation/common/config.sh"

VAL_DIR="$DATA_DIR/dataset/mc4"

for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME="${MODEL_NAMES[$i]}"
    MODEL_ABBR="${MODEL_ABBRS[$i]}"

    for SEED in "${SEEDS[@]}"; do
        OUTPUT_DIR="$DATA_DIR/gamma_lang/$MODEL_ABBR/kgw_seed${SEED}"
        OUTPUT_FILE="$OUTPUT_DIR/gamma_lang.json"

        if [ -f "$OUTPUT_FILE" ]; then
            echo "✅ Already exists: $OUTPUT_FILE — skipping"
            continue
        fi

        echo "=== Computing γ_lang for ${MODEL_ABBR} (seed=${SEED}) ==="

        python3 "$WORK_DIR/steam/compute_gamma_lang.py" \
            --base_model "$MODEL_NAME" \
            --input_dir "$VAL_DIR" \
            --output_file "$OUTPUT_FILE" \
            --gamma 0.25 \
            --seed "$SEED" \
            --seeding_scheme minhash

        echo ""
    done
done
