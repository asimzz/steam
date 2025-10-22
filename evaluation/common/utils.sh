#!/bin/bash
# Common utility functions for evaluation scripts

# Get watermark flags for a given method and mapping file
get_watermark_flags() {
    local method="$1"
    local mapping_file="$2"

    case "$method" in
        "kgw")
            echo "--watermark_method kgw"
            ;;
        "xsir")
            echo "--watermark_method xsir --transform_model $TRANSFORM_MODEL --embedding_model $EMBEDDING_MODEL --mapping_file $mapping_file"
            ;;
        "xkgw")
            echo "--watermark_method xkgw --cluster_mapping_file $mapping_file"
            ;;
        *)
            echo "❌ Unknown watermark method: $method"
            exit 1
            ;;
    esac
}

# Run text generation
run_generation() {
    local model_name="$1"
    local input_file="$2"
    local output_file="$3"
    local watermark_flags="$4"
    local seed="$5"

    python3 "$WORK_DIR/gen.py" \
        --base_model "$model_name" \
        --fp16 \
        --batch_size "$BATCH_SIZE" \
        --seed "$seed" \
        --input_file "$input_file" \
        --output_file "$output_file" \
        $watermark_flags
}

# Run detection
run_detection() {
    local model_name="$1"
    local detect_file="$2"
    local output_file="$3"
    local watermark_flags="$4"
    local seed="$5"

    python3 "$WORK_DIR/detect.py" \
        --base_model "$model_name" \
        --seed "$seed" \
        --detect_file "$detect_file" \
        --output_file "$output_file" \
        $watermark_flags
}

# Run translation
run_translation() {
    local input_file="$1"
    local output_file="$2"
    local translation_part="$3"
    local src_lang="$4"
    local tgt_lang="$5"

    python3 "$ATTACK_DIR/google_translate.py" \
        --input_file "$input_file" \
        --output_file "$output_file" \
        --translation_part "$translation_part" \
        --src_lang "$src_lang" \
        --tgt_lang "$tgt_lang"
}

# Evaluate detection results
evaluate_detection() {
    local hm_zscore="$1"
    local wm_zscore="$2"

    python3 "$WORK_DIR/evaluation/eval_detection.py" \
        --hm_zscore "$hm_zscore" \
        --wm_zscore "$wm_zscore"
}