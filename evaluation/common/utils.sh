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

# Validate translator name against TRANSLATORS list in config.sh
validate_translator() {
    local name="$1"
    local valid=false

    for t in "${TRANSLATORS[@]}"; do
        if [ "$t" == "$name" ]; then
            valid=true
            break
        fi
    done

    if [ "$valid" == false ]; then
        echo "❌ Unknown translator: '$name'"
        echo "   Supported translators: ${TRANSLATORS[*]}"
        echo "   To add a new translator:"
        echo "     1. Add the translator script in attack/ (see attack/deepseek_translate.py)"
        echo "     2. Update TRANSLATORS in evaluation/common/config.sh"
        exit 1
    fi
}

# Resolve input file per attacker (google uses .mod, others use .{attacker})
get_mod_file() {
    local watermark_dir="$1"
    local tgt_lang="$2"
    local attacker="$3"

    if [ "$attacker" == "google" ]; then
        echo "$watermark_dir/mc4.en-${tgt_lang}.mod.jsonl"
    else
        echo "$watermark_dir/mc4.en-${tgt_lang}.${attacker}.jsonl"
    fi
}

# Run STEAM BO detection
# Args: model_name model_abbr detect_file human_file output_dir tgt_lang translator seed watermark_flags [output_prefix]
run_steam() {
    local model_name="$1"
    local model_abbr="$2"
    local detect_file="$3"
    local human_file="$4"
    local output_dir="$5"
    local tgt_lang="$6"
    local translator="$7"
    local seed="$8"
    local watermark_flags="$9"
    local output_prefix="${10:-}"

    local gamma_lang_file="$DATA_DIR/gamma_lang/$model_abbr/kgw_seed${seed}/gamma_lang.json"

    local prefix_flag=""
    if [ -n "$output_prefix" ]; then
        prefix_flag="--output_prefix $output_prefix"
    fi

    python3 "$WORK_DIR/steam/detector.py" \
        --base_model "$model_name" \
        --detect_file "$detect_file" \
        --human_file "$human_file" \
        --output_dir "$output_dir" \
        --tgt_lang "$tgt_lang" \
        --gamma_lang_file "$gamma_lang_file" \
        --translator "$translator" \
        --seed "$seed" \
        $watermark_flags \
        $prefix_flag
}

# Evaluate detection results
evaluate_detection() {
    local hm_zscore="$1"
    local wm_zscore="$2"

    python3 "$WORK_DIR/evaluation/eval_detection.py" \
        --hm_zscore "$hm_zscore" \
        --wm_zscore "$wm_zscore"
}