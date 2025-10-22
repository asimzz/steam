#!/bin/bash
# Shared configuration for all evaluation scripts

# Model names and abbreviations
MODEL_NAMES=(
    "meta-llama/Llama-3.2-1B"
    "CohereForAI/aya-23-8B"
    "LLaMAX/LLaMAX3-8B"
)

MODEL_ABBRS=(
    "llama-3.2-1B"
    "aya-23-8B"
    "llamax3-8B"
)

# Watermark methods and seeds
WATERMARK_METHODS=("xsir" "xkgw")
SEEDS=(0 42 123)


DATA_DIR="$WORK_DIR/data"
GEN_DIR="$WORK_DIR/evaluation/gen"
ATTACK_DIR="$WORK_DIR/attack"
MAPPING_DIR="$DATA_DIR/mapping"

# Model configuration
BATCH_SIZE=32
TRANSFORM_MODEL="$DATA_DIR/model/transform_model_x-sbert.pth"
EMBEDDING_MODEL="paraphrase-multilingual-mpnet-base-v2"

# Validate model configuration
if [ ${#MODEL_NAMES[@]} -ne ${#MODEL_ABBRS[@]} ]; then
    echo "❌ MODEL_NAMES and MODEL_ABBRS length mismatch."
    exit 1
fi