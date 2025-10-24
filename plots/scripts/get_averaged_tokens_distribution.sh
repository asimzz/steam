set -e
set -u

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR=$SCRIPT_DIR/..
DATA_DIR=$WORK_DIR/data
GEN_DIR=$WORK_DIR/gen
FIGURE_DIR=$WORK_DIR/data/figures

MODEL_NAMES=(
    "meta-llama/Llama-3.2-1B"
    "CohereForAI/aya-23-8B"
)

MODEL_ABBRS=(
    "llama-3.2-1B"
    "aya-23-8B"
)

TGT_LANGS=(
    # High-resource languages
    "fr" # French
    "de" # German
    "it" # Italian
    "es" # Spanish
    "pt" # Portuguese
    # Medium-resource languages
    "pl" # Polish
    "nl" # Dutch
    "ru" # Russian
    "hi" # Hindi
    "ko" # Korean
    "ja" # Japanese
    # Low-resource languages
    "bn" # Bengali
    "fa" # Persian
    "vi" # Vietnamese
    "iw" # Hebrew
    "uk" # Ukrainian
    "ta" # Tamil
)

ORG_LANGS=(
    # Low-resource languages
    "bn" # Bengali
    "ta" # Tamil
)

WATERMARK_METHOD="kgw"
SEEDS=(0)

if [ ${#MODEL_NAMES[@]} -ne ${#MODEL_ABBRS[@]} ]; then
    echo "Length of MODEL_NAMES and MODEL_ABBRS should be the same"
    exit 1
fi

for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME=${MODEL_NAMES[$i]}
    MODEL_ABBR=${MODEL_ABBRS[$i]}

    for SEED in "${SEEDS[@]}"; do
        for ORG_LANG in "${ORG_LANGS[@]}"; do
            echo "Computing averaged token distribution for $MODEL_NAME $WATERMARK_METHOD (seed=$SEED) across all target languages for $ORG_LANG"

            python3 $WORK_DIR/src_watermark/$WATERMARK_METHOD/average_token_distribution.py \
                --base_model $MODEL_ABBR \
                --model_name $MODEL_NAME \
                --org_lang $ORG_LANG \
                --figure_dir $FIGURE_DIR \
                --gen_dir $GEN_DIR \
                --seed $SEED \
                --watermark_method $WATERMARK_METHOD \
                --compute_average \
                --tgt_langs "${TGT_LANGS[@]}"
        done
    done
done
