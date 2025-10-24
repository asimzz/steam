set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error when substituting.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORK_DIR=$SCRIPT_DIR/..
DATA_DIR=$WORK_DIR/data
DICT_DIR=$DATA_DIR/dictionary/download

# Parameters for SIR/X-SIR
MAPPING_DIR=$DATA_DIR/mapping

MODEL_NAMES=(
    # "meta-llama/Llama-3.2-1B"
    "CohereForAI/aya-23-8B"
    # "LLaMAX/LLaMAX3-8B"
)

MODEL_ABBRS=(
    # "llama-3.2-1B"
    "aya-23-8B"
    # "llamax3-8B"
)

SEEDS=(0)

if [ ${#MODEL_NAMES[@]} -ne ${#MODEL_ABBRS[@]} ]; then
    echo "Length of MODEL_NAMES and MODEL_ABBRS should be the same"
    exit 1
fi

for i in "${!MODEL_NAMES[@]}"; do
    MODEL_NAME=${MODEL_NAMES[$i]}
    MODEL_ABBR=${MODEL_ABBRS[$i]}

    for SEED in "${SEEDS[@]}"; do
        echo "Generating semantic mappings with language stats for $MODEL_NAME with seed $SEED"

        python3 $WORK_DIR/src_watermark/xsir/generate_semantic_mappings_with_lang_stats.py \
            --model "$MODEL_NAME" \
            --dictionary_files \
                $DICT_DIR/en-de.txt \
                $DICT_DIR/en-fr.txt \
                $DICT_DIR/en-es.txt \
                $DICT_DIR/en-it.txt \
                $DICT_DIR/en-pt.txt \
                $DICT_DIR/de-en.txt \
                $DICT_DIR/de-fr.txt \
                $DICT_DIR/de-es.txt \
                $DICT_DIR/de-it.txt \
                $DICT_DIR/de-pt.txt \
                $DICT_DIR/fr-en.txt \
                $DICT_DIR/fr-de.txt \
                $DICT_DIR/fr-es.txt \
                $DICT_DIR/fr-it.txt \
                $DICT_DIR/fr-pt.txt \
                $DICT_DIR/es-en.txt \
                $DICT_DIR/es-de.txt \
                $DICT_DIR/es-fr.txt \
                $DICT_DIR/es-it.txt \
                $DICT_DIR/es-pt.txt \
                $DICT_DIR/it-en.txt \
                $DICT_DIR/it-de.txt \
                $DICT_DIR/it-fr.txt \
                $DICT_DIR/it-es.txt \
                $DICT_DIR/it-pt.txt \
                $DICT_DIR/pt-en.txt \
                $DICT_DIR/pt-de.txt \
                $DICT_DIR/pt-fr.txt \
                $DICT_DIR/pt-es.txt \
                $DICT_DIR/pt-it.txt \
                $DICT_DIR/en-pl.txt \
                $DICT_DIR/en-nl.txt \
                $DICT_DIR/en-ru.txt \
                $DICT_DIR/en-hi.txt \
                $DICT_DIR/en-ko.txt \
                $DICT_DIR/en-ja.txt \
                $DICT_DIR/pl-en.txt \
                $DICT_DIR/nl-en.txt \
                $DICT_DIR/ru-en.txt \
                $DICT_DIR/hi-en.txt \
                $DICT_DIR/ko-en.txt \
                $DICT_DIR/ja-en.txt \
                $DICT_DIR/en-bn.txt \
                $DICT_DIR/en-fa.txt \
                $DICT_DIR/en-vi.txt \
                $DICT_DIR/en-iw.txt \
                $DICT_DIR/en-uk.txt \
                $DICT_DIR/en-ta.txt \
                $DICT_DIR/bn-en.txt \
                $DICT_DIR/fa-en.txt \
                $DICT_DIR/vi-en.txt \
                $DICT_DIR/iw-en.txt \
                $DICT_DIR/uk-en.txt \
                $DICT_DIR/ta-en.txt \
            --output_file "$MAPPING_DIR/xsir/300_mapping_${MODEL_ABBR}_seed${SEED}.json" \
            --seed "$SEED"

        # Generate plots
        echo "Generating coverage plots for $MODEL_NAME with seed $SEED"
        python3 $WORK_DIR/src_watermark/xsir/plot_lang_coverage.py \
            --lang_stats_file "$MAPPING_DIR/xsir/300_mapping_${MODEL_ABBR}_seed${SEED}_lang_stats.json" \
            --output_file "$DATA_DIR/figures/xsir/${MODEL_ABBR}_seed${SEED}_lang_coverage.png"
    done
done
