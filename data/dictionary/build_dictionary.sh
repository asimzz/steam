#!/bin/bash

set -e
set -u

# Usage: ./build_dictionaries.sh <category> [holdout_lang]
# Examples:
#   ./build_dictionaries.sh new_supported
#   ./build_dictionaries.sh original_supported
#   ./build_dictionaries.sh holdout en   # build holdout for 'en'

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <category> [holdout_lang]"
    echo "Categories: new_supported, original_supported, holdout"
    exit 1
fi

CATEGORY="$1"
HOLDOUT_LANG="${2:-}"

# Set working directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DICT_DIR="$SCRIPT_DIR"
DOWNLOAD_DIR="$DICT_DIR/download"

# Prepare dictionary list and output based on category
case "$CATEGORY" in
    "original_supported")
        echo "🛠️  Building merged dictionary for original supported languages..."
        DICTS=(
            "en-de" "en-fr" "en-ja" "en-zh"
            "de-en" "de-fr"
            "fr-en" "fr-de"
            "ja-en" "zh-en"
        )
        OUTPUT_FILE="$DICT_DIR/original_dictionary.txt"
        ;;
    "new_supported")
        echo "🛠️  Building merged dictionary for new supported languages..."
        DICTS=(
            # High-resource languages
            "en-de" "en-fr" "en-es" "en-it" "en-pt"
            "de-en" "de-fr" "de-es" "de-it" "de-pt"
            "fr-en" "fr-de" "fr-es" "fr-it" "fr-pt"
            "es-en" "es-de" "es-fr" "es-it" "es-pt"
            "it-en" "it-de" "it-fr" "it-es" "it-pt"
            "pt-en" "pt-de" "pt-fr" "pt-es" "pt-it"
            # Medium-resource languages
            "en-pl" "en-nl" "en-ru" "en-hi" "en-ko" "en-ja"
            "pl-en" "nl-en" "ru-en" "hi-en" "ko-en" "ja-en"
            # Low-resource languages
            "en-bn" "en-fa" "en-vi" "en-he" "en-uk" "en-ta"
            "bn-en" "fa-en" "vi-en" "he-en" "uk-en" "ta-en"
        )
        OUTPUT_FILE="$DICT_DIR/new_supported_dictionary.txt"
        ;;
    "holdout")
        if [ -z "$HOLDOUT_LANG" ]; then
            echo "❌ Holdout category requires a language argument. Example: $0 holdout en"
            exit 1
        fi
        echo "🛠️  Building holdout dictionary by excluding language '$HOLDOUT_LANG'..."
        BASE_DICTIONARIES=(
            "fr-de" "fr-en" "zh-en" "ja-en" "de-en" "de-fr" "en-fr" "en-de" "en-zh" "en-ja"
        )
        DICTS=()
        for d in "${BASE_DICTIONARIES[@]}"; do
            if [[ $d != *"$HOLDOUT_LANG"* ]]; then
                DICTS+=("$d")
            fi
        done
        OUTPUT_FILE="$DICT_DIR/holdout_dictionary_${HOLDOUT_LANG}.txt"
        ;;
    *)
        echo "❌ Unknown category: $CATEGORY"
        echo "Available categories: new_supported, original_supported, holdout"
        exit 1
        ;;
esac

# Resolve full paths to dictionary files
DICT_PATHS=()
for dict in "${DICTS[@]}"; do
    DICT_PATHS+=("$DOWNLOAD_DIR/${dict}.txt")
done

echo "➡️  Using $((${#DICT_PATHS[@]})) dictionaries"

# Build the dictionary (single call)
python3 "$DICT_DIR/build_dictionary.py" \
    --dicts "${DICT_PATHS[@]}" \
    --output_file "$OUTPUT_FILE"

echo "✅ Built: $OUTPUT_FILE"


