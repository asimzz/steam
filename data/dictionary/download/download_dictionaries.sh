#!/bin/bash

set -e
set -u

# Usage: ./download_dictionaries.sh <category>
# Example: ./download_dictionaries.sh new_supported

if [ $# -ne 1 ]; then
    echo "Usage: $0 <category>"
    echo "Categories: new_supported, original_supported"
    exit 1
fi

CATEGORY="$1"

# Set working directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DOWNLOAD_DIR="$SCRIPT_DIR"
mkdir -p "$DOWNLOAD_DIR"

# Base URL for dictionaries
BASE_URL="https://dl.fbaipublicfiles.com/arrival/dictionaries"

# Change to download directory
cd "$DOWNLOAD_DIR"

# Define dictionary lists based on category
case "$CATEGORY" in
    "original_supported")
        echo "📥 Downloading dictionaries for original supported languages..."
        DICTS=(
            "en-de" "en-fr" "en-ja" "en-zh"
            "de-en" "de-fr"
            "fr-en" "fr-de"
            "ja-en" "zh-en"
        )
        ;;
    "new_supported")
        echo "📥 Downloading dictionaries for new supported languages..."
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
        ;;
    *)
        echo "❌ Unknown category: $CATEGORY"
        echo "Available categories: new_supported, original_supported"
        exit 1
        ;;
esac

# Download dictionaries
total=${#DICTS[@]}
current=0

for dict in "${DICTS[@]}"; do
    current=$((current + 1))
    echo "[$current/$total] Downloading ${dict}.txt..."

    if [ -f "${dict}.txt" ]; then
        echo "✅ ${dict}.txt already exists, skipping"
    else
        wget -q "$BASE_URL/${dict}.txt" || echo "⚠️  Failed to download ${dict}.txt"
    fi
done

echo "🎉 Dictionary download complete for $CATEGORY!"