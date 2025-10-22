#!/bin/bash
# Language definitions for different evaluation categories

# New supported languages (medium to low resource)
NEW_SUPPORTED_LANGS=(
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

# Original supported languages
ORIGINAL_SUPPORTED_LANGS=("fr" "de" "zh" "ja")

# Unsupported languages for cross-lingual evaluation
UNSUPPORTED_LANGS=("it" "es" "pt" "pl" "nl" "hr" "cs" "da" "ko")