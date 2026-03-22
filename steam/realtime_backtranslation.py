#!/usr/bin/env python3
"""
Real-time Backtranslation Pipeline for STEAM BO Integration

This module provides real-time backtranslation using Google Translate
for the BO-enhanced STEAM watermark detection system.
"""

import time
import logging
from typing import Dict, Any, Optional
from deep_translator import GoogleTranslator


class RealtimeBacktranslator:
    """
    Real-time backtranslation system using Google Translate.

    Handles translation: target_lang → intermediate_lang → target_lang
    """

    def __init__(self, rate_limit_delay: float = 0.1):
        """
        Initialize backtranslator.

        Args:
            rate_limit_delay: Delay between API calls to avoid rate limiting
        """
        self.rate_limit_delay = rate_limit_delay
        self.translation_cache = {}  # Cache for repeated translations

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Language code normalization
        self.lang_code_map = {
            "zh": "zh-CN",  # Chinese simplified
            "iw": "he",     # Hebrew code mapping
        }

    def normalize_lang_code(self, lang_code: str) -> str:
        """Normalize language codes for Google Translate compatibility."""
        return self.lang_code_map.get(lang_code, lang_code)

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """
        Translate text using Google Translate.

        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Translated text or None if translation failed
        """
        # Normalize language codes
        source_lang = self.normalize_lang_code(source_lang)
        target_lang = self.normalize_lang_code(target_lang)

        # Check cache
        cache_key = (text[:100], source_lang, target_lang)  # Use first 100 chars as key
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]

        try:
            # Rate limiting
            time.sleep(self.rate_limit_delay)

            # Handle special cases for language codes
            if source_lang == "zh":
                source_lang = "zh-CN"
            if target_lang == "zh":
                target_lang = "zh-CN"
            if source_lang == "he":
                source_lang = "iw"
            if target_lang == "he":
                target_lang = "iw"

            # Initialize translator
            translator = GoogleTranslator(source=source_lang, target=target_lang)

            # Perform translation
            translated_text = translator.translate(text)

            # Cache result
            self.translation_cache[cache_key] = translated_text

            self.logger.debug(f"Translated {source_lang}→{target_lang}: {text[:50]}...")
            return translated_text

        except Exception as e:
            self.logger.error(f"Translation failed {source_lang}→{target_lang}: {e}")
            return None

    def translate_to_intermediate(self, text: str, target_lang: str, intermediate_lang: str) -> Dict[str, Any]:
        """
        Translate to intermediate language: target_lang → intermediate_lang

        Args:
            text: Original text in target language
            target_lang: Target language code (e.g., 'fr' for French)
            intermediate_lang: Intermediate language code (e.g., 'de' for German)

        Returns:
            Dictionary containing:
            - 'translated_text': Text in intermediate language
            - 'success': Boolean indicating if translation succeeded
            - 'error': Error message if failed
        """
        result = {
            'translated_text': None,
            'success': False,
            'error': None
        }

        try:
            # Single translation: target_lang → intermediate_lang
            self.logger.info(f"Translating {target_lang} → {intermediate_lang}")
            translated_text = self.translate_text(text, target_lang, intermediate_lang)

            if translated_text is None:
                result['error'] = f"Failed to translate {target_lang} → {intermediate_lang}"
                return result

            result['translated_text'] = translated_text
            result['success'] = True

            self.logger.info(f"Translation successful to {intermediate_lang}")

        except Exception as e:
            result['error'] = f"Translation error: {str(e)}"
            self.logger.error(result['error'])

        return result

    def translate_and_detect(self, text: str, target_lang: str,
                            intermediate_lang: str, watermark_detector) -> Dict[str, Any]:
        """
        Translate to intermediate language and perform watermark detection.

        Args:
            text: Original text in target language
            target_lang: Target language code
            intermediate_lang: Intermediate language for translation
            watermark_detector: Watermark detector instance (XSIR/KGW/UW)

        Returns:
            Dictionary containing:
            - 'z_score': Watermark detection z-score
            - 'success': Boolean indicating success
            - 'intermediate_lang': Language used for translation
            - 'translation_result': Full translation result
            - 'detection_result': Full detection result
            - 'error': Error message if failed
        """
        result = {
            'z_score': None,
            'success': False,
            'intermediate_lang': intermediate_lang,
            'translation_result': None,
            'detection_result': None,
            'error': None
        }

        try:
            # Perform translation to intermediate language
            translation_result = self.translate_to_intermediate(text, target_lang, intermediate_lang)
            result['translation_result'] = translation_result

            if not translation_result['success']:
                result['error'] = translation_result['error']
                return result

            # Perform watermark detection on translated text
            translated_text = translation_result['translated_text']
            detection_result = watermark_detector.detect(translated_text)
            result['detection_result'] = detection_result

            # Extract z-score
            if isinstance(detection_result, dict) and 'z_score' in detection_result:
                result['z_score'] = detection_result['z_score']
                result['success'] = True

                self.logger.info(f"Detection complete via {intermediate_lang}: z_score = {result['z_score']:.4f}")
            else:
                result['error'] = "Invalid detection result format"

        except Exception as e:
            result['error'] = f"Detection error: {str(e)}"
            self.logger.error(result['error'])

        return result


def test_translation():
    """Test function for the translation pipeline."""

    # Test text
    test_text = "Hello, this is a test sentence for watermark detection."

    # Initialize translator
    translator = RealtimeBacktranslator()

    # Test simple translation
    print("Testing translation pipeline...")
    result = translator.translate_to_intermediate(
        text=test_text,
        target_lang="en",
        intermediate_lang="fr"
    )

    print(f"Original English: {test_text}")
    print(f"Translated to French: {result['translated_text']}")
    print(f"Success: {result['success']}")

    if not result['success']:
        print(f"Error: {result['error']}")


if __name__ == "__main__":
    test_translation()