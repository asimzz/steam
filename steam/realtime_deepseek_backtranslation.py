#!/usr/bin/env python3
"""
Real-time Backtranslation using DeepSeek API for STEAM BO Integration.

Drop-in replacement for RealtimeBacktranslator that uses DeepSeek
instead of Google Translate.
"""

import os
import time
import logging
from typing import Optional
from openai import OpenAI
from langcodes import Language


class RealtimeDeepseekTranslator:
    """
    Real-time backtranslation system using DeepSeek API.

    API-compatible with RealtimeBacktranslator (same translate_text interface).
    """

    def __init__(self, model: str = "deepseek-chat", temperature: float = 1.3,
                 rate_limit_delay: float = 0.1):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("Please set the DEEP_SEEK_API_KEY environment variable.")

        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.model = model
        self.temperature = temperature
        self.rate_limit_delay = rate_limit_delay
        self.translation_cache = {}

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Language code normalization (match Google Translate conventions)
        self.lang_code_map = {
            "zh": "zh-CN",
            "iw": "he",
        }

    def normalize_lang_code(self, lang_code: str) -> str:
        """Normalize language codes for display names."""
        return self.lang_code_map.get(lang_code, lang_code)

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """
        Translate text using DeepSeek API.

        Args:
            text: Text to translate
            source_lang: Source language code (ISO 639-1)
            target_lang: Target language code (ISO 639-1)

        Returns:
            Translated text or None if translation failed
        """
        cache_key = (text[:100], source_lang, target_lang)
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]

        try:
            time.sleep(self.rate_limit_delay)

            src_name = Language.make(language=source_lang).display_name()
            tgt_name = Language.make(language=target_lang).display_name()

            prompt = f"Translate the following {src_name} text to {tgt_name}:\n\n{text}"

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful translator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
            )

            translated_text = response.choices[0].message.content.strip()
            self.translation_cache[cache_key] = translated_text

            self.logger.debug(f"Translated {source_lang}->{target_lang}: {text[:50]}...")
            return translated_text

        except Exception as e:
            self.logger.error(f"DeepSeek translation failed {source_lang}->{target_lang}: {e}")
            return None
