#!/usr/bin/env python3
"""
Real-time Backtranslation using OpenAI API (GPT-4o-mini) for STEAM BO Integration.

Drop-in replacement for RealtimeBacktranslator that uses GPT-4o-mini
instead of Google Translate.
"""

import os
import time
import logging
from typing import Optional
from openai import OpenAI
from langcodes import Language


class RealtimeOpenAITranslator:
    """
    Real-time backtranslation system using OpenAI API (GPT-4o-mini).

    API-compatible with RealtimeBacktranslator (same translate_text interface).
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0,
                 rate_limit_delay: float = 0.1):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please set the OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.rate_limit_delay = rate_limit_delay
        self.translation_cache = {}

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.lang_code_map = {
            "zh": "zh-CN",
            "iw": "he",
        }

    def normalize_lang_code(self, lang_code: str) -> str:
        """Normalize language codes for display names."""
        return self.lang_code_map.get(lang_code, lang_code)

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """
        Translate text using OpenAI API (GPT-4o-mini).

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
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
            )

            translated_text = response.choices[0].message.content.strip()
            self.translation_cache[cache_key] = translated_text

            self.logger.debug(f"Translated {source_lang}->{target_lang}: {text[:50]}...")
            return translated_text

        except Exception as e:
            self.logger.error(f"OpenAI translation failed {source_lang}->{target_lang}: {e}")
            return None
