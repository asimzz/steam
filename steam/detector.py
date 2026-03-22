#!/usr/bin/env python3
"""
STEAM Detector - Watermark detection with γ_lang correction and BO-based
back-translation language selection.

For each text:
1. Use BO to find optimal back-translation language (see bayesian_optimization.py)
2. Translate target_lang → back-translation language
3. Detect watermark, recompute z-score using per-language γ_lang

Output:
  - mc4.{prefix}.bo.z_score.jsonl      (watermarked texts)
  - mc4.{prefix}.bo.hum.z_score.jsonl  (human texts, matched back-translation language from BO)

Author: Asim
"""

import os
import json
import math
import numpy as np
import logging
import argparse
from typing import Dict, Tuple

import torch
from transformers import AutoTokenizer

from language_features import LanguageFeatures
from language_codes import iso3_to_iso1, iso1_to_iso3, is_valid_iso3
from realtime_backtranslation import RealtimeBacktranslator
from realtime_deepseek_backtranslation import RealtimeDeepseekTranslator
from realtime_openai_backtranslation import RealtimeOpenAITranslator
from bayesian_optimization import optimize_single_text

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import read_jsonl
from watermarks.kgw.extended_watermark_processor import WatermarkDetector as KGWDetector


def get_watermark_detector(base_model: str, **kwargs):
    """Create KGW watermark detector."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    return KGWDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=kwargs.get('gamma', 0.25),
        seed=kwargs.get('seed', 0),
        seeding_scheme=kwargs.get('seeding_scheme', 'minhash'),
        device=device,
        tokenizer=tokenizer,
        z_threshold=kwargs.get('z_threshold', 4.0),
        normalizers=kwargs.get('normalizers', []),
        ignore_repeated_ngrams=kwargs.get('ignore_repeated_ngrams', True),
    )


class STEAMDetector:
    """
    STEAM Detector with γ_lang-corrected z-scores.

    Handles translation, watermark detection, z-score correction,
    and delegates back-translation language search to bayesian_optimization.py.
    """

    def __init__(self,
                 watermark_detector,
                 target_lang: str,
                 input_dir: str,
                 output_dir: str,
                 gamma_lang_file: str,
                 n_initial: int = 3,
                 max_evaluations: int = 15,
                 random_state: int = 42,
                 translator: str = "google"):
        self.watermark_detector = watermark_detector
        self.target_lang = target_lang
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.n_initial = n_initial
        self.max_evaluations = max_evaluations
        self.random_state = random_state

        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Load γ_lang values
        with open(gamma_lang_file, 'r') as f:
            self._gamma_lang_data = json.load(f)
        self.logger.info(f"Loaded γ_lang for {len(self._gamma_lang_data)} languages")

        # Initialize backtranslator
        if translator == "deepseek":
            self.backtranslator = RealtimeDeepseekTranslator()
        elif translator == "gpt4o":
            self.backtranslator = RealtimeOpenAITranslator()
        else:
            self.backtranslator = RealtimeBacktranslator()

        # Initialize language features
        self.lang_features = LanguageFeatures(feature_sets=['syntax_knn', 'phonology_knn'])

        # Convert target language to ISO-3 for URIEL
        self.target_lang_iso3 = self._normalize_to_iso3(target_lang)

        # Load supported languages
        supported_file = os.path.join(os.path.dirname(__file__), "back_translation_languages.txt")
        with open(supported_file, 'r') as f:
            supported_iso1 = [line.strip() for line in f if line.strip()]

        # Build available back-translation languages
        self.available_bt_langs = []
        for lang_iso1 in supported_iso1:
            try:
                lang_iso3 = iso1_to_iso3(lang_iso1)
            except ValueError:
                continue
            if lang_iso3 == self.target_lang_iso3:
                continue
            if lang_iso3 in self.lang_features.available_languages:
                self.available_bt_langs.append(lang_iso3)

        self.logger.info(f"Available back-translation languages: {len(self.available_bt_langs)}")

        # Pre-compute feature vectors
        self._feature_vectors = {}
        for lang in self.available_bt_langs:
            self._feature_vectors[lang] = self.lang_features.get_feature_vector(lang)

        self.feature_dim = len(next(iter(self._feature_vectors.values())))

        os.makedirs(output_dir, exist_ok=True)

    def _normalize_to_iso3(self, lang_code: str) -> str:
        """Convert language code to ISO-3 format."""
        if is_valid_iso3(lang_code):
            return lang_code
        iso3 = iso1_to_iso3(lang_code)
        if iso3:
            return iso3
        raise ValueError(f"Cannot normalize language code {lang_code} to ISO-3")

    def _get_gamma_lang(self, bt_lang: str) -> float:
        """Get γ_lang for a back-translation language. Falls back to default gamma if not available."""
        try:
            bt_iso1 = iso3_to_iso1(bt_lang)
        except ValueError:
            self.logger.warning(f"Cannot convert {bt_lang} to ISO-1, using default gamma")
            return self.watermark_detector.gamma

        if bt_iso1 in self._gamma_lang_data:
            return self._gamma_lang_data[bt_iso1]["gamma_lang"]

        self.logger.warning(f"No γ_lang for {bt_iso1}, using default gamma={self.watermark_detector.gamma}")
        return self.watermark_detector.gamma

    def _recompute_z_score(self, num_green_tokens: int, num_tokens_scored: int, gamma_lang: float) -> float:
        """Recompute z-score using language-specific γ_lang."""
        numer = num_green_tokens - gamma_lang * num_tokens_scored
        denom = math.sqrt(num_tokens_scored * gamma_lang * (1 - gamma_lang))
        return numer / denom

    def _translate_and_detect(self, text: str, bt_lang: str) -> Tuple[float, str, bool]:
        """
        Translate text to back-translation language and detect watermark.
        Z-score is recomputed using γ_lang instead of default γ=0.25.
        """
        try:
            bt_iso1 = iso3_to_iso1(bt_lang)
            target_iso1 = iso3_to_iso1(self.target_lang_iso3)

            if not bt_iso1 or not target_iso1:
                self.logger.error(f"Language code conversion failed: {bt_lang} or {self.target_lang_iso3}")
                return 0.0, "", False

            translated_text = self.backtranslator.translate_text(text, target_iso1, bt_iso1)

            if translated_text is None:
                self.logger.error(f"Translation failed: {target_iso1} → {bt_iso1}")
                return 0.0, "", False

            detection_result = self.watermark_detector.detect(translated_text)
            num_green = detection_result.get('num_green_tokens')
            num_scored = detection_result.get('num_tokens_scored')

            if num_green is None or num_scored is None or num_scored == 0:
                return 0.0, "", False

            gamma_lang = self._get_gamma_lang(bt_lang)
            corrected_z = self._recompute_z_score(int(num_green), int(num_scored), gamma_lang)

            return corrected_z, translated_text, True

        except Exception as e:
            self.logger.error(f"Error in translate_and_detect for {bt_lang}: {e}")
            return 0.0, "", False

    def _evaluate_bt_lang(self, text: str, bt_lang: str) -> Dict:
        """Evaluate a back-translation language for a specific text."""
        z_score, translated_text, success = self._translate_and_detect(text, bt_lang)

        return {
            'bt_lang': bt_lang,
            'z_score': z_score,
            'feature_vector': self._feature_vectors.get(bt_lang, np.zeros(self.feature_dim)),
            'translated_text': translated_text if success else '',
            'success': success
        }

    def run(self, num_texts: int = 500, input_mod: str = None,
            input_hum: str = None, output_prefix: str = None) -> str:
        """
        Run STEAM on watermarked texts, then apply the selected back-translation
        language to corresponding human texts.
        """
        self.logger.info(f"Starting STEAM for {num_texts} texts")

        # Load watermarked texts
        mod_file = input_mod or os.path.join(self.input_dir, f"mc4.en-{self.target_lang}.mod.jsonl")
        if not os.path.exists(mod_file):
            raise FileNotFoundError(f"Input file not found: {mod_file}")
        mod_data = read_jsonl(mod_file)[:num_texts]

        # Load corresponding human texts
        hum_file = input_hum or os.path.join(self.input_dir, f"mc4.en-{self.target_lang}.hum.jsonl")
        if not os.path.exists(hum_file):
            raise FileNotFoundError(f"Human text file not found: {hum_file}")
        hum_data = read_jsonl(hum_file)[:num_texts]

        prefix = output_prefix or self.target_lang
        mod_output = os.path.join(self.output_dir, f"mc4.{prefix}.bo.z_score.jsonl")
        hum_output = os.path.join(self.output_dir, f"mc4.{prefix}.bo.hum.z_score.jsonl")

        # Resume: count existing lines
        start_idx = 0
        if os.path.exists(mod_output) and os.path.exists(hum_output):
            with open(mod_output, 'r') as f:
                start_idx = sum(1 for _ in f)
            with open(hum_output, 'r') as f:
                hum_count = sum(1 for _ in f)
            start_idx = min(start_idx, hum_count)
            if start_idx > 0:
                self.logger.info(f"Resuming from text {start_idx}")

        with open(mod_output, 'a') as f_mod, open(hum_output, 'a') as f_hum:
            for i, item in enumerate(mod_data):
                if i < start_idx:
                    continue
                text_content = item.get('response', '')
                prompt = item.get('prompt', '')

                if not text_content:
                    self.logger.warning(f"Empty text at index {i}, skipping")
                    continue

                try:
                    # BO on watermarked text
                    result = optimize_single_text(
                        text=text_content,
                        prompt=prompt,
                        text_id=i,
                        available_bt_langs=self.available_bt_langs,
                        feature_vectors=self._feature_vectors,
                        evaluate_fn=self._evaluate_bt_lang,
                        n_initial=self.n_initial,
                        max_evaluations=self.max_evaluations,
                        random_state=self.random_state,
                    )

                    best_bt_lang_iso3 = result['best_bt_lang_iso3']
                    try:
                        best_bt_lang_iso1 = iso3_to_iso1(best_bt_lang_iso3)
                    except (ValueError, TypeError):
                        best_bt_lang_iso1 = best_bt_lang_iso3

                    mod_result = {
                        'z_score': result['z_score'],
                        'best_bt_lang': best_bt_lang_iso1,
                        'prompt': prompt,
                        'response': result['best_translated_text'],
                    }
                    f_mod.write(json.dumps(mod_result) + '\n')

                    # Apply same back-translation language to human text
                    if best_bt_lang_iso3 and i < len(hum_data):
                        hum_text = hum_data[i].get('response', '')
                        hum_prompt = hum_data[i].get('prompt', '')
                        z_score, translated, success = self._translate_and_detect(hum_text, best_bt_lang_iso3)
                        hum_result = {
                            'z_score': z_score if success else 0.0,
                            'prompt': hum_prompt,
                            'response': translated if success else hum_text,
                        }
                    else:
                        hum_result = {'z_score': 0.0, 'prompt': '', 'response': ''}
                    f_hum.write(json.dumps(hum_result) + '\n')

                except Exception as e:
                    self.logger.error(f"Error processing text {i}: {e}")
                    f_mod.write(json.dumps({'z_score': 0.0, 'prompt': prompt, 'response': text_content}) + '\n')
                    f_hum.write(json.dumps({'z_score': 0.0, 'prompt': '', 'response': ''}) + '\n')

        self.logger.info(f"Watermarked results: {mod_output}")
        self.logger.info(f"Human results:       {hum_output}")
        return mod_output


def main():
    parser = argparse.ArgumentParser(description="STEAM Detector")
    parser.add_argument("--base_model", type=str, required=True, help="Base model name")
    parser.add_argument("--tgt_lang", type=str, required=True, help="Target language")
    parser.add_argument("--detect_file", type=str, required=True, help="Watermarked file to detect")
    parser.add_argument("--human_file", type=str, required=True, help="Corresponding human file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--gamma_lang_file", type=str, required=True, help="Path to gamma_lang.json")
    parser.add_argument("--n_initial", type=int, default=3, help="Initial back-translation languages")
    parser.add_argument("--max_evaluations", type=int, default=20, help="Max BO evaluations")
    parser.add_argument("--num_texts", type=int, default=500, help="Number of texts to process")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--seed", type=int, default=0, help="Watermark seed")
    parser.add_argument("--translator", type=str, default="google", choices=["google", "deepseek", "gpt4o"],
                        help="Translator for back-translation (default: google)")
    parser.add_argument("--input_mod", type=str, default=None,
                        help="Override watermarked input file path")
    parser.add_argument("--input_hum", type=str, default=None,
                        help="Override human input file path")
    parser.add_argument("--output_prefix", type=str, default=None,
                        help="Override output prefix (e.g. en-de.gpt4o)")

    # Watermark
    parser.add_argument('--watermark_method', type=str, choices=["xsir", "kgw", "xkgw", "sir"], default="kgw")
    parser.add_argument('--gamma', type=float, default=0.25)
    parser.add_argument('--seeding_scheme', type=str, default="minhash")

    args = parser.parse_args()

    watermark_detector = get_watermark_detector(
        base_model=args.base_model,
        gamma=args.gamma,
        seed=args.seed,
        seeding_scheme=args.seeding_scheme,
    )

    steam_detector = STEAMDetector(
        watermark_detector=watermark_detector,
        target_lang=args.tgt_lang,
        input_dir=os.path.dirname(args.detect_file),
        output_dir=args.output_dir,
        gamma_lang_file=args.gamma_lang_file,
        n_initial=args.n_initial,
        max_evaluations=args.max_evaluations,
        random_state=args.random_state,
        translator=args.translator
    )

    output_file = steam_detector.run(
        num_texts=args.num_texts,
        input_mod=args.detect_file,
        input_hum=args.human_file,
        output_prefix=args.output_prefix
    )
    print(f"\n✅ Output: {output_file}")


if __name__ == "__main__":
    main()
