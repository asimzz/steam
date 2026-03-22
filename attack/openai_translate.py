import os
import json
import argparse
from tqdm import tqdm
from openai import OpenAI
from langcodes import Language


def read_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class OpenAITranslation:
    """GPT-4o-mini translation with the same interface as GoogleTranslation."""

    def __init__(self, model="gpt-4o-mini", temperature=0.0):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please set the OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def translate_text(self, text, source_lang, target_lang):
        try:
            src_name = Language.make(language=source_lang).display_name()
            tgt_name = Language.make(language=target_lang).display_name()
            prompt = f"Translate the following {src_name} text to {tgt_name}:\n\n{text}"
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ OpenAI translation failed {source_lang}→{target_lang}: {e}")
            return None


def main(args):
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    translation_part = args.translation_part

    # Load data
    input_data = read_jsonl(args.input_file)
    if os.path.exists(args.output_file):
        translated_data = read_jsonl(args.output_file)
    else:
        translated_data = []

    total = len(input_data)
    done = len(translated_data)
    print(f"Translating '{translation_part}' from {src_lang} to {tgt_lang} (GPT-4o-mini)")
    print(f"{total} samples found. {done} already translated.")

    if total == done:
        print("✅ Translation already completed. Skipping.")
        return

    translator = OpenAITranslation()

    with open(args.output_file, "a", encoding="utf-8") as output_file:
        for idx in tqdm(range(done, total), desc="Translating", unit="line"):
            data = input_data[idx]
            if translation_part in data:
                translation = translator.translate_text(data[translation_part], src_lang, tgt_lang)
                if translation:
                    data[translation_part] = translation
            output_file.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"✅ Translated file saved to: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--src_lang", type=str, required=True)
    parser.add_argument("--tgt_lang", type=str, required=True)
    parser.add_argument("--translation_part", type=str, required=True)
    args = parser.parse_args()
    main(args)
