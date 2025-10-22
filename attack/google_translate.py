import os
import json
import argparse
from tqdm import tqdm
from deep_translator import GoogleTranslator


def read_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main(args):
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    translation_part = args.translation_part

    # Normalize Chinese language codes
    if src_lang == "zh":
        src_lang = "zh-CN"
    if tgt_lang == "zh":
        tgt_lang = "zh-CN"

    # Load input data
    input_data = read_jsonl(args.input_file)

    # Load existing translations if present
    if os.path.exists(args.output_file):
        translated_data = read_jsonl(args.output_file)
    else:
        translated_data = []

    total = len(input_data)
    done = len(translated_data)
    print(f"Translating '{translation_part}' from {args.src_lang} to {args.tgt_lang}")
    print(f"{total} samples found. {done} already translated.")

    if total == done:
        print("✅ Translation already completed. Skipping.")
        return

    # Initialize translator
    translator = GoogleTranslator(source=src_lang, target=tgt_lang)

    # Open output file in append mode
    with open(args.output_file, "a", encoding="utf-8") as output_file:
        for idx in tqdm(range(done, total), desc="Translating", unit="line"):
            data = input_data[idx]
            if translation_part in data:
                try:
                    translation = translator.translate(data[translation_part])
                    data[translation_part] = translation
                except Exception as e:
                    print(f"⚠️ Error on index {idx}: {e}")
                    # If translation fails, keep the original text
                    data[translation_part] = data[translation_part]
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