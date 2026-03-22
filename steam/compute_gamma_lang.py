import os
import json
import argparse
import torch
import tqdm
import numpy as np

from transformers import AutoTokenizer

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import read_jsonl
from watermarks.kgw.extended_watermark_processor import WatermarkDetector as KGWDetector


def compute_gamma_for_language(detector, val_data):
    """Compute average green_fraction for a list of validation texts."""
    green_fractions = []
    for item in val_data:
        text = item.get("response", "")
        if not text:
            continue
        try:
            result = detector.detect(text)
            gf = result.get("green_fraction")
            if gf is not None and gf == gf:
                green_fractions.append(float(gf))
        except (ValueError, RuntimeError):
            continue
    return green_fractions


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    detector = KGWDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=args.gamma,
        seed=args.seed,
        seeding_scheme=args.seeding_scheme,
        device=device,
        tokenizer=tokenizer,
        z_threshold=4.0,
        normalizers=[],
        ignore_repeated_ngrams=True,
    )

    # Find all validation files
    val_files = sorted([
        f for f in os.listdir(args.input_dir)
        if f.startswith("mc4.") and f.endswith(".val.jsonl")
    ])

    if not val_files:
        print(f"❌ No validation files found in {args.input_dir}")
        return

    print(f"Found {len(val_files)} validation files")

    gamma_lang = {}
    with torch.no_grad():
        for val_file in tqdm.tqdm(val_files):
            lang = val_file.replace("mc4.", "").replace(".val.jsonl", "")

            val_data = read_jsonl(os.path.join(args.input_dir, val_file))
            green_fractions = compute_gamma_for_language(detector, val_data)

            if green_fractions:
                mean_gf = float(np.mean(green_fractions))
                std_gf = float(np.std(green_fractions))
                gamma_lang[lang] = {
                    "gamma_lang": mean_gf,
                    "std": std_gf,
                    "n_samples": len(green_fractions),
                }
                print(f"  {lang}: γ_lang={mean_gf:.4f} (std={std_gf:.4f}, n={len(green_fractions)})")
            else:
                print(f"  {lang}: no valid samples")

    # Save
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(gamma_lang, f, indent=2)

    print(f"\n✅ Saved γ_lang for {len(gamma_lang)} languages to {args.output_file}")

    # Summary
    if gamma_lang:
        sorted_langs = sorted(gamma_lang.items(), key=lambda x: x[1]["gamma_lang"])
        print("\nLowest γ_lang (most biased):")
        for lang, info in sorted_langs[:5]:
            print(f"  {lang}: {info['gamma_lang']:.4f}")
        print("\nHighest γ_lang:")
        for lang, info in sorted_langs[-5:]:
            print(f"  {lang}: {info['gamma_lang']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute γ_lang from validation data.')
    parser.add_argument('--base_model', type=str, required=True, help="Model name for tokenizer.")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory with mc4.{lang}.val.jsonl files.")
    parser.add_argument('--output_file', type=str, required=True, help="Output JSON file.")
    parser.add_argument('--gamma', type=float, default=0.25, help="KGW gamma for green list.")
    parser.add_argument('--seed', type=int, default=0, help="KGW seed.")
    parser.add_argument('--seeding_scheme', type=str, default="minhash", help="KGW seeding scheme.")
    args = parser.parse_args()
    main(args)
