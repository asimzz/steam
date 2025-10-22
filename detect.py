import os
import tqdm
import torch
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM
from watermarks.xsir.watermark import (
    WatermarkWindow as XSIRWindow,
    WatermarkContext as XSIRContext,
)
from watermarks.kgw.extended_watermark_processor import (
    WatermarkDetector as KGWDetector
)
from watermarks.xkgw.watermark_processor import (
    WatermarkDetector as XKGWDetector
)

from utils import read_jsonl, append_jsonl

def get_length(text, tokenizer):
    return len(tokenizer.encode(text))

def is_nan(nan):
    return nan != nan

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # Load watermark detector
    if args.watermark_method in ["xsir", "sir"]:
        if args.watermark_type == "window": # use a window of previous tokens to hash, e.g. KGW
            watermark_detector = XSIRWindow(
                device,
                args.window_size,
                tokenizer
            )
        elif args.watermark_type == "context":
            watermark_detector = XSIRContext(
                device,
                args.chunk_size,
                tokenizer,
                mapping_file=args.mapping_file,
                delta=args.delta,
                transform_model_path=args.transform_model,
                embedding_model=args.embedding_model
            )
        else:
            raise ValueError(f"Incorrect watermark type: {args.watermark_type}")
    elif args.watermark_method == "kgw":
        watermark_detector = KGWDetector(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=args.gamma, # should match original setting
            seed=args.seed, # should match original setting
            seeding_scheme=args.seeding_scheme, # should match original setting
            device=device, # must match the original rng device type
            tokenizer=tokenizer,
            z_threshold=4.0,
            normalizers=[],
            ignore_repeated_ngrams=True,
        )
    elif args.watermark_method == "xkgw":
        watermark_detector = XKGWDetector(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=args.gamma, # should match original setting
            hash_key=args.hash_key, # should match original setting
            context_width=args.context_width, # should match original setting
            cluster_mapping_file=args.cluster_mapping_file,
            device=device, # must match the original rng device type
            tokenizer=tokenizer,
            z_threshold=4.0,
            ignore_repeated_ngrams=True,
        )
    else:
        raise ValueError(f"Incorrect watermark method: {args.watermark_method}")

    # Load data
    done_data = read_jsonl(args.output_file) if os.path.isfile(args.output_file) else []
    detect_data = read_jsonl(args.detect_file)
    if len(detect_data) == len(done_data):
        print("All data has been processed. Exiting...")
        return

    # Detect
    detect_data = detect_data[len(done_data):]
    with torch.no_grad():
        for dd in tqdm.tqdm(detect_data):
            try:
                detect_res = watermark_detector.detect(dd["response"])
            except ValueError as e:
                if "Must have at least" in str(e):
                    # Input is too short
                    detect_res = {"z_score": None}
                else:
                    raise e
            z_score = detect_res["z_score"]
            biases = detect_res["biases"] if "biases" in detect_res else None
            if is_nan(z_score):
                z_score = None
            append_jsonl(args.output_file, {"z_score": z_score, "prompt": dd["prompt"], "response": dd["response"], "biases": biases})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare the z-scores of strings in detect_file.')
    # Model
    parser.add_argument('--base_model', type=str, required=True, help="Base model path. Only tokenizer is used.")

    # Data
    parser.add_argument('--detect_file', type=str, required=True, help="File to detect the z-scores.")
    parser.add_argument('--output_file', type=str, required=True, help="Output file to write the z-scores.")

    # Watermark
    parser.add_argument('--watermark_method', type=str, choices=["xsir", "kgw", "xkgw", "sir", "uw"], required=True, help="Watermarking method")
    parser.add_argument('--delta', type=float, default=None, help="bias of logit")
    parser.add_argument('--seed', type=int, default=0, help="Seed for watermarking")

    # X-SIR
    parser.add_argument('--watermark_type', type=str, default="context")
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--chunk_size', type=int, default=10)
    parser.add_argument('--mapping_file', type=str, default="mapping.json")
    parser.add_argument('--transform_model', type=str, default="model/transform_model_x-sbert_test.pth")
    parser.add_argument('--embedding_model', type=str, default="paraphrase-multilingual-mpnet-base-v2")

    # KGW
    parser.add_argument('--gamma', type=float, default=0.25)
    parser.add_argument('--seeding_scheme', type=str, default="minhash")

    # X-KGW
    parser.add_argument('--cluster_mapping_file', type=str, default=None, help="Path to cluster mapping JSON file (num_clusters auto-inferred)")
    parser.add_argument('--context_width', type=int, default=1, help="Context width for X-KGW")
    parser.add_argument('--hash_key', type=int, default=15485863, help="Hash key for X-KGW")

    args = parser.parse_args()

    # Manually set default value for delta based on watermark_method
    if args.watermark_method == "kgw" and args.delta is None:
        args.delta = 2
    elif args.watermark_method == "xkgw" and args.delta is None:
        args.delta = 2
    elif args.watermark_method in ["xsir", "sir"] and args.delta is None:
        args.delta = 1

    main(args)