# Build a unified dictionary from external dictionaries
import argparse
from tqdm import tqdm
from utils import transform

def augment_dictionary(raw_entries, append_meta_symbols):
    augmented_entries = []
    for src, tgt in tqdm(raw_entries, desc="Augmenting dictionary"):
        src_tokens = transform(src, append_meta_symbols)
        tgt_tokens = transform(tgt, append_meta_symbols)
        for src_token in src_tokens:
            for tgt_token in tgt_tokens:
                if src_token != tgt_token:
                    augmented_entries.append((src_token, tgt_token))

    # deduplicate
    augmented_entries = list(set(augmented_entries))
    return augmented_entries


def main(args):
    # Read data
    raw_entries = [] # list of tuples (src, tgt)
    for d_path in args.dicts:
        with open(d_path, "r") as f:
            for line in f:
                src, tgt = line.strip().split()
                if src != tgt:
                    raw_entries.append((src, tgt))

    # Augment dictionary
    augmented_entries = augment_dictionary(raw_entries, args.append_meta_symbols)

    # Write data
    with open(args.output_file, "w") as f:
        for src, tgt in augmented_entries:
            f.write(f"{src}\t{tgt}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a unified dictionary from external dictionaries"
    )
    parser.add_argument(
        "--dicts", type=str, nargs="+", help="multiple external dictionaries"
    )
    parser.add_argument("--output_file", type=str, help="output dictionary")
    parser.add_argument(
        "--append_meta_symbols",
        action="store_true",
        help="append meta symbols to the tokens",
    )

    args = parser.parse_args()
    main(args)
