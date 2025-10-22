import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import json


def read_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]


def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def build_token_to_cluster_map(clusters):
    """Build a mapping from token to clusters it belongs to."""
    token_to_clusters = {}
    for cluster in clusters:
        if len(cluster) >= 2:  # Only keep valid clusters
            for token in cluster:
                if token not in token_to_clusters:
                    token_to_clusters[token] = []
                token_to_clusters[token].append(cluster)
    return token_to_clusters


def count_tokens_in_valid_clusters(t_tokens, w_tokens, token_to_clusters):
    matching_tokens = []

    w_tokens_set = set(w_tokens)  # Faster lookup

    for t_token in t_tokens:
        if t_token in token_to_clusters:
            candidate_clusters = token_to_clusters[t_token]
            for cluster in candidate_clusters:
                if any(w_token in cluster for w_token in w_tokens_set):
                    matching_tokens.append(t_token)
                    break  # No need to check further clusters once matched

    token_count = len(matching_tokens)
    return token_count, matching_tokens


def main(args):
    # Load data
    t_data = read_jsonl(args.translation_file)
    w_data = read_jsonl(args.watermark_file)

    t_responses = [d["response"] for d in t_data]
    w_responses = [d["response"] for d in w_data]
    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    t_responses_tokens = [
        tokenizer.convert_ids_to_tokens(
            tokenizer(text, add_special_tokens=False)["input_ids"]
        )
        for text in t_responses
    ]
    w_responses_tokens = [
        tokenizer.convert_ids_to_tokens(
            tokenizer(text, add_special_tokens=False)["input_ids"]
        )
        for text in w_responses
    ]

    assert len(t_responses_tokens) == len(t_responses), "Tokenization mismatch"
    assert len(w_responses_tokens) == len(w_responses), "Tokenization mismatch"

    clusters = read_json(args.clusters_file)
    token_to_clusters = build_token_to_cluster_map(clusters)

    nb_total_tokens = 0
    nb_matched_tokens = 0

    for t_tokens, w_tokens in zip(t_responses_tokens, w_responses_tokens):
        token_count, _ = count_tokens_in_valid_clusters(
            t_tokens, w_tokens, token_to_clusters
        )
        nb_total_tokens += len(t_tokens)
        nb_matched_tokens += token_count

    print(f"Total tokens: {nb_total_tokens}")
    print(f"Matched tokens: {nb_matched_tokens}")
    print(f"Percentage of matched tokens: {nb_matched_tokens / nb_total_tokens:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count tokens in valid clusters")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--clusters_file", type=str, required=True)
    parser.add_argument("--watermark_file", type=str, required=True)
    parser.add_argument("--translation_file", type=str, required=True)

    args = parser.parse_args()
    main(args)
