import argparse
from transformers import AutoTokenizer
import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def read_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

def plot_token_distribution(tokens_list, base_model, tgt_lang, org_lang, seed, figure_dir,top_percent=90,top_k=10):
    # Count token frequency
    total_tokens = len(tokens_list)
    token_counts = Counter(tokens_list)

    # Convert to percentage
    token_percentages = {token: (count / total_tokens) * 100 for token, count in token_counts.items()}
    sorted_tokens = sorted(token_percentages.items(), key=lambda x: x[1], reverse=True)

    # Compute number of tokens to cover top_percent%
    cumulative = 0.0
    selected_tokens = []
    for token, pct in sorted_tokens:
        if cumulative >= top_percent:
            break
        selected_tokens.append((token, pct))
        cumulative += pct

    # tokens, percentages = zip(*selected_tokens)
    # print(f"Number of tokens covering top {top_percent}%: {len(tokens)}")
    # print(f"Summation of selected token percentages: {sum(percentages):.3f}% of total tokens")
    # print(f"Percentage of selected tokens: {len(tokens) / len(set(tokens_list)) * 100:.3f}%")
    # print(f"Total tokens: {len(set(tokens_list))}")
    most_common = sorted_tokens[:top_k]
    
    # Prepare data for plotting
    tokens, percentages = zip(*most_common)
    plt.figure(figsize=(7, 5))
    sns.barplot(y=list(tokens[:top_k]), x=list(percentages[:top_k]), hue=list(tokens[:top_k]), palette="viridis", legend=False)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("Percentage (%)", fontsize=20)
    plt.ylabel("Token", fontsize=20)
    # plt.title("Token Distribution")
    # plt.title(f"Top {top_k} Token Distribution for {base_model} in {tgt_lang} with seed {seed}")
    plt.savefig(f"{figure_dir}/{base_model}/{tgt_lang}/{org_lang}_top_{top_k}_token_distribution.png", bbox_inches='tight')
    return

def main(args):
    t_data = read_jsonl(args.translation_file)
    t_responses = [d["response"] for d in t_data]
    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    t_responses_tokens = [
        tokenizer.convert_ids_to_tokens(
            tokenizer(text, add_special_tokens=False)["input_ids"]
        )
        for text in t_responses
    ]

    tokens_list = []
    for text_tokens in t_responses_tokens:
        tokens_list.extend(text_tokens)

    plot_token_distribution(tokens_list, args.base_model, args.tgt_lang, args.org_lang, args.seed, args.figure_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Token Distribution Plotter")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--tgt_lang", type=str, required=True)
    parser.add_argument("--org_lang", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--figure_dir", type=str, required=True)
    parser.add_argument("--translation_file", type=str, required=True)

    args = parser.parse_args()
    main(args)
