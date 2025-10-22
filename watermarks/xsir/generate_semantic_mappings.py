import os
import json
import random
import argparse
import networkx as nx
from transformers import AutoTokenizer

# Maximum size of connected components (used to decide clustering granularity)
CC_MAX_SIZE = 250

def main():
    parser = argparse.ArgumentParser(description='Generate token-to-cluster mappings.')
    parser.add_argument('--model', type=str, required=True, help='Model name or path')
    parser.add_argument('--dictionary', type=str, required=True, help='Dictionary file (one edge per line)')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save token->cluster mapping JSON')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    args = parser.parse_args()

    random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    vocab = tokenizer.get_vocab()

    # Load edges from dictionary
    edges = []
    with open(args.dictionary) as f:
        for line in f:
            src_token, tgt_token = line.strip().split()
            edges.append((src_token, tgt_token))

    # Add self-loop for all vocab tokens
    for token in vocab:
        edges.append((token, token))

    graph = nx.Graph(edges)
    connected_components_node = list(nx.connected_components(graph))
    connected_components_node.sort(key=len, reverse=True)
    connected_components_graph = [graph.subgraph(ccn) for ccn in connected_components_node]

    clusters = []
    for ccg in connected_components_graph:
        if len(ccg) <= CC_MAX_SIZE:
            clusters.append(list(ccg))
        else:
            resolution = 1 if len(ccg) <= 10000 else 10
            cs = nx.community.louvain_communities(ccg, seed=args.seed, resolution=resolution)
            clusters.extend(cs)
            print(f"[Seed {args.seed}] Split {len(ccg)} nodes into {len(cs)} clusters")

    # Filter clusters to keep only tokens in vocab
    valid_clusters = []
    for c in clusters:
        valid_c = [token for token in c if token in vocab]
        if valid_c:
            valid_clusters.append(valid_c)

    # Flatten clusters to check vocabulary coverage
    all_valid_tokens = [token for cluster in valid_clusters for token in cluster]
    assert len(all_valid_tokens) == len(vocab), \
        f"Vocabulary mismatch: {len(all_valid_tokens)} tokens mapped vs {len(vocab)} in vocab"

    # Generate mapping
    cluster_ids = [random.randint(0, 299) for _ in range(len(valid_clusters))]
    mapping = [None] * len(vocab)
    for i, cluster in enumerate(valid_clusters):
        for token in cluster:
            token_id = tokenizer.convert_tokens_to_ids(token)
            assert mapping[token_id] is None
            mapping[token_id] = cluster_ids[i]

    assert all(x is not None for x in mapping), "Incomplete token-to-cluster mapping"

    # Save token-to-cluster mapping
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(mapping, f, indent=4)

    # Save clusters
    cluster_path = args.output_file.replace(".json", "_clusters.json")
    valid_clusters.sort(key=len, reverse=True)
    with open(cluster_path, "w") as f:
        json.dump(valid_clusters, f, indent=4, ensure_ascii=False)

    # Print stats
    print(f"[Seed {args.seed}] Vocabulary size: {len(vocab)}")
    print(f"[Seed {args.seed}] Number of clusters: {len(valid_clusters)}")
    print(f"[Seed {args.seed}] Clusters with ≥2 tokens: {sum(len(c) >= 2 for c in valid_clusters)}")
    print(f"[Seed {args.seed}] Vocab coverage (%): {sum(len(c) for c in valid_clusters if len(c) >= 2) / len(vocab) * 100:.2f}")
    print(f"[Seed {args.seed}] Top 5 largest clusters: {[len(c) for c in sorted(valid_clusters, key=len, reverse=True)[:5]]}")

if __name__ == '__main__':
    main()