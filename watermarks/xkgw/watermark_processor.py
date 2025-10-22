# coding=utf-8
# X-KGW: Extended KGW with Semantic Cluster-based Watermarking
# Combines KGW's hash-based approach with XSIR's semantic clustering

from __future__ import annotations
import json
import collections
from math import sqrt
from functools import lru_cache
import scipy.stats
import torch
from transformers import LogitsProcessor


class WatermarkBase:
    def __init__(
        self,
        vocab: list[int] = None,
        gamma: float = 0.25,
        delta: float = 2.0,
        hash_key: int = 15485863,
        context_width: int = 1,
        cluster_mapping_file: str = None,
        num_clusters: int = None,  # Auto-inferred from mapping file if not provided
        select_green_tokens: bool = True,
        vocab_size: int = None,  # Optional: use if vocab is None
    ):
        # Vocabulary setup
        if vocab_size is not None:
            self.vocab_size = vocab_size
        elif vocab is not None:
            self.vocab_size = len(vocab)
        else:
            raise ValueError("Must provide either vocab or vocab_size")

        # Watermark parameters
        self.gamma = gamma
        self.delta = delta
        self.hash_key = hash_key
        self.context_width = context_width
        self.select_green_tokens = select_green_tokens

        # Cluster setup
        self.num_clusters = num_clusters  # Will be overridden if mapping file is loaded
        self.cluster_mapping = None
        self.rng = None

        # Load cluster mapping
        if cluster_mapping_file:
            self._load_cluster_mapping(cluster_mapping_file)

    def _load_cluster_mapping(self, mapping_file: str):
        """Load token_id -> cluster_id mapping from JSON file and infer number of clusters."""
        with open(mapping_file, 'r') as f:
            self.cluster_mapping = json.load(f)

        # Infer actual number of unique clusters from mapping
        unique_clusters = set(self.cluster_mapping)
        self.num_clusters = len(unique_clusters)
        print(f"Loaded mapping with {self.num_clusters} unique clusters for {len(self.cluster_mapping)} tokens")

        # Verify mapping covers all vocab tokens
        if len(self.cluster_mapping) != self.vocab_size:
            raise ValueError(
                f"Mapping size mismatch: {len(self.cluster_mapping)} in file != {self.vocab_size} vocab size. "
                f"Ensure the mapping file was generated for the same tokenizer/model."
            )

    def _seed_rng(self, input_ids: torch.LongTensor) -> None:
        """Seed RNG from local context."""
        if input_ids.shape[-1] < self.context_width:
            raise ValueError(f"Need at least {self.context_width} token prefix to seed RNG.")

        # Simple hash-based seeding (similar to KGW)
        tokens = input_ids[-self.context_width:]
        seed = self.hash_key * torch.prod(tokens).item()
        seed = int(seed) % (2**32 - 1)
        self.rng.manual_seed(seed)

    def _get_green_cluster_ids(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """Generate green cluster IDs based on context."""
        self._seed_rng(input_ids)

        green_cluster_count = int(self.num_clusters * self.gamma)
        cluster_permutation = torch.randperm(self.num_clusters, device=input_ids.device, generator=self.rng)

        if self.select_green_tokens:
            green_cluster_ids = cluster_permutation[:green_cluster_count]
        else:
            green_cluster_ids = cluster_permutation[(self.num_clusters - green_cluster_count):]

        return green_cluster_ids

    def _get_greenlist_ids(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """Get all token IDs that belong to green clusters."""
        if self.cluster_mapping is None:
            raise ValueError("Cluster mapping not loaded. Provide cluster_mapping_file.")

        green_cluster_ids = self._get_green_cluster_ids(input_ids)
        green_cluster_set = set(green_cluster_ids.cpu().tolist())

        # Find all tokens belonging to green clusters
        greenlist_ids = []
        for token_id in range(self.vocab_size):
            cluster_id = self.cluster_mapping[token_id]
            if cluster_id in green_cluster_set:
                greenlist_ids.append(token_id)

        return torch.tensor(greenlist_ids, device=input_ids.device)


class WatermarkLogitsProcessor(WatermarkBase, LogitsProcessor):
    """LogitsProcessor for X-KGW watermarking using semantic clusters."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _calc_greenlist_mask(self, scores: torch.FloatTensor, greenlist_token_ids) -> torch.BoolTensor:
        """Create boolean mask for green tokens."""
        green_tokens_mask = torch.zeros_like(scores, dtype=torch.bool)
        for b_idx, greenlist in enumerate(greenlist_token_ids):
            if len(greenlist) > 0:
                green_tokens_mask[b_idx][greenlist] = True
        return green_tokens_mask

    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float) -> torch.Tensor:
        """Apply bias to green tokens."""
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Apply watermark bias to logits."""
        # Initialize RNG on device
        self.rng = torch.Generator(device=input_ids.device) if self.rng is None else self.rng

        # Get greenlist for each sequence in batch
        list_of_greenlist_ids = []
        for input_seq in input_ids:
            greenlist_ids = self._get_greenlist_ids(input_seq)
            list_of_greenlist_ids.append(greenlist_ids)

        # Apply bias
        green_tokens_mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=list_of_greenlist_ids)
        scores = self._bias_greenlist_logits(scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.delta)

        return scores


class WatermarkDetector(WatermarkBase):
    """Detector for X-KGW watermarked text using semantic clusters."""

    def __init__(
        self,
        *args,
        device: torch.device = None,
        tokenizer = None,
        z_threshold: float = 4.0,
        ignore_repeated_ngrams: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        assert device, "Must pass device"
        assert tokenizer, "Need tokenizer instance for detection"

        self.tokenizer = tokenizer
        self.device = device
        self.z_threshold = z_threshold
        self.ignore_repeated_ngrams = ignore_repeated_ngrams
        self.rng = torch.Generator(device=self.device)

    def _compute_z_score(self, observed_count, T):
        """Compute z-score for watermark detection."""
        expected_count = self.gamma
        numer = observed_count - expected_count * T
        denom = sqrt(T * expected_count * (1 - expected_count))
        z = numer / denom
        return z

    def _compute_p_value(self, z):
        """Compute p-value from z-score."""
        p_value = scipy.stats.norm.sf(z)
        return p_value

    def _is_green_token(self, prefix: tuple[int], target: int) -> bool:
        """Check if token is green by checking its cluster membership."""
        if self.cluster_mapping is None:
            raise ValueError("Cluster mapping not loaded.")

        # Get green cluster IDs (fast - just permutation of ~100 clusters)
        green_cluster_ids = self._get_green_cluster_ids(torch.as_tensor(prefix, device=self.device))
        green_cluster_set = set(green_cluster_ids.cpu().tolist())

        # Check if token's cluster is green (O(1) lookup)
        target_cluster = self.cluster_mapping[target]
        return target_cluster in green_cluster_set

    def _score_sequence(
        self,
        input_ids: torch.Tensor,
        return_num_tokens_scored: bool = True,
        return_num_green_tokens: bool = True,
        return_green_fraction: bool = True,
        return_green_token_mask: bool = False,
        return_z_score: bool = True,
        return_p_value: bool = True,
    ):
        """Score a sequence for watermark detection."""
        if len(input_ids) - self.context_width < 1:
            raise ValueError(
                f"Must have at least 1 token to score after "
                f"the first {self.context_width} tokens required for seeding."
            )

        # Convert to list once (like KGW does) to avoid repeated .cpu().tolist() calls
        input_ids_list = input_ids.cpu().tolist()

        # Score each token based on its cluster
        green_token_mask = []
        ngram_to_watermark = {}
        frequencies = collections.Counter()

        for idx in range(self.context_width, len(input_ids_list)):
            prefix = tuple(input_ids_list[idx - self.context_width:idx])
            target = input_ids_list[idx]

            ngram = prefix + (target,)

            if ngram not in ngram_to_watermark:
                is_green = self._is_green_token(prefix, target)
                ngram_to_watermark[ngram] = is_green

            green_token_mask.append(ngram_to_watermark[ngram])
            frequencies[ngram] += 1

        # Count green tokens
        if self.ignore_repeated_ngrams:
            num_tokens_scored = len(ngram_to_watermark)
            green_token_count = sum(ngram_to_watermark.values())
        else:
            num_tokens_scored = sum(frequencies.values())
            green_token_count = sum(freq * ngram_to_watermark[ngram] for ngram, freq in frequencies.items())

        # Build output dictionary
        score_dict = {}
        if return_num_tokens_scored:
            score_dict['num_tokens_scored'] = num_tokens_scored
        if return_num_green_tokens:
            score_dict['num_green_tokens'] = green_token_count
        if return_green_fraction:
            score_dict['green_fraction'] = green_token_count / num_tokens_scored if num_tokens_scored > 0 else 0
        if return_z_score:
            z_score = self._compute_z_score(green_token_count, num_tokens_scored)
            score_dict['z_score'] = z_score
        if return_p_value:
            z_score = score_dict.get('z_score', self._compute_z_score(green_token_count, num_tokens_scored))
            score_dict['p_value'] = self._compute_p_value(z_score)
        if return_green_token_mask:
            score_dict['green_token_mask'] = green_token_mask

        return score_dict

    def detect(
        self,
        text: str = None,
        tokenized_text: list[int] = None,
        return_prediction: bool = True,
        return_scores: bool = True,
        z_threshold: float = None,
        **kwargs,
    ) -> dict:
        """Detect watermark in text."""
        assert (text is not None) ^ (tokenized_text is not None), "Must pass either raw or tokenized text"

        if return_prediction:
            kwargs['return_p_value'] = True

        # Tokenize if needed
        if tokenized_text is None:
            tokenized_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.device)
            if tokenized_text[0] == self.tokenizer.bos_token_id:
                tokenized_text = tokenized_text[1:]
        else:
            if isinstance(tokenized_text, list):
                tokenized_text = torch.tensor(tokenized_text, device=self.device)

        # Score sequence
        score_dict = self._score_sequence(tokenized_text, **kwargs)

        output_dict = {}
        if return_scores:
            output_dict.update(score_dict)

        # Make prediction
        if return_prediction:
            z_threshold = z_threshold if z_threshold else self.z_threshold
            assert z_threshold is not None, "Need threshold for detection"
            output_dict['prediction'] = score_dict['z_score'] > z_threshold
            if output_dict['prediction']:
                output_dict['confidence'] = 1 - score_dict['p_value']

        return output_dict
