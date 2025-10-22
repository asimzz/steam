from __future__ import annotations
import os
import json
import torch
import random
import scipy.stats
import numpy as np
import sentence_transformers

from math import sqrt
from transformers import LogitsProcessor
from transformers import BertModel, AutoTokenizer
from .train_watermark_model import TransformModel
from sentence_transformers import SentenceTransformer


class WatermarkBase:
    def __init__(self, gamma: float, delta: float, target_tokenizer, vocab_size=None):
        self.target_tokenizer = target_tokenizer
        self.vocab_size = self.target_tokenizer.vocab_size
        self.vocab_size = (
            vocab_size
            if vocab_size is not None
            else max(
                len(self.target_tokenizer.get_vocab()), self.target_tokenizer.vocab_size
            )
        )
        self.gamma = gamma
        self.delta = delta

    def _get_greenlist_ids(self, input_ids: torch.LongTensor):
        pass

    def _compute_z_score(self, observed_count, T):
        expected_count = self.gamma
        numer = observed_count - expected_count * T
        denom = sqrt(T * expected_count * (1 - expected_count))
        z = numer / denom
        return z

    def _compute_p_value(self, z):
        p_value = scipy.stats.norm.sf(z)
        return p_value

    def detect(self, text):
        pass

    def _get_bias(self, input_ids: torch.LongTensor) -> list[int]:
        green_list_ids = self._get_greenlist_ids(input_ids).cpu().numpy()
        bias = np.zeros(self.vocab_size, dtype=int)
        bias[green_list_ids] = 1
        return bias


class WatermarkContext(WatermarkBase):
    def __init__(
        self,
        device: torch.device,
        chunk_length,
        target_tokenizer,
        delta: float = 4.0,
        gamma: float = 0.5,
        embedding_model: str = "",
        mapping_file: str = "",
        transform_model_path: str = "transform_model.pth",
        vocab_size=None,
    ):
        super().__init__(gamma, delta, target_tokenizer, vocab_size)
        assert embedding_model in [
            "perceptiveshawty/compositional-bert-large-uncased",
            "paraphrase-multilingual-mpnet-base-v2",
        ], f"embedding_model {embedding_model} not supported"

        self.device = device
        if embedding_model == "perceptiveshawty/compositional-bert-large-uncased":
            self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model)
            self.embedding_model = BertModel.from_pretrained(embedding_model).to(
                self.device
            )
            self.input_dim = 1024
        elif embedding_model == "paraphrase-multilingual-mpnet-base-v2":
            self.embedding_tokenizer = None
            self.embedding_model = SentenceTransformer(embedding_model)
            self.input_dim = 768
        self.chunk_length = chunk_length
        self.transform_model = TransformModel(input_dim=self.input_dim)
        self.transform_model.load_state_dict(torch.load(transform_model_path))
        self.transform_model.to(self.device)
        self.transform_model.to(torch.float32)

        if os.path.exists(mapping_file):
            print(f"Loading mapping from {mapping_file}")
            with open(mapping_file, "r") as f:
                self.mapping = json.load(f)
        else:
            raise ValueError(f"Mapping file {mapping_file} not found")
            print(
                f"No mapping found, creating new mapping and saving to {mapping_file}"
            )
            water_mark_dim = self.transform_model.layers[-1].out_features
            self.mapping = [
                random.randint(0, water_mark_dim - 1)
                for _ in range(len(target_tokenizer))
            ]
            os.makedirs(os.path.dirname(mapping_file), exist_ok=True)
            with open(mapping_file, "w") as f:
                json.dump(self.mapping, f, indent=4)

    def get_embedding(self, sentence):
        if isinstance(self.embedding_model, sentence_transformers.SentenceTransformer):
            # SentenceTransformer
            emb = self.embedding_model.encode(
                sentence, show_progress_bar=False, convert_to_tensor=True
            ).to(self.device)
            return emb[None, :]
        else:
            # C-BERT
            input_ids = self.embedding_tokenizer.encode(
                sentence,
                return_tensors="pt",
                max_length=512,
                truncation="longest_first",
            )
            input_ids = input_ids.to(self.device)
            with torch.no_grad():
                output = self.embedding_model(input_ids)
            return output[0][:, 0, :]

    def get_context_sentence(self, input_ids: torch.LongTensor):
        input_sentence = self.target_tokenizer.decode(
            input_ids, skip_special_tokens=True
        )
        input_tokens = self.target_tokenizer.tokenize(input_sentence)

        word_2d = [
            input_tokens[x : x + self.chunk_length]
            for x in range(0, len(input_tokens), self.chunk_length)
        ]

        if len(word_2d) > 0 and len(word_2d[-1]) == self.chunk_length:
            return input_sentence
        else:
            return self.target_tokenizer.convert_tokens_to_string(
                [tok for group in word_2d[:-1] for tok in group]
            )

    def get_text_split(self, sentence):
        words = self.target_tokenizer.tokenize(sentence)
        return [
            words[x : x + self.chunk_length]
            for x in range(0, len(words), self.chunk_length)
        ]

    def scale_vector(self, v):
        mean = np.mean(v)
        v_minus_mean = v - mean
        v_minus_mean = np.tanh(1000 * v_minus_mean)
        return v_minus_mean

    def detect(self, text: str = None):
        word_2d = self.get_text_split(text)
        all_value = []
        biases = []
        for i in range(1, len(word_2d)):
            context_sentence = self.target_tokenizer.convert_tokens_to_string(
                [tok for group in word_2d[0:i] for tok in group]
            )
            context_embedding = self.get_embedding(context_sentence)
            output = self.transform_model(context_embedding).cpu()[0].detach().numpy()
            similarity_array = self.scale_vector(output)[self.mapping]

            tokens = word_2d[i]
            token_ids = self.target_tokenizer.convert_tokens_to_ids(tokens)

            for tok, tok_ids in zip(tokens, token_ids):
                all_value.append(-float(similarity_array[tok_ids]))
                biases.append((tok, -float(similarity_array[tok_ids])))

        return {"z_score": np.mean(all_value), "biases": biases}

    def _get_bias(self, input_ids: torch.LongTensor) -> list[int]:
        context_sentence = self.get_context_sentence(input_ids)
        context_embedding = self.get_embedding(context_sentence)
        output = self.transform_model(context_embedding).cpu()[0].numpy()
        similarity_array = self.scale_vector(output)[self.mapping]
        return -similarity_array


class WatermarkWindow(WatermarkBase):
    def __init__(
        self,
        device,
        window_size,
        target_tokenizer,
        gamma: float = 0.5,
        delta: float = 2.0,
        hash_key: int = 15485863,
        vocab_size=None,
    ):
        super().__init__(gamma, delta, target_tokenizer, vocab_size)
        self.device = device
        self.rng = torch.Generator(device=device)
        self.hash_key = hash_key
        self.window_size = window_size

    def detect(self, text: str = None):
        input_ids = self.target_tokenizer.encode(text, add_special_tokens=False)
        count, total = 0, 0
        t_v_pair = []
        input_symbols = self.target_tokenizer.convert_ids_to_tokens(input_ids)
        for i in range(self.window_size, len(input_ids)):
            greenlist_ids = self._get_greenlist_ids(torch.tensor(input_ids[:i]))
            if input_ids[i] in greenlist_ids:
                count += 1
                t_v_pair.append((input_symbols[i], 1))
            else:
                t_v_pair.append((input_symbols[i], 0))
            total += 1
        return {"z_score": (count - (total - count)) / total}

    def _seed_rng(self, input_ids: torch.LongTensor):
        if self.window_size == 0:
            seed = self.hash_key
        else:
            tokens = input_ids[-self.window_size :]
            seed = self.hash_key * torch.prod(tokens).item()
            seed = seed % (2**32 - 1)
        self.rng.manual_seed(int(seed))

    def _get_greenlist_ids(self, input_ids: torch.LongTensor) -> list[int]:
        self._seed_rng(input_ids)
        greenlist_size = int(self.vocab_size * self.gamma)
        vocab_permutation = torch.randperm(
            self.vocab_size, device=self.device, generator=self.rng
        )
        greenlist_ids = vocab_permutation[:greenlist_size]
        return greenlist_ids


class WatermarkLogitsProcessor(LogitsProcessor):

    def __init__(self, watermark_base: WatermarkBase, *args, **kwargs):
        self.watermark_base = watermark_base

    def _bias_logits(
        self, scores: torch.Tensor, batched_bias: torch.Tensor, greenlist_bias: float
    ) -> torch.Tensor:
        scores_vocab_size = scores.shape[1]
        batched_bias_np = np.array(batched_bias)
        batched_bias_tensor = torch.Tensor(batched_bias_np).to(
            self.watermark_base.device
        )
        bias_vocab_size = batched_bias_tensor.shape[1]

        if scores_vocab_size > bias_vocab_size:
            # Expand bias tensor to match scores tensor size
            bias_padded = torch.zeros(
                (batched_bias_tensor.shape[0], scores_vocab_size), device=scores.device
            )
            bias_padded[:, :bias_vocab_size] = (
                batched_bias_tensor  # Fill with actual bias values
            )
            batched_bias_tensor = bias_padded

        scores = scores + batched_bias_tensor * greenlist_bias
        return scores

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        batched_bias = [None for _ in range(input_ids.shape[0])]

        for b_idx in range(input_ids.shape[0]):
            current_bias = self.watermark_base._get_bias(input_ids[b_idx])
            batched_bias[b_idx] = current_bias

        scores = self._bias_logits(
            scores=scores,
            batched_bias=batched_bias,
            greenlist_bias=self.watermark_base.delta,
        )
        return scores
