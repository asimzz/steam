import json
import jieba
from opencc import OpenCC
from datasets import load_dataset
from tqdm import tqdm
import os

T2S = OpenCC('t2s')
N = 500  # Number of new samples to generate

for lang in ["en"]:
    # === Load existing prompt-response pairs ===
    existing_file = f"mc4.{lang}.jsonl"
    existing_pairs = set()
    if os.path.exists(existing_file):
        with open(existing_file, "r") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    pair = (d["prompt"], d["response"])
                    existing_pairs.add(pair)
                except Exception:
                    continue

    # === Begin new data collection ===
    bar = tqdm(total=N, desc=f"Generating new for {lang}")
    ds = load_dataset("allenai/c4", lang, streaming=True, split="validation")

    new_prompts = []
    new_responses = []

    for s in ds:
        text = s["text"]
        if lang == "zh":
            text = T2S.convert(text)

        tokens = list(jieba.cut(text))

        if 195 <= len(tokens) <= 205:
            split_index = int(len(tokens) * 0.1)
            prompt = "".join(tokens[:split_index])
            response = "".join(tokens[split_index:])

            pair = (prompt, response)
            if pair in existing_pairs:
                continue

            new_prompts.append(prompt)
            new_responses.append(response)
            bar.update(1)

            if len(new_prompts) == N:
                break

    assert len(new_prompts) == N, f"Only found {len(new_prompts)} new samples."

    # === Save new samples to a separate file ===
    new_file = f"mc4.{lang}.val.jsonl"
    with open(new_file, "w") as f:
        for p, r in zip(new_prompts, new_responses):
            f.write(json.dumps({"prompt": p, "response": r}, ensure_ascii=False) + "\n")

    print(f"✅ Saved {N} new samples to {new_file}")
