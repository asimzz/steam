# STEAM: Simple Translation-Enhanced Approach for Multilingual Watermarking

Official implementation of **"Is Multilingual LLM Watermarking Truly Multilingual? A Simple Back-Translation Solution"**.

In this work, we introduce STEAM (Simple Translation-Enhanced Approach for Multilingual watermarking), a novel defense mechanism designed to enhance the robustness of LLM watermarks against translation-based attacks

---

## Installation

This project requires **Python 3.10.17**.

### 1. Create and activate a virtual environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Overview

STEAM is designed as a plug-in defense layer that works with existing watermarking frameworks such as **X-SIR**, **X-KGW**, and **KGW**.

---

## Basic Workflow

```bash
# 1. Prepare bilingual dictionaries
bash data/dictionary/download_dictionaries.sh new_supported
bash data/dictionary/build_dictionaries.sh new_supported

# 2. Generate semantic mappings
bash evaluation/scripts/generate_mapping.sh new_supported

# 3. Generate watermarked and human text
bash evaluation/scripts/generate_watermark.sh new_supported
bash evaluation/scripts/generate_human.sh new_supported

# 4. Evaluate detection performance
bash evaluation/scripts/evaluate_detection.sh new_supported
```

### Categories

| Category             | Description                                                              |
| -------------------- | ------------------------------------------------------------------------ |
| `new_supported`      | Run experiments with the new set of supported languages                  |
| `original_supported` | Use only the original supported languages (`en`, `fr`, `de`, `zh`, `ja`) |
| `unsupported`        | Evaluate unsupported languages                                           |

Languages for each category can be configured in `evaluation/common/languages.sh` files.

---

## Code Structure

```
STEAM/
├── gen.py                        # Generate watermarked text
├── detect.py                     # Compute z-scores for detection
├── utils.py                      # Shared utility functions
│
├── data/
│   ├── dataset/                  # MC4 prompts (en, fr, de, zh, etc.)
│   ├── dictionary/               # Bilingual dictionaries (MUSE-based)
│   ├── mapping/                  # Semantic mappings (X-SIR / X-KGW)
│   └── model/                    # Pretrained transform models
│
├── evaluation/
│   ├── scripts/                  # Automated generation & evaluation scripts
│   ├── common/                   # Shared configs (models, languages)
│   └── eval_detection.py         # Computes AUC, TPR@FPR, F1
│
└── watermarks/
    ├── xsir/                     # X-SIR implementation
    ├── xkgw/                     # X-KGW implementation
    └── kgw/                      # KGW implementation
```

---

## Core Components

### 1. Text Generation (`gen.py`)

Generates watermarked or baseline text from prompts.

#### Example: Generate X-SIR Watermarked Text

```bash
python gen.py \
  --base_model meta-llama/Llama-3.2-1B \
  --input_file data/dataset/mc4.en.jsonl \
  --output_file evaluation/gen/llama-3.2-1B/new_supported/xsir_seed0/mc4.en.mod.jsonl \
  --watermark_method xsir \
  --watermark_type context \
  --mapping_file data/mapping/xsir/new_supported/mapping.json \
  --transform_model data/model/transform_model_x-sbert.pth
```

**Key Arguments**

- `--watermark_method`: `xsir`, `xkgw`, `kgw`, or `none`
- `--mapping_file`: Required for X-SIR and X-KGW methods

---

### 2. Watermark Detection (`detect.py`)

Computes z-scores for watermark detection.

#### Example

```bash
python detect.py \
  --base_model meta-llama/Llama-3.2-1B \
  --detect_file evaluation/gen/llama-3.2-1B/new_supported/xsir_seed0/mc4.en.mod.jsonl \
  --output_file evaluation/gen/llama-3.2-1B/new_supported/xsir_seed0/mc4.en.mod.z_score.jsonl \
  --watermark_method xsir \
  --watermark_type context \
  --mapping_file data/mapping/xsir/new_supported/mapping.json \
  --transform_model data/model/transform_model_x-sbert.pth
```

---

### 3. Evaluation (`eval_detection.py`)

Computes detection performance metrics including **AUC**, **TPR@FPR**, **F1**, and **ROC curves**.

#### Example

```bash
python evaluation/eval_detection.py \
  --hm_zscore evaluation/gen/llama-3.2-1B/new_supported/xsir_seed0/mc4.en-fr.hum.z_score.jsonl \
  --wm_zscore evaluation/gen/llama-3.2-1B/new_supported/xsir_seed0/mc4.en-fr.mod.z_score.jsonl
```

---

## Evaluation Workflow

### Step 1: Prepare Dictionaries

```bash
bash data/dictionary/download_dictionaries.sh new_supported
bash data/dictionary/build_dictionaries.sh new_supported

# to build a holdout dictionary by excluding a specific language:
bash data/dictionary/build_dictionaries.sh holdout en
```

### Step 2: Generate Data

```bash
bash evaluation/scripts/generate_mapping.sh new_supported
bash evaluation/scripts/generate_watermark.sh new_supported
bash evaluation/scripts/generate_human.sh new_supported

# for holdout settings, pass the excluded language as an argument
bash evaluation/scripts/generate_mapping.sh holdout en
bash evaluation/scripts/generate_watermark_holdout.sh en
bash evaluation/scripts/generate_human_holdout.sh en
```

### Step 3: Evaluate Detection

```bash
bash evaluation/scripts/evaluate_detection.sh new_supported

# for holdout settings:
bash evaluation/scripts/evaluate_detection_holdout.sh en
```

This will iterate over:

- Base models (defined in `evaluation/common/config.sh`)
- Seeds (default: 0, 42, 123)
- Watermark methods (`xsir`, `xkgw`, `kgw`)
- Languages (defined in `evaluation/common/languages.sh`)

Outputs are stored under:

```
evaluation/gen/<model>/<category>/<method>_seed<seed>/
```

---

## Configuration

To modify experiment settings:

Edit `evaluation/common/config.sh` and `evaluation/common/utils.sh` to change:

- Base models
- Seeds
- Watermark schemes
- Generation parameters

---
