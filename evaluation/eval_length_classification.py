import argparse
import os

from scipy import interpolate
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import read_jsonl


def tpr_at_fpr(fpr, tpr, fpr_target):
    if len(fpr) == 0 or len(tpr) == 0:
        return 0.0
    f = interpolate.interp1d(fpr, tpr, kind="linear")
    return float(f(fpr_target))


def f1_at_fpr(y_true, y_scores, fpr_target):
    if len(set(y_true)) <= 1 or len(y_scores) == 0:
        return 0.0

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    threshold = thresholds[next(i for i in range(len(fpr)) if fpr[i] > fpr_target) - 1]
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_scores)

    p_interp = interpolate.interp1d(thresholds_pr, precision[:-1], fill_value="extrapolate")
    r_interp = interpolate.interp1d(thresholds_pr, recall[:-1], fill_value="extrapolate")
    p_val = p_interp(threshold)
    r_val = r_interp(threshold)

    if p_val + r_val == 0:
        return 0.0
    return float(2 * p_val * r_val / (p_val + r_val))


def classify_text_length_by_percentiles(token_length, percentiles):
    p33, p67 = percentiles
    if token_length <= p33:
        return "short"
    elif token_length <= p67:
        return "medium"
    else:
        return "long"


def calculate_length_percentiles(text_items, tokenizer):
    token_lengths = sorted(
        len(tokenizer.encode(item.get("response", ""), add_special_tokens=False))
        for item in text_items
    )
    n = len(token_lengths)
    if n == 0:
        return (0, 0)
    return (token_lengths[int(n * 0.33)], token_lengths[int(n * 0.67)])


def evaluate_bo_by_length(base_dir, tgt_lang, tokenizer_name="CohereForAI/aya-23-8B"):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    wm_file = f"{base_dir}/mc4.{tgt_lang}.bo.z_score.jsonl"
    hum_file = f"{base_dir}/mc4.{tgt_lang}.bo.hum.z_score.jsonl"

    if not os.path.exists(wm_file) or not os.path.exists(hum_file):
        print(f"BO files not found for {tgt_lang}")
        return {}, (0, 0)

    wm_data = read_jsonl(wm_file)
    hum_data = read_jsonl(hum_file)

    n = min(len(wm_data), len(hum_data))
    wm_data = wm_data[:n]
    hum_data = hum_data[:n]

    percentiles = calculate_length_percentiles(hum_data, tokenizer)

    bins = {cat: {"wm": [], "hum": []} for cat in ["short", "medium", "long"]}

    for i in range(n):
        hum_response = hum_data[i].get("response", "")
        token_length = len(tokenizer.encode(hum_response, add_special_tokens=False))
        category = classify_text_length_by_percentiles(token_length, percentiles)

        bins[category]["wm"].append(wm_data[i].get("z_score", 0) or 0)
        bins[category]["hum"].append(hum_data[i].get("z_score", 0) or 0)

    results = {}
    for category in ["short", "medium", "long"]:
        wm_scores = bins[category]["wm"]
        hum_scores = bins[category]["hum"]
        num_samples = len(wm_scores)

        if num_samples == 0:
            continue

        y_true = [0] * len(hum_scores) + [1] * len(wm_scores)
        y_scores = hum_scores + wm_scores

        if len(set(y_true)) > 1:
            auc = roc_auc_score(y_true, y_scores)
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            tpr_01 = tpr_at_fpr(fpr, tpr, 0.1)
            tpr_001 = tpr_at_fpr(fpr, tpr, 0.01)
            f1_01 = f1_at_fpr(y_true, y_scores, 0.1)
            f1_001 = f1_at_fpr(y_true, y_scores, 0.01)
        else:
            auc = tpr_01 = tpr_001 = f1_01 = f1_001 = 0.0

        results[category] = {
            'num_samples': num_samples,
            'auc': auc,
            'tpr_01': tpr_01,
            'tpr_001': tpr_001,
            'f1_01': f1_01,
            'f1_001': f1_001,
        }

    return results, percentiles


def main(args):
    results, percentiles = evaluate_bo_by_length(
        args.base_wm_dir, args.tgt_lang, args.tokenizer
    )

    if not results:
        print(f"No results for {args.tgt_lang}")
        return

    p33, p67 = percentiles
    print(f"{args.tgt_lang} | P33={p33}, P67={p67}")
    print(f"{'Length':<8} {'N':<6} {'AUC':<8} {'TPR@0.1':<10} {'TPR@0.01':<10}")
    for category in ["short", "medium", "long"]:
        if category in results:
            r = results[category]
            print(f"{category:<8} {r['num_samples']:<6} {r['auc']:<8.3f} {r['tpr_01']:<10.3f} {r['tpr_001']:<10.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate watermark detection by text length.')
    parser.add_argument("--base_wm_dir", type=str, required=True, help="Directory with BO z-score files.")
    parser.add_argument("--tgt_lang", type=str, required=True, help="Target language code.")
    parser.add_argument("--tokenizer", type=str, default="CohereForAI/aya-23-8B", help="Tokenizer model name.")

    args = parser.parse_args()
    main(args)
