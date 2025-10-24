#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Paper-style token distribution plots with gradient palettes.
Fixes overlapped x-ticks by:
  - Wider default figure size (9.5 x 5.5 in)
  - Explicit major/minor x tick spacing (5% / 2.5% by default)
  - Extra bottom padding before saving

Usage examples at bottom.
"""

import argparse
import os
import json
import math
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter, MultipleLocator
import matplotlib.patheffects as pe
from matplotlib import colors as mcolors
from transformers import AutoTokenizer


# ----------------------------
# Palettes (same family as your scatter script)
# ----------------------------
PALETTES = {
    "okabe_ito": [
        "#0072B2", "#E69F00", "#56B4E9", "#009E73",
        "#F0E442", "#D55E00", "#CC79A7", "#000000"
    ],
    "tableau10": [
        "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
        "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"
    ],
    "set2": [
        "#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3",
        "#A6D854", "#FFD92F", "#E5C494", "#B3B3B3"
    ],
    "paired": [
        "#A6CEE3", "#1F78B4", "#B2DF8A", "#33A02C", "#FB9A99", "#E31A1C",
        "#FDBF6F", "#FF7F00", "#CAB2D6", "#6A3D9A", "#FFFF99", "#B15928"
    ],
    "dark2": [
        "#1B9E77", "#D95F02", "#7570B3", "#E7298A",
        "#66A61E", "#E6AB02", "#A6761D", "#666666"
    ],
    "colorblind10": [
        "#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC",
        "#CA9161", "#FBAFE4", "#949494", "#ECE133", "#56B4E9"
    ],
}

def use_palette(name: str, n: int):
    if name in PALETTES:
        base = PALETTES[name]
        if n <= len(base):
            return base[:n]
        return (base * ((n + len(base) - 1) // len(base)))[:n]
    return sns.color_palette(name, n)


# ----------------------------
# JSONL helper
# ----------------------------
def read_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ----------------------------
# Style helpers (paper-ready, like your scatter)
# ----------------------------
def set_paper_style(axis_fontsize=18, label_fontsize=20):
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.family": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.labelsize": label_fontsize,
        "xtick.labelsize": axis_fontsize,
        "ytick.labelsize": axis_fontsize,
        "legend.fontsize": axis_fontsize,
        "axes.linewidth": 1.3,
        "figure.dpi": 300,
        "savefig.transparent": True,
        "xtick.major.width": 1.3,
        "ytick.major.width": 1.3,
    })


def _nice_upper_limit(values, step=5.0):
    """Ceil max(values) to the next multiple of `step` for a clean percent axis."""
    vmax = max(values) if values else 100.0
    return math.ceil(vmax / step) * step


def _annotate_bars(ax, fmt="{:.1f}%"):
    """Value labels at bar ends with a white halo for legibility on print."""
    for p in ax.patches:
        width = p.get_width()
        y = p.get_y() + p.get_height() / 2.0
        ax.text(
            width + 0.5, y, fmt.format(width),
            va="center", ha="left",
            fontsize=ax.yaxis.get_ticklabels()[0].get_fontsize() if ax.yaxis.get_ticklabels() else 12,
            path_effects=[pe.withStroke(linewidth=2.0, foreground="white")],
        )


# ----------------------------
# Gradient palette builders
# ----------------------------
def gradient_palette(base_color, n, lightness_range=(0.35, 0.85)):
    """
    Generate n gradient shades of a base color using HLS lightness ramp.
    base_color: hex or any matplotlib color
    lightness_range: (min_L, max_L) in [0,1]
    """
    import colorsys
    base_rgb = mcolors.to_rgb(base_color)
    h, l, s = colorsys.rgb_to_hls(*base_rgb)
    lo, hi = lightness_range
    if n <= 1:
        return [colorsys.hls_to_rgb(h, (lo+hi)/2.0, s)]
    lightnesses = [lo + (hi - lo) * (i / (n - 1)) for i in range(n)]
    return [colorsys.hls_to_rgb(h, li, s) for li in lightnesses]


def multi_gradient_cycle(base_colors, n, lightness_range=(0.35, 0.85)):
    """
    Make a palette of length n by cycling through base_colors and
    expanding each into a local gradient chunk.
    """
    shades = []
    if n <= 0:
        return shades
    per = max(1, n // max(1, len(base_colors)))
    remainder = n - per * len(base_colors)
    for idx, base in enumerate(base_colors):
        k = per + (1 if idx < remainder else 0)
        if k <= 0:
            continue
        shades.extend(gradient_palette(base, k, lightness_range))
    return shades[:n]


def build_bar_palette(n_bars, mode="single", base_color="#0072B2",
                      base_colors=None, palette_name="okabe_ito",
                      lightness_range=(0.35, 0.85)):
    """
    Return a list of n_bars colors according to the chosen mode:
      - mode="single": one base color -> gradient across bars
      - mode="cycle" : cycle multiple base colors, each expanded to a gradient
    """
    if mode == "single":
        return gradient_palette(base_color, n_bars, lightness_range)
    if not base_colors:
        base_colors = use_palette(palette_name, n=8)
    base_colors = [mcolors.to_hex(c) for c in base_colors]
    return multi_gradient_cycle(base_colors, n_bars, lightness_range)


# ----------------------------
# Shared plotting scaffold
# ----------------------------
def _barplot_percent(ax, tokens, percentages, palette, x_label,
                     x_major_step=5.0, x_minor_step=2.5):
    """Shared barplot styling for single and averaged plots."""
    sns.barplot(
        x=list(percentages), y=list(tokens), hue=list(tokens), legend=False,
        ax=ax, orient="h", palette=palette, edgecolor="black"
    )

    # Axis labels & grids (major strong, minor faint) like your scatter
    ax.set_xlabel(x_label, labelpad=10)
    ax.set_ylabel("Token", labelpad=8)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))

    # Explicit tick spacing to avoid label crowding
    if x_major_step and x_major_step > 0:
        ax.xaxis.set_major_locator(MultipleLocator(x_major_step))
    if x_minor_step and x_minor_step > 0:
        ax.xaxis.set_minor_locator(MultipleLocator(x_minor_step))

    ax.grid(True, axis="x", which="major", linestyle="--", linewidth=0.9, alpha=0.45)
    ax.grid(True, axis="x", which="minor", linestyle=":", linewidth=0.6, alpha=0.25)
    ax.tick_params(axis="both", which="major", width=1.3, length=6, direction="out")
    ax.tick_params(axis="both", which="minor", width=1.0, length=4, direction="out")
    ax.set_axisbelow(True)

    xmax = _nice_upper_limit(list(percentages), step=max(1.0, x_major_step)) if percentages else 100.0
    ax.set_xlim(0, xmax)

    _annotate_bars(ax, fmt="{:.1f}%")


# ----------------------------
# Main plotting functions
# ----------------------------
def plot_token_distribution(tokens_list, base_model, tgt_lang, org_lang, figure_dir,
                            top_percent=90, top_k=10,
                            axis_fontsize=18, label_fontsize=20,
                            palette_mode="single",         # "single" or "cycle"
                            base_color="#0072B2",          # used if mode="single"
                            base_colors_name="okabe_ito",  # used if mode="cycle"
                            lightness_min=0.35, lightness_max=0.85,
                            fig_width=9.5, fig_height=5.5,  # wider defaults to prevent overlap
                            x_major_step=5.0, x_minor_step=2.5,
                            bottom_pad=0.25):
    """Single-file top-k token percentage plot."""
    set_paper_style(axis_fontsize=axis_fontsize, label_fontsize=label_fontsize)

    total_tokens = max(len(tokens_list), 1)
    token_counts = Counter(tokens_list)
    token_percentages = {tok: (cnt / total_tokens) * 100.0 for tok, cnt in token_counts.items()}
    sorted_tokens = sorted(token_percentages.items(), key=lambda x: x[1], reverse=True)

    # Optional coverage calc to top_percent
    cumulative = 0.0
    selected_tokens = []
    for token, pct in sorted_tokens:
        if cumulative >= top_percent:
            break
        selected_tokens.append((token, pct))
        cumulative += pct

    most_common = sorted_tokens[:top_k]
    tokens, percentages = zip(*most_common) if most_common else ([], [])

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    palette = build_bar_palette(
        n_bars=len(tokens),
        mode=palette_mode,
        base_color=base_color,
        base_colors=PALETTES.get(base_colors_name),
        palette_name=base_colors_name,
        lightness_range=(lightness_min, lightness_max),
    )

    _barplot_percent(
        ax, tokens, percentages, palette,
        x_label="Percentage (%)",
        x_major_step=x_major_step, x_minor_step=x_minor_step
    )

    # Adjust layout (extra bottom pad to guarantee xtick breathing room)
    fig.tight_layout()
    fig.subplots_adjust(bottom=max(0.11, bottom_pad))

    # Save (PNG, PDF, SVG)
    output_dir = os.path.join(figure_dir, base_model, tgt_lang)
    os.makedirs(output_dir, exist_ok=True)
    stem = f"{org_lang}_top_{top_k}_token_distribution"
    for ext in ("png", "pdf", "svg"):
        out_path = os.path.join(output_dir, f"{stem}.{ext}")
        fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved to {output_dir}/{stem}.(png|pdf)")


def compute_average_distribution(base_model, org_lang, tgt_langs, model_name, gen_dir,
                                 watermark_method, seed, figure_dir, top_k=10,
                                 axis_fontsize=18, label_fontsize=20,
                                 palette_mode="single",
                                 base_color="#0072B2",
                                 base_colors_name="okabe_ito",
                                 lightness_min=0.35, lightness_max=0.85,
                                 fig_width=9.5, fig_height=5.5,
                                 x_major_step=5.0, x_minor_step=2.5,
                                 bottom_pad=0.25):
    """
    Average token distribution across all target languages for a given origin language.
    """
    set_paper_style(axis_fontsize=axis_fontsize, label_fontsize=label_fontsize)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    all_token_counts = defaultdict(float)
    total_files = 0

    for tgt_lang in tgt_langs:
        if tgt_lang == org_lang:
            continue

        translation_file = os.path.join(
            gen_dir, base_model, f"{watermark_method}_seed{seed}",
            f"mc4.{tgt_lang}-{org_lang}-back.mod.z_score.jsonl"
        )

        if not os.path.exists(translation_file):
            print(f"Warning: File not found: {translation_file}")
            continue

        t_data = read_jsonl(translation_file)
        t_responses = [d.get("response", "") for d in t_data if isinstance(d, dict)]

        # Tokenize responses
        t_responses_tokens = [
            tokenizer.convert_ids_to_tokens(
                tokenizer(text, add_special_tokens=False)["input_ids"]
            )
            for text in t_responses
        ]
        tokens_list = [tok for toks in t_responses_tokens for tok in toks]

        total_tokens = max(len(tokens_list), 1)
        token_counts = Counter(tokens_list)

        # Accumulate simple mean across langs
        for token, count in token_counts.items():
            all_token_counts[token] += (count / total_tokens) * 100.0

        total_files += 1

    if total_files == 0:
        print(f"No files found for {org_lang}")
        return

    avg_token_percentages = {token: pct / total_files for token, pct in all_token_counts.items()}
    sorted_tokens = sorted(avg_token_percentages.items(), key=lambda x: x[1], reverse=True)
    most_common = sorted_tokens[:top_k]
    tokens, percentages = zip(*most_common) if most_common else ([], [])

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    palette = build_bar_palette(
        n_bars=len(tokens),
        mode=palette_mode,
        base_color=base_color,
        base_colors=PALETTES.get(base_colors_name),
        palette_name=base_colors_name,
        lightness_range=(lightness_min, lightness_max),
    )

    _barplot_percent(
        ax, tokens, percentages, palette,
        x_label="Average (%)",
        x_major_step=x_major_step, x_minor_step=x_minor_step
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=max(0.11, bottom_pad))

    output_dir = os.path.join(figure_dir, base_model, "averaged")
    os.makedirs(output_dir, exist_ok=True)
    stem = f"{org_lang}_avg_top_{top_k}_token_distribution"
    for ext in ("png", "pdf"):
        out_path = os.path.join(output_dir, f"{stem}.{ext}")
        fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved to {output_dir}/{stem}.(png|pdf)")


# ----------------------------
# Single-file entry (reads + tokenizes responses)
# ----------------------------
def main_single(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    t_data = read_jsonl(args.translation_file)
    t_responses = [d.get("response", "") for d in t_data if isinstance(d, dict)]
    t_responses_tokens = [
        tokenizer.convert_ids_to_tokens(
            tokenizer(text, add_special_tokens=False)["input_ids"]
        )
        for text in t_responses
    ]
    tokens_list = [tok for toks in t_responses_tokens for tok in toks]

    plot_token_distribution(
        tokens_list=tokens_list,
        base_model=args.base_model,
        tgt_lang=args.tgt_lang,
        org_lang=args.org_lang,
        figure_dir=args.figure_dir,
        top_percent=args.top_percent,
        top_k=args.top_k,
        axis_fontsize=args.axis_fontsize,
        label_fontsize=args.label_fontsize,
        palette_mode=args.palette_mode,
        base_color=args.base_color,
        base_colors_name=args.base_colors_name,
        lightness_min=args.lightness_min,
        lightness_max=args.lightness_max,
        fig_width=args.fig_width,
        fig_height=args.fig_height,
        x_major_step=args.x_major_step,
        x_minor_step=args.x_minor_step,
        bottom_pad=args.bottom_pad,
    )


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paper-style Token Distribution with Gradient Palettes (wider fig, spaced ticks)")

    # Required
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--org_lang", type=str, required=True)
    parser.add_argument("--figure_dir", type=str, required=True)

    # Single-file mode
    parser.add_argument("--tgt_lang", type=str, help="Target language for single-file plotting")
    parser.add_argument("--translation_file", type=str, help="Path to translation jsonl for single-file plotting")

    # Averaging mode
    parser.add_argument("--compute_average", action="store_true", help="Compute average across target languages")
    parser.add_argument("--tgt_langs", type=str, nargs='+', help="Target languages for averaging")
    parser.add_argument("--gen_dir", type=str, help="Base directory for generated files per language")
    parser.add_argument("--watermark_method", type=str, default="kgw")
    parser.add_argument("--seed", type=int, default=42)

    # Plot controls
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--top_percent", type=float, default=90.0)
    parser.add_argument("--axis_fontsize", type=int, default=18)
    parser.add_argument("--label_fontsize", type=int, default=20)

    # Figure size + spacing (wider defaults)
    parser.add_argument("--fig_width", type=float, default=9.5)
    parser.add_argument("--fig_height", type=float, default=5.5)
    parser.add_argument("--x_major_step", type=float, default=5.0, help="Major x tick step in percent units")
    parser.add_argument("--x_minor_step", type=float, default=2.5, help="Minor x tick step in percent units")
    parser.add_argument("--bottom_pad", type=float, default=0.25, help="Extra bottom margin fraction for x tick labels")

    # Gradient palette controls
    parser.add_argument("--palette_mode", type=str, default="single", choices=["single", "cycle"],
                        help='Use "single" for one base color gradient; "cycle" to cycle multiple bases with gradients.')
    parser.add_argument("--base_color", type=str, default="#1B9E77",
                        help="Hex/named base color when palette_mode=single.")
    parser.add_argument("--base_colors_name", type=str, default="okabe_ito",
                        help="Base palette name when palette_mode=cycle.")
    parser.add_argument("--lightness_min", type=float, default=0.35, help="Gradient min lightness (0–1)")
    parser.add_argument("--lightness_max", type=float, default=0.85, help="Gradient max lightness (0–1)")

    args = parser.parse_args()

    if args.compute_average:
        if not args.tgt_langs or not args.gen_dir:
            raise SystemExit("Error: --tgt_langs and --gen_dir are required when using --compute_average")
        compute_average_distribution(
            base_model=args.base_model,
            org_lang=args.org_lang,
            tgt_langs=args.tgt_langs,
            model_name=args.model_name,
            gen_dir=args.gen_dir,
            watermark_method=args.watermark_method,
            seed=args.seed,
            figure_dir=args.figure_dir,
            top_k=args.top_k,
            axis_fontsize=args.axis_fontsize,
            label_fontsize=args.label_fontsize,
            palette_mode=args.palette_mode,
            base_color=args.base_color,
            base_colors_name=args.base_colors_name,
            lightness_min=args.lightness_min,
            lightness_max=args.lightness_max,
            fig_width=args.fig_width,
            fig_height=args.fig_height,
            x_major_step=args.x_major_step,
            x_minor_step=args.x_minor_step,
            bottom_pad=args.bottom_pad,
        )
    else:
        if not args.tgt_lang or not args.translation_file:
            raise SystemExit("Error: --tgt_lang and --translation_file are required when not using --compute_average")
        main_single(args)
