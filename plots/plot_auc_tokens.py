import argparse
import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from utils import read_jsonl
import matplotlib.patheffects as pe
from matplotlib import colors as mcolors

# --- Config ---
TGT_LANGS = [
    "fr", "de", "it", "es", "pt",
    "pl", "nl", "ru", "hi", "ko", "ja",
    "bn", "fa", "vi", "iw", "uk", "ta",
]

MODEL_ABBR = [
    "llama-3.2-1B",
    "aya-23-8B",
    "llamax3-8B",
]

SEEDS = ["0", "42", "123"]
MARKERS = ["o", "s", "D", "^", "v", "P", "X"]  # marker shapes per model

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
    "pastel1": [
        "#FBB4AE", "#B3CDE3", "#CCEBC5", "#DECBE4", "#FED9A6",
        "#FFFFCC", "#E5D8BD", "#FDDAEC", "#F2F2F2"
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

# Choose your palette here:
PALETTE = use_palette("dark2", n=len(MODEL_ABBR))


# --- Helpers ---
def compute_auc(base_dir, model_abbr, tgt_lang, seed):
    hm_z = os.path.join(base_dir, model_abbr, f"xsir_seed{seed}", f"mc4.en-{tgt_lang}.hum.z_score.jsonl")
    wm_z = os.path.join(base_dir, model_abbr, f"xsir_seed{seed}", f"mc4.en-{tgt_lang}.mod.z_score.jsonl")
    hm_list = read_jsonl(hm_z)
    wm_list = read_jsonl(wm_z)
    if len(hm_list) != len(wm_list):
        print(f"⚠️ zscore lengths differ for {model_abbr} {tgt_lang} seed={seed}")
        return None

    hm_scores = [x["z_score"] if x["z_score"] is not None else 0 for x in hm_list]
    wm_scores = [x["z_score"] if x["z_score"] is not None else 0 for x in wm_list]
    y_true = [0]*len(hm_list) + [1]*len(wm_list)
    y_scores = hm_scores + wm_scores
    return roc_auc_score(y_true, y_scores)


def repel_text(ax, texts, anchors, max_iter=250, step=0.01):
    """Simple force-based label repulsion; draws leader lines afterward."""
    fig = ax.figure
    fig.canvas.draw()
    for _ in range(max_iter):
        moved = False
        bboxes = [t.get_window_extent(renderer=fig.canvas.get_renderer()).expanded(1.08, 1.25)
                  for t in texts]
        for i in range(len(texts)):
            for j in range(i+1, len(texts)):
                if not bboxes[i].overlaps(bboxes[j]):
                    continue
                xi, yi = texts[i].get_position()
                xj, yj = texts[j].get_position()
                dx, dy = xi - xj, yi - yj
                dist = math.hypot(dx, dy) or 1e-6
                ux, uy = dx / dist, dy / dist
                texts[i].set_position((xi + ux * step, yi + uy * step))
                texts[j].set_position((xj - ux * step, yj - uy * step))
                moved = True
        if not moved:
            break

    # Leader lines after settling
    for t, (xa, ya) in zip(texts, anchors):
        xt, yt = t.get_position()
        ax.plot([xa, xt], [ya, yt], lw=0.6, alpha=0.45, color="#666")


def lighten(color, amount=0.6):
    """
    Lighten a color by mixing with white.
    amount in [0,1]: 0 -> original, 1 -> white.
    """
    r, g, b = mcolors.to_rgb(color)
    r = 1 - (1 - r) * (1 - amount)
    g = 1 - (1 - g) * (1 - amount)
    b = 1 - (1 - b) * (1 - amount)
    return (r, g, b)


# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="Scatter (with per-model linear fit + highlighted line): AUC vs tokenizer words.")
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--stats_file", type=str, default="stats.json")
    # Figure / aesthetics
    parser.add_argument("--latex_width_in", type=float, default=6.5)
    parser.add_argument("--latex_height_in", type=float, default=5.0)
    parser.add_argument("--marker_size", type=float, default=110)        # scatter s=
    parser.add_argument("--marker_edgewidth", type=float, default=1.5)   # stronger outlines
    parser.add_argument("--alpha", type=float, default=0.75)             # transparency
    parser.add_argument("--lang_fontsize", type=int, default=9)
    parser.add_argument("--axis_fontsize", type=int, default=10)
    parser.add_argument("--title_fontsize", type=int, default=12)
    parser.add_argument("--x_jitter", type=float, default=85.0)          # per-model x jitter
    parser.add_argument("--y_jitter", type=float, default=0.003)         # small y jitter
    # Regression line style
    parser.add_argument("--fit_linewidth", type=float, default=2.0)
    parser.add_argument("--fit_alpha", type=float, default=0.9)
    parser.add_argument("--fit_linestyle", type=str, default="-")
    # Highlight (glow) around the line
    parser.add_argument("--glow_mult", type=float, default=3.8, help="Thickness multiplier for the highlight underlay.")
    parser.add_argument("--glow_alpha", type=float, default=0.5, help="Alpha of the highlight underlay.")
    parser.add_argument("--glow_lighten", type=float, default=0.5, help="How much to lighten the line color for the glow (0–1).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Seaborn + Matplotlib paper-ish style
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.family": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.labelsize": args.axis_fontsize + 2,
        "xtick.labelsize": args.axis_fontsize,
        "ytick.labelsize": args.axis_fontsize,
        "legend.fontsize": args.axis_fontsize,
        "figure.dpi": 300,
    })

    fig, ax = plt.subplots(figsize=(args.latex_width_in, args.latex_height_in))

    # Improved grids: light major + very light minor
    ax.set_axisbelow(True)
    ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.45)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.25)

    labeled_models = set()
    texts, anchors = [], []
    all_x_for_limits = []

    # Alternating initial label offsets (before repulsion)
    dx_dy = [(0.0, +0.010), (0.0, -0.010), (0.0, +0.014), (0.0, -0.014),
             (0.0, +0.018), (0.0, -0.018), (0.0, +0.022), (0.0, -0.022)]

    for midx, model_abbr in enumerate(MODEL_ABBR):
        color = PALETTE[midx % len(PALETTE)]
        marker = MARKERS[midx % len(MARKERS)]

        stats_path = os.path.join(args.data_dir, model_abbr, args.stats_file)
        if not os.path.exists(stats_path):
            print(f"Stats file not found: {stats_path}")
            continue
        with open(stats_path, "r") as f:
            stats = json.load(f)

        # Build raw (x,y) without jitter for fitting, plus jittered for plotting
        x_raw, y_raw, x_plot, y_plot, langs = [], [], [], [], []
        for tgt_lang in TGT_LANGS:
            auc_values = []
            for seed in SEEDS:
                auc = compute_auc(args.base_dir, model_abbr, tgt_lang, seed)
                if auc is not None:
                    auc_values.append(auc)
            if not auc_values:
                continue
            avg_auc = sum(auc_values) / len(auc_values)
            x_tokens = stats["per_language"][tgt_lang]["unique_words_hit_single_token"]

            # jitter for visible separation (plotting only)
            x_jitter = (midx - (len(MODEL_ABBR) - 1) / 2.0) * args.x_jitter
            y_jitter = (midx - (len(MODEL_ABBR) - 1) / 2.0) * args.y_jitter

            x_raw.append(x_tokens)
            y_raw.append(avg_auc)
            x_plot.append(x_tokens + x_jitter)
            y_plot.append(avg_auc + y_jitter)
            langs.append(tgt_lang)

        if not x_plot:
            continue

        all_x_for_limits.extend(x_plot)

        # --- regression line (fit on raw x/y to avoid jitter bias) ---
        xr = np.array(x_raw, dtype=float)
        yr = np.array(y_raw, dtype=float)
        if xr.size >= 2:
            slope, intercept = np.polyfit(xr, yr, 1)
            xfit = np.linspace(xr.min(), xr.max(), 200)
            yfit = slope * xfit + intercept

            # Highlight (glow) underlay: thicker, lighter, semi-transparent, drawn first
            glow_color = lighten(color, amount=args.glow_lighten)
            ax.plot(
                xfit, yfit,
                color=glow_color,
                linestyle=args.fit_linestyle,
                linewidth=args.fit_linewidth * args.glow_mult,
                alpha=args.glow_alpha,
                solid_capstyle="round",
                zorder=1
            )
            # Main line on top
            ax.plot(
                xfit, yfit,
                color=color,
                linestyle=args.fit_linestyle,
                linewidth=args.fit_linewidth,
                alpha=args.fit_alpha,
                solid_capstyle="round",
                zorder=2
            )

        # --- scatter markers (on top) ---
        label = model_abbr if model_abbr not in labeled_models else None
        ax.scatter(
            x_plot, y_plot,
            s=args.marker_size,
            marker=marker,
            label=label,
            alpha=args.alpha,
            facecolors=color,
            edgecolors="white",
            linewidths=args.marker_edgewidth,
            zorder=4,
        )
        ax.scatter(
            x_plot, y_plot,
            s=args.marker_size,
            marker=marker,
            alpha=args.alpha * 0.65,
            facecolors="none",
            edgecolors="black",
            linewidths=args.marker_edgewidth * 0.5,
            zorder=3,
        )
        if label:
            labeled_models.add(model_abbr)

        # --- language labels (with halo) ---
        for i, (x, y, lang) in enumerate(zip(x_plot, y_plot, langs)):
            dx, dy = dx_dy[i % len(dx_dy)]
            t = ax.text(
                x + dx, y + dy, lang,
                fontsize=args.lang_fontsize, ha="center", va="center",
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")]
            )
            texts.append(t)
            anchors.append((x, y))

    # Repel labels & draw leader lines
    repel_text(ax, texts, anchors, max_iter=300, step=0.012)

    # Axis labels with larger, bold font
    ax.set_xlabel("Number of Words in the Tokenizer Vocabulary",
                  fontsize=args.axis_fontsize + 3, labelpad=10)
    ax.set_ylabel("AUC",
                  fontsize=args.axis_fontsize + 3, labelpad=10)

    # Tick parameters for better visibility
    ax.tick_params(axis="both", which="major",
                   labelsize=args.axis_fontsize + 2,
                   width=1.4, length=6, direction="out")
    ax.tick_params(axis="both", which="minor",
                   labelsize=args.axis_fontsize + 1,
                   width=1.0, length=4, direction="out")
    ax.legend(frameon=True, fontsize=args.axis_fontsize + 1, loc="lower right")
    sns.despine(ax=ax)

    # widen x-axis slightly (after plotting/repel)
    if all_x_for_limits:
        xmin, xmax = min(all_x_for_limits), max(all_x_for_limits)
        pad = 0.06 * (xmax - xmin) if xmax > xmin else 100.0
        ax.set_xlim(xmin - pad, xmax + pad)

    fig.tight_layout()

    png_path = os.path.join(args.output_dir, "avg_auc_vs_tokens_scatter_with_glow_fit.png")
    pdf_path = os.path.join(args.output_dir, "avg_auc_vs_tokens_scatter_with_glow_fit.pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    print(f"✅ Saved {png_path}")
    print(f"✅ Saved {pdf_path}")


if __name__ == "__main__":
    main()
