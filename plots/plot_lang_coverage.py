import json
import argparse
import os
import colorsys
import math

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as pe
from matplotlib import colors as mcolors
from matplotlib.ticker import PercentFormatter, MultipleLocator


# ----------------------------
# Palettes (same as token distribution script)
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


def set_paper_style(axis_fontsize=18, label_fontsize=20):
    """Set paper-ready matplotlib style."""
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
        "savefig.transparent": False,
        "savefig.facecolor": "white",
        "xtick.major.width": 1.3,
        "ytick.major.width": 1.3,
    })


def gradient_palette(base_color, n, lightness_range=(0.35, 0.85)):
    """
    Generate n gradient shades of a base color using HLS lightness ramp.
    """
    base_rgb = mcolors.to_rgb(base_color)
    hue, _, sat = colorsys.rgb_to_hls(*base_rgb)
    lo, hi = lightness_range
    if n <= 1:
        return [colorsys.hls_to_rgb(hue, (lo+hi)/2.0, sat)]
    lightnesses = [lo + (hi - lo) * (i / (n - 1)) for i in range(n)]
    return [colorsys.hls_to_rgb(hue, li, sat) for li in lightnesses]


def _nice_upper_limit(values, step=1.0):
    """Ceil max(values) to the next multiple of `step` for a clean percent axis."""
    vmax = max(values) if values else 100.0
    return math.ceil(vmax / step) * step


def _annotate_bars(ax, horizontal=True, fmt="{:.2f}%", min_threshold=0.3, label_fontsize=14):
    """
    Value labels at bar ends with a white halo for legibility.
    Only shows labels for bars above min_threshold to avoid overlap.
    """
    for p in ax.patches:
        if horizontal:
            width = p.get_width()
            y = p.get_y() + p.get_height() / 2.0
            # Only show label if bar is wide enough
            if width >= min_threshold:
                ax.text(
                    width + 0.1, y, fmt.format(width),
                    va="center", ha="left",
                    fontsize=label_fontsize,
                    path_effects=[pe.withStroke(linewidth=2.0, foreground="white")],
                )
        else:
            height = p.get_height()
            x = p.get_x() + p.get_width() / 2.0
            # Only show label if bar is tall enough
            if height >= min_threshold:
                ax.text(
                    x, height + 0.15, fmt.format(height),
                    ha="center", va="bottom", rotation=90,
                    fontsize=label_fontsize,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
                )


def plot_vocab_contribution(lang_stats_file, output_file,
                           axis_fontsize=18, label_fontsize=20,
                           base_color="#1B9E77", lightness_min=0.35, lightness_max=0.85,
                           fig_width=14, fig_height=6,
                           y_major_step=1.0, bottom_pad=0.25, bar_width=0.85):
    """
    Create paper-style bar plot showing vocabulary contribution per language.
    """
    set_paper_style(axis_fontsize=axis_fontsize, label_fontsize=label_fontsize)

    # Load language stats
    with open(lang_stats_file, 'r') as f:
        lang_stats = json.load(f)

    # Sort languages by vocab contribution
    langs = []
    contributions = []
    for lang, stats in sorted(lang_stats.items(), key=lambda x: x[1]['vocab_contribution_percent'], reverse=True):
        langs.append(lang)
        contributions.append(stats['vocab_contribution_percent'])

    # Generate gradient palette
    palette = gradient_palette(base_color, len(langs), lightness_range=(lightness_min, lightness_max))

    # Create figure with white background
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='white')
    ax.set_facecolor('white')

    # Create vertical bar plot using matplotlib for better spacing control
    x_positions = range(len(langs))
    bars = ax.bar(x_positions, contributions, width=bar_width,
                   color=palette, edgecolor="black", linewidth=1.0)

    # Set x-tick positions and labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(langs, rotation=0)

    # Axis labels
    ax.set_xlabel("Language", labelpad=10)
    ax.set_ylabel("Dictionnary Coverage (%)", labelpad=10)

    # Y-axis formatting with percent
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))

    # Explicit tick spacing
    if y_major_step and y_major_step > 0:
        ax.yaxis.set_major_locator(MultipleLocator(y_major_step))

    # Remove box border (spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Grid styling (only horizontal grid lines, no vertical)
    ax.grid(True, axis="y", which="major", linestyle="--", linewidth=0.9, alpha=0.45)
    ax.grid(False, axis="x")  # Disable x-axis grid to remove vertical lines
    ax.set_axisbelow(True)

    # Set y-axis limits with nice upper bound
    ymax = _nice_upper_limit(contributions, step=max(1.0, y_major_step)) if contributions else 5.0
    ax.set_ylim(0, ymax)

    # Add percentage labels on bars (vertical orientation)
    _annotate_bars(ax, horizontal=False, fmt="{:.2f}%", min_threshold=0.0, label_fontsize=axis_fontsize-2)

    # Adjust layout
    fig.tight_layout()
    fig.subplots_adjust(bottom=max(0.15, bottom_pad))

    # Save in multiple formats
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    base_path = output_file.rsplit('.', 1)[0]
    for ext in ("png", "pdf", "svg"):
        out_path = f"{base_path}.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches='tight')

    plt.close(fig)
    print(f"Saved bar plot to {base_path}.(png|pdf|svg)")


def main():
    parser = argparse.ArgumentParser(description='Plot language coverage bar chart (paper-style)')
    parser.add_argument('--lang_stats_file', type=str, required=True,
                        help='Path to the language stats JSON file')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to save the plot')

    # Style parameters
    parser.add_argument('--axis_fontsize', type=int, default=24)
    parser.add_argument('--label_fontsize', type=int, default=26)
    parser.add_argument('--base_color', type=str, default='#7570B3',
                        help='Base color for gradient palette (hex or named color)')
    parser.add_argument('--lightness_min', type=float, default=0.35,
                        help='Gradient min lightness (0-1)')
    parser.add_argument('--lightness_max', type=float, default=0.85,
                        help='Gradient max lightness (0-1)')

    # Figure parameters
    parser.add_argument('--fig_width', type=float, default=14,
                        help='Figure width in inches')
    parser.add_argument('--fig_height', type=float, default=10,
                        help='Figure height in inches')
    parser.add_argument('--y_major_step', type=float, default=1.0,
                        help='Major y-axis tick step in percent units')
    parser.add_argument('--bottom_pad', type=float, default=0.25,
                        help='Extra bottom margin fraction')
    parser.add_argument('--bar_width', type=float, default=0.85,
                        help='Width of bars (0-1, lower = more spacing)')

    args = parser.parse_args()

    plot_vocab_contribution(
        lang_stats_file=args.lang_stats_file,
        output_file=args.output_file,
        axis_fontsize=args.axis_fontsize,
        label_fontsize=args.label_fontsize,
        base_color=args.base_color,
        lightness_min=args.lightness_min,
        lightness_max=args.lightness_max,
        fig_width=args.fig_width,
        fig_height=args.fig_height,
        y_major_step=args.y_major_step,
        bottom_pad=args.bottom_pad,
        bar_width=args.bar_width,
    )


if __name__ == '__main__':
    main()
