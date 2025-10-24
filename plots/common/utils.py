"""
Shared plotting utilities and helper functions.
"""

import json
import math
from typing import List, Tuple, Any

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
from matplotlib import colors as mcolors
from matplotlib.ticker import PercentFormatter, MultipleLocator

from .config import PAPER_STYLE_PARAMS


def read_jsonl(file_path: str) -> List[dict]:
    """
    Read a JSONL file and return a list of dictionaries.

    Args:
        file_path: Path to the JSONL file

    Returns:
        List of parsed JSON objects
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def set_paper_style(axis_fontsize: int = 18, label_fontsize: int = 20):
    """
    Set paper-ready matplotlib style.

    Args:
        axis_fontsize: Font size for axis tick labels
        label_fontsize: Font size for axis labels
    """
    sns.set_theme(style="whitegrid", context="paper")

    params = PAPER_STYLE_PARAMS.copy()
    params.update({
        "axes.labelsize": label_fontsize,
        "xtick.labelsize": axis_fontsize,
        "ytick.labelsize": axis_fontsize,
        "legend.fontsize": axis_fontsize,
    })

    plt.rcParams.update(params)


def nice_upper_limit(values: List[float], step: float = 5.0) -> float:
    """
    Ceil max(values) to the next multiple of `step` for a clean axis limit.

    Args:
        values: List of numeric values
        step: Step size for rounding

    Returns:
        Rounded upper limit
    """
    vmax = max(values) if values else 100.0
    return math.ceil(vmax / step) * step


def annotate_bars(ax, horizontal: bool = True, fmt: str = "{:.1f}%",
                  min_threshold: float = 0.3, fontsize: int = None):
    """
    Add value labels at bar ends with a white halo for legibility.

    Args:
        ax: Matplotlib axes object
        horizontal: Whether bars are horizontal
        fmt: Format string for labels
        min_threshold: Minimum bar size to show label
        fontsize: Font size for labels (defaults to axis tick fontsize)
    """
    if fontsize is None:
        tick_labels = ax.yaxis.get_ticklabels() if horizontal else ax.xaxis.get_ticklabels()
        fontsize = tick_labels[0].get_fontsize() if tick_labels else 12

    for p in ax.patches:
        if horizontal:
            width = p.get_width()
            y = p.get_y() + p.get_height() / 2.0
            if width >= min_threshold:
                ax.text(
                    width + 0.5, y, fmt.format(width),
                    va="center", ha="left", fontsize=fontsize,
                    path_effects=[pe.withStroke(linewidth=2.0, foreground="white")],
                )
        else:
            height = p.get_height()
            x = p.get_x() + p.get_width() / 2.0
            if height >= min_threshold:
                ax.text(
                    x, height + 0.15, fmt.format(height),
                    ha="center", va="bottom", rotation=90, fontsize=fontsize,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
                )


def barplot_percent(ax, tokens: List[str], percentages: List[float], palette: List[str],
                    x_label: str, x_major_step: float = 5.0, x_minor_step: float = 2.5):
    """
    Shared barplot styling for horizontal percentage plots.

    Args:
        ax: Matplotlib axes object
        tokens: List of token/category names
        percentages: List of percentage values
        palette: List of colors for bars
        x_label: X-axis label
        x_major_step: Major tick step size
        x_minor_step: Minor tick step size
    """
    sns.barplot(
        x=list(percentages), y=list(tokens), hue=list(tokens), legend=False,
        ax=ax, orient="h", palette=palette, edgecolor="black"
    )

    # Axis labels & formatting
    ax.set_xlabel(x_label, labelpad=10)
    ax.set_ylabel("Token", labelpad=8)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))

    # Tick spacing
    if x_major_step and x_major_step > 0:
        ax.xaxis.set_major_locator(MultipleLocator(x_major_step))
    if x_minor_step and x_minor_step > 0:
        ax.xaxis.set_minor_locator(MultipleLocator(x_minor_step))

    # Grid styling
    ax.grid(True, axis="x", which="major", linestyle="--", linewidth=0.9, alpha=0.45)
    ax.grid(True, axis="x", which="minor", linestyle=":", linewidth=0.6, alpha=0.25)
    ax.tick_params(axis="both", which="major", width=1.3, length=6, direction="out")
    ax.tick_params(axis="both", which="minor", width=1.0, length=4, direction="out")
    ax.set_axisbelow(True)

    # Set x-axis limits
    xmax = nice_upper_limit(list(percentages), step=max(1.0, x_major_step)) if percentages else 100.0
    ax.set_xlim(0, xmax)

    annotate_bars(ax, horizontal=True, fmt="{:.1f}%")


def repel_text(ax, texts: List[Any], anchors: List[Tuple[float, float]],
               max_iter: int = 250, step: float = 0.01):
    """
    Simple force-based label repulsion with leader lines.

    Args:
        ax: Matplotlib axes object
        texts: List of text objects to repel
        anchors: List of (x, y) anchor points for leader lines
        max_iter: Maximum iterations for repulsion algorithm
        step: Step size for repulsion
    """
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

    # Draw leader lines after settling
    for t, (xa, ya) in zip(texts, anchors):
        xt, yt = t.get_position()
        ax.plot([xa, xt], [ya, yt], lw=0.6, alpha=0.45, color="#666")


def lighten(color: str, amount: float = 0.6) -> Tuple[float, float, float]:
    """
    Lighten a color by mixing with white.

    Args:
        color: Color string (hex, named, etc.)
        amount: Amount to lighten (0=original, 1=white)

    Returns:
        RGB tuple of lightened color
    """
    r, g, b = mcolors.to_rgb(color)
    r = 1 - (1 - r) * (1 - amount)
    g = 1 - (1 - g) * (1 - amount)
    b = 1 - (1 - b) * (1 - amount)
    return (r, g, b)