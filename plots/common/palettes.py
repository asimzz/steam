"""
Color palette definitions and utilities for plotting.
"""

import colorsys
import seaborn as sns
from matplotlib import colors as mcolors


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
    """
    Get n colors from a named palette.

    Args:
        name: Palette name from PALETTES or seaborn palette name
        n: Number of colors needed

    Returns:
        List of color values
    """
    if name in PALETTES:
        base = PALETTES[name]
        if n <= len(base):
            return base[:n]
        return (base * ((n + len(base) - 1) // len(base)))[:n]
    return sns.color_palette(name, n)


def gradient_palette(base_color, n, lightness_range=(0.35, 0.85)):
    """
    Generate n gradient shades of a base color using HLS lightness ramp.

    Args:
        base_color: hex or any matplotlib color
        n: Number of gradient steps
        lightness_range: (min_L, max_L) in [0,1]

    Returns:
        List of RGB color tuples
    """
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

    Args:
        base_colors: List of base colors to cycle through
        n: Total number of colors needed
        lightness_range: (min_L, max_L) in [0,1]

    Returns:
        List of RGB color tuples
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
    Return a list of n_bars colors according to the chosen mode.

    Args:
        n_bars: Number of colors needed
        mode: "single" for one base color gradient, "cycle" for multiple base colors
        base_color: Base color when mode="single"
        base_colors: List of base colors when mode="cycle"
        palette_name: Palette name to use if base_colors not provided
        lightness_range: (min_L, max_L) for gradient generation

    Returns:
        List of colors for the bars
    """
    if mode == "single":
        return gradient_palette(base_color, n_bars, lightness_range)

    if not base_colors:
        base_colors = use_palette(palette_name, n=8)

    base_colors = [mcolors.to_hex(c) for c in base_colors]
    return multi_gradient_cycle(base_colors, n_bars, lightness_range)