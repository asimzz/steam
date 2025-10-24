"""
Centralized configuration for plotting scripts.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any


@dataclass
class PlotConfig:
    """Configuration class for plot styling and parameters."""

    # Figure dimensions
    fig_width: float = 9.5
    fig_height: float = 5.5

    # Font sizes
    axis_fontsize: int = 18
    label_fontsize: int = 20
    title_fontsize: int = 22
    legend_fontsize: int = 18

    # Color settings
    base_color: str = "#1B9E77"
    palette_name: str = "okabe_ito"
    lightness_min: float = 0.35
    lightness_max: float = 0.85

    # Grid and ticks
    x_major_step: float = 5.0
    x_minor_step: float = 2.5
    y_major_step: float = 1.0

    # Layout
    bottom_pad: float = 0.25
    bar_width: float = 0.85

    # Style parameters
    marker_size: float = 110
    marker_edgewidth: float = 1.5
    alpha: float = 0.75
    fit_linewidth: float = 2.0
    fit_alpha: float = 0.9

    # File output
    dpi: int = 300
    output_formats: Tuple[str, ...] = ("png", "pdf", "svg")

    @classmethod
    def for_token_distribution(cls, **kwargs) -> "PlotConfig":
        """Configuration optimized for token distribution plots."""
        defaults = {
            "fig_width": 9.5,
            "fig_height": 5.5,
            "x_major_step": 5.0,
            "x_minor_step": 2.5,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def for_auc_scatter(cls, **kwargs) -> "PlotConfig":
        """Configuration optimized for AUC scatter plots."""
        defaults = {
            "fig_width": 6.5,
            "fig_height": 5.0,
            "marker_size": 110,
            "alpha": 0.75,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def for_lang_coverage(cls, **kwargs) -> "PlotConfig":
        """Configuration optimized for language coverage plots."""
        defaults = {
            "fig_width": 14,
            "fig_height": 6,
            "axis_fontsize": 24,
            "label_fontsize": 26,
            "y_major_step": 1.0,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    def get_lightness_range(self) -> Tuple[float, float]:
        """Get the lightness range for gradient palettes."""
        return (self.lightness_min, self.lightness_max)

    def get_figure_size(self) -> Tuple[float, float]:
        """Get the figure size as a tuple."""
        return (self.fig_width, self.fig_height)


# Default configurations for different plot types
DEFAULT_CONFIG = PlotConfig()
TOKEN_DISTRIBUTION_CONFIG = PlotConfig.for_token_distribution()
AUC_SCATTER_CONFIG = PlotConfig.for_auc_scatter()
LANG_COVERAGE_CONFIG = PlotConfig.for_lang_coverage()


# Common matplotlib rcParams for paper-style plots
PAPER_STYLE_PARAMS = {
    "font.family": "serif",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 1.3,
    "figure.dpi": 300,
    "savefig.transparent": True,
    "xtick.major.width": 1.3,
    "ytick.major.width": 1.3,
}