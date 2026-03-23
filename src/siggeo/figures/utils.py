from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import ticker

from siggeo.io.atomic import save_figure_atomic


REGIME_COLORS = {
    "aligned": "#1F4E79",
    "inverse": "#B04A2F",
    "orthogonal": "#3C7A3D",
    "asymmetric_ORF": "#C88A1B",
    "asymmetric_CRISPR": "#2D7F8D",
    "inactive": "#6B7280",
    "ambiguous": "#B8C0C7",
}


def apply_global_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#FCFBF8",
            "axes.facecolor": "#FCFBF8",
            "savefig.facecolor": "#FCFBF8",
            "axes.edgecolor": "#36454F",
            "axes.labelcolor": "#1F2933",
            "axes.titlecolor": "#14213D",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.color": "#D9E2EC",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.8,
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "xtick.color": "#1F2933",
            "ytick.color": "#1F2933",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def prepare_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#52606D")
    ax.spines["bottom"].set_color("#52606D")
    ax.tick_params(length=0)


def annotate_bar_values(ax: plt.Axes, *, fmt: str = "{:.0f}", padding: float = 0.0) -> None:
    for patch in ax.patches:
        width = patch.get_width()
        if width is None:
            continue
        y = patch.get_y() + patch.get_height() / 2.0
        ax.text(width + padding, y, fmt.format(width), va="center", ha="left", fontsize=9, color="#243B53")


def format_small_metric_axis(ax: plt.Axes, *, axis: str = "y") -> None:
    formatter = ticker.FuncFormatter(lambda value, _: f"{value * 1e3:.2f}")
    if axis == "x":
        ax.xaxis.set_major_formatter(formatter)
    else:
        ax.yaxis.set_major_formatter(formatter)


def finalize_figure(path: Path) -> None:
    apply_global_style()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    save_figure_atomic(path, lambda temp_path: plt.savefig(temp_path, dpi=300, bbox_inches="tight"))
    plt.close()
