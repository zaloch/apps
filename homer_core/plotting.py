# Homer - Plotting module using Plotly for interactive charts
# and Matplotlib for static/PDF-ready figures

import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from io import BytesIO
from typing import Optional


# ── Color palette ────────────────────────────────────────────────────────────

HOMER_COLORS = [
    "#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B",
    "#44BBA4", "#E94F37", "#393E41", "#D4A373", "#6B4226",
    "#1B998B", "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
    "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE",
]

HOMER_TEMPLATE = "plotly_white"


# ── Plotly interactive figures ───────────────────────────────────────────────

def create_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    orientation: str = "v",
    barmode: str = "group",
    title: str = "",
    x_label: str = "",
    y_label: str = "",
) -> go.Figure:
    """Create an interactive bar chart."""
    if orientation == "h":
        fig = px.bar(
            df, x=y, y=x, color=color, orientation="h",
            barmode=barmode, title=title,
            color_discrete_sequence=HOMER_COLORS,
            template=HOMER_TEMPLATE,
        )
        fig.update_layout(xaxis_title=y_label or y, yaxis_title=x_label or x)
    else:
        fig = px.bar(
            df, x=x, y=y, color=color, orientation="v",
            barmode=barmode, title=title,
            color_discrete_sequence=HOMER_COLORS,
            template=HOMER_TEMPLATE,
        )
        fig.update_layout(xaxis_title=x_label or x, yaxis_title=y_label or y)

    fig.update_layout(
        legend_title_text=color or "",
        font=dict(family="Arial, sans-serif", size=12),
        margin=dict(t=50, b=50, l=50, r=50),
    )
    return fig


def create_stacked_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str,
    orientation: str = "v",
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    normalize: bool = False,
) -> go.Figure:
    """Create a stacked bar chart (vertical or horizontal)."""
    barnorm = "percent" if normalize else None

    if orientation == "h":
        fig = px.bar(
            df, x=y, y=x, color=color, orientation="h",
            barmode="stack", barnorm=barnorm, title=title,
            color_discrete_sequence=HOMER_COLORS,
            template=HOMER_TEMPLATE,
        )
        fig.update_layout(xaxis_title=y_label or y, yaxis_title=x_label or x)
    else:
        fig = px.bar(
            df, x=x, y=y, color=color, orientation="v",
            barmode="stack", barnorm=barnorm, title=title,
            color_discrete_sequence=HOMER_COLORS,
            template=HOMER_TEMPLATE,
        )
        fig.update_layout(xaxis_title=x_label or x, yaxis_title=y_label or y)

    fig.update_layout(
        legend_title_text=color,
        font=dict(family="Arial, sans-serif", size=12),
    )
    return fig


def create_scatter_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    size: Optional[str] = None,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    trendline: Optional[str] = None,
) -> go.Figure:
    """Create an interactive scatter plot."""
    fig = px.scatter(
        df, x=x, y=y, color=color, size=size,
        title=title, trendline=trendline,
        color_discrete_sequence=HOMER_COLORS,
        template=HOMER_TEMPLATE,
    )
    fig.update_layout(
        xaxis_title=x_label or x,
        yaxis_title=y_label or y,
        legend_title_text=color or "",
        font=dict(family="Arial, sans-serif", size=12),
    )
    return fig


def create_box_plot(
    df: pd.DataFrame,
    x: Optional[str],
    y: str,
    color: Optional[str] = None,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    points: str = "outliers",
) -> go.Figure:
    """Create an interactive box plot."""
    fig = px.box(
        df, x=x, y=y, color=color, points=points,
        title=title,
        color_discrete_sequence=HOMER_COLORS,
        template=HOMER_TEMPLATE,
    )
    fig.update_layout(
        xaxis_title=x_label or (x if x else ""),
        yaxis_title=y_label or y,
        legend_title_text=color or "",
        font=dict(family="Arial, sans-serif", size=12),
    )
    return fig


def create_violin_plot(
    df: pd.DataFrame,
    x: Optional[str],
    y: str,
    color: Optional[str] = None,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
) -> go.Figure:
    """Create an interactive violin plot."""
    fig = px.violin(
        df, x=x, y=y, color=color, box=True, points="outliers",
        title=title,
        color_discrete_sequence=HOMER_COLORS,
        template=HOMER_TEMPLATE,
    )
    fig.update_layout(
        xaxis_title=x_label or (x if x else ""),
        yaxis_title=y_label or y,
        font=dict(family="Arial, sans-serif", size=12),
    )
    return fig


def create_histogram(
    df: pd.DataFrame,
    x: str,
    color: Optional[str] = None,
    nbins: int = 50,
    title: str = "",
    x_label: str = "",
    y_label: str = "Count",
) -> go.Figure:
    """Create an interactive histogram."""
    fig = px.histogram(
        df, x=x, color=color, nbins=nbins,
        title=title,
        color_discrete_sequence=HOMER_COLORS,
        template=HOMER_TEMPLATE,
    )
    fig.update_layout(
        xaxis_title=x_label or x,
        yaxis_title=y_label,
        font=dict(family="Arial, sans-serif", size=12),
    )
    return fig


def create_heatmap(
    df: pd.DataFrame,
    x: str,
    y: str,
    z: str,
    title: str = "",
    colorscale: str = "Viridis",
) -> go.Figure:
    """Create a heatmap from pivoted data."""
    pivot = df.pivot_table(values=z, index=y, columns=x, aggfunc="mean")
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=colorscale,
    ))
    fig.update_layout(
        title=title,
        xaxis_title=x,
        yaxis_title=y,
        template=HOMER_TEMPLATE,
        font=dict(family="Arial, sans-serif", size=12),
    )
    return fig


def create_xy_line_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    markers: bool = True,
) -> go.Figure:
    """Create an XY line plot."""
    fig = px.line(
        df.sort_values(x), x=x, y=y, color=color,
        title=title, markers=markers,
        color_discrete_sequence=HOMER_COLORS,
        template=HOMER_TEMPLATE,
    )
    fig.update_layout(
        xaxis_title=x_label or x,
        yaxis_title=y_label or y,
        font=dict(family="Arial, sans-serif", size=12),
    )
    return fig


# ── Figure export ────────────────────────────────────────────────────────────

def fig_to_png_bytes(fig: go.Figure, width: int = 1200, height: int = 700) -> bytes:
    """Export a Plotly figure to PNG bytes."""
    return fig.to_image(format="png", width=width, height=height, scale=2)


def fig_to_svg_bytes(fig: go.Figure, width: int = 1200, height: int = 700) -> bytes:
    """Export a Plotly figure to SVG bytes."""
    return fig.to_image(format="svg", width=width, height=height)


def fig_to_html(fig: go.Figure) -> str:
    """Export a Plotly figure to standalone HTML."""
    return fig.to_html(include_plotlyjs="cdn", full_html=False)


# ── Matplotlib static figures (for PDF reports) ─────────────────────────────

def _apply_homer_style():
    """Apply Homer styling to matplotlib plots."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def mpl_bar_chart(
    df: pd.DataFrame, x: str, y: str, color: Optional[str] = None,
    title: str = "", figsize: tuple = (10, 6),
) -> plt.Figure:
    """Create a matplotlib bar chart for PDF reports."""
    _apply_homer_style()
    fig, ax = plt.subplots(figsize=figsize)

    if color and color in df.columns:
        groups = df.groupby(color)
        width = 0.8 / len(groups)
        for i, (name, group) in enumerate(groups):
            positions = np.arange(len(group[x].unique())) + i * width
            ax.bar(positions, group.groupby(x)[y].mean().values, width,
                   label=str(name), color=HOMER_COLORS[i % len(HOMER_COLORS)])
        ax.set_xticks(np.arange(len(df[x].unique())) + width * len(groups) / 2)
        ax.set_xticklabels(df[x].unique(), rotation=45, ha="right")
        ax.legend(title=color)
    else:
        values = df.groupby(x)[y].mean()
        ax.bar(range(len(values)), values.values, color=HOMER_COLORS[0])
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(values.index, rotation=45, ha="right")

    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fig.tight_layout()
    return fig


def mpl_scatter_plot(
    df: pd.DataFrame, x: str, y: str, color: Optional[str] = None,
    title: str = "", figsize: tuple = (10, 6),
) -> plt.Figure:
    """Create a matplotlib scatter plot for PDF reports."""
    _apply_homer_style()
    fig, ax = plt.subplots(figsize=figsize)

    if color and color in df.columns:
        for i, (name, group) in enumerate(df.groupby(color)):
            ax.scatter(group[x], group[y], label=str(name),
                      color=HOMER_COLORS[i % len(HOMER_COLORS)], alpha=0.7, s=20)
        ax.legend(title=color, fontsize=8)
    else:
        ax.scatter(df[x], df[y], color=HOMER_COLORS[0], alpha=0.7, s=20)

    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fig.tight_layout()
    return fig


def mpl_box_plot(
    df: pd.DataFrame, x: Optional[str], y: str, title: str = "",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Create a matplotlib box plot for PDF reports."""
    _apply_homer_style()
    fig, ax = plt.subplots(figsize=figsize)

    if x and x in df.columns:
        groups = [group[y].dropna().values for _, group in df.groupby(x)]
        labels = [str(name) for name, _ in df.groupby(x)]
        bp = ax.boxplot(groups, labels=labels, patch_artist=True)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(HOMER_COLORS[i % len(HOMER_COLORS)])
            patch.set_alpha(0.7)
        ax.set_xlabel(x)
    else:
        bp = ax.boxplot(df[y].dropna().values, patch_artist=True)
        bp["boxes"][0].set_facecolor(HOMER_COLORS[0])
        bp["boxes"][0].set_alpha(0.7)

    ax.set_title(title)
    ax.set_ylabel(y)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    return fig


def mpl_fig_to_bytes(fig: plt.Figure, fmt: str = "png", dpi: int = 150) -> bytes:
    """Convert matplotlib figure to bytes."""
    buf = BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf.read()
