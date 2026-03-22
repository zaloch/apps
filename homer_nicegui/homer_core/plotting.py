# Homer - Plotting module using Plotly for interactive charts
# and Matplotlib/Seaborn for static/PDF-ready figures
# Includes anima-style histology analysis plots (stripplot, pairplot, before/after)

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import pandas as pd
from io import BytesIO
from typing import Optional


# ── Color palette ────────────────────────────────────────────────────────────

HOMER_COLORS = [
    "#4FC3F7", "#FF6B9D", "#FFB74D", "#81C784", "#BA68C8",
    "#4DD0E1", "#FF8A65", "#AED581", "#F06292", "#FFD54F",
    "#7986CB", "#E57373", "#4DB6AC", "#DCE775", "#90A4AE",
    "#CE93D8", "#80DEEA", "#FFAB91", "#C5E1A5", "#EF9A9A",
]

HOMER_TEMPLATE = "plotly_dark"

DARK_LAYOUT = dict(
    paper_bgcolor="#0E1117",
    plot_bgcolor="#1B2028",
    font=dict(family="Inter, Arial, sans-serif", size=12, color="#E0E0E0"),
    title_font_color="#FFFFFF",
    legend_font_color="#E0E0E0",
    xaxis=dict(gridcolor="#2A2F3A", zerolinecolor="#2A2F3A"),
    yaxis=dict(gridcolor="#2A2F3A", zerolinecolor="#2A2F3A"),
    margin=dict(t=50, b=50, l=50, r=50),
)


def apply_dark_theme(fig):
    """Apply the dark theme layout to any Plotly figure and return it."""
    fig.update_layout(**DARK_LAYOUT)
    return fig


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
    fig.update_layout(**DARK_LAYOUT)
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
    fig.update_layout(**DARK_LAYOUT)
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
    fig.update_layout(**DARK_LAYOUT)
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
    fig.update_layout(**DARK_LAYOUT)
    return fig


def create_violin_plot(
    df: pd.DataFrame,
    x: Optional[str],
    y: str,
    color: Optional[str] = None,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    points: str = "outliers",
) -> go.Figure:
    """Create an interactive violin plot."""
    fig = px.violin(
        df, x=x, y=y, color=color, box=True, points=points,
        title=title,
        color_discrete_sequence=HOMER_COLORS,
        template=HOMER_TEMPLATE,
    )
    fig.update_layout(
        xaxis_title=x_label or (x if x else ""),
        yaxis_title=y_label or y,
        font=dict(family="Arial, sans-serif", size=12),
    )
    fig.update_layout(**DARK_LAYOUT)
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
    fig.update_layout(**DARK_LAYOUT)
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
    fig.update_layout(**DARK_LAYOUT)
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
    fig.update_layout(**DARK_LAYOUT)
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
        "font.sans-serif": ["Inter", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.facecolor": "#0E1117",
        "axes.facecolor": "#1B2028",
        "text.color": "#E0E0E0",
        "axes.labelcolor": "#E0E0E0",
        "xtick.color": "#E0E0E0",
        "ytick.color": "#E0E0E0",
        "axes.grid": True,
        "grid.color": "#2A2F3A",
        "grid.alpha": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#2A2F3A",
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


# ── Plotly: Strip plot (mirrors anima __full_report) ─────────────────────────

def create_strip_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    auto_ylim: bool = True,
) -> go.Figure:
    """Create a Plotly strip/jitter plot (equivalent to seaborn stripplot).

    Mirrors the ClusterCleaner.__full_report visualization from anima.
    """
    fig = px.strip(
        df, x=x, y=y, color=color or x,
        title=title,
        color_discrete_sequence=HOMER_COLORS,
        template=HOMER_TEMPLATE,
    )
    fig.update_traces(marker=dict(size=4, opacity=0.7))
    fig.update_layout(
        xaxis_title=x_label or x,
        yaxis_title=y_label or y,
        font=dict(family="Arial, sans-serif", size=12),
        xaxis_tickangle=-45,
    )
    # Auto-scale Y axis: 0 to max + 40% headroom for non-percentage data
    if auto_ylim and "%" not in y:
        y_max = df[y].max()
        if pd.notna(y_max) and y_max > 0:
            fig.update_yaxes(range=[0, y_max + 0.4 * y_max])
    elif "%" in y:
        fig.update_yaxes(range=[0, 100])

    fig.update_layout(**DARK_LAYOUT)
    return fig


def create_swarm_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    title: str = "",
) -> go.Figure:
    """Create a Plotly strip plot with jitter (approximates seaborn swarmplot)."""
    fig = px.strip(
        df, x=x, y=y, color=color or x,
        title=title,
        color_discrete_sequence=HOMER_COLORS,
        template=HOMER_TEMPLATE,
    )
    fig.update_traces(jitter=0.4, marker=dict(size=5, opacity=0.65))
    fig.update_layout(
        font=dict(family="Arial, sans-serif", size=12),
        xaxis_tickangle=-45,
    )
    fig.update_layout(**DARK_LAYOUT)
    return fig


# ── Plotly: Outlier before/after visualization ───────────────────────────────

def create_outlier_comparison(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    metric: str,
    lower_bound: float,
    upper_bound: float,
    n_removed: int,
    method: str = "IQR",
    title: str = "",
) -> go.Figure:
    """Create a side-by-side histogram showing data before and after outlier removal.

    Mirrors ClusterCleaner.plot_before_after from anima.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f"Before Cleaning ({len(before_df)} pts)", f"After Cleaning ({len(after_df)} pts)"],
        shared_yaxes=True,
    )

    # Before histogram
    fig.add_trace(
        go.Histogram(x=before_df[metric], nbinsx=50, name="Before",
                     marker_color=HOMER_COLORS[0], opacity=0.75),
        row=1, col=1,
    )

    # After histogram
    fig.add_trace(
        go.Histogram(x=after_df[metric], nbinsx=50, name="After",
                     marker_color=HOMER_COLORS[1], opacity=0.75),
        row=1, col=2,
    )

    # Bound lines on the after plot
    if lower_bound != 0 or upper_bound != 0:
        for bound, label, color in [
            (lower_bound, "Lower", "red"),
            (upper_bound, "Upper", "red"),
        ]:
            fig.add_vline(x=bound, line_dash="dash", line_color=color,
                         annotation_text=f"{label}: {bound:.2f}",
                         row=1, col=2)

    fig.update_layout(
        title=title or f"Outlier Removal ({method}): {metric} | {n_removed} removed",
        template=HOMER_TEMPLATE,
        font=dict(family="Arial, sans-serif", size=12),
        showlegend=False,
    )
    fig.update_xaxes(title_text=metric, row=1, col=1)
    fig.update_xaxes(title_text=metric, row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)

    fig.update_layout(**DARK_LAYOUT)
    return fig


# ── Plotly: Pairplot (mirrors anima broad_describe) ──────────────────────────

def create_pairplot_matrix(
    df: pd.DataFrame,
    columns: list[str],
    color: Optional[str] = None,
    title: str = "Pairwise Relationships",
    max_cols: int = 6,
) -> go.Figure:
    """Create a Plotly scatter matrix (pairplot).

    Mirrors HistologyIngest.broad_describe pairplot from anima.
    """
    cols = columns[:max_cols]
    fig = px.scatter_matrix(
        df, dimensions=cols, color=color,
        title=title,
        color_discrete_sequence=HOMER_COLORS,
        template=HOMER_TEMPLATE,
    )
    fig.update_traces(diagonal_visible=True, marker=dict(size=3, opacity=0.5))
    fig.update_layout(
        font=dict(family="Arial, sans-serif", size=10),
        height=150 * len(cols) + 100,
        width=150 * len(cols) + 100,
    )
    fig.update_layout(**DARK_LAYOUT)
    return fig


# ── Plotly: Multi-metric strip plot (full report style) ─────────────────────

def create_sample_overview_strip(
    df: pd.DataFrame,
    metrics: list[str],
    sample_col: str = "Sample ID",
    title: str = "Sample Overview",
    max_metrics: int = 4,
) -> go.Figure:
    """Create a multi-panel strip plot showing multiple metrics across samples.

    Mirrors the full_report style from ClusterCleaner in anima.
    """
    metrics = metrics[:max_metrics]
    n = len(metrics)

    fig = make_subplots(
        rows=1, cols=n,
        subplot_titles=metrics,
        shared_yaxes=False,
    )

    for i, metric in enumerate(metrics):
        for j, sample in enumerate(df[sample_col].unique()):
            subset = df[df[sample_col] == sample]
            fig.add_trace(
                go.Box(
                    y=subset[metric],
                    name=str(sample),
                    marker_color=HOMER_COLORS[j % len(HOMER_COLORS)],
                    boxpoints="outliers",
                    showlegend=(i == 0),
                    legendgroup=str(sample),
                ),
                row=1, col=i + 1,
            )
        # Y-axis scaling
        if "%" in metric:
            fig.update_yaxes(range=[0, 100], row=1, col=i + 1)

    fig.update_layout(
        title=title,
        template=HOMER_TEMPLATE,
        font=dict(family="Arial, sans-serif", size=11),
        height=500,
        boxmode="group",
    )
    fig.update_layout(**DARK_LAYOUT)
    return fig


# ── Seaborn/Matplotlib: Pairplot for PDF reports ────────────────────────────

def mpl_pairplot(
    df: pd.DataFrame,
    columns: list[str],
    hue: Optional[str] = None,
    max_cols: int = 7,
) -> plt.Figure:
    """Create a seaborn pairplot for PDF reports.

    Mirrors HistologyIngest.broad_describe from anima.
    """
    _apply_homer_style()
    cols = columns[:max_cols]
    plot_df = df[cols + ([hue] if hue and hue not in cols else [])].copy()

    g = sns.pairplot(plot_df, hue=hue, palette=HOMER_COLORS[:df[hue].nunique()] if hue else None)
    g.fig.suptitle("Pairwise Relationships", y=1.02, fontsize=14, fontweight="bold")
    g.fig.tight_layout()
    return g.fig


def mpl_strip_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    title: str = "",
    figsize: tuple = (10, 10),
) -> plt.Figure:
    """Create a seaborn strip plot for PDF reports.

    Mirrors ClusterCleaner.__full_report from anima.
    """
    _apply_homer_style()
    fsize = 20
    fig, ax = plt.subplots(tight_layout=True, figsize=figsize)
    sns.stripplot(data=df, y=y, x=x, hue=hue or x, ax=ax,
                  palette=HOMER_COLORS[:df[x].nunique()])

    fig.suptitle(title or y, fontsize=fsize)
    plt.yticks(fontsize=fsize * 0.75)
    plt.xticks(fontsize=fsize * 0.75, rotation=90)
    plt.ylabel(y, fontsize=fsize * 0.75)
    plt.xlabel("", fontsize=fsize * 0.75)
    plt.legend(loc="upper right", prop={"size": 10})

    if "%" not in y:
        y_max = df[y].max()
        if pd.notna(y_max) and y_max > 0:
            ax.set_ylim([0, y_max + round(0.40 * y_max)])
    else:
        ax.set_ylim([0, 100])

    fig.tight_layout()
    return fig


def mpl_outlier_comparison(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    metric: str,
    lower_bound: float,
    upper_bound: float,
    removed_df: pd.DataFrame = None,
    title: str = "",
    figsize: tuple = (16, 6),
) -> plt.Figure:
    """Create matplotlib before/after histograms for outlier removal.

    Mirrors ClusterCleaner.plot_before_after from anima.
    """
    _apply_homer_style()
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    ax[0].hist(before_df[metric].dropna(), bins=50, color=HOMER_COLORS[0], alpha=0.75)
    ax[0].set_title(f"Before Cleaning: {metric}")
    ax[0].set_xlabel(metric)
    ax[0].set_ylabel("Count")

    ax[1].hist(after_df[metric].dropna(), bins=50, color=HOMER_COLORS[1], alpha=0.75)
    ax[1].set_title(f"After Cleaning: {metric}")
    ax[1].set_xlabel(metric)

    # Preserve axes limits from before
    ax[1].set_xlim(ax[0].get_xlim())
    ax[1].set_ylim(ax[0].get_ylim())

    # Draw bound lines
    if lower_bound != 0 or upper_bound != 0:
        ax[1].axvline(lower_bound, color="yellow", linestyle="dashed", label=f"Lower: {lower_bound:.2f}")
        ax[1].axvline(upper_bound, color="yellow", linestyle="dashed", label=f"Upper: {upper_bound:.2f}")

    if removed_df is not None and len(removed_df) > 0:
        r_min = removed_df[metric].min()
        r_max = removed_df[metric].max()
        if pd.notna(r_min):
            ax[1].axvline(r_min, color="red", linestyle="dashed")
        if pd.notna(r_max):
            ax[1].axvline(r_max, color="red", linestyle="dashed")
        ax[1].text(0.02, 0.95, f"{len(removed_df)} removed",
                   transform=ax[1].transAxes, fontsize=12, verticalalignment="top", color="red")

    ax[1].legend(fontsize=8)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig
