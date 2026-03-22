# Homer - PDF Report Generator
# Creates downloadable PDF reports with all dashboard figures

import io
import datetime
from typing import Optional

import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

from . import plotting


class ReportBuilder:
    """Builds a multi-page PDF report from dashboard figures."""

    def __init__(self, title: str = "Homer Data Report", dataset_name: str = ""):
        self.title = title
        self.dataset_name = dataset_name
        self.figures: list[dict] = []  # {"title": str, "plotly_fig": go.Figure, "config": dict}

    def add_figure(
        self,
        title: str,
        plotly_fig: go.Figure,
        config: Optional[dict] = None,
    ):
        """Add a figure to the report."""
        self.figures.append({
            "title": title,
            "plotly_fig": plotly_fig,
            "config": config or {},
        })

    def _create_title_page(self) -> plt.Figure:
        """Create a title page for the report."""
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")

        ax.text(0.5, 0.7, "HOMER", transform=ax.transAxes,
                fontsize=48, fontweight="bold", ha="center", va="center",
                color="#2E86AB")
        ax.text(0.5, 0.60, "Histology Output Mapper & Explorer for Research",
                transform=ax.transAxes, fontsize=16, ha="center", va="center",
                color="#555555")
        ax.text(0.5, 0.45, self.title, transform=ax.transAxes,
                fontsize=20, ha="center", va="center", color="#333333")
        if self.dataset_name:
            ax.text(0.5, 0.38, f"Dataset: {self.dataset_name}",
                    transform=ax.transAxes, fontsize=14, ha="center", va="center",
                    color="#666666")
        ax.text(0.5, 0.25, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                transform=ax.transAxes, fontsize=12, ha="center", va="center",
                color="#888888")
        ax.text(0.5, 0.20, f"Total figures: {len(self.figures)}",
                transform=ax.transAxes, fontsize=12, ha="center", va="center",
                color="#888888")

        fig.patch.set_facecolor("white")
        return fig

    def _plotly_to_mpl_image(self, plotly_fig: go.Figure) -> plt.Figure:
        """Convert a Plotly figure to a matplotlib figure by rendering to image."""
        try:
            img_bytes = plotly_fig.to_image(format="png", width=1100, height=650, scale=2)
            from PIL import Image
            img = Image.open(io.BytesIO(img_bytes))

            fig, ax = plt.subplots(figsize=(11, 7))
            ax.imshow(img)
            ax.axis("off")
            fig.patch.set_facecolor("white")
            fig.tight_layout(pad=0.5)
            return fig
        except Exception:
            # Fallback: create a placeholder
            fig, ax = plt.subplots(figsize=(11, 7))
            ax.text(0.5, 0.5, "Figure could not be rendered\n(install kaleido for image export)",
                    transform=ax.transAxes, fontsize=14, ha="center", va="center")
            ax.axis("off")
            return fig

    def generate_pdf(self) -> bytes:
        """Generate the complete PDF report and return as bytes."""
        buf = io.BytesIO()

        with PdfPages(buf) as pdf:
            # Title page
            title_fig = self._create_title_page()
            pdf.savefig(title_fig, dpi=150)
            plt.close(title_fig)

            # Each figure on its own page
            for entry in self.figures:
                fig = self._plotly_to_mpl_image(entry["plotly_fig"])

                # Add figure title as suptitle
                fig.suptitle(entry["title"], fontsize=14, fontweight="bold", y=0.98)

                pdf.savefig(fig, dpi=150)
                plt.close(fig)

        buf.seek(0)
        return buf.read()

    def clear(self):
        """Clear all figures from the report."""
        self.figures.clear()


def generate_data_summary_page(df: pd.DataFrame, data_type: str) -> plt.Figure:
    """Create a summary statistics page for the dataset."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")

    ax.text(0.5, 0.95, "Dataset Summary", transform=ax.transAxes,
            fontsize=20, fontweight="bold", ha="center", va="top", color="#2E86AB")

    summary_lines = [
        f"Data Type: {data_type.title()}",
        f"Rows: {len(df):,}",
        f"Columns: {len(df.columns)}",
        f"Numeric Columns: {len(df.select_dtypes(include='number').columns)}",
        f"Categorical Columns: {len(df.select_dtypes(include='object').columns)}",
        "",
    ]

    # Add numeric column stats
    numeric_cols = df.select_dtypes(include="number").columns[:10]
    if len(numeric_cols) > 0:
        summary_lines.append("Numeric Column Statistics (first 10):")
        summary_lines.append(f"{'Column':<35} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
        summary_lines.append("-" * 83)
        for col in numeric_cols:
            mean = f"{df[col].mean():.2f}" if pd.notna(df[col].mean()) else "N/A"
            std = f"{df[col].std():.2f}" if pd.notna(df[col].std()) else "N/A"
            mn = f"{df[col].min():.2f}" if pd.notna(df[col].min()) else "N/A"
            mx = f"{df[col].max():.2f}" if pd.notna(df[col].max()) else "N/A"
            col_display = col[:33] if len(col) > 33 else col
            summary_lines.append(f"{col_display:<35} {mean:>12} {std:>12} {mn:>12} {mx:>12}")

    text = "\n".join(summary_lines)
    ax.text(0.05, 0.85, text, transform=ax.transAxes, fontsize=9,
            fontfamily="monospace", va="top", color="#333333")

    fig.patch.set_facecolor("white")
    return fig
