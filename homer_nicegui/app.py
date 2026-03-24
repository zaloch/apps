# Homer - Histology Output Mapper & Explorer for Research (NiceGUI Version)
# A data dashboard for histology image analysis data
# Aligned with anima/HistologyAnalysis workflows and column conventions
__author__ = "Gonzalo Zeballos"
__license__ = "GNU GPLv3"
__version__ = "1.0"

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nicegui import ui, app, events
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import BytesIO, StringIO
import base64
import tempfile

from homer_core.data_parser import (
    load_uploaded_file, load_file, parse_histology_data, apply_filters,
    get_filterable_columns, get_plottable_numeric_columns, get_grouping_columns,
    get_phenotype_columns, dezero, remove_outliers, sample_for_plotting,
    get_memory_usage_mb, HistologyDataset, MAX_INTERACTIVE_ROWS,
)
from homer_core.plotting import (
    create_bar_chart, create_stacked_bar_chart, create_scatter_plot,
    create_box_plot, create_violin_plot, create_histogram, create_heatmap,
    create_xy_line_plot, create_strip_plot, create_swarm_plot,
    create_outlier_comparison, create_pairplot_matrix,
    create_sample_overview_strip,
    fig_to_png_bytes, fig_to_svg_bytes,
)
from homer_core.report_generator import ReportBuilder, generate_data_summary_page
from homer_core.sample_data import generate_object_data, generate_summary_data, generate_cluster_data, PROFILE_NAMES, DEFAULT_PROFILE
from homer_core.metadata import (
    load_metadata_csv, merge_metadata, create_empty_metadata,
    metadata_template_csv, calculate_per_image_percentages,
    aggregate_object_data, generate_demo_metadata,
    ExperimentMetadata, STANDARD_METADATA_FIELDS,
)


# ── Application State ────────────────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.dataset: HistologyDataset | None = None
        self.filters: dict = {}
        self.report_figures: list[dict] = []
        self.plot_counter: int = 0
        self.current_fig: go.Figure | None = None
        self.metadata: ExperimentMetadata | None = None
        self.aggregated_df: pd.DataFrame | None = None

state = AppState()


# ── Custom Styling ───────────────────────────────────────────────────────────

HOMER_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

body, .q-page, .nicegui-content {
    font-family: 'Inter', sans-serif !important;
}

/* ── Header (minimal bar) ──────────────────────────────────────────────── */
.homer-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.15rem 1rem;
    margin: 0;
    border-bottom: 1px solid rgba(99, 179, 237, 0.08);
    background: #0f172a;
}
.homer-header h1 {
    margin: 0;
    font-size: 0.9rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #4FC3F7, #81D4FA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.homer-header p {
    margin: 0;
    font-size: 0.6rem;
    color: #64748b;
    font-weight: 400;
}
.homer-header .version-tag {
    background: rgba(99, 179, 237, 0.08);
    color: #4FC3F7;
    padding: 0.05rem 0.35rem;
    border-radius: 6px;
    font-size: 0.5rem;
    font-weight: 600;
    border: 1px solid rgba(99, 179, 237, 0.1);
    margin-left: auto;
}

/* ── Data Badges ────────────────────────────────────────────────────────── */
.data-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.badge-object { background: rgba(129, 199, 132, 0.12); color: #66BB6A; border: 1px solid rgba(129, 199, 132, 0.25); }
.badge-summary { background: rgba(79, 195, 247, 0.12); color: #4FC3F7; border: 1px solid rgba(79, 195, 247, 0.25); }
.badge-cluster { background: rgba(255, 183, 77, 0.12); color: #FFB74D; border: 1px solid rgba(255, 183, 77, 0.25); }

/* ── Metric Cards ───────────────────────────────────────────────────────── */
.metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    padding: 1.1rem 1.3rem;
    border-radius: 12px;
    border: 1px solid rgba(99, 179, 237, 0.1);
    text-align: center;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(99, 179, 237, 0.3);
}
.metric-card .value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #4FC3F7;
    line-height: 1.2;
}
.metric-card .label {
    font-size: 0.7rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 0.25rem;
}

/* ── Getting Started Cards ──────────────────────────────────────────────── */
.getting-started-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
}
.gs-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid rgba(99, 179, 237, 0.1);
    border-radius: 12px;
    padding: 1.3rem;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.gs-card:hover {
    transform: translateY(-2px);
    border-color: rgba(99, 179, 237, 0.25);
}
.gs-card .gs-icon { font-size: 1.6rem; margin-bottom: 0.5rem; }
.gs-card h4 { margin: 0 0 0.3rem 0; font-size: 0.95rem; font-weight: 600; color: #e2e8f0; }
.gs-card p { margin: 0; font-size: 0.8rem; color: #94a3b8; line-height: 1.5; }

/* ── Workflow Steps ──────────────────────────────────────────────────────── */
.workflow-steps {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin: 1rem 0;
}
.workflow-step {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(30, 41, 59, 0.8);
    padding: 0.4rem 0.8rem;
    border-radius: 20px;
    border: 1px solid rgba(99, 179, 237, 0.08);
    font-size: 0.75rem;
    color: #cbd5e1;
}
.workflow-step .step-num {
    background: rgba(79, 195, 247, 0.15);
    color: #4FC3F7;
    width: 20px; height: 20px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.65rem;
    font-weight: 700;
    flex-shrink: 0;
}

/* ── Plot Chips ──────────────────────────────────────────────────────────── */
.plot-types-row { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.6rem; }
.plot-chip {
    background: rgba(30, 41, 59, 0.9);
    border: 1px solid rgba(99, 179, 237, 0.1);
    padding: 0.25rem 0.6rem;
    border-radius: 6px;
    font-size: 0.68rem;
    color: #94a3b8;
    font-weight: 500;
}

/* ── Section Title ──────────────────────────────────────────────────────── */
.sidebar-section-title {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    margin-top: 0.8rem;
    margin-bottom: 0.3rem;
}

/* ── Sidebar Tweaks ─────────────────────────────────────────────────────── */
.q-drawer { border-right: 1px solid rgba(99, 179, 237, 0.08) !important; }

/* ── Tab Tweaks ──────────────────────────────────────────────────────────── */
.q-tabs--dense .q-tab { min-height: 40px; }
</style>
"""

PLOT_TYPES = [
    "Bar Chart", "Stacked Bar Chart", "Scatter Plot",
    "Box Plot", "Violin Plot", "Strip Plot", "Swarm Plot",
    "Histogram", "XY Line Plot", "Heatmap",
    "Pairplot Matrix", "Sample Overview",
]


# ── File handling ────────────────────────────────────────────────────────────

async def handle_upload(e: events.UploadEventArguments, force_type_select, max_job_cb):
    try:
        content = e.content.read()
        filename = e.name
        file_size_mb = len(content) / (1024 * 1024)
        suffix = os.path.splitext(filename)[1]

        if file_size_mb > 50:
            ui.notify(f"Large file ({file_size_mb:.0f} MB). Loading with memory optimization...", type="info")

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        df, actual_file_size_mb, total_rows = load_file(tmp_path)
        os.unlink(tmp_path)

        ft = None
        force_val = force_type_select.value
        if "Object" in force_val:
            ft = "object"
        elif "Summary" in force_val:
            ft = "summary"
        elif "Cluster" in force_val:
            ft = "cluster"

        state.dataset = parse_histology_data(
            df, filename, force_type=ft, max_job=max_job_cb.value,
            file_size_mb=actual_file_size_mb, total_rows=total_rows,
        )
        state.filters = {}

        msg = f"Loaded {filename}: {len(df):,} rows, {len(df.columns)} columns"
        if state.dataset.is_sampled:
            msg += f" (sampled from {total_rows:,} total rows)"
        ui.notify(msg, type="positive")
        main_content.refresh()
        sidebar_info.refresh()
    except Exception as ex:
        ui.notify(f"Error loading file: {ex}", type="negative")


def load_demo(demo_type: str, n_samples: int = 8, n_objects: int = 5000, auto_agg: bool = True, profile: str | None = None):
    if demo_type == "object":
        df = generate_object_data(n_cells=n_objects, n_images=n_samples, profile=profile)
        dataset = parse_histology_data(df, "demo_object_data.csv", force_type="object")
        state.dataset = dataset
        state.filters = {}

        if auto_agg:
            group_cols = []
            for col in ["Sample ID", "Analysis Region"]:
                if col in df.columns:
                    group_cols.append(col)
            meta_factors = ["Treatment Group", "Genotype", "Timepoint",
                            "Subject ID", "Sex", "Age", "Dose", "Cohort"]
            for col in meta_factors:
                if col in df.columns and col not in group_cols:
                    group_cols.append(col)

            agg_df = aggregate_object_data(
                df,
                group_cols=group_cols,
                classification_cols=dataset.classification_columns,
                phenotype_combo_cols=dataset.phenotype_combo_columns,
                intensity_cols=(dataset.nucleus_intensity_columns +
                                dataset.cell_intensity_columns),
                morphology_cols=dataset.morphology_columns,
            )
            state.aggregated_df = agg_df
            state.dataset = parse_histology_data(
                agg_df, "demo_object_data_aggregated.csv",
                force_type="summary",
            )
    elif demo_type == "summary":
        df = generate_summary_data(n_images=n_samples, profile=profile)
        state.dataset = parse_histology_data(df, "demo_summary_data.csv")
        state.filters = {}
    elif demo_type == "cluster":
        df = generate_cluster_data(n_clusters=n_objects, n_images=n_samples, profile=profile)
        state.dataset = parse_histology_data(df, "demo_cluster_data.csv", force_type="cluster")
        state.filters = {}
    ui.notify(f"Loaded {demo_type} demo data ({n_samples} samples)", type="positive")
    main_content.refresh()
    sidebar_info.refresh()


# ── Download helpers ─────────────────────────────────────────────────────────

def download_png():
    if state.current_fig is None:
        ui.notify("No figure to download", type="warning")
        return
    try:
        png_bytes = fig_to_png_bytes(state.current_fig)
        b64 = base64.b64encode(png_bytes).decode()
        ui.download(src=f"data:image/png;base64,{b64}",
                    filename=f"homer_plot_{state.plot_counter}.png")
    except Exception as ex:
        ui.notify(f"PNG export failed (install kaleido): {ex}", type="warning")


def download_svg():
    if state.current_fig is None:
        ui.notify("No figure to download", type="warning")
        return
    try:
        svg_bytes = fig_to_svg_bytes(state.current_fig)
        b64 = base64.b64encode(svg_bytes).decode()
        ui.download(src=f"data:image/svg+xml;base64,{b64}",
                    filename=f"homer_plot_{state.plot_counter}.svg")
    except Exception as ex:
        ui.notify(f"SVG export failed (install kaleido): {ex}", type="warning")


def add_to_report(title: str):
    if state.current_fig is None:
        ui.notify("No figure to add", type="warning")
        return
    state.report_figures.append({"title": title or f"Plot {state.plot_counter}", "fig": state.current_fig})
    ui.notify(f"Added to report ({len(state.report_figures)} figures)", type="positive")


def download_report(report_title: str):
    if not state.report_figures:
        ui.notify("No figures in report", type="warning")
        return
    try:
        builder = ReportBuilder(title=report_title,
                                dataset_name=state.dataset.filename if state.dataset else "")
        for entry in state.report_figures:
            builder.add_figure(entry["title"], entry["fig"])
        pdf_bytes = builder.generate_pdf()
        b64 = base64.b64encode(pdf_bytes).decode()
        ui.download(src=f"data:application/pdf;base64,{b64}", filename="homer_report.pdf")
        ui.notify("PDF report downloaded!", type="positive")
    except Exception as ex:
        ui.notify(f"Report generation failed: {ex}", type="negative")


def download_filtered_csv():
    if state.dataset is None:
        return
    filtered_df = apply_filters(state.dataset.df, state.filters)
    csv_str = filtered_df.to_csv(index=False)
    b64 = base64.b64encode(csv_str.encode()).decode()
    ui.download(src=f"data:text/csv;base64,{b64}", filename="homer_filtered_data.csv")


# ── Plot generation ──────────────────────────────────────────────────────────

def generate_plot(
    plot_type, x_col, y_col, color_col, title,
    orientation, barmode, normalize, trendline, points, nbins, agg_func,
    plot_container,
    pair_cols=None, sample_col=None, overview_metrics=None,
):
    if state.dataset is None:
        ui.notify("No data loaded", type="warning")
        return

    filtered_df = apply_filters(state.dataset.df, state.filters)
    plot_df = filtered_df.copy()
    color = color_col if color_col and color_col != "(None)" else None

    # Downsample for performance on point-heavy plot types
    if len(plot_df) > 50_000 and plot_type in (
        "Scatter Plot", "Strip Plot", "Swarm Plot", "Pairplot Matrix",
    ):
        plot_df = sample_for_plotting(plot_df, max_points=50_000, stratify_col=color)
        ui.notify(f"Showing {len(plot_df):,} sampled points for performance", type="info")

    try:
        if plot_type in ("Bar Chart", "Stacked Bar Chart") and y_col and x_col:
            group_cols = [x_col]
            if color:
                group_cols.append(color)
            plot_df = plot_df.groupby(group_cols, as_index=False)[y_col].agg(agg_func)

        fig = None

        if plot_type == "Bar Chart":
            fig = create_bar_chart(plot_df, x_col, y_col, color=color,
                                   orientation=orientation, barmode=barmode, title=title)
        elif plot_type == "Stacked Bar Chart":
            if not color:
                ui.notify("Stacked bar requires Color/Group By", type="warning")
                return
            fig = create_stacked_bar_chart(plot_df, x_col, y_col, color=color,
                                           orientation=orientation, title=title, normalize=normalize)
        elif plot_type == "Scatter Plot":
            fig = create_scatter_plot(plot_df, x_col, y_col, color=color,
                                      title=title, trendline=trendline)
        elif plot_type == "Box Plot":
            x = x_col if x_col and x_col != "(None)" else None
            fig = create_box_plot(plot_df, x, y_col, color=color, title=title, points=points)
        elif plot_type == "Violin Plot":
            x = x_col if x_col and x_col != "(None)" else None
            fig = create_violin_plot(plot_df, x, y_col, color=color, title=title, points=points)
        elif plot_type == "Strip Plot":
            x = x_col if x_col and x_col != "(None)" else "index"
            fig = create_strip_plot(plot_df, x, y_col, color=color, title=title)
        elif plot_type == "Swarm Plot":
            x = x_col if x_col and x_col != "(None)" else "index"
            fig = create_swarm_plot(plot_df, x, y_col, color=color, title=title)
        elif plot_type == "Histogram":
            fig = create_histogram(plot_df, x_col, color=color, nbins=nbins, title=title)
        elif plot_type == "XY Line Plot":
            fig = create_xy_line_plot(plot_df, x_col, y_col, color=color, title=title)
        elif plot_type == "Heatmap":
            fig = create_heatmap(plot_df, x_col, y_col, y_col, title=title)
        elif plot_type == "Pairplot Matrix":
            if pair_cols and len(pair_cols) >= 2:
                fig = create_pairplot_matrix(plot_df, pair_cols, color=color, title=title)
            else:
                ui.notify("Select at least 2 columns for pairplot", type="warning")
                return
        elif plot_type == "Sample Overview":
            if overview_metrics:
                fig = create_sample_overview_strip(
                    plot_df, overview_metrics,
                    sample_col=sample_col or "Sample ID", title=title)
            else:
                ui.notify("Select at least 1 metric", type="warning")
                return

        if fig:
            state.current_fig = fig
            state.plot_counter += 1
            plot_container.clear()
            with plot_container:
                ui.plotly(fig).classes("w-full")
            ui.notify("Plot generated!", type="positive")

    except Exception as ex:
        ui.notify(f"Error generating plot: {ex}", type="negative")


# ── UI Components ────────────────────────────────────────────────────────────

@ui.refreshable
def sidebar_info():
    if state.dataset is None:
        return

    ds = state.dataset
    badge_map = {"object": "badge-object", "summary": "badge-summary", "cluster": "badge-cluster"}
    badge_class = badge_map.get(ds.data_type, "badge-summary")
    ui.html(f'<span class="data-badge {badge_class}">{ds.data_type.upper()} DATA</span>')
    ui.label(f"File: {ds.filename}").classes("text-sm text-gray-600")
    if ds.file_size_mb > 0:
        ui.label(f"File Size: {ds.file_size_mb:.1f} MB").classes("text-sm text-gray-500")

    filtered_df = apply_filters(ds.df, state.filters)
    if ds.is_sampled:
        ui.label(f"Rows: {len(filtered_df):,} / {len(ds.df):,} (of {ds.total_rows:,})").classes("text-sm")
    else:
        ui.label(f"Rows: {len(filtered_df):,} / {len(ds.df):,}").classes("text-sm")
    ui.label(f"Columns: {len(ds.df.columns)}").classes("text-sm")
    mem_mb = get_memory_usage_mb(ds.df)
    ui.label(f"Memory: {mem_mb:.1f} MB").classes("text-sm text-gray-500")

    if ds.algorithm_names:
        ui.label(f"Algorithms: {len(ds.algorithm_names)}").classes("text-sm")
    if ds.sample_ids:
        ui.label(f"Samples: {len(ds.sample_ids)}").classes("text-sm")

    # Column groups
    with ui.expansion("Column Groups").classes("w-full"):
        groups = [
            ("Intensity", ds.intensity_columns),
            ("Cell", ds.cell_columns),
            ("Phenotype %", ds.phenotype_fraction_columns),
            ("Spatial", ds.spatial_columns),
        ]
        for name, cols in groups:
            if cols:
                ui.label(f"{name} ({len(cols)})").classes("text-xs font-bold")
                ui.label(", ".join(cols[:3]) + ("..." if len(cols) > 3 else "")).classes("text-xs text-gray-500")

    # Filters
    filterable = get_filterable_columns(ds)
    filterable = [c for c in filterable if c in ds.df.columns and ds.df[c].nunique() <= 100]

    if filterable:
        ui.separator()
        ui.label("Filters").classes("text-sm font-bold")

        for col in filterable[:6]:
            unique_vals = sorted(ds.df[col].dropna().unique().tolist(), key=str)
            str_vals = [str(v) for v in unique_vals]
            vals_map = {str(v): v for v in unique_vals}

            def make_handler(column, vmap):
                def handler(e):
                    if e.value:
                        state.filters[column] = [vmap[v] for v in e.value]
                    else:
                        state.filters.pop(column, None)
                return handler

            ui.select(str_vals, multiple=True, label=col,
                      on_change=make_handler(col, vals_map)).classes("w-full").props("dense")


@ui.refreshable
def main_content():
    if state.dataset is None:
        with ui.column().classes("w-full max-w-5xl mx-auto mt-6 px-4"):
            ui.html("""
            <div class="workflow-steps">
                <div class="workflow-step"><span class="step-num">1</span> Upload data</div>
                <div class="workflow-step"><span class="step-num">2</span> Add metadata</div>
                <div class="workflow-step"><span class="step-num">3</span> Merge &amp; aggregate</div>
                <div class="workflow-step"><span class="step-num">4</span> Plot &amp; explore</div>
                <div class="workflow-step"><span class="step-num">5</span> Export report</div>
            </div>

            <div class="getting-started-grid">
                <div class="gs-card">
                    <div class="gs-icon">📂</div>
                    <h4>Upload Histology Data</h4>
                    <p>Import CSV, TSV, or Excel files exported from histology software. Auto-detects object, summary, or cluster data types.</p>
                </div>
                <div class="gs-card">
                    <div class="gs-icon">🧪</div>
                    <h4>Try Demo Data</h4>
                    <p>Click <strong>Object</strong>, <strong>Summary</strong>, or <strong>Cluster</strong> in the sidebar to explore with sample data.</p>
                </div>
                <div class="gs-card">
                    <div class="gs-icon">🧬</div>
                    <h4>Add Metadata</h4>
                    <p>Map samples to experimental factors (Treatment, Genotype, Subject ID, Timepoint) for biological analysis.</p>
                </div>
                <div class="gs-card">
                    <div class="gs-icon">📊</div>
                    <h4>Build Plots</h4>
                    <p>Create interactive visualizations: Bar, Scatter, Box, Violin, Strip, Heatmap, Pairplot, and more.</p>
                </div>
                <div class="gs-card">
                    <div class="gs-icon">🔬</div>
                    <h4>Process Data</h4>
                    <p>Clean data with de-zero filtering and outlier removal (IQR, percentile, std dev, winsorize).</p>
                </div>
                <div class="gs-card">
                    <div class="gs-icon">📄</div>
                    <h4>Export Reports</h4>
                    <p>Download individual plots as PNG/SVG, or compile a multi-figure PDF report for publication.</p>
                </div>
            </div>

            <div style="margin-top: 1rem;">
                <div style="font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.4rem;">Supported Data Types</div>
                <div class="plot-types-row">
                    <span class="plot-chip" style="border-color: rgba(129, 199, 132, 0.25); color: #66BB6A;">Object Data</span>
                    <span class="plot-chip" style="border-color: rgba(79, 195, 247, 0.25); color: #4FC3F7;">Summary Data</span>
                    <span class="plot-chip" style="border-color: rgba(255, 183, 77, 0.25); color: #FFB74D;">Cluster Data</span>
                </div>
            </div>

            <div style="margin-top: 0.8rem;">
                <div style="font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.4rem;">Available Plot Types</div>
                <div class="plot-types-row">
                    <span class="plot-chip">Bar Chart</span>
                    <span class="plot-chip">Stacked Bar</span>
                    <span class="plot-chip">Scatter Plot</span>
                    <span class="plot-chip">Box Plot</span>
                    <span class="plot-chip">Violin Plot</span>
                    <span class="plot-chip">Strip Plot</span>
                    <span class="plot-chip">Swarm Plot</span>
                    <span class="plot-chip">Histogram</span>
                    <span class="plot-chip">XY Line</span>
                    <span class="plot-chip">Heatmap</span>
                    <span class="plot-chip">Pairplot Matrix</span>
                    <span class="plot-chip">Sample Overview</span>
                </div>
            </div>
            """)
        return

    ds = state.dataset
    filtered_df = apply_filters(ds.df, state.filters)
    numeric_cols = get_plottable_numeric_columns(ds)
    grouping_cols = get_grouping_columns(ds)
    phenotype_cols = get_phenotype_columns(ds)

    # Metrics row
    row_label = "Total Rows"
    row_value = f"{len(filtered_df):,}"
    if ds.is_sampled:
        row_label = "Loaded Rows"
        row_value = f"{len(filtered_df):,} of {ds.total_rows:,}"

    with ui.row().classes("w-full gap-3 mb-4"):
        for label, value in [
            (row_label, row_value),
            ("Data Type", ds.data_type.title()),
            ("Numeric Cols", str(len(ds.numeric_columns))),
            ("Samples", str(len(ds.sample_ids)) if ds.sample_ids else "N/A"),
            ("Memory", f"{get_memory_usage_mb(ds.df):.1f} MB"),
        ]:
            with ui.element("div").classes("flex-1"):
                ui.html(f'<div class="metric-card"><div class="value">{value}</div><div class="label">{label}</div></div>')

    with ui.tabs().classes("w-full") as tabs:
        table_tab = ui.tab("Data Table", icon="table_chart")
        metadata_tab = ui.tab("Metadata", icon="biotech")
        process_tab = ui.tab("Processing", icon="cleaning_services")
        stats_tab = ui.tab("Statistics", icon="analytics")
        plot_tab = ui.tab("Plot Builder", icon="bar_chart")
        multi_plot_tab = ui.tab("Multi Plot", icon="dashboard")
        report_tab = ui.tab("Report", icon="picture_as_pdf")

    with ui.tab_panels(tabs, value=table_tab).classes("w-full"):

        # ── Plot Builder ─────────────────────────────────────────────
        with ui.tab_panel(plot_tab):
            # Plot area (full width, top)
            plot_display = ui.column().classes("w-full")
            with plot_display:
                ui.label("Configure and click Generate to see your plot here.").classes("text-gray-400 text-center mt-4")

            # Download / report row
            with ui.row().classes("w-full gap-2 mt-1"):
                ui.button("PNG", on_click=download_png, color="secondary").props("dense")
                ui.button("SVG", on_click=download_svg, color="secondary").props("dense")

                # title_input defined below but referenced here via closure
                title_ref = {"value": ""}
                ui.button("Add to Report",
                          on_click=lambda: add_to_report(title_ref["value"]),
                          color="accent").props("dense")

            # Collapsible config panel
            with ui.expansion("Plot Configuration", icon="tune", value=True).classes("w-full mt-2"):
                x_options = grouping_cols + numeric_cols
                y_options = numeric_cols
                color_options = ["(None)"] + grouping_cols

                with ui.row().classes("w-full gap-4"):
                    plot_type_sel = ui.select(PLOT_TYPES, label="Plot Type", value="Bar Chart").classes("w-48")
                    x_sel = ui.select(x_options, label="X Axis",
                                      value=x_options[0] if x_options else None).classes("flex-1")
                    y_sel = ui.select(y_options, label="Y Axis",
                                      value=y_options[0] if y_options else None).classes("flex-1")
                    color_sel = ui.select(color_options, label="Color / Group",
                                          value="(None)").classes("w-48")

                with ui.row().classes("w-full gap-4"):
                    title_input = ui.input("Chart Title", value="").classes("flex-1")
                    title_input.on("update:model-value", lambda e: title_ref.update({"value": e.args}))
                    agg_sel = ui.select(["mean", "median", "sum", "count"],
                                        label="Aggregation", value="mean").classes("w-40")
                    orient_sel = ui.select(["v", "h"], label="Orient", value="v").classes("w-24")
                    barmode_sel = ui.select(["group", "overlay"], label="Bar Mode", value="group").classes("w-32")

                with ui.expansion("More Options", icon="settings").classes("w-full"):
                    with ui.row().classes("w-full gap-4"):
                        normalize_cb = ui.checkbox("Normalize to 100%", value=False)
                        trendline_sel = ui.select([None, "ols", "lowess"], label="Trendline", value=None).classes("w-40")
                        points_sel = ui.select(["outliers", "all", "suspectedoutliers"],
                                                label="Show Points", value="outliers").classes("w-40")
                        nbins_slider = ui.slider(min=10, max=200, value=50).props("label").classes("w-48")
                        ui.label("Bins").classes("text-xs")

                    # Pairplot / Sample Overview
                    with ui.row().classes("w-full gap-4"):
                        default_pair = phenotype_cols[:5] if phenotype_cols else numeric_cols[:5]
                        pair_sel = ui.select(numeric_cols, multiple=True, label="Pairplot Columns",
                                             value=default_pair).classes("flex-1")
                        sample_col_sel = ui.select(grouping_cols, label="Sample Column",
                                                    value="Sample ID" if "Sample ID" in grouping_cols else (grouping_cols[0] if grouping_cols else None)).classes("w-48")
                        overview_sel = ui.select(numeric_cols, multiple=True, label="Overview Metrics",
                                                  value=phenotype_cols[:4] if phenotype_cols else numeric_cols[:4]).classes("flex-1")

                def on_generate():
                    title_ref["value"] = title_input.value
                    generate_plot(
                        plot_type_sel.value, x_sel.value, y_sel.value,
                        color_sel.value, title_input.value,
                        orient_sel.value, barmode_sel.value,
                        normalize_cb.value, trendline_sel.value,
                        points_sel.value, int(nbins_slider.value),
                        agg_sel.value, plot_display,
                        pair_cols=pair_sel.value,
                        sample_col=sample_col_sel.value,
                        overview_metrics=overview_sel.value,
                    )

                ui.button("Generate Plot", on_click=on_generate, color="primary").classes("w-full mt-2")

        # ── Metadata & Aggregation ──────────────────────────────────
        with ui.tab_panel(metadata_tab):
            ui.label("Experimental Metadata").classes("text-xl font-bold")
            ui.label(
                "Map each image/sample to experimental factors (Subject ID, Treatment, "
                "Genotype, Timepoint, etc.) then aggregate and plot by these factors."
            ).classes("text-sm text-gray-500 mb-4")

            with ui.row().classes("w-full gap-8"):
                # ── Upload metadata CSV ──
                with ui.card().classes("flex-1 p-4"):
                    ui.label("Upload Metadata CSV").classes("text-lg font-bold")
                    ui.label(
                        "CSV with one row per sample. Must include a Sample ID column."
                    ).classes("text-sm text-gray-500")

                    async def handle_meta_upload(e: events.UploadEventArguments):
                        try:
                            content = e.content.read()
                            meta_io = BytesIO(content)
                            meta = load_metadata_csv(meta_io, filename=e.name)
                            state.metadata = meta
                            ui.notify(
                                f"Loaded metadata: {len(meta.df)} samples, "
                                f"factors: {', '.join(meta.factor_columns)}",
                                type="positive",
                            )
                            main_content.refresh()
                        except Exception as ex:
                            ui.notify(f"Error loading metadata: {ex}", type="negative")

                    ui.upload(
                        label="Upload metadata CSV",
                        auto_upload=True,
                        on_upload=handle_meta_upload,
                    ).classes("w-full").props('accept=".csv,.tsv,.xlsx"')

                    # Demo metadata
                    if ds.sample_ids:
                        def load_demo_meta():
                            demo_df = generate_demo_metadata(ds.sample_ids)
                            state.metadata = ExperimentMetadata(
                                df=demo_df,
                                join_key="Sample ID",
                                factor_columns=[c for c in demo_df.columns if c != "Sample ID"],
                                filename="demo_metadata.csv",
                            )
                            ui.notify("Loaded demo metadata", type="positive")
                            main_content.refresh()

                        ui.button("Load Demo Metadata", on_click=load_demo_meta,
                                  color="secondary").classes("w-full mt-2")

                    # Download template
                    if ds.sample_ids:
                        def dl_template():
                            csv_str = metadata_template_csv(ds.sample_ids)
                            b64 = base64.b64encode(csv_str.encode()).decode()
                            ui.download(src=f"data:text/csv;base64,{b64}",
                                        filename="metadata_template.csv")

                        ui.button("Download Template CSV", on_click=dl_template,
                                  color="secondary").classes("w-full mt-2").props("dense")

                # ── Current metadata status ──
                with ui.card().classes("flex-1 p-4"):
                    if state.metadata is not None:
                        meta = state.metadata
                        ui.label("Current Metadata").classes("text-lg font-bold")

                        with ui.row().classes("gap-4 mb-4"):
                            ui.html(f'<div class="metric-card"><div class="value">{len(meta.df)}</div><div class="label">Samples</div></div>')
                            ui.html(f'<div class="metric-card"><div class="value">{len(meta.factor_columns)}</div><div class="label">Factors</div></div>')
                            matched = set(meta.df[meta.join_key].astype(str)) & set(str(s) for s in ds.sample_ids)
                            ui.html(f'<div class="metric-card"><div class="value">{len(matched)}/{len(ds.sample_ids)}</div><div class="label">Matched</div></div>')

                        ui.label(f"Join key: {meta.join_key}").classes("text-sm")
                        ui.label(f"Factors: {', '.join(meta.factor_columns)}").classes("text-sm text-gray-600")

                        # Preview table
                        with ui.expansion("Preview Metadata").classes("w-full"):
                            meta_cols_def = [{"headerName": c, "field": c, "sortable": True}
                                             for c in meta.df.columns]
                            meta_rows = meta.df.to_dict("records")
                            ui.aggrid({
                                "columnDefs": meta_cols_def,
                                "rowData": meta_rows,
                                "defaultColDef": {"flex": 1, "minWidth": 100},
                            }).classes("w-full").style("height: 250px")

                        # Merge button
                        def do_merge():
                            try:
                                merged_df = merge_metadata(filtered_df, meta)
                                state.dataset = parse_histology_data(
                                    merged_df, ds.filename,
                                    force_type=ds.data_type,
                                    file_size_mb=ds.file_size_mb,
                                    total_rows=ds.total_rows,
                                )
                                state.filters = {}
                                ui.notify(
                                    f"Merged! {len(meta.factor_columns)} columns added. "
                                    f"Use {', '.join(meta.factor_columns)} as plot groupings.",
                                    type="positive",
                                )
                                main_content.refresh()
                                sidebar_info.refresh()
                            except Exception as ex:
                                ui.notify(f"Merge failed: {ex}", type="negative")

                        ui.button("Merge Metadata into Histology Data",
                                  on_click=do_merge, color="primary").classes("w-full mt-4")
                    else:
                        ui.label("No Metadata Loaded").classes("text-lg font-bold")
                        ui.label("Upload a metadata CSV or load demo metadata.").classes("text-sm text-gray-500")

            # ── Object data aggregation ──
            if ds.data_type == "object" and state.metadata is not None:
                ui.separator()
                ui.label("Aggregate Object Data to Per-Image Percentages").classes("text-lg font-bold mt-4")
                ui.label(
                    "Convert per-cell binary data into per-image percentages, "
                    "then plot by Treatment, Genotype, Subject ID, etc."
                ).classes("text-sm text-gray-500")

                agg_options = []
                if "Sample ID" in filtered_df.columns:
                    agg_options.append("Sample ID")
                if "Analysis Region" in filtered_df.columns:
                    agg_options.append("Analysis Region")
                if state.metadata:
                    for fc in state.metadata.factor_columns:
                        if fc in filtered_df.columns and fc not in agg_options:
                            agg_options.append(fc)

                agg_group_sel = ui.select(
                    agg_options, multiple=True, label="Group by",
                    value=["Sample ID"] if "Sample ID" in agg_options else agg_options[:1],
                ).classes("w-full max-w-lg")

                agg_result_container = ui.column().classes("w-full")

                def do_aggregate():
                    try:
                        agg_df = aggregate_object_data(
                            filtered_df,
                            group_cols=agg_group_sel.value,
                            classification_cols=ds.classification_columns,
                            phenotype_combo_cols=ds.phenotype_combo_columns,
                            intensity_cols=(ds.nucleus_intensity_columns + ds.cell_intensity_columns),
                            morphology_cols=ds.morphology_columns,
                        )
                        state.aggregated_df = agg_df
                        ui.notify(
                            f"Aggregated {len(filtered_df):,} objects into {len(agg_df):,} groups",
                            type="positive",
                        )

                        agg_result_container.clear()
                        with agg_result_container:
                            pct_cols = [c for c in agg_df.columns if c.startswith("% ")]
                            show_cols = list(agg_group_sel.value) + ["Object Count"] + pct_cols
                            show_cols = [c for c in show_cols if c in agg_df.columns]
                            display_agg = agg_df[show_cols]

                            agg_col_defs = [{"headerName": c, "field": c, "sortable": True}
                                            for c in display_agg.columns]
                            agg_rows = display_agg.round(2).to_dict("records")
                            ui.aggrid({
                                "columnDefs": agg_col_defs,
                                "rowData": agg_rows,
                                "defaultColDef": {"flex": 1, "minWidth": 100},
                            }).classes("w-full").style("height: 300px")

                            with ui.row().classes("gap-4 mt-4"):
                                def use_agg():
                                    state.dataset = parse_histology_data(
                                        state.aggregated_df,
                                        f"{ds.filename}_aggregated",
                                        force_type="summary",
                                    )
                                    state.filters = {}
                                    ui.notify("Switched to aggregated data for plotting", type="positive")
                                    main_content.refresh()
                                    sidebar_info.refresh()

                                ui.button("Use Aggregated Data for Plotting",
                                          on_click=use_agg, color="primary")

                                def dl_agg():
                                    csv_str = state.aggregated_df.to_csv(index=False)
                                    b64 = base64.b64encode(csv_str.encode()).decode()
                                    ui.download(src=f"data:text/csv;base64,{b64}",
                                                filename="homer_aggregated.csv")

                                ui.button("Download Aggregated CSV",
                                          on_click=dl_agg, color="secondary")

                    except Exception as ex:
                        ui.notify(f"Aggregation failed: {ex}", type="negative")

                ui.button("Aggregate", on_click=do_aggregate, color="primary").classes("mt-2")

        # ── Data Processing ──────────────────────────────────────────
        with ui.tab_panel(process_tab):
            with ui.row().classes("w-full gap-8"):
                with ui.card().classes("flex-1 p-4"):
                    ui.label("De-zero Rows").classes("text-lg font-bold")
                    ui.label("Remove rows where a metric equals zero (noise).").classes("text-sm text-gray-500")
                    dezero_sel = ui.select(numeric_cols, label="Metric", value=numeric_cols[0] if numeric_cols else None).classes("w-full")

                    def apply_dezero():
                        before = len(filtered_df)
                        cleaned = dezero(filtered_df, dezero_sel.value)
                        removed = before - len(cleaned)
                        state.dataset = parse_histology_data(cleaned, ds.filename, force_type=ds.data_type)
                        state.filters = {}
                        ui.notify(f"Removed {removed} rows. {len(cleaned)} remaining.", type="positive")
                        main_content.refresh()
                        sidebar_info.refresh()

                    ui.button("Apply De-zero", on_click=apply_dezero, color="primary").classes("w-full mt-2")

                with ui.card().classes("flex-1 p-4"):
                    ui.label("Outlier Removal").classes("text-lg font-bold")
                    ui.label("IQR, percentile, std dev, or winsorize.").classes("text-sm text-gray-500")
                    out_metric_sel = ui.select(numeric_cols, label="Metric", value=numeric_cols[0] if numeric_cols else None).classes("w-full")
                    out_method_sel = ui.select(["iqr", "percentile", "std", "winsorize"], label="Method", value="iqr").classes("w-full")
                    iqr_factor = ui.slider(min=0.5, max=5.0, value=1.5, step=0.1).props("label")
                    ui.label("IQR Factor / Std Factor").classes("text-xs")

                    outlier_plot_container = ui.column().classes("w-full")

                    def preview_outliers():
                        cleaned, removed, lower, upper = remove_outliers(
                            filtered_df, out_metric_sel.value,
                            method=out_method_sel.value, factor=iqr_factor.value,
                            std_factor=iqr_factor.value,
                        )
                        fig = create_outlier_comparison(
                            filtered_df, cleaned, out_metric_sel.value,
                            lower, upper, len(removed), method=out_method_sel.value.upper(),
                        )
                        outlier_plot_container.clear()
                        with outlier_plot_container:
                            ui.plotly(fig).classes("w-full")
                        ui.notify(f"Lower: {lower:.4f} | Upper: {upper:.4f} | Removed: {len(removed)}", type="info")

                    def apply_outliers():
                        cleaned, removed, lower, upper = remove_outliers(
                            filtered_df, out_metric_sel.value,
                            method=out_method_sel.value, factor=iqr_factor.value,
                            std_factor=iqr_factor.value,
                        )
                        state.dataset = parse_histology_data(cleaned, ds.filename, force_type=ds.data_type)
                        state.filters = {}
                        ui.notify(f"Applied. {len(cleaned)} rows remaining.", type="positive")
                        main_content.refresh()
                        sidebar_info.refresh()

                    with ui.row().classes("w-full gap-2 mt-2"):
                        ui.button("Preview", on_click=preview_outliers, color="secondary").classes("flex-1")
                        ui.button("Apply", on_click=apply_outliers, color="primary").classes("flex-1")

        # ── Data Table ───────────────────────────────────────────────
        with ui.tab_panel(table_tab):
            with ui.row().classes("w-full gap-4 mb-4"):
                max_rows_input = ui.number("Max rows", value=100, min=10, max=10000).classes("w-40")
                search_input = ui.input("Search in data").classes("flex-1")

            table_container = ui.column().classes("w-full")

            def refresh_table():
                display_df = filtered_df.copy()
                search = search_input.value
                if search:
                    mask = display_df.astype(str).apply(
                        lambda col: col.str.contains(search, case=False, na=False)
                    ).any(axis=1)
                    display_df = display_df[mask]

                max_r = int(max_rows_input.value) if max_rows_input.value else 100
                display_df = display_df.head(max_r)

                table_container.clear()
                with table_container:
                    columns = [{"headerName": col, "field": col, "sortable": True,
                               "filter": True, "resizable": True}
                               for col in display_df.columns[:30]]
                    rows = display_df.to_dict("records")
                    ui.aggrid({
                        "columnDefs": columns,
                        "rowData": rows,
                        "defaultColDef": {"flex": 1, "minWidth": 100},
                    }).classes("w-full").style("height: 500px")

            ui.button("Refresh Table", on_click=refresh_table, color="primary")
            ui.button("Download CSV", on_click=download_filtered_csv, color="secondary")
            refresh_table()

        # ── Statistics ───────────────────────────────────────────────
        with ui.tab_panel(stats_tab):
            desc = filtered_df.describe()
            stats_cols = [{"headerName": "Statistic", "field": "index", "pinned": "left"}] + [
                {"headerName": col, "field": col, "sortable": True}
                for col in desc.columns[:20]
            ]
            stats_rows = desc.reset_index().round(4).to_dict("records")
            ui.aggrid({
                "columnDefs": stats_cols,
                "rowData": stats_rows,
                "defaultColDef": {"flex": 1, "minWidth": 120},
            }).classes("w-full").style("height: 400px")

            if ds.algorithm_names:
                ui.label(f"Algorithms: {', '.join(str(a) for a in ds.algorithm_names)}").classes("mt-4 text-sm")
            if ds.sample_ids:
                ui.label(f"Samples ({len(ds.sample_ids)}): {', '.join(str(s) for s in ds.sample_ids[:10])}").classes("text-sm")
            if ds.marker_columns:
                ui.label("Marker Columns:").classes("font-bold mt-4")
                ui.label(", ".join(ds.marker_columns[:20])).classes("text-sm text-gray-600")
            if ds.phenotype_fraction_columns:
                ui.label("Phenotype Fractions (broad_describe):").classes("font-bold mt-2")
                clean_pheno = get_phenotype_columns(ds, include_weak_strong=False)
                ui.label(", ".join(clean_pheno[:15])).classes("text-sm text-gray-600")

        # ── Multi Plot Builder ─────────────────────────────────────────
        with ui.tab_panel(multi_plot_tab):
            ui.label("Multi Plot Builder").classes("text-lg font-bold")
            ui.label("Generate multiple plots at once — each Y metric becomes a separate plot.").classes("text-sm text-gray-500 mb-2")

            multi_plot_container = ui.column().classes("w-full")

            with ui.row().classes("w-full gap-4"):
                multi_type_sel = ui.select(PLOT_TYPES, label="Plot Type", value="Bar Chart").classes("w-48")
                multi_x_sel = ui.select(
                    grouping_cols + numeric_cols, label="X / Grouping",
                    value=grouping_cols[0] if grouping_cols else (numeric_cols[0] if numeric_cols else None),
                ).classes("w-48")
                multi_color_sel = ui.select(
                    ["(None)"] + grouping_cols, label="Color / Group",
                    value="(None)",
                ).classes("w-48")
                multi_agg_sel = ui.select(["mean", "median", "sum", "count"],
                                           label="Aggregation", value="mean").classes("w-40")
                multi_points_sel = ui.select(
                    ["outliers", "suspectedoutliers", "all", "False"],
                    label="Show Points", value="outliers",
                ).classes("w-44").tooltip("Show individual data points on Box/Violin plots")

            with ui.row().classes("w-full gap-4"):
                default_multi_y = phenotype_cols[:6] if phenotype_cols else numeric_cols[:6]
                multi_y_sel = ui.select(numeric_cols, multiple=True, label="Y Metrics (one plot each)",
                                         value=default_multi_y).classes("flex-1")
                multi_grid_sel = ui.select([1, 2, 3, 4], label="Grid columns", value=2).classes("w-32")

            def gen_multi_plots():
                if not multi_y_sel.value:
                    ui.notify("Select at least one Y metric", type="warning")
                    return
                color = multi_color_sel.value if multi_color_sel.value != "(None)" else None
                points_val = multi_points_sel.value if multi_points_sel.value != "False" else False
                figs = []
                for y_col in multi_y_sel.value:
                    title = f"{multi_type_sel.value}: {y_col}"
                    # Use generate_plot logic inline
                    plot_df = filtered_df.copy()
                    if multi_type_sel.value in ("Bar Chart", "Stacked Bar Chart") and y_col and multi_x_sel.value:
                        gcols = [multi_x_sel.value]
                        if color:
                            gcols.append(color)
                        plot_df = plot_df.groupby(gcols, as_index=False)[y_col].agg(multi_agg_sel.value)
                    fig = None
                    try:
                        if multi_type_sel.value == "Bar Chart":
                            fig = create_bar_chart(plot_df, multi_x_sel.value, y_col, color=color, title=title)
                        elif multi_type_sel.value == "Box Plot":
                            fig = create_box_plot(plot_df, multi_x_sel.value, y_col, color=color, title=title, points=points_val)
                        elif multi_type_sel.value == "Violin Plot":
                            fig = create_violin_plot(plot_df, multi_x_sel.value, y_col, color=color, title=title, points=points_val)
                        elif multi_type_sel.value == "Strip Plot":
                            fig = create_strip_plot(plot_df, multi_x_sel.value or "index", y_col, color=color, title=title)
                        elif multi_type_sel.value == "Histogram":
                            fig = create_histogram(plot_df, y_col, color=color, title=title)
                        elif multi_type_sel.value == "Scatter Plot":
                            fig = create_scatter_plot(plot_df, multi_x_sel.value, y_col, color=color, title=title)
                        else:
                            fig = create_bar_chart(plot_df, multi_x_sel.value, y_col, color=color, title=title)
                    except Exception:
                        pass
                    if fig:
                        figs.append({"title": title, "fig": fig})
                        state.report_figures.append({"title": title, "fig": fig})

                # Display grid
                grid_c = int(multi_grid_sel.value)
                multi_plot_container.clear()
                with multi_plot_container:
                    for row_start in range(0, len(figs), grid_c):
                        row_items = figs[row_start:row_start + grid_c]
                        with ui.row().classes("w-full gap-2"):
                            for item in row_items:
                                with ui.column().classes("flex-1 min-w-0"):
                                    ui.plotly(item["fig"]).classes("w-full")

                ui.notify(
                    f"Generated {len(figs)} plots, all added to Report ({len(state.report_figures)} total)",
                    type="positive",
                )

            ui.button("Generate All Plots", on_click=gen_multi_plots, color="primary").classes("mt-2")

        # ── Report ───────────────────────────────────────────────────
        with ui.tab_panel(report_tab):
            n_figs = len(state.report_figures)
            ui.label("Report").classes("text-lg font-bold")

            with ui.row().classes("w-full gap-4"):
                report_title_input = ui.input("Report Title", value="Homer Data Report").classes("flex-1")
                report_grid_sel = ui.select([1, 2, 3, 4], label="PPTX Grid Cols", value=2).classes("w-40")
                report_table_cb = ui.checkbox("Include data table", value=True)

            ui.label(
                f"Report contains {n_figs} figure(s). "
                "Add via Plot Builder, Multi Plot, or Auto-Generate below."
            ).classes("text-sm text-gray-500 mt-2")

            # Auto-generate section
            ui.separator()
            ui.label("Auto-Generate Report").classes("text-md font-bold mt-2")
            ui.label("Automatically create plots for all metrics and export.").classes("text-sm text-gray-500")

            with ui.row().classes("w-full gap-4"):
                auto_color_sel = ui.select(
                    ["(None)"] + grouping_cols, label="Color / Group",
                    value="(None)",
                ).classes("w-48")
                auto_types_sel = ui.select(
                    ["Bar Chart", "Box Plot", "Violin Plot", "Strip Plot", "Histogram"],
                    multiple=True, label="Plot types",
                    value=["Bar Chart", "Box Plot"],
                ).classes("flex-1")

            def auto_generate():
                color = auto_color_sel.value if auto_color_sel.value != "(None)" else None
                x_col = None
                for candidate in ["Treatment Group", "Genotype", "Analysis Region", "Sample ID"]:
                    if candidate in grouping_cols:
                        x_col = candidate
                        break
                if not x_col and grouping_cols:
                    x_col = grouping_cols[0]

                targets = phenotype_cols if phenotype_cols else numeric_cols[:12]
                new_figs = []
                for pt in auto_types_sel.value:
                    for y_col in targets:
                        title = f"{pt}: {y_col}"
                        plot_df = filtered_df.copy()
                        fig = None
                        try:
                            if pt in ("Bar Chart",) and x_col:
                                gcols = [x_col]
                                if color:
                                    gcols.append(color)
                                agg_df = plot_df.groupby(gcols, as_index=False)[y_col].agg("mean")
                                fig = create_bar_chart(agg_df, x_col, y_col, color=color, title=title)
                            elif pt == "Box Plot" and x_col:
                                fig = create_box_plot(plot_df, x_col, y_col, color=color, title=title)
                            elif pt == "Violin Plot" and x_col:
                                fig = create_violin_plot(plot_df, x_col, y_col, color=color, title=title)
                            elif pt == "Strip Plot" and x_col:
                                fig = create_strip_plot(plot_df, x_col, y_col, color=color, title=title)
                            elif pt == "Histogram":
                                fig = create_histogram(plot_df, y_col, color=color, title=title)
                        except Exception:
                            pass
                        if fig:
                            new_figs.append({"title": title, "fig": fig})

                state.report_figures.extend(new_figs)
                ui.notify(
                    f"Generated {len(new_figs)} plots. Report now has {len(state.report_figures)} figures.",
                    type="positive",
                )
                main_content.refresh()

            ui.button("Auto-Generate All Plots", on_click=auto_generate, color="primary").classes("mt-2")

            # Export buttons
            ui.separator()
            ui.label("Export").classes("text-md font-bold mt-2")
            with ui.row().classes("gap-4 mt-2"):
                def download_pptx():
                    if not state.report_figures:
                        ui.notify("No figures in report", type="warning")
                        return
                    try:
                        builder = ReportBuilder(
                            title=report_title_input.value,
                            dataset_name=ds.filename,
                        )
                        for entry in state.report_figures:
                            builder.add_figure(entry["title"], entry["fig"])
                        pptx_bytes = builder.generate_pptx(
                            grid_cols=int(report_grid_sel.value),
                            include_data_table=report_table_cb.value,
                            df=filtered_df,
                        )
                        b64 = base64.b64encode(pptx_bytes).decode()
                        ui.download(
                            src=f"data:application/vnd.openxmlformats-officedocument.presentationml.presentation;base64,{b64}",
                            filename="homer_report.pptx",
                        )
                        ui.notify("PPTX downloaded!", type="positive")
                    except Exception as ex:
                        ui.notify(f"PPTX failed: {ex}", type="negative")

                ui.button("Generate PPTX", on_click=download_pptx, color="primary")
                ui.button("Generate PDF",
                          on_click=lambda: download_report(report_title_input.value),
                          color="secondary")
                ui.button("Download Data CSV", on_click=download_filtered_csv, color="secondary")

                def clear_report():
                    state.report_figures.clear()
                    ui.notify("Report cleared", type="info")
                    main_content.refresh()

                ui.button("Clear All Figures", on_click=clear_report, color="secondary")

            if state.report_figures:
                with ui.expansion(f"Figures in report ({n_figs})", icon="list").classes("w-full mt-2"):
                    for i, entry in enumerate(state.report_figures):
                        ui.label(f"  {i+1}. {entry['title']}").classes("text-sm")


# ── Main Page ────────────────────────────────────────────────────────────────

@ui.page("/")
def index():
    ui.html(HOMER_CSS)

    with ui.header().classes("shadow-none p-0 m-0").style("background: #0f172a; min-height: 0;"):
        ui.html(
            '<div class="homer-header">'
            '<h1>HOMER</h1>'
            '<p>Histology Output Mapper &amp; Explorer for Research</p>'
            '<span class="version-tag">v1.0</span>'
            '</div>'
        )

    with ui.left_drawer(value=False).classes("p-4").style("background: #0f172a;"):
        ui.html('<div class="sidebar-section-title">Data Upload</div>')

        force_type_select = ui.select(
            ["Auto-detect", "Force Object", "Force Summary", "Force Cluster"],
            label="Data Type", value="Auto-detect",
        ).classes("w-full mb-2")

        max_job_cb = ui.checkbox("Latest Job Id only", value=False)

        ui.upload(
            label="Upload histology data file",
            auto_upload=True,
            on_upload=lambda e: handle_upload(e, force_type_select, max_job_cb),
        ).classes("w-full").props('accept=".csv,.tsv,.txt,.xlsx,.xls"')

        ui.separator()
        ui.html('<div class="sidebar-section-title">Quick Start &mdash; Demo Data</div>')

        demo_profile_sel = ui.select(
            PROFILE_NAMES, label="Organ / Panel Profile",
            value=DEFAULT_PROFILE,
        ).classes("w-full").tooltip("Select a tissue/organ panel with literature-based markers")

        with ui.expansion("Simulation Settings", icon="tune").classes("w-full"):
            demo_n_samples = ui.number("Samples / images", value=8, min=2, max=100, step=1).classes("w-full")
            demo_n_objects = ui.number("Objects / cells", value=5000, min=100, max=100000, step=500).classes("w-full")
            demo_auto_agg = ui.checkbox("Auto-aggregate object data", value=True).tooltip(
                "Automatically aggregate per-cell object data into per-image percentages for immediate plotting by Treatment, Genotype, etc.")

        with ui.row().classes("w-full gap-2"):
            ui.button("Object", on_click=lambda: load_demo("object", int(demo_n_samples.value), int(demo_n_objects.value), demo_auto_agg.value, demo_profile_sel.value), color="primary").props("dense").classes("flex-1")
            ui.button("Summary", on_click=lambda: load_demo("summary", int(demo_n_samples.value), int(demo_n_objects.value), profile=demo_profile_sel.value), color="primary").props("dense").classes("flex-1")
            ui.button("Cluster", on_click=lambda: load_demo("cluster", int(demo_n_samples.value), int(demo_n_objects.value), profile=demo_profile_sel.value), color="primary").props("dense").classes("flex-1")

        ui.separator()
        sidebar_info()

    with ui.column().classes("w-full p-4").style("background: #0f172a; min-height: 100vh;"):
        main_content()

    with ui.footer().classes("text-center p-2").style(
        "background: linear-gradient(180deg, transparent, #0f172a); "
        "border-top: 1px solid rgba(99, 179, 237, 0.08);"
    ):
        ui.label("Homer v1.0  ·  Histology Data Dashboard  ·  Built with NiceGUI").classes("text-xs").style("color: #475569;")


ui.run(
    title="Homer - Histology Data Dashboard",
    port=int(os.environ.get("PORT", 8080)),
    host=os.environ.get("HOST", "0.0.0.0"),
    reload=os.environ.get("HOMER_DEV", "").lower() in ("1", "true"),
    storage_secret=os.environ.get("STORAGE_SECRET", "homer-halo-dashboard"),
    dark=True,
)
