# Homer - Histology Output Mapper & Explorer for Research (Streamlit Version)
# A data dashboard for histology image analysis data
# Aligned with anima/HistologyAnalysis workflows and column conventions
__author__ = "Gonzalo Zeballos"
__license__ = "GNU GPLv3"
__version__ = "1.0"

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import tempfile
from io import BytesIO

from homer_core.data_parser import (
    load_file, parse_histology_data, apply_filters,
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
from homer_core.sample_data import generate_object_data, generate_summary_data, generate_cluster_data
from homer_core.metadata import (
    load_metadata_csv, merge_metadata, create_empty_metadata,
    metadata_template_csv, calculate_per_image_percentages,
    aggregate_object_data, generate_demo_metadata,
    ExperimentMetadata, STANDARD_METADATA_FIELDS,
)


# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Homer - Histology Data Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Kill Streamlit top padding ──────────────────────────────────────── */
.stApp > header { display: none; }
section.main > div.block-container {
    padding-top: 0.5rem !important;
    max-width: 100% !important;
}

/* ── Header (minimal bar) ────────────────────────────────────────────── */
.homer-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.2rem 0;
    margin-bottom: 0.3rem;
    border-bottom: 1px solid rgba(99, 179, 237, 0.08);
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
    padding: 0.3rem 0.85rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 0.5rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}
.badge-object { background: rgba(129, 199, 132, 0.12); color: #66BB6A; border: 1px solid rgba(129, 199, 132, 0.25); }
.badge-summary { background: rgba(79, 195, 247, 0.12); color: #4FC3F7; border: 1px solid rgba(79, 195, 247, 0.25); }
.badge-cluster { background: rgba(255, 183, 77, 0.12); color: #FFB74D; border: 1px solid rgba(255, 183, 77, 0.25); }

/* ── Metric Cards ───────────────────────────────────────────────────────── */
.metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    padding: 1.2rem 1.5rem;
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
    font-size: 1.6rem;
    font-weight: 700;
    color: #4FC3F7;
    line-height: 1.2;
}
.metric-card .label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 0.3rem;
}

/* ── Section Headers ────────────────────────────────────────────────────── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid rgba(99, 179, 237, 0.1);
}
.section-header .icon {
    font-size: 1.3rem;
}
.section-header h3 {
    margin: 0;
    font-size: 1.15rem;
    font-weight: 700;
    color: #e2e8f0;
}

/* ── Getting Started Cards ──────────────────────────────────────────────── */
.getting-started-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
}
.gs-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid rgba(99, 179, 237, 0.1);
    border-radius: 12px;
    padding: 1.5rem;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.gs-card:hover {
    transform: translateY(-2px);
    border-color: rgba(99, 179, 237, 0.25);
}
.gs-card .gs-icon {
    font-size: 1.8rem;
    margin-bottom: 0.6rem;
}
.gs-card h4 {
    margin: 0 0 0.4rem 0;
    font-size: 1rem;
    font-weight: 600;
    color: #e2e8f0;
}
.gs-card p {
    margin: 0;
    font-size: 0.85rem;
    color: #94a3b8;
    line-height: 1.5;
}

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
    gap: 0.5rem;
    background: rgba(30, 41, 59, 0.8);
    padding: 0.5rem 1rem;
    border-radius: 20px;
    border: 1px solid rgba(99, 179, 237, 0.08);
    font-size: 0.8rem;
    color: #cbd5e1;
}
.workflow-step .step-num {
    background: rgba(79, 195, 247, 0.15);
    color: #4FC3F7;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.7rem;
    font-weight: 700;
    flex-shrink: 0;
}
.workflow-step .step-arrow {
    color: #475569;
    margin-left: 0.3rem;
}

/* ── Plot Types Grid ────────────────────────────────────────────────────── */
.plot-types-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 0.8rem;
}
.plot-chip {
    background: rgba(30, 41, 59, 0.9);
    border: 1px solid rgba(99, 179, 237, 0.1);
    padding: 0.3rem 0.7rem;
    border-radius: 6px;
    font-size: 0.72rem;
    color: #94a3b8;
    font-weight: 500;
}

/* ── Sidebar Styling ────────────────────────────────────────────────────── */
.sidebar-section-title {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    margin-top: 1rem;
    margin-bottom: 0.4rem;
}

/* ── Footer ─────────────────────────────────────────────────────────────── */
.footer {
    text-align: center;
    padding: 0.3rem 0;
    color: #475569;
    font-size: 0.65rem;
    letter-spacing: 0.02em;
    margin-top: 2rem;
}

/* ── Streamlit Overrides ────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.3rem;
    background: rgba(15, 23, 42, 0.5);
    padding: 0.3rem;
    border-radius: 10px;
    border: 1px solid rgba(99, 179, 237, 0.08);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-size: 0.85rem;
    font-weight: 500;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ── Session State ────────────────────────────────────────────────────────────

if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "filters" not in st.session_state:
    st.session_state.filters = {}
if "report_figures" not in st.session_state:
    st.session_state.report_figures = []
if "plot_counter" not in st.session_state:
    st.session_state.plot_counter = 0
if "metadata" not in st.session_state:
    st.session_state.metadata = None  # ExperimentMetadata or None
if "aggregated_df" not in st.session_state:
    st.session_state.aggregated_df = None  # per-image aggregated data
if "multi_plots" not in st.session_state:
    st.session_state.multi_plots = []  # list of {"title", "fig"} for multi-plot builder


# ── Header / Footer ─────────────────────────────────────────────────────────

def display_header():
    st.markdown(
        '<div class="homer-header">'
        '<h1>HOMER</h1>'
        '<p>Histology Output Mapper &amp; Explorer for Research</p>'
        '<span class="version-tag">v1.0</span>'
        '</div>',
        unsafe_allow_html=True,
    )


def display_footer():
    st.markdown("""
    <div class="footer">
        Homer v1.0 &nbsp;&middot;&nbsp; Histology Data Dashboard &nbsp;&middot;&nbsp; Built with Streamlit
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar():
    st.sidebar.markdown('<div class="sidebar-section-title">Data Upload</div>', unsafe_allow_html=True)

    uploaded_file = st.sidebar.file_uploader(
        "Upload histology data file",
        type=["csv", "tsv", "txt", "xlsx", "xls"],
        help="Upload CSV, TSV, or Excel files exported from histology software",
    )

    force_type = st.sidebar.radio(
        "Data type detection",
        ["Auto-detect", "Force Object Data", "Force Summary Data", "Force Cluster Data"],
        index=0,
    )

    # Histology-specific options
    with st.sidebar.expander("Histology Options"):
        max_job = st.checkbox("Keep only latest Job Id per sample", value=False, key="max_job")
        analysis_area = st.text_input("Filter Analysis Region (blank = all)", value="", key="analysis_area")

    # Demo data
    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="sidebar-section-title">Quick Start &mdash; Demo Data</div>', unsafe_allow_html=True)

    with st.sidebar.expander("Simulation Settings"):
        demo_n_samples = st.number_input("Number of samples / images", min_value=2, max_value=100, value=8, step=1, key="demo_n_samples")
        demo_n_objects = st.number_input("Number of objects / cells", min_value=100, max_value=100000, value=5000, step=500, key="demo_n_objects")
        demo_auto_agg = st.checkbox("Auto-aggregate object data", value=True, key="demo_auto_agg",
                                    help="Automatically aggregate per-cell object data into per-image percentages for immediate plotting by Treatment, Genotype, etc.")

    dc1, dc2, dc3 = st.sidebar.columns(3)
    load_demo_object = dc1.button("Object", use_container_width=True)
    load_demo_summary = dc2.button("Summary", use_container_width=True)
    load_demo_cluster = dc3.button("Cluster", use_container_width=True)

    if load_demo_object:
        df = generate_object_data(n_cells=demo_n_objects, n_images=demo_n_samples)
        dataset = parse_histology_data(df, "demo_object_data.csv", force_type="object")
        st.session_state.dataset = dataset
        st.session_state.filters = {}

        if demo_auto_agg:
            # Build grouping columns: Sample ID + Analysis Region + all metadata factors
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
            st.session_state.aggregated_df = agg_df
            # Switch directly to aggregated view for plotting
            st.session_state.dataset = parse_histology_data(
                agg_df, "demo_object_data_aggregated.csv",
                force_type="summary",
            )

        st.rerun()

    if load_demo_summary:
        df = generate_summary_data(n_images=demo_n_samples)
        st.session_state.dataset = parse_histology_data(df, "demo_summary_data.csv")
        st.session_state.filters = {}
        st.rerun()

    if load_demo_cluster:
        df = generate_cluster_data(n_clusters=demo_n_objects, n_images=demo_n_samples)
        st.session_state.dataset = parse_histology_data(df, "demo_cluster_data.csv", force_type="cluster")
        st.session_state.filters = {}
        st.rerun()

    if uploaded_file is not None:
        try:
            file_size_mb = uploaded_file.size / (1024 * 1024) if hasattr(uploaded_file, 'size') else 0.0
            if file_size_mb > 50:
                st.sidebar.info(f"Large file detected ({file_size_mb:.0f} MB). Loading with memory optimization...")

            # Write to temp file so load_file can use dask for parallel reading
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name
            try:
                df, actual_file_size_mb, total_rows = load_file(tmp_path)
            finally:
                os.unlink(tmp_path)
            ft = None
            if "Object" in force_type:
                ft = "object"
            elif "Summary" in force_type:
                ft = "summary"
            elif "Cluster" in force_type:
                ft = "cluster"

            area = analysis_area if analysis_area else None
            dataset = parse_histology_data(
                df, uploaded_file.name, force_type=ft,
                max_job=max_job, analysis_area=area,
                file_size_mb=file_size_mb, total_rows=total_rows,
            )
            st.session_state.dataset = dataset
            st.session_state.filters = {}

            if dataset.is_sampled:
                st.sidebar.warning(
                    f"File has {total_rows:,} rows. Loaded {len(df):,} rows "
                    f"({MAX_INTERACTIVE_ROWS:,} max) for interactive analysis."
                )
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")

    # Filters
    dataset = st.session_state.dataset
    if dataset is not None:
        st.sidebar.markdown("---")
        st.sidebar.markdown('<div class="sidebar-section-title">Filters</div>', unsafe_allow_html=True)

        filterable = get_filterable_columns(dataset)
        filterable = [c for c in filterable if c in dataset.df.columns and dataset.df[c].nunique() <= 100]

        new_filters = {}
        for col in filterable[:8]:
            unique_vals = sorted(dataset.df[col].dropna().unique().tolist(), key=str)
            selected = st.sidebar.multiselect(f"{col}", options=unique_vals, default=[], key=f"filter_{col}")
            if selected:
                new_filters[col] = selected
        st.session_state.filters = new_filters

        # Dataset info
        st.sidebar.markdown("---")
        st.sidebar.markdown('<div class="sidebar-section-title">Dataset Info</div>', unsafe_allow_html=True)
        badge_map = {"object": "badge-object", "summary": "badge-summary", "cluster": "badge-cluster"}
        badge_class = badge_map.get(dataset.data_type, "badge-summary")
        st.sidebar.markdown(
            f'<span class="data-badge {badge_class}">{dataset.data_type.upper()} DATA</span>',
            unsafe_allow_html=True,
        )
        st.sidebar.markdown(f"**File:** {dataset.filename}")
        if dataset.file_size_mb > 0:
            st.sidebar.markdown(f"**File Size:** {dataset.file_size_mb:.1f} MB")
        filtered_df = apply_filters(dataset.df, st.session_state.filters)
        if dataset.is_sampled:
            st.sidebar.markdown(
                f"**Rows:** {len(filtered_df):,} / {len(dataset.df):,} "
                f"(sampled from {dataset.total_rows:,})"
            )
        else:
            st.sidebar.markdown(f"**Rows:** {len(filtered_df):,} / {len(dataset.df):,}")
        st.sidebar.markdown(f"**Columns:** {len(dataset.df.columns)}")
        mem_mb = get_memory_usage_mb(dataset.df)
        st.sidebar.markdown(f"**Memory:** {mem_mb:.1f} MB")

        if dataset.algorithm_names:
            st.sidebar.markdown(f"**Algorithms:** {', '.join(str(a) for a in dataset.algorithm_names)}")
        if dataset.sample_ids:
            st.sidebar.markdown(f"**Samples:** {len(dataset.sample_ids)}")

        if dataset.fluorophore_channels:
            st.sidebar.markdown(f"**Fluorophores:** {', '.join(dataset.fluorophore_channels)}")

        # Column classification summary
        with st.sidebar.expander("Column Groups"):
            groups = [
                ("Phenotype Combos (DAPI+ C1+...)", dataset.phenotype_combo_columns),
                ("Classification", dataset.classification_columns),
                ("Nucleus Intensity", dataset.nucleus_intensity_columns),
                ("Cell Intensity", dataset.cell_intensity_columns),
                ("% Nucleus Completeness", dataset.completeness_columns),
                ("Morphology", dataset.morphology_columns),
                ("Intensity (H-Score/Intensity)", dataset.intensity_columns),
                ("Cell Columns", dataset.cell_columns),
                ("Total (counts)", dataset.total_columns),
                ("Fraction (%)", dataset.fraction_columns),
                ("Phenotype Totals", dataset.phenotype_total_columns),
                ("Phenotype Fractions", dataset.phenotype_fraction_columns),
                ("Channel Totals", dataset.channel_total_columns),
                ("Channel Fractions", dataset.channel_fraction_columns),
                ("Spatial", dataset.spatial_columns),
                ("Coordinate", dataset.coordinate_columns),
                ("Area", dataset.area_columns),
            ]
            for name, cols in groups:
                if cols:
                    st.markdown(f"**{name}** ({len(cols)})")
                    st.caption(", ".join(cols[:5]) + ("..." if len(cols) > 5 else ""))


# ── Plot Builder ─────────────────────────────────────────────────────────────

PLOT_TYPES = [
    "Bar Chart",
    "Stacked Bar Chart",
    "Scatter Plot",
    "Box Plot",
    "Violin Plot",
    "Strip Plot",
    "Swarm Plot",
    "Histogram",
    "XY Line Plot",
    "Heatmap",
    "Pairplot Matrix",
    "Sample Overview",
]


def render_plot_builder(dataset: HistologyDataset, filtered_df: pd.DataFrame):
    # ── Plot display area (full width, top) ──
    plot_container = st.container()

    # ── Collapsible config panel below ──
    with st.expander("Plot Configuration", expanded=True):
        numeric_cols = get_plottable_numeric_columns(dataset)
        grouping_cols = get_grouping_columns(dataset)
        phenotype_cols = get_phenotype_columns(dataset)

        # Row 1: type + axes (compact horizontal layout)
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])

        with c1:
            plot_type = st.selectbox("Plot Type", PLOT_TYPES)

        x_col = y_col = None
        pair_cols = None
        sample_col = None
        overview_metrics = None

        with c2:
            if plot_type == "Histogram":
                x_col = st.selectbox("Variable", numeric_cols, key="x_col")
            elif plot_type in ("Box Plot", "Violin Plot", "Strip Plot", "Swarm Plot"):
                x_col = st.selectbox("Grouping (X)", ["(None)"] + grouping_cols, key="x_col")
                if x_col == "(None)":
                    x_col = None
            elif plot_type == "Pairplot Matrix":
                default_cols = phenotype_cols[:5] if phenotype_cols else numeric_cols[:5]
                pair_cols = st.multiselect("Columns", numeric_cols, default=default_cols, key="pair_cols")
            elif plot_type == "Sample Overview":
                sample_col = st.selectbox("Sample Column", grouping_cols,
                                           index=grouping_cols.index("Sample ID") if "Sample ID" in grouping_cols else 0,
                                           key="sample_col")
            elif plot_type in ("Bar Chart", "Stacked Bar Chart"):
                x_col = st.selectbox("X Axis", grouping_cols + numeric_cols, key="x_col")
            else:
                x_col = st.selectbox("X Axis", numeric_cols + grouping_cols, key="x_col")

        with c3:
            if plot_type in ("Box Plot", "Violin Plot", "Strip Plot", "Swarm Plot"):
                y_col = st.selectbox("Value (Y)", numeric_cols, key="y_col")
            elif plot_type == "Sample Overview":
                overview_metrics = st.multiselect("Metrics", numeric_cols,
                                                  default=phenotype_cols[:4] if phenotype_cols else numeric_cols[:4],
                                                  key="overview_metrics")
            elif plot_type not in ("Histogram", "Pairplot Matrix"):
                y_col = st.selectbox("Y Axis", numeric_cols, key="y_col")

        with c4:
            color_col = None
            if plot_type not in ("Pairplot Matrix", "Sample Overview"):
                color_col = st.selectbox("Color / Group", ["(None)"] + grouping_cols, key="color_col")
                if color_col == "(None)":
                    color_col = None

        # Row 2: title + options + generate
        o1, o2, o3 = st.columns([2, 2, 1])

        orientation = "v"
        barmode = "group"
        normalize = False
        trendline = None
        points = "outliers"
        nbins = 50
        agg_func = "mean"
        title = ""

        with o1:
            title = st.text_input("Chart Title", value="", key="chart_title")

        with o2:
            if plot_type == "Bar Chart":
                oo1, oo2, oo3 = st.columns(3)
                with oo1:
                    orientation = "v" if st.radio("Orient", ["V", "H"], key="orient", horizontal=True) == "V" else "h"
                with oo2:
                    barmode = st.radio("Mode", ["group", "overlay"], key="barmode", horizontal=True)
                with oo3:
                    agg_func = st.selectbox("Agg", ["mean", "median", "sum", "count"], key="agg_func")
            elif plot_type == "Stacked Bar Chart":
                oo1, oo2, oo3 = st.columns(3)
                with oo1:
                    orientation = "v" if st.radio("Orient", ["V", "H"], key="orient", horizontal=True) == "V" else "h"
                with oo2:
                    normalize = st.checkbox("Normalize %", key="normalize")
                with oo3:
                    agg_func = st.selectbox("Agg", ["mean", "median", "sum", "count"], key="agg_func")
            elif plot_type == "Scatter Plot":
                trend = st.selectbox("Trendline", ["None", "OLS", "LOWESS"], key="trend")
                trendline = None if trend == "None" else trend.lower()
            elif plot_type in ("Box Plot", "Violin Plot"):
                points = st.selectbox("Points", ["outliers", "all", "suspectedoutliers", False], key="points")
            elif plot_type == "Histogram":
                nbins = st.slider("Bins", 10, 200, 50, key="nbins")

        with o3:
            st.write("")  # vertical align
            generate_btn = st.button("Generate Plot", type="primary", use_container_width=True)

    # ── Render plot in full-width container ──
    with plot_container:
        if generate_btn:
            st.session_state.plot_counter += 1
            try:
                fig = _build_single_plot(
                    dataset, filtered_df, plot_type,
                    x_col=x_col, y_col=y_col, color_col=color_col,
                    title=title, agg_func=agg_func, orientation=orientation,
                    barmode=barmode, normalize=normalize, trendline=trendline,
                    points=points, nbins=nbins, pair_cols=pair_cols,
                    sample_col=sample_col, overview_metrics=overview_metrics,
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    _render_download_buttons(fig, title, plot_type, x_col, y_col)
            except Exception as e:
                st.error(f"Error generating plot: {e}")


def _render_download_buttons(fig, title, plot_type, x_col, y_col):
    """Render download and report buttons for a figure."""
    dl_col1, dl_col2, dl_col3 = st.columns(3)
    with dl_col1:
        try:
            png_bytes = fig_to_png_bytes(fig)
            st.download_button("Download PNG", data=png_bytes,
                               file_name=f"homer_plot_{st.session_state.plot_counter}.png",
                               mime="image/png")
        except Exception:
            st.info("Install kaleido for PNG export: pip install kaleido")

    with dl_col2:
        try:
            svg_bytes = fig_to_svg_bytes(fig)
            st.download_button("Download SVG", data=svg_bytes,
                               file_name=f"homer_plot_{st.session_state.plot_counter}.svg",
                               mime="image/svg+xml")
        except Exception:
            st.info("Install kaleido for SVG export: pip install kaleido")

    with dl_col3:
        if st.button("Add to Report", key=f"add_report_{st.session_state.plot_counter}"):
            report_title = title if title else f"{plot_type}: {y_col or x_col}"
            st.session_state.report_figures.append({"title": report_title, "fig": fig})
            st.success(f"Added to report ({len(st.session_state.report_figures)} figures)")


# ── Multi Plot Builder ──────────────────────────────────────────────────────

def _build_single_plot(dataset, filtered_df, plot_type, x_col, y_col, color_col,
                       title, agg_func="mean", orientation="v", barmode="group",
                       normalize=False, trendline=None, points="outliers",
                       nbins=50, pair_cols=None, sample_col=None,
                       overview_metrics=None):
    """Build and return a single plotly figure (shared by Plot Builder and Multi Plot Builder)."""
    plot_df = filtered_df.copy()

    if len(plot_df) > 50_000 and plot_type in (
        "Scatter Plot", "Strip Plot", "Swarm Plot", "Pairplot Matrix",
    ):
        plot_df = sample_for_plotting(plot_df, max_points=50_000, stratify_col=color_col)

    # Aggregate for bar charts
    if plot_type in ("Bar Chart", "Stacked Bar Chart") and y_col and x_col:
        group_cols = [x_col]
        if color_col:
            group_cols.append(color_col)
        plot_df = plot_df.groupby(group_cols, as_index=False)[y_col].agg(agg_func)

    fig = None
    if plot_type == "Bar Chart":
        fig = create_bar_chart(plot_df, x_col, y_col, color=color_col,
                               orientation=orientation, barmode=barmode, title=title)
    elif plot_type == "Stacked Bar Chart" and color_col:
        fig = create_stacked_bar_chart(plot_df, x_col, y_col, color=color_col,
                                       orientation=orientation, title=title, normalize=normalize)
    elif plot_type == "Scatter Plot":
        fig = create_scatter_plot(plot_df, x_col, y_col, color=color_col,
                                  title=title, trendline=trendline)
    elif plot_type == "Box Plot":
        fig = create_box_plot(plot_df, x_col, y_col, color=color_col,
                              title=title, points=points)
    elif plot_type == "Violin Plot":
        fig = create_violin_plot(plot_df, x_col, y_col, color=color_col, title=title)
    elif plot_type == "Strip Plot":
        fig = create_strip_plot(plot_df, x_col or "index", y_col, color=color_col, title=title)
    elif plot_type == "Swarm Plot":
        fig = create_swarm_plot(plot_df, x_col or "index", y_col, color=color_col, title=title)
    elif plot_type == "Histogram":
        fig = create_histogram(plot_df, x_col, color=color_col, nbins=nbins, title=title)
    elif plot_type == "XY Line Plot":
        fig = create_xy_line_plot(plot_df, x_col, y_col, color=color_col, title=title)
    elif plot_type == "Heatmap" and y_col:
        fig = create_heatmap(plot_df, x_col, y_col, y_col, title=title)
    elif plot_type == "Pairplot Matrix" and pair_cols and len(pair_cols) >= 2:
        fig = create_pairplot_matrix(plot_df, pair_cols, color=color_col, title=title)
    elif plot_type == "Sample Overview" and overview_metrics:
        fig = create_sample_overview_strip(plot_df, overview_metrics,
                                           sample_col=sample_col, title=title)
    return fig


def render_multi_plot_builder(dataset: HistologyDataset, filtered_df: pd.DataFrame):
    """Multi Plot Builder: generate a grid of plots at once."""
    st.markdown('<div class="section-header"><span class="icon">📊</span><h3>Multi Plot Builder</h3></div>', unsafe_allow_html=True)
    st.caption("Generate multiple plots at once. Each Y metric becomes a separate plot.")

    numeric_cols = get_plottable_numeric_columns(dataset)
    grouping_cols = get_grouping_columns(dataset)
    phenotype_cols = get_phenotype_columns(dataset)

    mc1, mc2 = st.columns([1, 1])
    with mc1:
        multi_plot_type = st.selectbox("Plot Type", PLOT_TYPES, key="multi_plot_type")
        multi_x = st.selectbox("X Axis / Grouping",
                               grouping_cols + numeric_cols if multi_plot_type in (
                                   "Bar Chart", "Stacked Bar Chart", "Box Plot",
                                   "Violin Plot", "Strip Plot", "Swarm Plot",
                               ) else numeric_cols + grouping_cols,
                               key="multi_x")
        multi_color = st.selectbox("Color / Group By", ["(None)"] + grouping_cols, key="multi_color")
        if multi_color == "(None)":
            multi_color = None

    with mc2:
        default_y = phenotype_cols[:6] if phenotype_cols else numeric_cols[:6]
        multi_y_cols = st.multiselect(
            "Y Metrics (one plot each)", numeric_cols,
            default=default_y,
            key="multi_y_cols",
        )
        multi_grid_cols = st.slider("Grid columns", 1, 4, 2, key="multi_grid_cols")
        multi_agg = st.selectbox("Aggregation", ["mean", "median", "sum", "count"], key="multi_agg")

    if st.button("Generate All Plots", type="primary", key="multi_gen"):
        if not multi_y_cols:
            st.warning("Select at least one Y metric.")
        else:
            figs = []
            progress = st.progress(0)
            for i, y_col in enumerate(multi_y_cols):
                title = f"{multi_plot_type}: {y_col}"
                fig = _build_single_plot(
                    dataset, filtered_df, multi_plot_type,
                    x_col=multi_x, y_col=y_col, color_col=multi_color,
                    title=title, agg_func=multi_agg,
                )
                if fig:
                    figs.append({"title": title, "fig": fig})
                progress.progress((i + 1) / len(multi_y_cols))
            progress.empty()

            st.session_state.multi_plots = figs
            # Also add all to report
            for item in figs:
                st.session_state.report_figures.append(item)
            st.success(f"Generated {len(figs)} plots. All added to Report ({len(st.session_state.report_figures)} total).")

    # Display grid
    plots = st.session_state.multi_plots
    if plots:
        cols_per_row = multi_grid_cols if "multi_grid_cols" in st.session_state else 2
        for row_start in range(0, len(plots), cols_per_row):
            row_items = plots[row_start:row_start + cols_per_row]
            cols = st.columns(len(row_items))
            for col, item in zip(cols, row_items):
                with col:
                    st.plotly_chart(item["fig"], use_container_width=True)

        if st.button("Clear Multi Plots", key="clear_multi"):
            st.session_state.multi_plots = []
            st.rerun()


# ── Data Processing Tab ─────────────────────────────────────────────────────

def render_data_processing(dataset: HistologyDataset, filtered_df: pd.DataFrame):
    """Render data processing tools (dezero, outlier removal)."""
    st.markdown('<div class="section-header"><span class="icon">🧹</span><h3>Data Processing</h3></div>', unsafe_allow_html=True)
    st.caption("Tools for cleaning data, mirroring the anima HistologyMunger and ClusterCleaner workflows.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### De-zero Rows")
        st.caption("Remove rows where a metric equals zero (noise clusters).")
        dezero_metric = st.selectbox("Metric to de-zero", dataset.numeric_columns, key="dezero_metric")
        if st.button("Apply De-zero", key="dezero_btn"):
            before_count = len(filtered_df)
            cleaned = dezero(filtered_df, dezero_metric)
            removed = before_count - len(cleaned)
            st.session_state.dataset = parse_histology_data(cleaned, dataset.filename, force_type=dataset.data_type)
            st.success(f"Removed {removed} zero-valued rows. {len(cleaned)} rows remaining.")
            st.rerun()

    with col2:
        st.markdown("#### Outlier Removal")
        outlier_metric = st.selectbox("Metric", dataset.numeric_columns, key="outlier_metric")
        outlier_method = st.selectbox("Method", ["iqr", "percentile", "std", "winsorize"], key="outlier_method")

        # Method-specific parameters
        factor = 1.5
        lower_pct = 1.0
        upper_pct = 99.0
        std_factor = 2.0
        limits = (0.01, 0.01)

        if outlier_method == "iqr":
            factor = st.slider("IQR Factor", 0.5, 5.0, 1.5, 0.1, key="iqr_factor")
        elif outlier_method == "percentile":
            lower_pct = st.slider("Lower Percentile", 0.0, 25.0, 1.0, 0.5, key="lower_pct")
            upper_pct = st.slider("Upper Percentile", 75.0, 100.0, 99.0, 0.5, key="upper_pct")
        elif outlier_method == "std":
            std_factor = st.slider("Std Factor", 0.5, 5.0, 2.0, 0.1, key="std_factor")
        elif outlier_method == "winsorize":
            lim_lower = st.slider("Lower Limit", 0.0, 0.2, 0.01, 0.005, key="win_lower")
            lim_upper = st.slider("Upper Limit", 0.0, 0.2, 0.01, 0.005, key="win_upper")
            limits = (lim_lower, lim_upper)

        preview_btn = st.button("Preview Outlier Removal", key="outlier_preview")
        apply_btn = st.button("Apply Outlier Removal", type="primary", key="outlier_apply")

    if preview_btn or apply_btn:
        cleaned, removed, lower, upper = remove_outliers(
            filtered_df, outlier_metric, method=outlier_method,
            factor=factor, lower_pct=lower_pct, upper_pct=upper_pct,
            std_factor=std_factor, limits=limits,
        )

        fig = create_outlier_comparison(
            filtered_df, cleaned, outlier_metric,
            lower, upper, len(removed), method=outlier_method.upper(),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"Lower bound: {lower:.4f} | Upper bound: {upper:.4f} | Removed: {len(removed)} rows")

        if apply_btn:
            st.session_state.dataset = parse_histology_data(cleaned, dataset.filename, force_type=dataset.data_type)
            st.success(f"Applied {outlier_method} outlier removal. {len(cleaned)} rows remaining.")
            st.rerun()


# ── Data Table ───────────────────────────────────────────────────────────────

def render_data_table(filtered_df: pd.DataFrame):
    st.markdown('<div class="section-header"><span class="icon">📋</span><h3>Data Table</h3></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        max_rows = st.number_input("Max rows to display", 10, 10000, 100, key="max_rows")
    with col2:
        search_term = st.text_input("Search in data", "", key="search_term")
    with col3:
        selected_cols = st.multiselect(
            "Select columns", options=filtered_df.columns.tolist(),
            default=filtered_df.columns.tolist()[:10], key="table_cols",
        )

    display_df = filtered_df[selected_cols] if selected_cols else filtered_df

    if search_term:
        mask = display_df.astype(str).apply(
            lambda col: col.str.contains(search_term, case=False, na=False)
        ).any(axis=1)
        display_df = display_df[mask]

    st.dataframe(display_df.head(max_rows), use_container_width=True, height=400)

    csv_data = display_df.to_csv(index=False).encode()
    st.download_button("Download Filtered Data (CSV)", data=csv_data,
                       file_name="homer_filtered_data.csv", mime="text/csv")


# ── Summary Statistics ───────────────────────────────────────────────────────

def render_summary_stats(dataset: HistologyDataset, filtered_df: pd.DataFrame):
    st.markdown('<div class="section-header"><span class="icon">📈</span><h3>Summary Statistics</h3></div>', unsafe_allow_html=True)

    metric_cols = st.columns(5)
    with metric_cols[0]:
        if dataset.is_sampled:
            st.metric("Loaded Rows", f"{len(filtered_df):,}",
                       delta=f"of {dataset.total_rows:,} total", delta_color="off")
        else:
            st.metric("Total Rows", f"{len(filtered_df):,}")
    with metric_cols[1]:
        st.metric("Data Type", dataset.data_type.title())
    with metric_cols[2]:
        st.metric("Numeric Columns", len(dataset.numeric_columns))
    with metric_cols[3]:
        st.metric("Categorical Columns", len(dataset.categorical_columns))
    with metric_cols[4]:
        mem = get_memory_usage_mb(dataset.df)
        st.metric("Memory", f"{mem:.1f} MB")

    # Histology-specific info
    info_cols = st.columns(3)
    with info_cols[0]:
        if dataset.algorithm_names:
            st.markdown("**Algorithms:**")
            for a in dataset.algorithm_names:
                st.write(f"- {a}")
    with info_cols[1]:
        if dataset.sample_ids:
            st.markdown(f"**Samples ({len(dataset.sample_ids)}):**")
            for s in dataset.sample_ids[:10]:
                st.write(f"- {s}")
            if len(dataset.sample_ids) > 10:
                st.caption(f"... and {len(dataset.sample_ids) - 10} more")
    with info_cols[2]:
        if dataset.analysis_regions:
            st.markdown("**Analysis Regions:**")
            for r in dataset.analysis_regions:
                st.write(f"- {r}")

    # Column groups
    if dataset.marker_columns:
        st.markdown("#### Marker Columns")
        st.write(", ".join(dataset.marker_columns[:20]))

    if dataset.phenotype_fraction_columns:
        st.markdown("#### Phenotype Fraction Columns (for broad_describe)")
        clean_pheno = get_phenotype_columns(dataset, include_weak_strong=False)
        st.write(", ".join(clean_pheno[:15]))

    with st.expander("Descriptive Statistics", expanded=False):
        st.dataframe(filtered_df.describe(), use_container_width=True)


# ── Report Download ──────────────────────────────────────────────────────────

def _auto_generate_all_plots(dataset: HistologyDataset, filtered_df: pd.DataFrame,
                             plot_types: list[str], color_col: str | None,
                             engine: str = "plotly") -> list[dict]:
    """Auto-generate a comprehensive set of plots for all plottable metrics."""
    numeric_cols = get_plottable_numeric_columns(dataset)
    grouping_cols = get_grouping_columns(dataset)
    phenotype_cols = get_phenotype_columns(dataset)
    figs = []

    # Pick a sensible x grouping
    x_col = None
    for candidate in ["Treatment Group", "Genotype", "Analysis Region", "Sample ID"]:
        if candidate in grouping_cols:
            x_col = candidate
            break
    if not x_col and grouping_cols:
        x_col = grouping_cols[0]

    for pt in plot_types:
        targets = phenotype_cols if phenotype_cols else numeric_cols[:12]
        for y_col in targets:
            title = f"{pt}: {y_col}"
            if pt in ("Bar Chart", "Box Plot", "Violin Plot", "Strip Plot") and x_col:
                fig = _build_single_plot(dataset, filtered_df, pt,
                                         x_col=x_col, y_col=y_col, color_col=color_col,
                                         title=title)
            elif pt == "Histogram":
                fig = _build_single_plot(dataset, filtered_df, pt,
                                         x_col=y_col, y_col=None, color_col=color_col,
                                         title=title)
            elif pt == "Scatter Plot" and len(numeric_cols) >= 2:
                other = [c for c in numeric_cols if c != y_col]
                if other:
                    fig = _build_single_plot(dataset, filtered_df, pt,
                                             x_col=other[0], y_col=y_col, color_col=color_col,
                                             title=title)
                else:
                    fig = None
            else:
                fig = None
            if fig:
                figs.append({"title": title, "fig": fig})
    return figs


def render_report_section(dataset: HistologyDataset, filtered_df: pd.DataFrame):
    st.markdown('<div class="section-header"><span class="icon">📄</span><h3>Report</h3></div>', unsafe_allow_html=True)

    n_figs = len(st.session_state.report_figures)

    # ── Settings row ──
    rc1, rc2, rc3, rc4 = st.columns([2, 1, 1, 1])
    with rc1:
        report_title = st.text_input("Report Title", "Homer Data Report", key="report_title")
    with rc2:
        grid_cols = st.selectbox("Grid columns (PPTX)", [1, 2, 3, 4], index=1, key="report_grid_cols")
    with rc3:
        include_table = st.checkbox("Include data table", value=True, key="report_inc_table")
    with rc4:
        engine = st.selectbox("Render engine", ["plotly", "matplotlib"], key="report_engine")

    st.info(f"Report contains **{n_figs}** figure(s). Add via Plot Builder, Multi Plot Builder, or Auto-Generate below.")

    # ── Auto-generate ──
    st.markdown("#### Auto-Generate Report")
    st.caption("Automatically create plots for all metrics and export.")

    ac1, ac2, ac3 = st.columns(3)
    with ac1:
        grouping_cols = get_grouping_columns(dataset)
        auto_color = st.selectbox("Color / Group By", ["(None)"] + grouping_cols, key="auto_color")
        if auto_color == "(None)":
            auto_color = None
    with ac2:
        auto_types = st.multiselect(
            "Plot types to include",
            ["Bar Chart", "Box Plot", "Violin Plot", "Strip Plot", "Histogram"],
            default=["Bar Chart", "Box Plot"],
            key="auto_types",
        )
    with ac3:
        st.write("")  # spacer
        st.write("")
        if st.button("Auto-Generate All Plots", type="primary", key="auto_gen"):
            with st.spinner("Generating plots for all metrics..."):
                new_figs = _auto_generate_all_plots(dataset, filtered_df, auto_types, auto_color, engine)
            st.session_state.report_figures.extend(new_figs)
            st.success(f"Generated {len(new_figs)} plots. Report now has {len(st.session_state.report_figures)} figures.")
            st.rerun()

    # ── Export buttons ──
    st.markdown("---")
    st.markdown("#### Export")
    ex1, ex2, ex3, ex4 = st.columns(4)

    with ex1:
        if st.button("Generate PPTX", type="primary", disabled=(n_figs == 0), key="gen_pptx"):
            try:
                with st.spinner("Building PPTX..."):
                    builder = ReportBuilder(title=report_title, dataset_name=dataset.filename)
                    for entry in st.session_state.report_figures:
                        builder.add_figure(entry["title"], entry["fig"])
                    pptx_bytes = builder.generate_pptx(
                        grid_cols=grid_cols,
                        include_data_table=include_table,
                        df=filtered_df,
                    )
                st.download_button("Download PPTX", data=pptx_bytes,
                                   file_name="homer_report.pptx",
                                   mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                                   key="download_pptx")
            except Exception as e:
                st.error(f"PPTX generation failed: {e}")

    with ex2:
        if st.button("Generate PDF", disabled=(n_figs == 0), key="gen_pdf"):
            try:
                with st.spinner("Building PDF..."):
                    builder = ReportBuilder(title=report_title, dataset_name=dataset.filename)
                    for entry in st.session_state.report_figures:
                        builder.add_figure(entry["title"], entry["fig"])
                    pdf_bytes = builder.generate_pdf()
                st.download_button("Download PDF", data=pdf_bytes,
                                   file_name="homer_report.pdf", mime="application/pdf",
                                   key="download_pdf")
            except Exception as e:
                st.error(f"PDF generation failed: {e}")

    with ex3:
        csv_data = filtered_df.to_csv(index=False).encode()
        st.download_button("Download Data CSV", data=csv_data,
                           file_name="homer_data.csv", mime="text/csv", key="dl_report_csv")

    with ex4:
        if st.button("Clear All Figures", key="clear_report"):
            st.session_state.report_figures = []
            st.rerun()

    # ── Figures list ──
    if st.session_state.report_figures:
        with st.expander(f"Figures in report ({n_figs})", expanded=False):
            for i, entry in enumerate(st.session_state.report_figures):
                st.write(f"{i+1}. {entry['title']}")


# ── Metadata Tab ──────────────────────────────────────────────────────────────

def render_metadata_tab(dataset: HistologyDataset, filtered_df: pd.DataFrame):
    """Render metadata management: upload, manual entry, merge, and aggregation."""
    st.markdown('<div class="section-header"><span class="icon">🧬</span><h3>Experimental Metadata</h3></div>', unsafe_allow_html=True)
    st.caption(
        "Map each image/sample to experimental factors (Subject ID, Treatment, Genotype, "
        "Timepoint, etc.) then aggregate and plot as a function of these factors."
    )

    meta_col1, meta_col2 = st.columns(2)

    with meta_col1:
        st.markdown("#### Upload Metadata CSV")
        st.caption(
            "CSV with one row per sample. Must include a `Sample ID` column "
            "(matching your histology data) plus any experimental factors."
        )
        meta_file = st.file_uploader(
            "Upload metadata CSV",
            type=["csv", "tsv", "xlsx"],
            key="meta_upload",
        )
        if meta_file is not None:
            try:
                meta = load_metadata_csv(meta_file, filename=meta_file.name)
                st.session_state.metadata = meta
                st.success(
                    f"Loaded metadata: {len(meta.df)} samples, "
                    f"join key: **{meta.join_key}**, "
                    f"factors: {', '.join(meta.factor_columns)}"
                )
            except Exception as e:
                st.error(f"Error loading metadata: {e}")

        # Demo metadata button
        if dataset.sample_ids:
            if st.button("Load Demo Metadata", key="demo_meta"):
                demo_meta_df = generate_demo_metadata(dataset.sample_ids)
                st.session_state.metadata = ExperimentMetadata(
                    df=demo_meta_df,
                    join_key="Sample ID",
                    factor_columns=[c for c in demo_meta_df.columns if c != "Sample ID"],
                    filename="demo_metadata.csv",
                )
                st.success("Loaded demo metadata with Treatment, Genotype, Timepoint, etc.")
                st.rerun()

        # Download template
        if dataset.sample_ids:
            template_csv = metadata_template_csv(dataset.sample_ids)
            st.download_button(
                "Download Metadata Template",
                data=template_csv.encode(),
                file_name="metadata_template.csv",
                mime="text/csv",
                key="dl_meta_template",
            )

    with meta_col2:
        st.markdown("#### Manual Metadata Entry")
        if dataset.sample_ids:
            # If metadata exists, show editable table; otherwise create empty
            if st.session_state.metadata is not None:
                edit_df = st.session_state.metadata.df.copy()
            else:
                edit_df = create_empty_metadata(dataset.sample_ids)

            edited = st.data_editor(
                edit_df,
                use_container_width=True,
                num_rows="fixed",
                height=300,
                key="meta_editor",
            )

            if st.button("Apply Manual Metadata", key="apply_manual_meta"):
                # Remove empty columns
                non_empty = [c for c in edited.columns if not (edited[c] == "").all()]
                cleaned = edited[non_empty]
                meta = ExperimentMetadata(
                    df=cleaned,
                    join_key="Sample ID",
                    factor_columns=[c for c in cleaned.columns if c != "Sample ID" and cleaned[c].nunique() > 1],
                    filename="manual_entry",
                )
                st.session_state.metadata = meta
                st.success(f"Applied metadata with factors: {', '.join(meta.factor_columns)}")
                st.rerun()
        else:
            st.info("Load histology data first to see sample IDs for metadata entry.")

    # ── Show current metadata status and merge ──
    st.markdown("---")

    if st.session_state.metadata is not None:
        meta = st.session_state.metadata

        st.markdown("#### Current Metadata")
        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            st.metric("Samples in Metadata", len(meta.df))
        with mcol2:
            st.metric("Factor Columns", len(meta.factor_columns))
        with mcol3:
            matched = set(meta.df[meta.join_key].astype(str)) & set(
                str(s) for s in dataset.sample_ids
            )
            st.metric("Matched Samples", f"{len(matched)} / {len(dataset.sample_ids)}")

        with st.expander("Preview Metadata", expanded=False):
            st.dataframe(meta.df, use_container_width=True, height=200)

        st.markdown("#### Merge & Aggregate")

        # Merge metadata into histology data
        if st.button("Merge Metadata into Histology Data", type="primary", key="merge_meta"):
            try:
                merged_df = merge_metadata(filtered_df, meta)
                # Re-parse with merged data
                new_dataset = parse_histology_data(
                    merged_df, dataset.filename,
                    force_type=dataset.data_type,
                    file_size_mb=dataset.file_size_mb,
                    total_rows=dataset.total_rows,
                )
                st.session_state.dataset = new_dataset
                st.success(
                    f"Merged! {len(meta.factor_columns)} metadata columns added. "
                    f"You can now use {', '.join(meta.factor_columns)} as plot groupings."
                )
                st.rerun()
            except Exception as e:
                st.error(f"Merge failed: {e}")

        # Object data aggregation
        if dataset.data_type == "object":
            st.markdown("---")
            st.markdown("#### Aggregate Object Data to Per-Image Percentages")
            st.caption(
                "Convert per-cell binary data (0/1 classifications) into per-image "
                "percentages, then plot by Treatment, Genotype, Subject ID, etc."
            )

            agg_group_options = []
            if "Sample ID" in filtered_df.columns:
                agg_group_options.append("Sample ID")
            if "Analysis Region" in filtered_df.columns:
                agg_group_options.append("Analysis Region")
            # Add metadata factor columns if merged
            for fc in meta.factor_columns:
                if fc in filtered_df.columns and fc not in agg_group_options:
                    agg_group_options.append(fc)

            agg_groups = st.multiselect(
                "Group by",
                options=agg_group_options,
                default=["Sample ID"] if "Sample ID" in agg_group_options else agg_group_options[:1],
                key="agg_groups",
            )

            if st.button("Aggregate", type="primary", key="agg_btn") and agg_groups:
                try:
                    agg_df = aggregate_object_data(
                        filtered_df,
                        group_cols=agg_groups,
                        classification_cols=dataset.classification_columns,
                        phenotype_combo_cols=dataset.phenotype_combo_columns,
                        intensity_cols=(dataset.nucleus_intensity_columns +
                                        dataset.cell_intensity_columns),
                        morphology_cols=dataset.morphology_columns,
                    )
                    st.session_state.aggregated_df = agg_df

                    # Also parse as a new dataset for plotting
                    agg_dataset = parse_histology_data(
                        agg_df, f"{dataset.filename}_aggregated",
                        force_type="summary",
                        file_size_mb=0,
                        total_rows=len(agg_df),
                    )
                    st.success(
                        f"Aggregated {len(filtered_df):,} objects into "
                        f"{len(agg_df):,} groups. "
                        f"Switch to the aggregated view to plot by factors."
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Aggregation failed: {e}")

            if st.session_state.aggregated_df is not None:
                st.markdown("**Aggregated Data Preview:**")
                agg_df = st.session_state.aggregated_df
                # Show the % columns prominently
                pct_cols = [c for c in agg_df.columns if c.startswith("% ")]
                display_cols = agg_groups + ["Object Count"] + pct_cols
                display_cols = [c for c in display_cols if c in agg_df.columns]
                st.dataframe(agg_df[display_cols], use_container_width=True, height=300)

                if st.button("Use Aggregated Data for Plotting", key="use_agg"):
                    st.session_state.dataset = parse_histology_data(
                        agg_df, f"{dataset.filename}_aggregated",
                        force_type="summary",
                    )
                    st.session_state.filters = {}
                    st.success("Switched to aggregated data. Use Plot Builder to visualize.")
                    st.rerun()

                csv_data = agg_df.to_csv(index=False).encode()
                st.download_button(
                    "Download Aggregated Data (CSV)",
                    data=csv_data,
                    file_name="homer_aggregated.csv",
                    mime="text/csv",
                    key="dl_agg_csv",
                )

    else:
        st.info("Upload a metadata CSV or use manual entry to map samples to experimental factors.")


# ── Main App ─────────────────────────────────────────────────────────────────

def main():
    display_header()
    render_sidebar()

    dataset = st.session_state.dataset

    if dataset is None:
        st.markdown("""
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
                <p>Click <strong>Object</strong>, <strong>Summary</strong>, or <strong>Cluster</strong> in the sidebar to explore with sample data instantly.</p>
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
            <div style="font-size: 0.8rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.5rem;">Supported Data Types</div>
            <div class="plot-types-row">
                <span class="plot-chip" style="border-color: rgba(129, 199, 132, 0.25); color: #66BB6A;">Object Data</span>
                <span class="plot-chip" style="border-color: rgba(79, 195, 247, 0.25); color: #4FC3F7;">Summary Data</span>
                <span class="plot-chip" style="border-color: rgba(255, 183, 77, 0.25); color: #FFB74D;">Cluster Data</span>
            </div>
        </div>

        <div style="margin-top: 1rem;">
            <div style="font-size: 0.8rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.5rem;">Available Plot Types</div>
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
        """, unsafe_allow_html=True)
        display_footer()
        return

    filtered_df = apply_filters(dataset.df, st.session_state.filters)

    tab_table, tab_metadata, tab_process, tab_stats, tab_plots, tab_multi, tab_report = st.tabs([
        "📋 Data Table", "🧬 Metadata", "🧹 Processing",
        "📈 Statistics", "📊 Plot Builder", "📊 Multi Plot", "📄 Report",
    ])

    with tab_table:
        render_data_table(filtered_df)
    with tab_metadata:
        render_metadata_tab(dataset, filtered_df)
    with tab_process:
        render_data_processing(dataset, filtered_df)
    with tab_stats:
        render_summary_stats(dataset, filtered_df)
    with tab_plots:
        render_plot_builder(dataset, filtered_df)
    with tab_multi:
        render_multi_plot_builder(dataset, filtered_df)
    with tab_report:
        render_report_section(dataset, filtered_df)

    display_footer()


if __name__ == "__main__":
    main()
