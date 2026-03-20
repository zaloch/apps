# Homer - Halo Output Mapper & Explorer for Research (Streamlit Version)
# A data dashboard for HALO by Indica Labs image analysis data
# Aligned with anima/HaloAnalysis workflows and column conventions
__author__ = "Gonzalo Zeballos"
__license__ = "GNU GPLv3"
__version__ = "1.0"

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

from homer_core.data_parser import (
    load_uploaded_file, parse_halo_data, apply_filters,
    get_filterable_columns, get_plottable_numeric_columns, get_grouping_columns,
    get_phenotype_columns, dezero, remove_outliers,
    HaloDataset,
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


# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Homer - Halo Data Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.homer-header {
    background: linear-gradient(135deg, #2E86AB 0%, #1a5276 100%);
    padding: 1.5rem 2rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    color: white;
}

.homer-header h1 {
    margin: 0;
    font-size: 2.2rem;
    font-weight: 700;
    color: white;
}

.homer-header p {
    margin: 0.3rem 0 0 0;
    font-size: 1rem;
    opacity: 0.85;
    color: white;
}

.data-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 0.5rem;
}

.badge-object { background: #d4edda; color: #155724; }
.badge-summary { background: #cce5ff; color: #004085; }
.badge-cluster { background: #fff3cd; color: #856404; }

.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background-color: #2E86AB;
    text-align: center;
    padding: 0.3rem 0;
    z-index: 1000;
    color: white;
    font-size: 0.8rem;
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


# ── Header / Footer ─────────────────────────────────────────────────────────

def display_header():
    st.markdown("""
    <div class="homer-header">
        <h1>HOMER</h1>
        <p>Halo Output Mapper &amp; Explorer for Research</p>
    </div>
    """, unsafe_allow_html=True)


def display_footer():
    st.markdown("""
    <div class="footer">
        Homer v1.0 | HALO Data Dashboard | Built with Streamlit
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar():
    st.sidebar.markdown("## Data Upload")

    uploaded_file = st.sidebar.file_uploader(
        "Upload HALO data file",
        type=["csv", "tsv", "txt", "xlsx", "xls"],
        help="Upload CSV, TSV, or Excel files exported from HALO",
    )

    force_type = st.sidebar.radio(
        "Data type detection",
        ["Auto-detect", "Force Object Data", "Force Summary Data", "Force Cluster Data"],
        index=0,
    )

    # HALO-specific options
    with st.sidebar.expander("HALO Options"):
        max_job = st.checkbox("Keep only latest Job Id per sample", value=False, key="max_job")
        analysis_area = st.text_input("Filter Analysis Region (blank = all)", value="", key="analysis_area")

    # Demo data
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Demo Data")
    dc1, dc2, dc3 = st.sidebar.columns(3)
    load_demo_object = dc1.button("Object", use_container_width=True)
    load_demo_summary = dc2.button("Summary", use_container_width=True)
    load_demo_cluster = dc3.button("Cluster", use_container_width=True)

    if load_demo_object:
        df = generate_object_data(n_cells=5000, n_images=3)
        st.session_state.dataset = parse_halo_data(df, "demo_object_data.csv", force_type="object")
        st.session_state.filters = {}
        st.rerun()

    if load_demo_summary:
        df = generate_summary_data(n_images=12)
        st.session_state.dataset = parse_halo_data(df, "demo_summary_data.csv")
        st.session_state.filters = {}
        st.rerun()

    if load_demo_cluster:
        df = generate_cluster_data(n_clusters=200, n_images=4)
        st.session_state.dataset = parse_halo_data(df, "demo_cluster_data.csv", force_type="cluster")
        st.session_state.filters = {}
        st.rerun()

    if uploaded_file is not None:
        try:
            df = load_uploaded_file(uploaded_file, uploaded_file.name)
            ft = None
            if "Object" in force_type:
                ft = "object"
            elif "Summary" in force_type:
                ft = "summary"
            elif "Cluster" in force_type:
                ft = "cluster"

            area = analysis_area if analysis_area else None
            dataset = parse_halo_data(df, uploaded_file.name, force_type=ft,
                                      max_job=max_job, analysis_area=area)
            st.session_state.dataset = dataset
            st.session_state.filters = {}
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")

    # Filters
    dataset = st.session_state.dataset
    if dataset is not None:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Filters")

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
        st.sidebar.markdown("### Dataset Info")
        badge_map = {"object": "badge-object", "summary": "badge-summary", "cluster": "badge-cluster"}
        badge_class = badge_map.get(dataset.data_type, "badge-summary")
        st.sidebar.markdown(
            f'<span class="data-badge {badge_class}">{dataset.data_type.upper()} DATA</span>',
            unsafe_allow_html=True,
        )
        st.sidebar.markdown(f"**File:** {dataset.filename}")
        filtered_df = apply_filters(dataset.df, st.session_state.filters)
        st.sidebar.markdown(f"**Rows:** {len(filtered_df):,} / {len(dataset.df):,}")
        st.sidebar.markdown(f"**Columns:** {len(dataset.df.columns)}")

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


def render_plot_builder(dataset: HaloDataset, filtered_df: pd.DataFrame):
    st.markdown("### Plot Builder")

    col_config, col_preview = st.columns([1, 2])

    with col_config:
        plot_type = st.selectbox("Plot Type", PLOT_TYPES)

        numeric_cols = get_plottable_numeric_columns(dataset)
        grouping_cols = get_grouping_columns(dataset)
        phenotype_cols = get_phenotype_columns(dataset)

        # Axis selection based on plot type
        x_col = y_col = None

        if plot_type == "Histogram":
            x_col = st.selectbox("Variable", numeric_cols, key="x_col")
        elif plot_type in ("Box Plot", "Violin Plot", "Strip Plot", "Swarm Plot"):
            x_col = st.selectbox("Grouping (X)", ["(None)"] + grouping_cols, key="x_col")
            if x_col == "(None)":
                x_col = None
            y_col = st.selectbox("Value (Y)", numeric_cols, key="y_col")
        elif plot_type == "Pairplot Matrix":
            default_cols = phenotype_cols[:5] if phenotype_cols else numeric_cols[:5]
            pair_cols = st.multiselect("Columns for pairplot", numeric_cols, default=default_cols, key="pair_cols")
        elif plot_type == "Sample Overview":
            sample_col = st.selectbox("Sample Column", grouping_cols,
                                       index=grouping_cols.index("Sample ID") if "Sample ID" in grouping_cols else 0,
                                       key="sample_col")
            overview_metrics = st.multiselect("Metrics", numeric_cols,
                                              default=phenotype_cols[:4] if phenotype_cols else numeric_cols[:4],
                                              key="overview_metrics")
        else:
            if plot_type in ("Bar Chart", "Stacked Bar Chart"):
                x_col = st.selectbox("X Axis", grouping_cols + numeric_cols, key="x_col")
            else:
                x_col = st.selectbox("X Axis", numeric_cols + grouping_cols, key="x_col")
            y_col = st.selectbox("Y Axis", numeric_cols, key="y_col")

        # Color/group
        color_col = None
        if plot_type not in ("Pairplot Matrix", "Sample Overview"):
            color_col = st.selectbox("Color / Group By", ["(None)"] + grouping_cols, key="color_col")
            if color_col == "(None)":
                color_col = None

        title = st.text_input("Chart Title", value="", key="chart_title")

        # Plot-specific options
        orientation = "v"
        barmode = "group"
        normalize = False
        trendline = None
        points = "outliers"
        nbins = 50
        agg_func = "mean"

        if plot_type == "Bar Chart":
            orientation = "v" if st.radio("Orientation", ["Vertical", "Horizontal"], key="orient") == "Vertical" else "h"
            barmode = st.radio("Bar Mode", ["group", "overlay"], key="barmode")
            agg_func = st.selectbox("Aggregation", ["mean", "median", "sum", "count"], key="agg_func")

        elif plot_type == "Stacked Bar Chart":
            orientation = "v" if st.radio("Orientation", ["Vertical", "Horizontal"], key="orient") == "Vertical" else "h"
            normalize = st.checkbox("Normalize to 100%", key="normalize")
            agg_func = st.selectbox("Aggregation", ["mean", "median", "sum", "count"], key="agg_func")

        elif plot_type == "Scatter Plot":
            trend = st.selectbox("Trendline", ["None", "OLS", "LOWESS"], key="trend")
            trendline = None if trend == "None" else trend.lower()

        elif plot_type in ("Box Plot", "Violin Plot"):
            points = st.selectbox("Show Points", ["outliers", "all", "suspectedoutliers", False], key="points")

        elif plot_type == "Histogram":
            nbins = st.slider("Number of Bins", 10, 200, 50, key="nbins")

        generate_btn = st.button("Generate Plot", type="primary", use_container_width=True)

    with col_preview:
        if generate_btn:
            st.session_state.plot_counter += 1

            try:
                plot_df = filtered_df.copy()
                fig = None

                # Aggregate for bar charts
                if plot_type in ("Bar Chart", "Stacked Bar Chart") and y_col and x_col:
                    group_cols = [x_col]
                    if color_col:
                        group_cols.append(color_col)
                    plot_df = plot_df.groupby(group_cols, as_index=False)[y_col].agg(agg_func)

                if plot_type == "Bar Chart":
                    fig = create_bar_chart(plot_df, x_col, y_col, color=color_col,
                                           orientation=orientation, barmode=barmode, title=title)
                elif plot_type == "Stacked Bar Chart":
                    if not color_col:
                        st.warning("Stacked bar charts require a Color/Group By column.")
                    else:
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
                elif plot_type == "Heatmap":
                    z_col = st.selectbox("Value (Z/Color)", numeric_cols, key="z_col")
                    if y_col:
                        fig = create_heatmap(plot_df, x_col, y_col, z_col, title=title)
                elif plot_type == "Pairplot Matrix":
                    if pair_cols and len(pair_cols) >= 2:
                        fig = create_pairplot_matrix(plot_df, pair_cols, color=color_col, title=title)
                    else:
                        st.warning("Select at least 2 columns for pairplot.")
                elif plot_type == "Sample Overview":
                    if overview_metrics:
                        fig = create_sample_overview_strip(plot_df, overview_metrics,
                                                           sample_col=sample_col, title=title)
                    else:
                        st.warning("Select at least 1 metric.")

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


# ── Data Processing Tab ─────────────────────────────────────────────────────

def render_data_processing(dataset: HaloDataset, filtered_df: pd.DataFrame):
    """Render data processing tools (dezero, outlier removal)."""
    st.markdown("### Data Processing")
    st.caption("Tools for cleaning data, mirroring the anima HaloMunger and ClusterCleaner workflows.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### De-zero Rows")
        st.caption("Remove rows where a metric equals zero (noise clusters).")
        dezero_metric = st.selectbox("Metric to de-zero", dataset.numeric_columns, key="dezero_metric")
        if st.button("Apply De-zero", key="dezero_btn"):
            before_count = len(filtered_df)
            cleaned = dezero(filtered_df, dezero_metric)
            removed = before_count - len(cleaned)
            st.session_state.dataset = parse_halo_data(cleaned, dataset.filename, force_type=dataset.data_type)
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
            st.session_state.dataset = parse_halo_data(cleaned, dataset.filename, force_type=dataset.data_type)
            st.success(f"Applied {outlier_method} outlier removal. {len(cleaned)} rows remaining.")
            st.rerun()


# ── Data Table ───────────────────────────────────────────────────────────────

def render_data_table(filtered_df: pd.DataFrame):
    st.markdown("### Data Table")

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

def render_summary_stats(dataset: HaloDataset, filtered_df: pd.DataFrame):
    st.markdown("### Summary Statistics")

    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Total Rows", f"{len(filtered_df):,}")
    with metric_cols[1]:
        st.metric("Data Type", dataset.data_type.title())
    with metric_cols[2]:
        st.metric("Numeric Columns", len(dataset.numeric_columns))
    with metric_cols[3]:
        st.metric("Categorical Columns", len(dataset.categorical_columns))

    # HALO-specific info
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

def render_report_section(dataset: HaloDataset, filtered_df: pd.DataFrame):
    st.markdown("### Report Generation")

    n_figs = len(st.session_state.report_figures)
    st.info(f"Report contains {n_figs} figure(s). Add figures via 'Add to Report' in Plot Builder.")

    col1, col2, col3 = st.columns(3)

    with col1:
        report_title = st.text_input("Report Title", "Homer Data Report", key="report_title")

    with col2:
        if st.button("Generate PDF Report", type="primary", disabled=(n_figs == 0)):
            try:
                with st.spinner("Generating PDF report..."):
                    builder = ReportBuilder(title=report_title, dataset_name=dataset.filename)
                    for entry in st.session_state.report_figures:
                        builder.add_figure(entry["title"], entry["fig"])
                    pdf_bytes = builder.generate_pdf()

                st.download_button("Download PDF Report", data=pdf_bytes,
                                   file_name="homer_report.pdf", mime="application/pdf",
                                   key="download_pdf")
            except Exception as e:
                st.error(f"Error generating report: {e}")

    with col3:
        if st.button("Clear Report Figures"):
            st.session_state.report_figures = []
            st.rerun()

    if st.session_state.report_figures:
        st.markdown("**Figures in report:**")
        for i, entry in enumerate(st.session_state.report_figures):
            st.write(f"{i+1}. {entry['title']}")


# ── Main App ─────────────────────────────────────────────────────────────────

def main():
    display_header()
    render_sidebar()

    dataset = st.session_state.dataset

    if dataset is None:
        st.markdown("""
        ### Getting Started

        1. **Upload** a HALO data file (CSV, TSV, or Excel) using the sidebar
        2. Or click **Object** / **Summary** / **Cluster** demo buttons
        3. Use **filters** in the sidebar to subset your data
        4. Use **Data Processing** to clean data (de-zero, outlier removal)
        5. Build **interactive plots** with the Plot Builder
        6. **Download** individual figures (PNG/SVG) or a full PDF report

        #### Supported HALO Data Types
        - **Summary Data**: HALO analysis output (Algorithm Name, Job Id, Image Tag, cell counts, %, H-Scores)
        - **Object Data**: Cell-by-cell exports (Cell ID, coordinates, marker intensities, phenotypes)
        - **Cluster Data**: Aggregated object data (Total Cluster Count, Region Area, cell fractions)

        #### Column Classification (anima-compatible)
        Homer auto-detects and classifies columns following the same logic as `HaloAnalysis.set_analysis_metrics()`:
        - **Intensity**: H-Score, Intensity columns
        - **Cell/Total/Fraction**: Columns with "Cells", split by "%" presence
        - **Phenotype vs Channel**: Split by Spectrum/Cy5 presence
        - **Spatial/Coordinate/Area**: Non-cell numeric metrics

        #### Available Plot Types
        Bar, Stacked Bar, Scatter, Box, Violin, Strip, Swarm, Histogram, XY Line, Heatmap, Pairplot Matrix, Sample Overview
        """)
        display_footer()
        return

    filtered_df = apply_filters(dataset.df, st.session_state.filters)

    tab_plots, tab_process, tab_table, tab_stats, tab_report = st.tabs([
        "Plot Builder", "Data Processing", "Data Table", "Summary Statistics", "Report",
    ])

    with tab_plots:
        render_plot_builder(dataset, filtered_df)
    with tab_process:
        render_data_processing(dataset, filtered_df)
    with tab_table:
        render_data_table(filtered_df)
    with tab_stats:
        render_summary_stats(dataset, filtered_df)
    with tab_report:
        render_report_section(dataset, filtered_df)

    display_footer()


if __name__ == "__main__":
    main()
