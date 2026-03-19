# Homer - Halo Output Mapper & Explorer for Research (Streamlit Version)
# A data dashboard for HALO by Indica Labs image analysis data
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
    HaloDataset,
)
from homer_core.plotting import (
    create_bar_chart, create_stacked_bar_chart, create_scatter_plot,
    create_box_plot, create_violin_plot, create_histogram, create_heatmap,
    create_xy_line_plot, fig_to_png_bytes, fig_to_svg_bytes,
)
from homer_core.report_generator import ReportBuilder, generate_data_summary_page
from homer_core.sample_data import generate_object_data, generate_summary_data


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

.metric-card {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #2E86AB;
}

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


# ── Header ───────────────────────────────────────────────────────────────────

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
        ["Auto-detect", "Force Object Data", "Force Summary Data"],
        index=0,
    )

    # Demo data option
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Demo Data")
    demo_col1, demo_col2 = st.sidebar.columns(2)
    load_demo_object = demo_col1.button("Object Demo", use_container_width=True)
    load_demo_summary = demo_col2.button("Summary Demo", use_container_width=True)

    if load_demo_object:
        df = generate_object_data(n_cells=5000, n_images=3)
        dataset = parse_halo_data(df, "demo_object_data.csv", force_type="object")
        st.session_state.dataset = dataset
        st.session_state.filters = {}
        st.rerun()

    if load_demo_summary:
        df = generate_summary_data(n_images=12)
        dataset = parse_halo_data(df, "demo_summary_data.csv", force_type="summary")
        st.session_state.dataset = dataset
        st.session_state.filters = {}
        st.rerun()

    if uploaded_file is not None:
        try:
            df = load_uploaded_file(uploaded_file, uploaded_file.name)
            ft = None
            if force_type == "Force Object Data":
                ft = "object"
            elif force_type == "Force Summary Data":
                ft = "summary"
            dataset = parse_halo_data(df, uploaded_file.name, force_type=ft)
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
        # Limit to columns with reasonable cardinality
        filterable = [c for c in filterable if dataset.df[c].nunique() <= 100]

        new_filters = {}
        for col in filterable[:8]:  # Max 8 filter widgets
            unique_vals = sorted(dataset.df[col].dropna().unique().tolist(), key=str)
            selected = st.sidebar.multiselect(
                f"{col}",
                options=unique_vals,
                default=[],
                key=f"filter_{col}",
            )
            if selected:
                new_filters[col] = selected

        st.session_state.filters = new_filters

        # Data info
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Dataset Info")
        badge_class = "badge-object" if dataset.data_type == "object" else "badge-summary"
        st.sidebar.markdown(
            f'<span class="data-badge {badge_class}">{dataset.data_type.upper()} DATA</span>',
            unsafe_allow_html=True,
        )
        st.sidebar.markdown(f"**File:** {dataset.filename}")

        filtered_df = apply_filters(dataset.df, st.session_state.filters)
        st.sidebar.markdown(f"**Rows:** {len(filtered_df):,} / {len(dataset.df):,}")
        st.sidebar.markdown(f"**Columns:** {len(dataset.df.columns)}")


# ── Plot Builder ─────────────────────────────────────────────────────────────

PLOT_TYPES = [
    "Bar Chart",
    "Stacked Bar Chart",
    "Scatter Plot",
    "Box Plot",
    "Violin Plot",
    "Histogram",
    "XY Line Plot",
    "Heatmap",
]


def render_plot_builder(dataset: HaloDataset, filtered_df: pd.DataFrame):
    """Render the interactive plot configuration panel."""
    st.markdown("### Plot Builder")

    col_config, col_preview = st.columns([1, 2])

    with col_config:
        plot_type = st.selectbox("Plot Type", PLOT_TYPES)

        numeric_cols = get_plottable_numeric_columns(dataset)
        # Include categorical columns that exist in filtered_df
        all_cols = list(filtered_df.columns)
        grouping_cols = get_grouping_columns(dataset)

        # X axis
        if plot_type in ("Histogram",):
            x_col = st.selectbox("Variable", numeric_cols, key="x_col")
            y_col = None
        elif plot_type in ("Box Plot", "Violin Plot"):
            x_col = st.selectbox("Grouping (X)", ["(None)"] + grouping_cols, key="x_col")
            if x_col == "(None)":
                x_col = None
            y_col = st.selectbox("Value (Y)", numeric_cols, key="y_col")
        else:
            if plot_type in ("Bar Chart", "Stacked Bar Chart"):
                x_col = st.selectbox("X Axis", grouping_cols + numeric_cols, key="x_col")
            else:
                x_col = st.selectbox("X Axis", numeric_cols + grouping_cols, key="x_col")
            y_col = st.selectbox("Y Axis", numeric_cols, key="y_col")

        # Color/group
        color_col = st.selectbox(
            "Color / Group By",
            ["(None)"] + grouping_cols,
            key="color_col",
        )
        if color_col == "(None)":
            color_col = None

        # Plot-specific options
        title = st.text_input("Chart Title", value="", key="chart_title")

        orientation = "v"
        barmode = "group"
        normalize = False
        trendline = None
        points = "outliers"
        nbins = 50

        if plot_type == "Bar Chart":
            orientation = st.radio("Orientation", ["Vertical", "Horizontal"], key="orient")
            orientation = "v" if orientation == "Vertical" else "h"
            barmode = st.radio("Bar Mode", ["group", "overlay"], key="barmode")

        elif plot_type == "Stacked Bar Chart":
            orientation = st.radio("Orientation", ["Vertical", "Horizontal"], key="orient")
            orientation = "v" if orientation == "Vertical" else "h"
            normalize = st.checkbox("Normalize to 100%", key="normalize")

        elif plot_type == "Scatter Plot":
            trendline_opt = st.selectbox("Trendline", ["None", "OLS", "LOWESS"], key="trend")
            trendline = None if trendline_opt == "None" else trendline_opt.lower()

        elif plot_type in ("Box Plot", "Violin Plot"):
            points = st.selectbox("Show Points", ["outliers", "all", "suspectedoutliers", False], key="points")

        elif plot_type == "Histogram":
            nbins = st.slider("Number of Bins", 10, 200, 50, key="nbins")

        # Aggregation for bar charts
        agg_func = "mean"
        if plot_type in ("Bar Chart", "Stacked Bar Chart"):
            agg_func = st.selectbox("Aggregation", ["mean", "median", "sum", "count"], key="agg_func")

        generate_btn = st.button("Generate Plot", type="primary", use_container_width=True)

    with col_preview:
        if generate_btn or st.session_state.plot_counter > 0:
            if generate_btn:
                st.session_state.plot_counter += 1

            try:
                plot_df = filtered_df.copy()

                # For bar charts, aggregate the data
                if plot_type in ("Bar Chart", "Stacked Bar Chart") and y_col:
                    group_cols = [x_col]
                    if color_col:
                        group_cols.append(color_col)
                    plot_df = plot_df.groupby(group_cols, as_index=False)[y_col].agg(agg_func)

                fig = None

                if plot_type == "Bar Chart":
                    fig = create_bar_chart(
                        plot_df, x_col, y_col, color=color_col,
                        orientation=orientation, barmode=barmode, title=title,
                    )
                elif plot_type == "Stacked Bar Chart":
                    if not color_col:
                        st.warning("Stacked bar charts require a Color/Group By column.")
                    else:
                        fig = create_stacked_bar_chart(
                            plot_df, x_col, y_col, color=color_col,
                            orientation=orientation, title=title, normalize=normalize,
                        )
                elif plot_type == "Scatter Plot":
                    fig = create_scatter_plot(
                        plot_df, x_col, y_col, color=color_col,
                        title=title, trendline=trendline,
                    )
                elif plot_type == "Box Plot":
                    fig = create_box_plot(
                        plot_df, x_col, y_col, color=color_col,
                        title=title, points=points,
                    )
                elif plot_type == "Violin Plot":
                    fig = create_violin_plot(
                        plot_df, x_col, y_col, color=color_col, title=title,
                    )
                elif plot_type == "Histogram":
                    fig = create_histogram(
                        plot_df, x_col, color=color_col,
                        nbins=nbins, title=title,
                    )
                elif plot_type == "XY Line Plot":
                    fig = create_xy_line_plot(
                        plot_df, x_col, y_col, color=color_col, title=title,
                    )
                elif plot_type == "Heatmap":
                    if not y_col:
                        st.warning("Heatmap requires both X and Y axes.")
                    else:
                        z_col = st.selectbox("Value (Z/Color)", numeric_cols, key="z_col")
                        fig = create_heatmap(plot_df, x_col, y_col, z_col, title=title)

                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                    # Download buttons
                    dl_col1, dl_col2, dl_col3 = st.columns(3)
                    with dl_col1:
                        try:
                            png_bytes = fig_to_png_bytes(fig)
                            st.download_button(
                                "Download PNG",
                                data=png_bytes,
                                file_name=f"homer_plot_{st.session_state.plot_counter}.png",
                                mime="image/png",
                            )
                        except Exception:
                            st.info("Install kaleido for PNG export: pip install kaleido")

                    with dl_col2:
                        try:
                            svg_bytes = fig_to_svg_bytes(fig)
                            st.download_button(
                                "Download SVG",
                                data=svg_bytes,
                                file_name=f"homer_plot_{st.session_state.plot_counter}.svg",
                                mime="image/svg+xml",
                            )
                        except Exception:
                            st.info("Install kaleido for SVG export: pip install kaleido")

                    with dl_col3:
                        add_to_report = st.button(
                            "Add to Report",
                            key=f"add_report_{st.session_state.plot_counter}",
                        )
                        if add_to_report:
                            report_title = title if title else f"{plot_type}: {y_col or x_col}"
                            st.session_state.report_figures.append({
                                "title": report_title,
                                "fig": fig,
                            })
                            st.success(f"Added to report ({len(st.session_state.report_figures)} figures)")

            except Exception as e:
                st.error(f"Error generating plot: {e}")


# ── Data Table ───────────────────────────────────────────────────────────────

def render_data_table(filtered_df: pd.DataFrame):
    """Render an interactive data table."""
    st.markdown("### Data Table")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        max_rows = st.number_input("Max rows to display", 10, 10000, 100, key="max_rows")
    with col2:
        search_term = st.text_input("Search in data", "", key="search_term")
    with col3:
        selected_cols = st.multiselect(
            "Select columns",
            options=filtered_df.columns.tolist(),
            default=filtered_df.columns.tolist()[:10],
            key="table_cols",
        )

    display_df = filtered_df[selected_cols] if selected_cols else filtered_df

    if search_term:
        mask = display_df.astype(str).apply(
            lambda col: col.str.contains(search_term, case=False, na=False)
        ).any(axis=1)
        display_df = display_df[mask]

    st.dataframe(display_df.head(max_rows), use_container_width=True, height=400)

    # Download table
    csv_data = display_df.to_csv(index=False).encode()
    st.download_button(
        "Download Filtered Data (CSV)",
        data=csv_data,
        file_name="homer_filtered_data.csv",
        mime="text/csv",
    )


# ── Summary Statistics ───────────────────────────────────────────────────────

def render_summary_stats(dataset: HaloDataset, filtered_df: pd.DataFrame):
    """Render summary statistics cards."""
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

    if dataset.marker_columns:
        st.markdown("#### Marker Columns Detected")
        st.write(", ".join(dataset.marker_columns[:20]))

    # Descriptive stats
    with st.expander("Descriptive Statistics", expanded=False):
        st.dataframe(filtered_df.describe(), use_container_width=True)


# ── Report Download ──────────────────────────────────────────────────────────

def render_report_section(dataset: HaloDataset, filtered_df: pd.DataFrame):
    """Render the report generation section."""
    st.markdown("### Report Generation")

    n_figs = len(st.session_state.report_figures)
    st.info(f"Report contains {n_figs} figure(s). Add figures using the 'Add to Report' button in the Plot Builder.")

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

                st.download_button(
                    "Download PDF Report",
                    data=pdf_bytes,
                    file_name="homer_report.pdf",
                    mime="application/pdf",
                    key="download_pdf",
                )
            except Exception as e:
                st.error(f"Error generating report: {e}")

    with col3:
        if st.button("Clear Report Figures"):
            st.session_state.report_figures = []
            st.rerun()


# ── Main App ─────────────────────────────────────────────────────────────────

def main():
    display_header()
    render_sidebar()

    dataset = st.session_state.dataset

    if dataset is None:
        st.markdown("""
        ### Getting Started

        1. **Upload** a HALO data file (CSV, TSV, or Excel) using the sidebar
        2. Or click **Object Demo** / **Summary Demo** to explore with sample data
        3. Use **filters** in the sidebar to subset your data
        4. Build **interactive plots** with the Plot Builder
        5. **Download** individual figures (PNG/SVG) or a full PDF report

        #### Supported HALO Data Types
        - **Object Data**: Cell-by-cell measurements (Cell ID, coordinates, marker intensities, phenotypes)
        - **Summary Data**: Aggregate statistics (total cells, percentages, densities, H-scores)

        #### Available Plot Types
        - Bar Charts (grouped, stacked, horizontal/vertical)
        - Scatter Plots (with optional trendlines)
        - Box Plots & Violin Plots
        - Histograms
        - XY Line Plots
        - Heatmaps
        """)
        display_footer()
        return

    # Apply filters
    filtered_df = apply_filters(dataset.df, st.session_state.filters)

    # Tabs
    tab_plots, tab_table, tab_stats, tab_report = st.tabs([
        "Plot Builder", "Data Table", "Summary Statistics", "Report",
    ])

    with tab_plots:
        render_plot_builder(dataset, filtered_df)

    with tab_table:
        render_data_table(filtered_df)

    with tab_stats:
        render_summary_stats(dataset, filtered_df)

    with tab_report:
        render_report_section(dataset, filtered_df)

    display_footer()


if __name__ == "__main__":
    main()
