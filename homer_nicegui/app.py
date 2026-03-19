# Homer - Halo Output Mapper & Explorer for Research (NiceGUI Version)
# A data dashboard for HALO by Indica Labs image analysis data
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


# ── Application State ────────────────────────────────────────────────────────

class AppState:
    """Global application state."""
    def __init__(self):
        self.dataset: HaloDataset | None = None
        self.filters: dict = {}
        self.report_figures: list[dict] = []
        self.plot_counter: int = 0
        self.current_fig: go.Figure | None = None

state = AppState()


# ── Custom Styling ───────────────────────────────────────────────────────────

HOMER_CSS = """
<style>
.homer-header {
    background: linear-gradient(135deg, #2E86AB 0%, #1a5276 100%);
    padding: 1.2rem 2rem;
    border-radius: 10px;
    color: white;
    margin-bottom: 1rem;
}
.homer-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
.homer-header p { margin: 0.2rem 0 0 0; font-size: 0.95rem; opacity: 0.85; }

.data-badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 15px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-object { background: #d4edda; color: #155724; }
.badge-summary { background: #cce5ff; color: #004085; }

.metric-card {
    background: #f8f9fa;
    padding: 0.8rem;
    border-radius: 8px;
    border-left: 4px solid #2E86AB;
    text-align: center;
}
.metric-card .value { font-size: 1.5rem; font-weight: 700; color: #2E86AB; }
.metric-card .label { font-size: 0.8rem; color: #666; }
</style>
"""


# ── Plot Configuration ───────────────────────────────────────────────────────

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


# ── File handling helpers ────────────────────────────────────────────────────

async def handle_upload(e: events.UploadEventArguments, force_type_select):
    """Handle file upload event."""
    try:
        content = e.content.read()
        filename = e.name

        # Write to temp file for pandas to read
        suffix = os.path.splitext(filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        from homer_core.data_parser import load_file
        df = load_file(tmp_path)
        os.unlink(tmp_path)

        ft = None
        force_val = force_type_select.value
        if force_val == "Force Object":
            ft = "object"
        elif force_val == "Force Summary":
            ft = "summary"

        state.dataset = parse_halo_data(df, filename, force_type=ft)
        state.filters = {}
        ui.notify(f"Loaded {filename}: {len(df):,} rows, {len(df.columns)} columns", type="positive")
        main_content.refresh()
        sidebar_info.refresh()
    except Exception as ex:
        ui.notify(f"Error loading file: {ex}", type="negative")


def load_demo_object():
    """Load demo object-level data."""
    df = generate_object_data(n_cells=5000, n_images=3)
    state.dataset = parse_halo_data(df, "demo_object_data.csv", force_type="object")
    state.filters = {}
    ui.notify("Loaded demo object data (5,000 cells)", type="positive")
    main_content.refresh()
    sidebar_info.refresh()


def load_demo_summary():
    """Load demo summary data."""
    df = generate_summary_data(n_images=12)
    state.dataset = parse_halo_data(df, "demo_summary_data.csv", force_type="summary")
    state.filters = {}
    ui.notify("Loaded demo summary data (12 images)", type="positive")
    main_content.refresh()
    sidebar_info.refresh()


# ── Download helpers ─────────────────────────────────────────────────────────

def download_png():
    """Download current figure as PNG."""
    if state.current_fig is None:
        ui.notify("No figure to download", type="warning")
        return
    try:
        png_bytes = fig_to_png_bytes(state.current_fig)
        b64 = base64.b64encode(png_bytes).decode()
        ui.download(
            src=f"data:image/png;base64,{b64}",
            filename=f"homer_plot_{state.plot_counter}.png",
        )
    except Exception as ex:
        ui.notify(f"PNG export failed (install kaleido): {ex}", type="warning")


def download_svg():
    """Download current figure as SVG."""
    if state.current_fig is None:
        ui.notify("No figure to download", type="warning")
        return
    try:
        svg_bytes = fig_to_svg_bytes(state.current_fig)
        b64 = base64.b64encode(svg_bytes).decode()
        ui.download(
            src=f"data:image/svg+xml;base64,{b64}",
            filename=f"homer_plot_{state.plot_counter}.svg",
        )
    except Exception as ex:
        ui.notify(f"SVG export failed (install kaleido): {ex}", type="warning")


def add_to_report(title: str):
    """Add current figure to report."""
    if state.current_fig is None:
        ui.notify("No figure to add", type="warning")
        return
    report_title = title if title else f"Plot {state.plot_counter}"
    state.report_figures.append({
        "title": report_title,
        "fig": state.current_fig,
    })
    ui.notify(f"Added to report ({len(state.report_figures)} figures)", type="positive")


def download_report(report_title: str):
    """Generate and download PDF report."""
    if not state.report_figures:
        ui.notify("No figures in report", type="warning")
        return
    try:
        builder = ReportBuilder(
            title=report_title,
            dataset_name=state.dataset.filename if state.dataset else "",
        )
        for entry in state.report_figures:
            builder.add_figure(entry["title"], entry["fig"])
        pdf_bytes = builder.generate_pdf()
        b64 = base64.b64encode(pdf_bytes).decode()
        ui.download(
            src=f"data:application/pdf;base64,{b64}",
            filename="homer_report.pdf",
        )
        ui.notify("PDF report downloaded!", type="positive")
    except Exception as ex:
        ui.notify(f"Report generation failed: {ex}", type="negative")


def download_filtered_csv():
    """Download filtered data as CSV."""
    if state.dataset is None:
        return
    filtered_df = apply_filters(state.dataset.df, state.filters)
    csv_str = filtered_df.to_csv(index=False)
    b64 = base64.b64encode(csv_str.encode()).decode()
    ui.download(
        src=f"data:text/csv;base64,{b64}",
        filename="homer_filtered_data.csv",
    )


# ── Plot generation ──────────────────────────────────────────────────────────

def generate_plot(
    plot_type, x_col, y_col, color_col, title,
    orientation, barmode, normalize, trendline, points, nbins, agg_func,
    plot_container,
):
    """Generate a plot based on current configuration."""
    if state.dataset is None:
        ui.notify("No data loaded", type="warning")
        return

    filtered_df = apply_filters(state.dataset.df, state.filters)
    plot_df = filtered_df.copy()

    color = color_col if color_col and color_col != "(None)" else None

    try:
        # Aggregate for bar charts
        if plot_type in ("Bar Chart", "Stacked Bar Chart") and y_col:
            group_cols = [x_col]
            if color:
                group_cols.append(color)
            plot_df = plot_df.groupby(group_cols, as_index=False)[y_col].agg(agg_func)

        fig = None

        if plot_type == "Bar Chart":
            fig = create_bar_chart(
                plot_df, x_col, y_col, color=color,
                orientation=orientation, barmode=barmode, title=title,
            )
        elif plot_type == "Stacked Bar Chart":
            if not color:
                ui.notify("Stacked bar requires Color/Group By", type="warning")
                return
            fig = create_stacked_bar_chart(
                plot_df, x_col, y_col, color=color,
                orientation=orientation, title=title, normalize=normalize,
            )
        elif plot_type == "Scatter Plot":
            fig = create_scatter_plot(
                plot_df, x_col, y_col, color=color, title=title, trendline=trendline,
            )
        elif plot_type == "Box Plot":
            x = x_col if x_col and x_col != "(None)" else None
            fig = create_box_plot(
                plot_df, x, y_col, color=color, title=title, points=points,
            )
        elif plot_type == "Violin Plot":
            x = x_col if x_col and x_col != "(None)" else None
            fig = create_violin_plot(
                plot_df, x, y_col, color=color, title=title,
            )
        elif plot_type == "Histogram":
            fig = create_histogram(
                plot_df, x_col, color=color, nbins=nbins, title=title,
            )
        elif plot_type == "XY Line Plot":
            fig = create_xy_line_plot(
                plot_df, x_col, y_col, color=color, title=title,
            )
        elif plot_type == "Heatmap":
            fig = create_heatmap(plot_df, x_col, y_col, y_col, title=title)

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
    """Render sidebar dataset information."""
    if state.dataset is None:
        return

    ds = state.dataset
    badge_class = "badge-object" if ds.data_type == "object" else "badge-summary"
    ui.html(f'<span class="data-badge {badge_class}">{ds.data_type.upper()} DATA</span>')
    ui.label(f"File: {ds.filename}").classes("text-sm text-gray-600")

    filtered_df = apply_filters(ds.df, state.filters)
    ui.label(f"Rows: {len(filtered_df):,} / {len(ds.df):,}").classes("text-sm")
    ui.label(f"Columns: {len(ds.df.columns)}").classes("text-sm")

    # Filters
    filterable = get_filterable_columns(ds)
    filterable = [c for c in filterable if ds.df[c].nunique() <= 100]

    if filterable:
        ui.separator()
        ui.label("Filters").classes("text-sm font-bold")

        for col in filterable[:6]:
            unique_vals = sorted(ds.df[col].dropna().unique().tolist(), key=str)
            str_vals = [str(v) for v in unique_vals]

            def make_filter_handler(column, vals_map):
                def handler(e):
                    if e.value:
                        state.filters[column] = [vals_map[v] for v in e.value]
                    else:
                        state.filters.pop(column, None)
                return handler

            vals_map = {str(v): v for v in unique_vals}
            ui.select(
                str_vals, multiple=True, label=col,
                on_change=make_filter_handler(col, vals_map),
            ).classes("w-full").props("dense")


@ui.refreshable
def main_content():
    """Render main content area."""
    if state.dataset is None:
        # Landing page
        with ui.card().classes("w-full max-w-3xl mx-auto mt-8 p-8"):
            ui.markdown("""
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
        return

    ds = state.dataset
    filtered_df = apply_filters(ds.df, state.filters)
    numeric_cols = get_plottable_numeric_columns(ds)
    grouping_cols = get_grouping_columns(ds)
    all_cols = list(filtered_df.columns)

    # Metrics row
    with ui.row().classes("w-full gap-4 mb-4"):
        for label, value in [
            ("Total Rows", f"{len(filtered_df):,}"),
            ("Data Type", ds.data_type.title()),
            ("Numeric Cols", str(len(ds.numeric_columns))),
            ("Categorical Cols", str(len(ds.categorical_columns))),
        ]:
            with ui.card().classes("flex-1"):
                ui.html(f"""
                <div class="metric-card">
                    <div class="value">{value}</div>
                    <div class="label">{label}</div>
                </div>
                """)

    # Tabs
    with ui.tabs().classes("w-full") as tabs:
        plot_tab = ui.tab("Plot Builder")
        table_tab = ui.tab("Data Table")
        stats_tab = ui.tab("Statistics")
        report_tab = ui.tab("Report")

    with ui.tab_panels(tabs, value=plot_tab).classes("w-full"):

        # ── Plot Builder Tab ─────────────────────────────────────────
        with ui.tab_panel(plot_tab):
            with ui.row().classes("w-full gap-4"):
                # Config column
                with ui.column().classes("w-80 shrink-0"):
                    with ui.card().classes("w-full p-4"):
                        ui.label("Plot Configuration").classes("text-lg font-bold mb-2")

                        plot_type_sel = ui.select(
                            PLOT_TYPES, label="Plot Type", value="Bar Chart",
                        ).classes("w-full")

                        x_options = grouping_cols + numeric_cols
                        y_options = numeric_cols
                        color_options = ["(None)"] + grouping_cols

                        x_sel = ui.select(x_options, label="X Axis", value=x_options[0] if x_options else None).classes("w-full")
                        y_sel = ui.select(y_options, label="Y Axis", value=y_options[0] if y_options else None).classes("w-full")
                        color_sel = ui.select(color_options, label="Color / Group By", value="(None)").classes("w-full")

                        title_input = ui.input("Chart Title", value="").classes("w-full")

                        # Additional options
                        with ui.expansion("Advanced Options", icon="settings").classes("w-full"):
                            orient_sel = ui.select(
                                ["v", "h"], label="Orientation",
                                value="v",
                            ).classes("w-full")
                            barmode_sel = ui.select(
                                ["group", "overlay"], label="Bar Mode",
                                value="group",
                            ).classes("w-full")
                            normalize_cb = ui.checkbox("Normalize to 100%", value=False)
                            trendline_sel = ui.select(
                                [None, "ols", "lowess"], label="Trendline",
                                value=None,
                            ).classes("w-full")
                            points_sel = ui.select(
                                ["outliers", "all", "suspectedoutliers"],
                                label="Show Points", value="outliers",
                            ).classes("w-full")
                            nbins_slider = ui.slider(min=10, max=200, value=50).props("label")
                            ui.label("Number of bins").classes("text-xs")
                            agg_sel = ui.select(
                                ["mean", "median", "sum", "count"],
                                label="Aggregation", value="mean",
                            ).classes("w-full")

                        plot_container = ui.column().classes("hidden")

                        def on_generate():
                            generate_plot(
                                plot_type_sel.value, x_sel.value, y_sel.value,
                                color_sel.value, title_input.value,
                                orient_sel.value, barmode_sel.value,
                                normalize_cb.value, trendline_sel.value,
                                points_sel.value, int(nbins_slider.value),
                                agg_sel.value, plot_display,
                            )

                        ui.button("Generate Plot", on_click=on_generate, color="primary").classes("w-full mt-2")

                        # Download buttons
                        with ui.row().classes("w-full gap-2 mt-2"):
                            ui.button("PNG", on_click=download_png, color="secondary").classes("flex-1").props("dense")
                            ui.button("SVG", on_click=download_svg, color="secondary").classes("flex-1").props("dense")
                            ui.button(
                                "Add to Report",
                                on_click=lambda: add_to_report(title_input.value),
                                color="accent",
                            ).classes("flex-1").props("dense")

                # Plot display column
                with ui.column().classes("flex-1 min-w-0"):
                    plot_display = ui.column().classes("w-full")
                    with plot_display:
                        ui.label("Configure and generate a plot to see it here.").classes("text-gray-400 text-center mt-8")

        # ── Data Table Tab ───────────────────────────────────────────
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
                    # Convert to list of dicts for ag_grid
                    columns = [{"headerName": col, "field": col, "sortable": True, "filter": True, "resizable": True}
                               for col in display_df.columns[:30]]
                    rows = display_df.head(max_r).to_dict("records")
                    ui.aggrid({
                        "columnDefs": columns,
                        "rowData": rows,
                        "defaultColDef": {"flex": 1, "minWidth": 100},
                    }).classes("w-full").style("height: 500px")

            ui.button("Refresh Table", on_click=refresh_table, color="primary")
            ui.button("Download CSV", on_click=download_filtered_csv, color="secondary")

            # Auto-load table
            refresh_table()

        # ── Statistics Tab ───────────────────────────────────────────
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

            if ds.marker_columns:
                ui.label("Detected Marker Columns:").classes("font-bold mt-4")
                ui.label(", ".join(ds.marker_columns[:20])).classes("text-sm text-gray-600")

        # ── Report Tab ───────────────────────────────────────────────
        with ui.tab_panel(report_tab):
            ui.label(f"Report contains {len(state.report_figures)} figure(s)").classes("text-lg")
            ui.label("Add figures using 'Add to Report' in the Plot Builder tab.").classes("text-sm text-gray-500")

            report_title_input = ui.input("Report Title", value="Homer Data Report").classes("w-96 mt-4")

            with ui.row().classes("gap-4 mt-4"):
                ui.button(
                    "Generate PDF Report",
                    on_click=lambda: download_report(report_title_input.value),
                    color="primary",
                ).props("" if state.report_figures else "disable")

                def clear_report():
                    state.report_figures.clear()
                    ui.notify("Report cleared", type="info")
                    main_content.refresh()

                ui.button("Clear Report", on_click=clear_report, color="secondary")

            if state.report_figures:
                ui.separator()
                ui.label("Figures in report:").classes("font-bold mt-2")
                for i, entry in enumerate(state.report_figures):
                    ui.label(f"  {i+1}. {entry['title']}").classes("text-sm")


# ── Main Page Layout ─────────────────────────────────────────────────────────

@ui.page("/")
def index():
    """Main page layout."""
    ui.html(HOMER_CSS)

    with ui.header().classes("bg-transparent shadow-none p-0"):
        ui.html("""
        <div class="homer-header">
            <h1>HOMER</h1>
            <p>Halo Output Mapper &amp; Explorer for Research</p>
        </div>
        """)

    with ui.left_drawer(value=True).classes("bg-gray-50 p-4"):
        ui.label("Data Upload").classes("text-lg font-bold mb-2")

        force_type_select = ui.select(
            ["Auto-detect", "Force Object", "Force Summary"],
            label="Data Type", value="Auto-detect",
        ).classes("w-full mb-2")

        ui.upload(
            label="Upload HALO data file",
            auto_upload=True,
            on_upload=lambda e: handle_upload(e, force_type_select),
        ).classes("w-full").props('accept=".csv,.tsv,.txt,.xlsx,.xls"')

        ui.separator()
        ui.label("Demo Data").classes("text-sm font-bold")
        with ui.row().classes("w-full gap-2"):
            ui.button("Object Demo", on_click=load_demo_object, color="primary").props("dense").classes("flex-1")
            ui.button("Summary Demo", on_click=load_demo_summary, color="primary").props("dense").classes("flex-1")

        ui.separator()
        sidebar_info()

    with ui.column().classes("w-full p-4"):
        main_content()

    with ui.footer().classes("bg-blue-800 text-white text-center p-2"):
        ui.label("Homer v1.0 | HALO Data Dashboard | Built with NiceGUI").classes("text-sm")


ui.run(title="Homer - Halo Data Dashboard", port=8080, reload=False)
