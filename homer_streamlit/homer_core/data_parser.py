# Homer - Data Parser for HALO by Indica Labs output files
# Handles CSV, TSV, and Excel files with auto-detection of object vs summary data

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class HaloDataset:
    """Container for parsed HALO data with metadata."""
    df: pd.DataFrame
    data_type: str  # "object" or "summary"
    filename: str
    numeric_columns: list = field(default_factory=list)
    categorical_columns: list = field(default_factory=list)
    marker_columns: list = field(default_factory=list)
    intensity_columns: list = field(default_factory=list)
    positivity_columns: list = field(default_factory=list)
    coordinate_columns: list = field(default_factory=list)
    area_columns: list = field(default_factory=list)
    percentage_columns: list = field(default_factory=list)
    count_columns: list = field(default_factory=list)
    id_columns: list = field(default_factory=list)
    annotation_columns: list = field(default_factory=list)

    @property
    def shape(self):
        return self.df.shape

    @property
    def columns(self):
        return list(self.df.columns)


# ── Column pattern definitions ──────────────────────────────────────────────

# Patterns that indicate object-level (cell-by-cell) data
OBJECT_INDICATOR_PATTERNS = [
    "cell id", "object id", "cell_id", "object_id",
    "xmin", "xmax", "ymin", "ymax",
    "x location", "y location",
    "cell area", "nucleus area", "cytoplasm area",
    "cell phenotype", "cell classification",
    "nucleus intensity", "cytoplasm intensity", "membrane intensity",
    "nucleus od", "cytoplasm od", "membrane od",
    "positive classification", "negative classification",
]

# Patterns that indicate summary-level (aggregate) data
SUMMARY_INDICATOR_PATTERNS = [
    "total cells", "total objects", "num cells", "num objects",
    "% positive", "percent positive", "percentage",
    "total area", "analyzed area", "annotation area",
    "density", "cells per",
    "positive cells", "negative cells",
    "h-score", "hscore",
    "average positive", "average negative",
]

# Column classification patterns
COORDINATE_PATTERNS = [
    "x location", "y location", "x_location", "y_location",
    "xmin", "xmax", "ymin", "ymax",
    "x centroid", "y centroid", "centroid x", "centroid y",
    "xlocation", "ylocation",
]

AREA_PATTERNS = [
    "cell area", "nucleus area", "cytoplasm area", "membrane area",
    "object area", "total area", "analyzed area", "annotation area",
    "cell_area", "nucleus_area", "cytoplasm_area",
]

INTENSITY_PATTERNS = [
    "intensity", "mean intensity", "median intensity",
    "min intensity", "max intensity", "std intensity",
    "optical density", " od",
]

POSITIVITY_PATTERNS = [
    "positive classification", "positive",
    "positivity", "is positive",
]

PERCENTAGE_PATTERNS = [
    "% ", "percent", "percentage", "fraction",
]

COUNT_PATTERNS = [
    "total cells", "total objects", "num cells", "num objects",
    "cell count", "object count", "count",
    "positive cells", "negative cells",
    "total positive", "total negative",
]

ID_PATTERNS = [
    "cell id", "object id", "cell_id", "object_id",
    "image id", "slide id", "sample id",
]

ANNOTATION_PATTERNS = [
    "image", "image location", "image name", "image file",
    "annotation", "annotation layer", "analysis region",
    "region", "layer", "slide", "sample",
    "classifier label",
]


def load_file(filepath: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Load a data file (CSV, TSV, or Excel) into a DataFrame."""
    path = Path(filepath)
    suffix = path.suffix.lower()

    if suffix in (".csv",):
        return pd.read_csv(filepath, low_memory=False)
    elif suffix in (".tsv", ".txt"):
        return pd.read_csv(filepath, sep="\t", low_memory=False)
    elif suffix in (".xls", ".xlsx", ".xlsm"):
        return pd.read_excel(filepath, sheet_name=sheet_name or 0)
    else:
        # Try CSV as default
        return pd.read_csv(filepath, low_memory=False)


def load_uploaded_file(file_obj, filename: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Load an uploaded file object into a DataFrame."""
    suffix = Path(filename).suffix.lower()

    if suffix in (".csv",):
        return pd.read_csv(file_obj, low_memory=False)
    elif suffix in (".tsv", ".txt"):
        return pd.read_csv(file_obj, sep="\t", low_memory=False)
    elif suffix in (".xls", ".xlsx", ".xlsm"):
        return pd.read_excel(file_obj, sheet_name=sheet_name or 0)
    else:
        return pd.read_csv(file_obj, low_memory=False)


def _match_patterns(column_name: str, patterns: list[str]) -> bool:
    """Check if a column name matches any of the given patterns (case-insensitive)."""
    col_lower = column_name.lower().strip()
    return any(p in col_lower for p in patterns)


def detect_data_type(df: pd.DataFrame) -> str:
    """Auto-detect whether the data is object-level or summary-level.

    Heuristics:
    - Object data has many rows (typically thousands+) with cell-level columns
    - Summary data has fewer rows with aggregate statistics columns
    - Column name patterns are the primary signal
    """
    columns_lower = [c.lower().strip() for c in df.columns]

    object_score = sum(
        1 for col in columns_lower
        for pattern in OBJECT_INDICATOR_PATTERNS
        if pattern in col
    )
    summary_score = sum(
        1 for col in columns_lower
        for pattern in SUMMARY_INDICATOR_PATTERNS
        if pattern in col
    )

    # Row count as secondary signal
    if len(df) > 5000:
        object_score += 3
    elif len(df) < 100:
        summary_score += 2

    # If there are "Cell ID" or coordinate columns, strong object signal
    if any("cell id" in c or "object id" in c for c in columns_lower):
        object_score += 5
    if any("xmin" in c or "x location" in c for c in columns_lower):
        object_score += 3

    # If there are percentage or "total" columns, strong summary signal
    if any("% " in c or "total cells" in c for c in columns_lower):
        summary_score += 5

    if object_score > summary_score:
        return "object"
    elif summary_score > object_score:
        return "summary"
    else:
        # Default: if many rows, assume object; otherwise summary
        return "object" if len(df) > 500 else "summary"


def classify_columns(df: pd.DataFrame, data_type: str) -> dict:
    """Classify each column into semantic categories."""
    result = {
        "numeric": [],
        "categorical": [],
        "marker": [],
        "intensity": [],
        "positivity": [],
        "coordinate": [],
        "area": [],
        "percentage": [],
        "count": [],
        "id": [],
        "annotation": [],
    }

    for col in df.columns:
        col_lower = col.lower().strip()
        is_numeric = pd.api.types.is_numeric_dtype(df[col])

        # Classify by pattern
        if _match_patterns(col, ID_PATTERNS):
            result["id"].append(col)
        elif _match_patterns(col, COORDINATE_PATTERNS):
            result["coordinate"].append(col)
        elif _match_patterns(col, AREA_PATTERNS):
            result["area"].append(col)
        elif _match_patterns(col, INTENSITY_PATTERNS):
            result["intensity"].append(col)
            result["marker"].append(col)
        elif _match_patterns(col, POSITIVITY_PATTERNS):
            result["positivity"].append(col)
            result["marker"].append(col)
        elif _match_patterns(col, PERCENTAGE_PATTERNS):
            result["percentage"].append(col)
        elif _match_patterns(col, COUNT_PATTERNS):
            result["count"].append(col)
        elif _match_patterns(col, ANNOTATION_PATTERNS):
            result["annotation"].append(col)

        # Also add to numeric/categorical
        if is_numeric:
            result["numeric"].append(col)
        else:
            result["categorical"].append(col)

    return result


def parse_halo_data(
    df: pd.DataFrame,
    filename: str = "unknown",
    force_type: Optional[str] = None,
) -> HaloDataset:
    """Parse a DataFrame as HALO data, auto-detecting type and classifying columns."""
    data_type = force_type if force_type else detect_data_type(df)
    classified = classify_columns(df, data_type)

    return HaloDataset(
        df=df,
        data_type=data_type,
        filename=filename,
        numeric_columns=classified["numeric"],
        categorical_columns=classified["categorical"],
        marker_columns=classified["marker"],
        intensity_columns=classified["intensity"],
        positivity_columns=classified["positivity"],
        coordinate_columns=classified["coordinate"],
        area_columns=classified["area"],
        percentage_columns=classified["percentage"],
        count_columns=classified["count"],
        id_columns=classified["id"],
        annotation_columns=classified["annotation"],
    )


def get_filterable_columns(dataset: HaloDataset) -> list[str]:
    """Return columns suitable for filtering (categorical + annotation)."""
    return dataset.categorical_columns + dataset.annotation_columns


def get_plottable_numeric_columns(dataset: HaloDataset) -> list[str]:
    """Return numeric columns suitable for plotting (exclude IDs)."""
    id_set = set(dataset.id_columns)
    return [c for c in dataset.numeric_columns if c not in id_set]


def get_grouping_columns(dataset: HaloDataset) -> list[str]:
    """Return columns suitable for grouping/coloring plots."""
    candidates = dataset.categorical_columns + dataset.annotation_columns
    # Deduplicate while preserving order
    seen = set()
    result = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply column-value filters to a DataFrame.

    filters: {column_name: [list of selected values]}
    """
    filtered = df.copy()
    for col, values in filters.items():
        if col in filtered.columns and values:
            filtered = filtered[filtered[col].isin(values)]
    return filtered


def get_column_summary(df: pd.DataFrame, column: str) -> dict:
    """Get summary statistics for a column."""
    if pd.api.types.is_numeric_dtype(df[column]):
        return {
            "type": "numeric",
            "count": int(df[column].count()),
            "mean": float(df[column].mean()),
            "std": float(df[column].std()),
            "min": float(df[column].min()),
            "max": float(df[column].max()),
            "median": float(df[column].median()),
            "null_count": int(df[column].isnull().sum()),
        }
    else:
        return {
            "type": "categorical",
            "count": int(df[column].count()),
            "unique": int(df[column].nunique()),
            "top": str(df[column].mode().iloc[0]) if not df[column].mode().empty else "N/A",
            "null_count": int(df[column].isnull().sum()),
            "value_counts": df[column].value_counts().head(20).to_dict(),
        }
