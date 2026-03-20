# Homer - Data Parser for HALO by Indica Labs output files
# Aligned with anima/HaloAnalysis column patterns and data workflows
# Handles CSV, TSV, and Excel files with auto-detection of object vs summary data

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass, field
from scipy.stats.mstats import winsorize as scipy_winsorize


@dataclass
class HaloDataset:
    """Container for parsed HALO data with metadata.
    Column classification mirrors HaloAnalysis.set_analysis_metrics()."""
    df: pd.DataFrame
    data_type: str  # "object", "summary", or "cluster"
    filename: str
    # Core column groups (matching anima naming)
    numeric_columns: list = field(default_factory=list)
    categorical_columns: list = field(default_factory=list)
    # HALO-specific column groups
    intensity_columns: list = field(default_factory=list)      # H-Score, Intensity
    spatial_columns: list = field(default_factory=list)         # non-cell numeric (area, coords, etc.)
    cell_columns: list = field(default_factory=list)            # columns with "Cells"
    total_columns: list = field(default_factory=list)           # cell columns without "%"
    fraction_columns: list = field(default_factory=list)        # cell columns with "%"
    channel_total_columns: list = field(default_factory=list)   # totals with Spectrum/Cy5
    channel_fraction_columns: list = field(default_factory=list)  # fractions with Spectrum/Cy5
    phenotype_total_columns: list = field(default_factory=list)   # totals without Spectrum/Cy5
    phenotype_fraction_columns: list = field(default_factory=list)  # fractions without Spectrum/Cy5
    # Additional semantic groups
    coordinate_columns: list = field(default_factory=list)
    area_columns: list = field(default_factory=list)
    positivity_columns: list = field(default_factory=list)
    marker_columns: list = field(default_factory=list)
    id_columns: list = field(default_factory=list)
    annotation_columns: list = field(default_factory=list)
    # HALO metadata
    algorithm_names: list = field(default_factory=list)
    sample_ids: list = field(default_factory=list)
    analysis_regions: list = field(default_factory=list)

    @property
    def shape(self):
        return self.df.shape

    @property
    def columns(self):
        return list(self.df.columns)


# ── Column pattern definitions (derived from anima HaloAnalysis) ─────────────

# Patterns indicating object-level data (cell-by-cell / cluster exports)
OBJECT_INDICATOR_PATTERNS = [
    "cell id", "object id", "cell_id", "object_id",
    "xmin", "xmax", "ymin", "ymax",
    "x location", "y location",
    "cell area", "nucleus area", "cytoplasm area",
    "cell phenotype", "cell classification",
    "nucleus intensity", "cytoplasm intensity", "membrane intensity",
    "nucleus od", "cytoplasm od", "membrane od",
    "positive classification", "negative classification",
    "classifier label",
]

# Patterns indicating summary-level data (HALO analysis output)
SUMMARY_INDICATOR_PATTERNS = [
    "algorithm name", "job id", "image tag",
    "total cells", "total objects", "num cells",
    "analysis region",
    "% positive", "percent positive",
    "total area", "analyzed area", "annotation area",
    "region area",
    "density", "cells per",
    "positive cells", "negative cells",
    "h-score", "hscore",
    "weak", "strong", "moderate",
]

# Patterns for cluster-level data (aggregated object data)
CLUSTER_INDICATOR_PATTERNS = [
    "total cluster count", "total cell count", "total area analyzed",
    "region area", "cluster",
]

# ── Semantic column classification patterns ──────────────────────────────────

COORDINATE_PATTERNS = [
    "x location", "y location", "x_location", "y_location",
    "xmin", "xmax", "ymin", "ymax",
    "x centroid", "y centroid", "centroid x", "centroid y",
    "xlocation", "ylocation",
]

AREA_PATTERNS = [
    "cell area", "nucleus area", "cytoplasm area", "membrane area",
    "object area", "total area", "analyzed area", "annotation area",
    "region area", "total area analyzed",
]

INTENSITY_PATTERNS = [
    "h-score", "hscore", "intensity",
    "mean intensity", "median intensity",
    "min intensity", "max intensity", "std intensity",
    "optical density", " od",
]

POSITIVITY_PATTERNS = [
    "positive classification", "positive",
    "positivity", "is positive",
    "negative classification",
]

ID_PATTERNS = [
    "cell id", "object id", "cell_id", "object_id",
    "job id", "job_id",
]

ANNOTATION_PATTERNS = [
    "image tag", "image location", "image name", "image file",
    "annotation", "annotation layer", "analysis region",
    "algorithm name", "sample id",
    "classifier label",
]


# ── File I/O ─────────────────────────────────────────────────────────────────

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


# ── Pattern matching ─────────────────────────────────────────────────────────

def _match_patterns(column_name: str, patterns: list[str]) -> bool:
    """Check if a column name matches any of the given patterns (case-insensitive)."""
    col_lower = column_name.lower().strip()
    return any(p in col_lower for p in patterns)


# ── Data type detection ──────────────────────────────────────────────────────

def detect_data_type(df: pd.DataFrame) -> str:
    """Auto-detect whether the data is object-level, summary-level, or cluster-level.

    Uses column name heuristics aligned with HALO export conventions and
    the anima HaloAnalysis workflows.
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
    cluster_score = sum(
        1 for col in columns_lower
        for pattern in CLUSTER_INDICATOR_PATTERNS
        if pattern in col
    )

    # Row count as secondary signal
    if len(df) > 5000:
        object_score += 3
    elif len(df) < 100:
        summary_score += 2

    # Strong signals
    if any("cell id" in c or "object id" in c for c in columns_lower):
        object_score += 5
    if any("xmin" in c or "x location" in c for c in columns_lower):
        object_score += 3

    # HALO summary file signature: "Algorithm Name" + "Job Id" + "Image Tag"
    if any("algorithm name" in c for c in columns_lower):
        summary_score += 4
    if any("job id" in c for c in columns_lower):
        summary_score += 3
    if any("image tag" in c for c in columns_lower):
        summary_score += 3

    # Cluster data: aggregated object data with cluster columns
    if any("total cluster count" in c for c in columns_lower):
        cluster_score += 5

    # Percentage columns with "%" in cell-related headers → summary
    pct_cell_cols = [c for c in columns_lower if "%" in c and "cell" in c]
    if len(pct_cell_cols) > 3:
        summary_score += 4

    scores = {"object": object_score, "summary": summary_score, "cluster": cluster_score}
    best = max(scores, key=scores.get)

    if scores[best] == 0:
        return "object" if len(df) > 500 else "summary"
    return best


# ── Column classification (mirrors anima set_analysis_metrics) ───────────────

def classify_columns(df: pd.DataFrame, data_type: str) -> dict:
    """Classify columns into semantic categories.

    Mirrors the logic in HaloAnalysis.set_analysis_metrics() from the anima module,
    with additional categories for dashboard use.
    """
    numeric_data = [h for h in df.select_dtypes(include=np.number).columns]
    categoric_data = [h for h in df.select_dtypes(include="object").columns]

    # ── anima-style classification ───────────────────────────────────────
    # Intensity: H-Score or Intensity
    intensity_data = [h for h in numeric_data if "H-Score" in h or "Intensity" in h]

    # Spatial: numeric columns that aren't Cells/Nuclei/Cytoplasms/H-Score/Intensity
    spatial_data = [
        h for h in numeric_data
        if "Cells" not in h and "Nuclei" not in h and "Cytoplasms" not in h
        and "H-Score" not in h and "Intensity" not in h
    ]

    # Cell-related columns
    cell_data = [h for h in numeric_data if "Cells" in h]
    total_data = [h for h in cell_data if "%" not in h]
    fraction_data = [h for h in cell_data if "%" in h]

    # Channel-specific (Spectrum/Cy5)
    channel_total_data = [h for h in total_data if "Spectrum" in h or "Cy5" in h]
    channel_fraction_data = [h for h in fraction_data if "Spectrum" in h or "Cy5" in h]

    # Phenotype-specific (not Spectrum/Cy5)
    phenotype_total_data = [h for h in total_data if "Spectrum" not in h and "Cy5" not in h]
    phenotype_fraction_data = [h for h in fraction_data if "Spectrum" not in h and "Cy5" not in h]

    # ── Additional semantic groups for dashboard ─────────────────────────
    coordinate_cols = [c for c in df.columns if _match_patterns(c, COORDINATE_PATTERNS)]
    area_cols = [c for c in df.columns if _match_patterns(c, AREA_PATTERNS)]
    positivity_cols = [c for c in df.columns if _match_patterns(c, POSITIVITY_PATTERNS)]
    id_cols = [c for c in df.columns if _match_patterns(c, ID_PATTERNS)]
    annotation_cols = [c for c in df.columns if _match_patterns(c, ANNOTATION_PATTERNS)]

    # Marker columns: intensity + positivity + phenotype-related
    marker_cols = list(dict.fromkeys(intensity_data + positivity_cols + channel_total_data + channel_fraction_data))

    return {
        "numeric": numeric_data,
        "categorical": categoric_data,
        "intensity": intensity_data,
        "spatial": spatial_data,
        "cell": cell_data,
        "total": total_data,
        "fraction": fraction_data,
        "channel_total": channel_total_data,
        "channel_fraction": channel_fraction_data,
        "phenotype_total": phenotype_total_data,
        "phenotype_fraction": phenotype_fraction_data,
        "coordinate": coordinate_cols,
        "area": area_cols,
        "positivity": positivity_cols,
        "marker": marker_cols,
        "id": id_cols,
        "annotation": annotation_cols,
    }


# ── HALO-specific preprocessing ─────────────────────────────────────────────

def preprocess_halo_summary(df: pd.DataFrame, max_job: bool = False) -> pd.DataFrame:
    """Preprocess a HALO summary file (same as HaloAnalysis.parse_halo_analysis).

    - Creates 'Sample ID' from 'Image Tag' (strips .scn)
    - Optionally keeps only the latest Job Id per Sample/Algorithm
    """
    if "Image Tag" in df.columns:
        df = df.copy()
        df["Sample ID"] = df["Image Tag"].str.replace(".scn", "", regex=False)

    if max_job and "Job Id" in df.columns and "Sample ID" in df.columns and "Algorithm Name" in df.columns:
        df = (
            df.sort_values("Job Id")
            .groupby(["Sample ID", "Algorithm Name"])
            .tail(1)
            .reset_index(drop=True)
        )

    return df


def dezero(df: pd.DataFrame, metric: str = "Total Cells") -> pd.DataFrame:
    """Remove rows with zero values in the given metric.
    Clusters with 0 cells are noise (mirrors HaloMunger.dezero)."""
    if metric in df.columns:
        return df[df[metric] > 0].copy()
    return df.copy()


# ── Outlier removal (mirrors ClusterCleaner) ─────────────────────────────────

def remove_outliers_iqr(
    df: pd.DataFrame,
    metric: str,
    factor: float = 1.5,
) -> Tuple[pd.DataFrame, pd.DataFrame, float, float]:
    """Remove outliers using IQR method. Returns (cleaned, removed, lower, upper)."""
    values = df[metric].dropna()
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    cleaned = df[(df[metric] >= lower) & (df[metric] <= upper)]
    removed = df[(df[metric] < lower) | (df[metric] > upper)]
    return cleaned, removed, float(lower), float(upper)


def remove_outliers_percentile(
    df: pd.DataFrame,
    metric: str,
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, float, float]:
    """Remove outliers using percentile bounds. Returns (cleaned, removed, lower, upper)."""
    values = df[metric].dropna()
    lower, upper = np.percentile(values, [lower_pct, upper_pct])
    cleaned = df[(df[metric] >= lower) & (df[metric] <= upper)]
    removed = df[(df[metric] < lower) | (df[metric] > upper)]
    return cleaned, removed, float(lower), float(upper)


def remove_outliers_std(
    df: pd.DataFrame,
    metric: str,
    std_factor: float = 2.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, float, float]:
    """Remove outliers using standard deviation bounds. Returns (cleaned, removed, lower, upper)."""
    values = df[metric].dropna()
    mean = values.mean()
    std = values.std()
    lower = mean - std_factor * std
    upper = mean + std_factor * std
    cleaned = df[(df[metric] >= lower) & (df[metric] <= upper)]
    removed = df[(df[metric] < lower) | (df[metric] > upper)]
    return cleaned, removed, float(lower), float(upper)


def winsorize_column(
    df: pd.DataFrame,
    metric: str,
    limits: Tuple[float, float] = (0.01, 0.01),
) -> pd.DataFrame:
    """Winsorize a column in-place. Returns the modified DataFrame."""
    df = df.copy()
    values = df[metric].dropna().values
    winsorized = scipy_winsorize(values, limits=limits)
    df.loc[df[metric].notna(), metric] = winsorized
    return df


def remove_outliers(
    df: pd.DataFrame,
    metric: str,
    method: str = "iqr",
    factor: float = 1.5,
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
    std_factor: float = 2.0,
    limits: Tuple[float, float] = (0.01, 0.01),
) -> Tuple[pd.DataFrame, pd.DataFrame, float, float]:
    """Remove outliers using the specified method.

    Methods: 'iqr', 'percentile', 'std', 'winsorize'
    Returns (cleaned_df, removed_df, lower_bound, upper_bound).
    For winsorize, returns the winsorized df with empty removed and 0 bounds.
    """
    if method == "iqr":
        return remove_outliers_iqr(df, metric, factor)
    elif method == "percentile":
        return remove_outliers_percentile(df, metric, lower_pct, upper_pct)
    elif method == "std":
        return remove_outliers_std(df, metric, std_factor)
    elif method == "winsorize":
        winsorized = winsorize_column(df, metric, limits)
        return winsorized, pd.DataFrame(columns=df.columns), 0.0, 0.0
    else:
        raise ValueError(f"Unknown outlier method: {method}")


# ── Main parse function ──────────────────────────────────────────────────────

def parse_halo_data(
    df: pd.DataFrame,
    filename: str = "unknown",
    force_type: Optional[str] = None,
    max_job: bool = False,
    analysis_area: Optional[str] = None,
) -> HaloDataset:
    """Parse a DataFrame as HALO data, auto-detecting type and classifying columns.

    Performs HALO-specific preprocessing:
    - Creates Sample ID from Image Tag
    - Optionally filters to latest Job Id
    - Optionally filters to specific Analysis Region
    """
    # Preprocess HALO summary-style data
    df = preprocess_halo_summary(df, max_job=max_job)

    # Filter by analysis area if specified
    if analysis_area and "Analysis Region" in df.columns:
        df = df[df["Analysis Region"] == analysis_area].copy()

    data_type = force_type if force_type else detect_data_type(df)
    classified = classify_columns(df, data_type)

    # Extract HALO metadata
    algorithm_names = df["Algorithm Name"].unique().tolist() if "Algorithm Name" in df.columns else []
    sample_ids = df["Sample ID"].unique().tolist() if "Sample ID" in df.columns else []
    analysis_regions = df["Analysis Region"].unique().tolist() if "Analysis Region" in df.columns else []

    return HaloDataset(
        df=df,
        data_type=data_type,
        filename=filename,
        numeric_columns=classified["numeric"],
        categorical_columns=classified["categorical"],
        intensity_columns=classified["intensity"],
        spatial_columns=classified["spatial"],
        cell_columns=classified["cell"],
        total_columns=classified["total"],
        fraction_columns=classified["fraction"],
        channel_total_columns=classified["channel_total"],
        channel_fraction_columns=classified["channel_fraction"],
        phenotype_total_columns=classified["phenotype_total"],
        phenotype_fraction_columns=classified["phenotype_fraction"],
        coordinate_columns=classified["coordinate"],
        area_columns=classified["area"],
        positivity_columns=classified["positivity"],
        marker_columns=classified["marker"],
        id_columns=classified["id"],
        annotation_columns=classified["annotation"],
        algorithm_names=algorithm_names,
        sample_ids=sample_ids,
        analysis_regions=analysis_regions,
    )


# ── Accessor helpers ─────────────────────────────────────────────────────────

def get_filterable_columns(dataset: HaloDataset) -> list[str]:
    """Return columns suitable for filtering (categorical + annotation)."""
    seen = set()
    result = []
    for c in dataset.categorical_columns + dataset.annotation_columns:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


def get_plottable_numeric_columns(dataset: HaloDataset) -> list[str]:
    """Return numeric columns suitable for plotting (exclude IDs)."""
    id_set = set(dataset.id_columns)
    return [c for c in dataset.numeric_columns if c not in id_set]


def get_grouping_columns(dataset: HaloDataset) -> list[str]:
    """Return columns suitable for grouping/coloring plots."""
    candidates = dataset.categorical_columns + dataset.annotation_columns
    seen = set()
    result = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


def get_phenotype_columns(dataset: HaloDataset, include_weak_strong: bool = False) -> list[str]:
    """Return phenotype fraction columns, optionally filtering out Weak/Strong/Moderate/Negative.
    Mirrors the broad_describe filtering from anima."""
    cols = dataset.phenotype_fraction_columns
    if not include_weak_strong:
        cols = [c for c in cols
                if "Weak" not in c and "Strong" not in c
                and "Moderate" not in c and "Negative" not in c]
    return cols


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
            "mean": float(df[column].mean()) if not df[column].empty else 0.0,
            "std": float(df[column].std()) if not df[column].empty else 0.0,
            "min": float(df[column].min()) if not df[column].empty else 0.0,
            "max": float(df[column].max()) if not df[column].empty else 0.0,
            "median": float(df[column].median()) if not df[column].empty else 0.0,
            "null_count": int(df[column].isnull().sum()),
        }
    else:
        mode = df[column].mode()
        return {
            "type": "categorical",
            "count": int(df[column].count()),
            "unique": int(df[column].nunique()),
            "top": str(mode.iloc[0]) if not mode.empty else "N/A",
            "null_count": int(df[column].isnull().sum()),
            "value_counts": df[column].value_counts().head(20).to_dict(),
        }
