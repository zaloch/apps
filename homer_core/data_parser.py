# Homer - Data Parser for HALO by Indica Labs output files
# Aligned with anima/HaloAnalysis column patterns and real HALO object/summary exports
# Handles CSV, TSV, and Excel files with auto-detection of object vs summary data
# Supports large files (100MB - 3GB) via chunked reading and intelligent sampling

import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass, field
from scipy.stats.mstats import winsorize as scipy_winsorize
import logging

try:
    import dask.dataframe as dd
    HAS_DASK = True
except ImportError:
    HAS_DASK = False

logger = logging.getLogger(__name__)


@dataclass
class HaloDataset:
    """Container for parsed HALO data with metadata.
    Column classification mirrors HaloAnalysis.set_analysis_metrics() and
    the real HALO object export format (Image Location, Object Id, per-channel metrics)."""
    df: pd.DataFrame
    data_type: str  # "object", "summary", or "cluster"
    filename: str
    # Core column groups (matching anima naming)
    numeric_columns: list = field(default_factory=list)
    categorical_columns: list = field(default_factory=list)
    # HALO-specific column groups (summary data)
    intensity_columns: list = field(default_factory=list)      # H-Score, *Intensity, *Cell Intensity
    spatial_columns: list = field(default_factory=list)         # non-cell numeric (area, coords, etc.)
    cell_columns: list = field(default_factory=list)            # columns with "Cells"
    total_columns: list = field(default_factory=list)           # cell columns without "%"
    fraction_columns: list = field(default_factory=list)        # cell columns with "%"
    channel_total_columns: list = field(default_factory=list)   # totals with Spectrum/Cy5/5-FAM/Rhodamine
    channel_fraction_columns: list = field(default_factory=list)
    phenotype_total_columns: list = field(default_factory=list)
    phenotype_fraction_columns: list = field(default_factory=list)
    # Object-data specific column groups
    phenotype_combo_columns: list = field(default_factory=list)   # DAPI+ C1+ C2+ ... binary columns
    classification_columns: list = field(default_factory=list)    # *Positive Classification, *Positive Nucleus Classification
    completeness_columns: list = field(default_factory=list)      # *% Nucleus Completeness
    nucleus_intensity_columns: list = field(default_factory=list) # *Nucleus Intensity
    cell_intensity_columns: list = field(default_factory=list)    # *Cell Intensity
    morphology_columns: list = field(default_factory=list)        # Cell/Nucleus/Cytoplasm Area, Perimeter, Roundness
    # Additional semantic groups
    coordinate_columns: list = field(default_factory=list)
    area_columns: list = field(default_factory=list)
    positivity_columns: list = field(default_factory=list)
    marker_columns: list = field(default_factory=list)
    id_columns: list = field(default_factory=list)
    annotation_columns: list = field(default_factory=list)
    # Detected fluorophore channels
    fluorophore_channels: list = field(default_factory=list)  # e.g. ["DAPI", "Cy5", "5-FAM", "Spectrum Aqua", "Rhodamine 6G"]
    # HALO metadata
    algorithm_names: list = field(default_factory=list)
    sample_ids: list = field(default_factory=list)
    analysis_regions: list = field(default_factory=list)
    # Large file metadata
    is_sampled: bool = False           # True if data was sampled for performance
    total_rows: int = 0                # Original row count before sampling
    file_size_mb: float = 0.0          # File size in MB

    @property
    def shape(self):
        return self.df.shape

    @property
    def columns(self):
        return list(self.df.columns)


# ── Known HALO fluorophore/channel names ─────────────────────────────────────

KNOWN_FLUOROPHORES = [
    "DAPI", "Cy5", "5-FAM", "Spectrum Aqua", "Spectrum Orange",
    "Rhodamine 6G", "Qdot 570", "Qdot 605", "Qdot 625", "Qdot 655",
    "Qdot 705", "FITC", "Texas Red", "Cy3", "Cy7",
    "Alexa 488", "Alexa 555", "Alexa 594", "Alexa 647",
    "Opal 480", "Opal 520", "Opal 540", "Opal 570",
    "Opal 620", "Opal 650", "Opal 690", "Opal 780",
]

# ── Column pattern definitions ───────────────────────────────────────────────

# Patterns indicating object-level data (per-object exports from HALO)
OBJECT_INDICATOR_PATTERNS = [
    "object id", "cell id",
    "image location",
    "xmin", "xmax", "ymin", "ymax",
    "x location", "y location",
    "cell area", "nucleus area", "cytoplasm area",
    "nucleus perimeter", "nucleus roundness",
    "cell phenotype", "cell classification",
    "nucleus intensity", "cell intensity",
    "nucleus completeness",
    "positive classification", "positive nucleus classification",
    "nucleus od", "cytoplasm od", "membrane od",
    "classifier label",
]

# Patterns indicating summary-level data (HALO analysis output)
SUMMARY_INDICATOR_PATTERNS = [
    "image tag", "job id",
    "total cells", "total objects", "num cells",
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
    "cluster",
]

# ── Semantic column classification patterns ──────────────────────────────────

COORDINATE_PATTERNS = [
    "x location", "y location", "x_location", "y_location",
    "xmin", "xmax", "ymin", "ymax",
    "x centroid", "y centroid", "centroid x", "centroid y",
]

AREA_PATTERNS = [
    "cell area", "nucleus area", "cytoplasm area", "membrane area",
    "object area", "total area", "analyzed area", "annotation area",
    "region area", "total area analyzed",
]

MORPHOLOGY_PATTERNS = [
    "cell area", "nucleus area", "cytoplasm area",
    "nucleus perimeter", "nucleus roundness",
    "nucleus circularity", "nucleus solidity",
    "cell perimeter", "cell roundness",
    "membrane area",
]

INTENSITY_PATTERNS = [
    "h-score", "hscore",
    "nucleus intensity", "cell intensity",
    "cytoplasm intensity", "membrane intensity",
    "mean intensity", "median intensity",
    "optical density", " od",
]

CLASSIFICATION_PATTERNS = [
    "positive classification",
    "positive nucleus classification",
]

COMPLETENESS_PATTERNS = [
    "% nucleus completeness",
    "nucleus completeness",
]

ID_PATTERNS = [
    "object id", "cell id", "object_id", "cell_id",
    "job id", "job_id",
]

ANNOTATION_PATTERNS = [
    "image tag", "image location", "image name", "image file",
    "annotation", "annotation layer", "analysis region",
    "algorithm name", "sample id",
    "classifier label",
]


# ── Phenotype combination column detection ───────────────────────────────────

# Pattern: "DAPI+ C1+ C2+ C3+ C4+" or "DAPI+ C1- C2+ C3- C4-" etc.
PHENOTYPE_COMBO_PATTERN = re.compile(
    r"^(DAPI[+-])\s+"                          # starts with DAPI+/DAPI-
    r"(C\d+[+-]\s*)+$"                         # followed by C1+/- C2+/- etc.
    r"|^DAPI[+-]$"                             # or just DAPI+ / DAPI-
    r"|^(DAPI[+-]\s+)?(C\d+[+-]\s*){2,}$",    # or multiple channel combos
    re.IGNORECASE,
)

def _is_phenotype_combo(col_name: str) -> bool:
    """Check if a column name is a phenotype combination (e.g. 'DAPI+ C1+ C2- C3+ C4-')."""
    name = col_name.strip()
    # Quick checks before regex
    if not any(marker in name for marker in ["DAPI", "C1", "C2", "C3", "C4", "C5"]):
        return False
    # Check for pattern: contains +/- and channel markers
    parts = name.split()
    if len(parts) < 1:
        return False
    # All parts should be marker+/- format
    return all(re.match(r"^(DAPI|C\d+)[+-]$", p, re.IGNORECASE) for p in parts)


def _detect_fluorophores(columns: list[str]) -> list[str]:
    """Detect which fluorophore channels are present in the column headers."""
    found = []
    col_text = " ".join(columns).lower()
    for fluor in KNOWN_FLUOROPHORES:
        if fluor.lower() in col_text:
            found.append(fluor)
    return found


# ── Large file thresholds ────────────────────────────────────────────────────

# Files larger than this (in MB) trigger chunked loading + memory optimization
LARGE_FILE_THRESHOLD_MB = 50
# Max rows to keep in memory for interactive work; larger files get sampled
MAX_INTERACTIVE_ROWS = 500_000
# Chunk size for reading large CSVs (rows per chunk)
CSV_CHUNK_SIZE = 100_000
# Columns that can be safely downcast to save memory
DOWNCAST_INT_COLS = True
DOWNCAST_FLOAT_COLS = True


# ── File I/O ─────────────────────────────────────────────────────────────────

def _get_file_size_mb(filepath: str) -> float:
    """Get file size in MB."""
    try:
        return Path(filepath).stat().st_size / (1024 * 1024)
    except (OSError, ValueError):
        return 0.0


def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to reduce memory usage.
    Typically saves 40-60% memory on large HALO exports."""
    for col in df.select_dtypes(include=["int64"]).columns:
        col_min, col_max = df[col].min(), df[col].max()
        if col_min >= 0 and col_max <= 255:
            df[col] = df[col].astype(np.uint8)
        elif col_min >= -128 and col_max <= 127:
            df[col] = df[col].astype(np.int8)
        elif col_min >= 0 and col_max <= 65535:
            df[col] = df[col].astype(np.uint16)
        elif col_min >= -32768 and col_max <= 32767:
            df[col] = df[col].astype(np.int16)
        elif col_min >= 0 and col_max <= 4294967295:
            df[col] = df[col].astype(np.uint32)
        else:
            df[col] = df[col].astype(np.int32)

    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    # Convert low-cardinality string columns to categories
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].nunique() < 100:
            df[col] = df[col].astype("category")

    return df


def _read_csv_dask(
    filepath_or_buffer,
    sep: str = ",",
    max_rows: Optional[int] = None,
) -> Tuple[pd.DataFrame, int]:
    """Read a CSV using dask for parallel processing.

    Only works with file paths (strings/Path objects), not file-like objects.
    Falls back to _read_csv_chunked_pandas for file-like objects.

    Returns (DataFrame, total_row_count).
    """
    # Dask cannot read file-like objects; fall back for those
    if not isinstance(filepath_or_buffer, (str, Path)):
        return _read_csv_chunked_pandas(filepath_or_buffer, sep=sep, max_rows=max_rows)

    filepath_str = str(filepath_or_buffer)
    logger.info(f"Reading CSV with dask (parallel): {filepath_str}")

    ddf = dd.read_csv(filepath_str, sep=sep, blocksize="64MB")
    total_rows = len(ddf)  # triggers a compute for row count

    if max_rows is not None and total_rows > max_rows:
        # Take the first max_rows rows
        df = ddf.head(max_rows, npartitions=-1, compute=True)
    else:
        df = ddf.compute()

    return df, total_rows


def dask_aggregate(
    filepath: str,
    sep: str = ",",
    group_cols: Optional[list] = None,
    agg_dict: Optional[dict] = None,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Perform parallel groupby aggregation on a large CSV using dask.

    Reads the file with dask, applies groupby().agg() lazily, then computes
    the result to a pandas DataFrame. This is useful when you want to aggregate
    a huge file without loading it all into memory.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    sep : str
        Column separator (default ',').
    group_cols : list
        Columns to group by.
    agg_dict : dict
        Aggregation specification, e.g. {'Total Cells': 'sum', 'H-Score': 'mean'}.
    max_rows : int, optional
        If set, only read up to this many rows before aggregating.

    Returns
    -------
    pd.DataFrame
        Aggregated result as a pandas DataFrame.
    """
    if not HAS_DASK:
        raise RuntimeError(
            "dask is required for dask_aggregate but is not installed. "
            "Install it with: pip install 'dask[dataframe]'"
        )
    if group_cols is None or agg_dict is None:
        raise ValueError("group_cols and agg_dict are required for dask_aggregate")

    ddf = dd.read_csv(str(filepath), sep=sep, blocksize="64MB")

    if max_rows is not None:
        ddf = ddf.head(max_rows, npartitions=-1, compute=False)
        # head returns a pandas DF when compute=True; wrap back into dask
        if isinstance(ddf, pd.DataFrame):
            ddf = dd.from_pandas(ddf, npartitions=1)

    result = ddf.groupby(group_cols).agg(agg_dict)
    return result.compute().reset_index()


def _read_csv_chunked_pandas(
    filepath_or_buffer,
    sep: str = ",",
    max_rows: Optional[int] = None,
    sample_frac: Optional[float] = None,
) -> Tuple[pd.DataFrame, int]:
    """Read a CSV in chunks using pandas, optionally sampling or capping rows.

    This is the pure-pandas fallback used when dask is unavailable or when
    the input is a file-like object that dask cannot handle.

    Returns (DataFrame, total_row_count).
    If file fits in memory, returns all rows.
    """
    chunks = []
    total_rows = 0

    reader = pd.read_csv(
        filepath_or_buffer,
        sep=sep,
        chunksize=CSV_CHUNK_SIZE,
        low_memory=True,
    )

    for chunk in reader:
        total_rows += len(chunk)

        if sample_frac is not None and sample_frac < 1.0:
            chunk = chunk.sample(frac=sample_frac, random_state=42)

        chunks.append(chunk)

        if max_rows is not None and total_rows >= max_rows:
            break

    df = pd.concat(chunks, ignore_index=True)

    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)

    return df, total_rows


def _read_csv_chunked(
    filepath_or_buffer,
    sep: str = ",",
    max_rows: Optional[int] = None,
    sample_frac: Optional[float] = None,
) -> Tuple[pd.DataFrame, int]:
    """Read a CSV in chunks, optionally sampling or capping rows.

    When dask is available and the input is a file path (not a file-like object),
    uses dask for parallel reading. Otherwise falls back to pandas chunked reading.

    Returns (DataFrame, total_row_count).
    If file fits in memory, returns all rows.
    """
    # Use dask for file paths when available and no sampling is requested
    if HAS_DASK and isinstance(filepath_or_buffer, (str, Path)) and sample_frac is None:
        try:
            return _read_csv_dask(filepath_or_buffer, sep=sep, max_rows=max_rows)
        except Exception as e:
            logger.warning(f"Dask read failed, falling back to pandas chunked: {e}")

    return _read_csv_chunked_pandas(
        filepath_or_buffer, sep=sep, max_rows=max_rows, sample_frac=sample_frac,
    )


def load_file(
    filepath: str,
    sheet_name: Optional[str] = None,
    max_rows: Optional[int] = None,
    optimize_memory: bool = True,
) -> Tuple[pd.DataFrame, float, int]:
    """Load a data file into a DataFrame with large file support.

    Returns (DataFrame, file_size_mb, total_rows).
    For files > LARGE_FILE_THRESHOLD_MB, automatically optimizes memory.
    """
    path = Path(filepath)
    suffix = path.suffix.lower()
    file_size_mb = _get_file_size_mb(filepath)
    total_rows = 0

    is_large = file_size_mb > LARGE_FILE_THRESHOLD_MB
    effective_max = max_rows
    if is_large and effective_max is None:
        effective_max = MAX_INTERACTIVE_ROWS
        dask_note = " (dask parallel reading enabled)" if HAS_DASK else ""
        logger.info(
            f"Large file detected ({file_size_mb:.0f} MB){dask_note}. "
            f"Capping at {MAX_INTERACTIVE_ROWS:,} rows for interactive use."
        )

    if suffix in (".csv",):
        if is_large or effective_max:
            df, total_rows = _read_csv_chunked(filepath, sep=",", max_rows=effective_max)
        else:
            df = pd.read_csv(filepath, low_memory=False)
            total_rows = len(df)
    elif suffix in (".tsv", ".txt"):
        if is_large or effective_max:
            df, total_rows = _read_csv_chunked(filepath, sep="\t", max_rows=effective_max)
        else:
            df = pd.read_csv(filepath, sep="\t", low_memory=False)
            total_rows = len(df)
    elif suffix in (".xls", ".xlsx", ".xlsm"):
        df = pd.read_excel(filepath, sheet_name=sheet_name or 0)
        total_rows = len(df)
        if effective_max and len(df) > effective_max:
            df = df.head(effective_max)
    else:
        if is_large or effective_max:
            df, total_rows = _read_csv_chunked(filepath, sep=",", max_rows=effective_max)
        else:
            df = pd.read_csv(filepath, low_memory=False)
            total_rows = len(df)

    if optimize_memory and (is_large or len(df) > 50_000):
        df = _optimize_dtypes(df)

    return df, file_size_mb, total_rows


def load_uploaded_file(
    file_obj,
    filename: str,
    sheet_name: Optional[str] = None,
    max_rows: Optional[int] = None,
    optimize_memory: bool = True,
    file_size_mb: float = 0.0,
) -> Tuple[pd.DataFrame, int]:
    """Load an uploaded file object into a DataFrame with large file support.

    Returns (DataFrame, total_rows).
    """
    suffix = Path(filename).suffix.lower()
    is_large = file_size_mb > LARGE_FILE_THRESHOLD_MB
    effective_max = max_rows
    if is_large and effective_max is None:
        effective_max = MAX_INTERACTIVE_ROWS

    total_rows = 0

    if suffix in (".csv",):
        if is_large or effective_max:
            df, total_rows = _read_csv_chunked(file_obj, sep=",", max_rows=effective_max)
        else:
            df = pd.read_csv(file_obj, low_memory=False)
            total_rows = len(df)
    elif suffix in (".tsv", ".txt"):
        if is_large or effective_max:
            df, total_rows = _read_csv_chunked(file_obj, sep="\t", max_rows=effective_max)
        else:
            df = pd.read_csv(file_obj, sep="\t", low_memory=False)
            total_rows = len(df)
    elif suffix in (".xls", ".xlsx", ".xlsm"):
        df = pd.read_excel(file_obj, sheet_name=sheet_name or 0)
        total_rows = len(df)
        if effective_max and len(df) > effective_max:
            df = df.head(effective_max)
    else:
        if is_large or effective_max:
            df, total_rows = _read_csv_chunked(file_obj, sep=",", max_rows=effective_max)
        else:
            df = pd.read_csv(file_obj, low_memory=False)
            total_rows = len(df)

    if optimize_memory and (is_large or len(df) > 50_000):
        df = _optimize_dtypes(df)

    return df, total_rows


# ── Pattern matching ─────────────────────────────────────────────────────────

def _match_patterns(column_name: str, patterns: list[str]) -> bool:
    """Check if a column name matches any of the given patterns (case-insensitive)."""
    col_lower = column_name.lower().strip()
    return any(p in col_lower for p in patterns)


# ── Data type detection ──────────────────────────────────────────────────────

def detect_data_type(df: pd.DataFrame) -> str:
    """Auto-detect whether the data is object-level, summary-level, or cluster-level.

    Uses column name heuristics aligned with real HALO export conventions.
    Object data: Image Location, Object Id, XMin/YMin, per-channel metrics, phenotype combos
    Summary data: Image Tag, Job Id, Total Cells, % columns, H-Score
    """
    columns = list(df.columns)
    columns_lower = [c.lower().strip() for c in columns]

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

    # Strong object signals
    if any("object id" in c for c in columns_lower):
        object_score += 8
    if any("image location" in c for c in columns_lower):
        object_score += 5
    if any("xmin" in c for c in columns_lower):
        object_score += 3
    # Phenotype combo columns are a very strong object signal
    n_phenotype_combos = sum(1 for c in columns if _is_phenotype_combo(c))
    if n_phenotype_combos > 3:
        object_score += 10
    # Per-channel nucleus intensity columns
    if any("nucleus intensity" in c for c in columns_lower):
        object_score += 3
    if any("% nucleus completeness" in c for c in columns_lower):
        object_score += 3
    if any("positive nucleus classification" in c for c in columns_lower):
        object_score += 3

    # Strong summary signals
    if any("image tag" in c for c in columns_lower):
        summary_score += 5
    if any("job id" in c for c in columns_lower):
        summary_score += 4
    # "Total Cells" without "Object Id" → summary
    if any("total cells" in c for c in columns_lower) and not any("object id" in c for c in columns_lower):
        summary_score += 5
    # Percentage columns with "%" in cell-related headers
    pct_cell_cols = [c for c in columns_lower if "%" in c and "cell" in c]
    if len(pct_cell_cols) > 3:
        summary_score += 4

    # Cluster data
    if any("total cluster count" in c for c in columns_lower):
        cluster_score += 5

    # Row count as secondary signal
    if len(df) > 5000:
        object_score += 2
    elif len(df) < 100:
        summary_score += 2

    scores = {"object": object_score, "summary": summary_score, "cluster": cluster_score}
    best = max(scores, key=scores.get)

    if scores[best] == 0:
        return "object" if len(df) > 500 else "summary"
    return best


# ── Column classification ────────────────────────────────────────────────────

def classify_columns(df: pd.DataFrame, data_type: str) -> dict:
    """Classify columns into semantic categories.

    Handles both:
    - Summary data (anima set_analysis_metrics style: Cells, %, H-Score)
    - Object data (per-channel: Nucleus Intensity, Cell Intensity, Classification,
      % Nucleus Completeness, phenotype combos like DAPI+ C1+ C2+)
    """
    columns = list(df.columns)
    numeric_data = [h for h in df.select_dtypes(include=np.number).columns]
    categoric_data = [h for h in df.select_dtypes(include="object").columns]

    # ── Detect fluorophores present in the data ──────────────────────────
    fluorophores = _detect_fluorophores(columns)

    # ── Object-data specific classification ──────────────────────────────
    phenotype_combo_cols = [c for c in columns if _is_phenotype_combo(c)]
    classification_cols = [c for c in columns if _match_patterns(c, CLASSIFICATION_PATTERNS)]
    completeness_cols = [c for c in columns if _match_patterns(c, COMPLETENESS_PATTERNS)]
    nucleus_intensity_cols = [c for c in columns
                             if "nucleus intensity" in c.lower() and "%" not in c.lower()]
    cell_intensity_cols = [c for c in columns
                          if "cell intensity" in c.lower() and "%" not in c.lower()]
    morphology_cols = [c for c in columns if _match_patterns(c, MORPHOLOGY_PATTERNS)]

    # ── anima-style classification (summary data) ────────────────────────
    # Intensity: H-Score, *Intensity, *Cell Intensity
    intensity_data = [h for h in numeric_data
                      if "H-Score" in h or "Intensity" in h]

    # Spatial: numeric non-cell/non-intensity/non-phenotype-combo columns
    phenotype_combo_set = set(phenotype_combo_cols)
    classification_set = set(classification_cols)
    spatial_data = [
        h for h in numeric_data
        if h not in phenotype_combo_set
        and h not in classification_set
        and "Cells" not in h and "Nuclei" not in h and "Cytoplasms" not in h
        and "H-Score" not in h and "Intensity" not in h
        and "Completeness" not in h
    ]

    # Cell-related columns (summary data)
    cell_data = [h for h in numeric_data if "Cells" in h]
    total_data = [h for h in cell_data if "%" not in h]
    fraction_data = [h for h in cell_data if "%" in h]

    # Channel-specific (Spectrum/Cy5/5-FAM/Rhodamine)
    channel_keywords = ["spectrum", "cy5", "5-fam", "rhodamine", "opal", "alexa"]
    channel_total_data = [h for h in total_data
                          if any(kw in h.lower() for kw in channel_keywords)]
    channel_fraction_data = [h for h in fraction_data
                             if any(kw in h.lower() for kw in channel_keywords)]

    # Phenotype-specific (not channel-specific)
    phenotype_total_data = [h for h in total_data if h not in set(channel_total_data)]
    phenotype_fraction_data = [h for h in fraction_data if h not in set(channel_fraction_data)]

    # ── Semantic groups ──────────────────────────────────────────────────
    coordinate_cols = [c for c in columns if _match_patterns(c, COORDINATE_PATTERNS)]
    area_cols = [c for c in columns if _match_patterns(c, AREA_PATTERNS)]
    positivity_cols = classification_cols  # For object data, classification IS positivity
    if not positivity_cols:
        positivity_cols = [c for c in columns if "positive" in c.lower()]
    id_cols = [c for c in columns if _match_patterns(c, ID_PATTERNS)]
    annotation_cols = [c for c in columns if _match_patterns(c, ANNOTATION_PATTERNS)]

    # Marker columns: all per-channel measurement columns
    marker_cols = list(dict.fromkeys(
        intensity_data + classification_cols + completeness_cols +
        nucleus_intensity_cols + cell_intensity_cols +
        channel_total_data + channel_fraction_data
    ))

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
        "phenotype_combo": phenotype_combo_cols,
        "classification": classification_cols,
        "completeness": completeness_cols,
        "nucleus_intensity": nucleus_intensity_cols,
        "cell_intensity": cell_intensity_cols,
        "morphology": morphology_cols,
        "coordinate": coordinate_cols,
        "area": area_cols,
        "positivity": positivity_cols,
        "marker": marker_cols,
        "id": id_cols,
        "annotation": annotation_cols,
        "fluorophores": fluorophores,
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

    # For object data: derive Sample ID from Image Location (extract filename, strip extension)
    if "Image Location" in df.columns and "Sample ID" not in df.columns:
        df = df.copy()
        df["Sample ID"] = (
            df["Image Location"]
            .str.rsplit("\\", n=1).str[-1]  # Windows paths
            .str.rsplit("/", n=1).str[-1]    # Unix paths
            .str.replace(".scn", "", regex=False)
            .str.replace(".ndpi", "", regex=False)
            .str.replace(".svs", "", regex=False)
        )

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

def sample_for_plotting(df: pd.DataFrame, max_points: int = 50_000,
                         stratify_col: Optional[str] = None) -> pd.DataFrame:
    """Downsample a DataFrame for plotting performance.

    For datasets with > max_points rows, takes a random sample.
    Optionally stratifies by a categorical column to preserve group proportions.
    """
    if len(df) <= max_points:
        return df

    if stratify_col and stratify_col in df.columns:
        # Stratified sampling preserving group proportions
        frac = max_points / len(df)
        sampled = df.groupby(stratify_col, group_keys=False).apply(
            lambda x: x.sample(frac=min(frac, 1.0), random_state=42)
        )
        if len(sampled) > max_points:
            sampled = sampled.sample(n=max_points, random_state=42)
        return sampled.reset_index(drop=True)
    else:
        return df.sample(n=max_points, random_state=42).reset_index(drop=True)


def get_memory_usage_mb(df: pd.DataFrame) -> float:
    """Get DataFrame memory usage in MB."""
    return df.memory_usage(deep=True).sum() / (1024 * 1024)


def parse_halo_data(
    df: pd.DataFrame,
    filename: str = "unknown",
    force_type: Optional[str] = None,
    max_job: bool = False,
    analysis_area: Optional[str] = None,
    file_size_mb: float = 0.0,
    total_rows: int = 0,
) -> HaloDataset:
    """Parse a DataFrame as HALO data, auto-detecting type and classifying columns.

    Handles both real HALO object exports (Image Location, Object Id, per-channel metrics,
    phenotype combos like DAPI+ C1+ C2+ C3+ C4+) and summary exports (Image Tag, Job Id,
    Total Cells, % columns, H-Score).
    """
    # Preprocess (derive Sample ID, filter max job)
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

    is_sampled = total_rows > 0 and len(df) < total_rows

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
        phenotype_combo_columns=classified["phenotype_combo"],
        classification_columns=classified["classification"],
        completeness_columns=classified["completeness"],
        nucleus_intensity_columns=classified["nucleus_intensity"],
        cell_intensity_columns=classified["cell_intensity"],
        morphology_columns=classified["morphology"],
        coordinate_columns=classified["coordinate"],
        area_columns=classified["area"],
        positivity_columns=classified["positivity"],
        marker_columns=classified["marker"],
        id_columns=classified["id"],
        annotation_columns=classified["annotation"],
        fluorophore_channels=classified["fluorophores"],
        algorithm_names=algorithm_names,
        sample_ids=sample_ids,
        analysis_regions=analysis_regions,
        is_sampled=is_sampled,
        total_rows=total_rows if total_rows > 0 else len(df),
        file_size_mb=file_size_mb,
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
    """Return numeric columns suitable for plotting (exclude IDs and phenotype combo binary cols)."""
    exclude = set(dataset.id_columns) | set(dataset.phenotype_combo_columns)
    return [c for c in dataset.numeric_columns if c not in exclude]


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


def get_per_channel_columns(dataset: HaloDataset, channel: str) -> dict:
    """Get all columns for a specific fluorophore channel.

    Returns dict with keys: nucleus_intensity, cell_intensity, classification,
    nucleus_classification, completeness
    """
    ch_lower = channel.lower()
    result = {}
    for col in dataset.df.columns:
        col_lower = col.lower()
        if ch_lower not in col_lower:
            continue
        if "nucleus intensity" in col_lower:
            result["nucleus_intensity"] = col
        elif "cell intensity" in col_lower:
            result["cell_intensity"] = col
        elif "positive nucleus classification" in col_lower:
            result["nucleus_classification"] = col
        elif "positive classification" in col_lower:
            result["classification"] = col
        elif "% nucleus completeness" in col_lower or "nucleus completeness" in col_lower:
            result["completeness"] = col
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
