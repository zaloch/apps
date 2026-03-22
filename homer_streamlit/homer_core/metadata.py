# Homer - Metadata Manager
# Handles experimental metadata (Subject ID, Treatment, Genotype, Timepoint, etc.)
# Supports both CSV upload and manual per-image entry
# Merges metadata with histology data and calculates per-image percentages from object data

import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass, field


# ── Standard metadata fields ────────────────────────────────────────────────

STANDARD_METADATA_FIELDS = [
    "Subject ID",
    "Sample ID",
    "Treatment Group",
    "Genotype",
    "Day",
    "Timepoint",
    "Sex",
    "Age",
    "Dose",
    "Region",
    "Cohort",
    "Notes",
]

# Common join keys between histology data and metadata
HISTOLOGY_JOIN_KEYS = [
    "Sample ID",       # derived from Image Tag or Image Location
    "Image Tag",       # raw histology field
    "Image Location",  # raw histology field (full path)
]


@dataclass
class ExperimentMetadata:
    """Container for experimental metadata that maps images to experimental factors."""
    df: pd.DataFrame
    join_key: str                  # column used to join with histology data
    factor_columns: list = field(default_factory=list)  # columns usable as plot groupings
    filename: str = ""


# ── Metadata I/O ────────────────────────────────────────────────────────────

def load_metadata_csv(filepath_or_buffer, filename: str = "") -> ExperimentMetadata:
    """Load a metadata CSV file.

    Expected format: one row per image/sample with columns like:
    Sample ID, Subject ID, Treatment Group, Genotype, Day, Timepoint, ...

    The first column that matches a known histology join key is used as the merge key.
    """
    if isinstance(filepath_or_buffer, str):
        df = pd.read_csv(filepath_or_buffer)
    else:
        df = pd.read_csv(filepath_or_buffer)

    join_key = _detect_join_key(df)
    factor_columns = _detect_factor_columns(df, join_key)

    return ExperimentMetadata(
        df=df,
        join_key=join_key,
        factor_columns=factor_columns,
        filename=filename,
    )


def _detect_join_key(df: pd.DataFrame) -> str:
    """Detect which column should be used to join metadata with histology data."""
    cols_lower = {c.lower().strip(): c for c in df.columns}

    # Priority order for join keys
    for key in HISTOLOGY_JOIN_KEYS:
        if key.lower() in cols_lower:
            return cols_lower[key.lower()]

    # Fall back: any column with "sample", "image", or "slide" in the name
    for lower, original in cols_lower.items():
        if any(kw in lower for kw in ["sample", "image", "slide", "specimen"]):
            return original

    # Last resort: first column
    return df.columns[0]


def _detect_factor_columns(df: pd.DataFrame, join_key: str) -> list[str]:
    """Detect which columns are experimental factors (suitable for plot grouping).

    Excludes the join key and any columns that are all unique (likely IDs)
    or all the same value (no variation).
    """
    factors = []
    for col in df.columns:
        if col == join_key:
            continue
        # Skip numeric-only columns that look like measurements
        if pd.api.types.is_numeric_dtype(df[col]):
            # Keep numeric factors with low cardinality (e.g., Day: 1,3,7,14)
            if df[col].nunique() > 20:
                continue
        # Skip columns where every value is unique (likely another ID)
        if df[col].nunique() == len(df) and len(df) > 3:
            continue
        # Skip columns with no variation
        if df[col].nunique() <= 1:
            continue
        factors.append(col)
    return factors


def create_empty_metadata(sample_ids: list[str]) -> pd.DataFrame:
    """Create an empty metadata template for manual entry.

    One row per sample with blank columns for standard metadata fields.
    """
    df = pd.DataFrame({"Sample ID": sample_ids})
    for field_name in STANDARD_METADATA_FIELDS:
        if field_name not in df.columns:
            df[field_name] = ""
    return df


def metadata_template_csv(sample_ids: list[str]) -> str:
    """Generate a CSV string template for metadata entry."""
    df = create_empty_metadata(sample_ids)
    return df.to_csv(index=False)


# ── Merge metadata with histology data ──────────────────────────────────────────

def merge_metadata(
    histology_df: pd.DataFrame,
    metadata: ExperimentMetadata,
) -> pd.DataFrame:
    """Merge experimental metadata into the histology dataframe.

    Joins on the detected join key. Metadata columns are added to each
    matching row in the histology data.
    """
    join_key = metadata.join_key

    # Find the corresponding column in the histology data
    histology_join_col = _find_histology_join_column(histology_df, join_key)
    if histology_join_col is None:
        raise ValueError(
            f"Cannot find matching column for metadata key '{join_key}' "
            f"in histology data. Available columns: {list(histology_df.columns[:10])}"
        )

    # Avoid duplicating columns that already exist
    meta_cols = [join_key] + [
        c for c in metadata.factor_columns
        if c not in histology_df.columns
    ]
    meta_subset = metadata.df[meta_cols].drop_duplicates(subset=[join_key])

    # Merge
    if histology_join_col == join_key:
        merged = histology_df.merge(meta_subset, on=join_key, how="left")
    else:
        merged = histology_df.merge(
            meta_subset, left_on=histology_join_col, right_on=join_key, how="left"
        )

    return merged


def _find_histology_join_column(histology_df: pd.DataFrame, meta_key: str) -> Optional[str]:
    """Find the histology dataframe column that corresponds to the metadata join key."""
    # Exact match
    if meta_key in histology_df.columns:
        return meta_key

    # Case-insensitive match
    cols_lower = {c.lower().strip(): c for c in histology_df.columns}
    if meta_key.lower() in cols_lower:
        return cols_lower[meta_key.lower()]

    # "Sample ID" is often derived from Image Tag/Location
    if meta_key.lower() == "sample id" and "Sample ID" in histology_df.columns:
        return "Sample ID"

    return None


# ── Object data aggregation ─────────────────────────────────────────────────

def aggregate_object_data(
    df: pd.DataFrame,
    group_cols: list[str],
    classification_cols: Optional[list[str]] = None,
    phenotype_combo_cols: Optional[list[str]] = None,
    intensity_cols: Optional[list[str]] = None,
    morphology_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Aggregate object-level data to per-image/per-group summary.

    For each group (e.g., per Sample ID, per Sample ID + Analysis Region):
    - Classification cols: calculate % positive (mean of binary 0/1 * 100)
    - Phenotype combo cols: calculate % positive (mean of binary 0/1 * 100)
    - Intensity cols: calculate mean and median
    - Morphology cols: calculate mean
    - Count total objects per group

    This mirrors the HistologyAnalysis aggregation workflow.
    """
    if not group_cols:
        raise ValueError("At least one grouping column is required")

    # Filter to only columns that exist
    group_cols = [c for c in group_cols if c in df.columns]
    if not group_cols:
        raise ValueError("None of the specified group columns exist in the data")

    agg_dict = {}

    # Classification columns: binary 0/1 → percentage
    if classification_cols:
        for col in classification_cols:
            if col in df.columns:
                agg_dict[col] = "mean"

    # Phenotype combo columns: binary 0/1 → percentage
    if phenotype_combo_cols:
        for col in phenotype_combo_cols:
            if col in df.columns:
                agg_dict[col] = "mean"

    # Intensity columns: mean
    if intensity_cols:
        for col in intensity_cols:
            if col in df.columns:
                agg_dict[col] = "mean"

    # Morphology columns: mean
    if morphology_cols:
        for col in morphology_cols:
            if col in df.columns:
                agg_dict[col] = "mean"

    if not agg_dict:
        # Fall back: aggregate all numeric columns by mean
        numeric = df.select_dtypes(include=np.number).columns
        for col in numeric:
            if col not in group_cols:
                agg_dict[col] = "mean"

    grouped = df.groupby(group_cols, as_index=False, observed=True).agg(agg_dict)

    # Add object count per group
    counts = df.groupby(group_cols, observed=True).size().reset_index(name="Object Count")
    grouped = grouped.merge(counts, on=group_cols, how="left")

    # Convert binary classification/phenotype means to percentages
    pct_cols = []
    if classification_cols:
        pct_cols.extend([c for c in classification_cols if c in grouped.columns])
    if phenotype_combo_cols:
        pct_cols.extend([c for c in phenotype_combo_cols if c in grouped.columns])

    for col in pct_cols:
        pct_name = f"% {col}"
        grouped[pct_name] = (grouped[col] * 100).round(2)

    return grouped


def calculate_per_image_percentages(
    df: pd.DataFrame,
    sample_col: str = "Sample ID",
    classification_cols: Optional[list[str]] = None,
    phenotype_combo_cols: Optional[list[str]] = None,
    extra_group_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Calculate per-image/per-sample percentages from object-level data.

    This is the key aggregation step: transforms per-cell binary data into
    per-image percentage data that can then be plotted as a factor of
    Treatment, Genotype, Subject ID, etc.

    Returns a DataFrame with one row per sample (or per sample + region),
    with % positive columns for each classification/phenotype.
    """
    group_cols = [sample_col]
    if extra_group_cols:
        group_cols.extend([c for c in extra_group_cols if c in df.columns and c != sample_col])

    # Remove duplicates while preserving order
    seen = set()
    unique_group_cols = []
    for c in group_cols:
        if c not in seen and c in df.columns:
            seen.add(c)
            unique_group_cols.append(c)

    return aggregate_object_data(
        df,
        group_cols=unique_group_cols,
        classification_cols=classification_cols,
        phenotype_combo_cols=phenotype_combo_cols,
    )


# ── Demo metadata ──────────────────────────────────────────────────────────

def generate_demo_metadata(sample_ids: list[str]) -> pd.DataFrame:
    """Generate demo metadata for testing.

    Creates realistic experimental metadata with Treatment, Genotype,
    Subject ID, Timepoint, Sex assignments.
    """
    rng = np.random.default_rng(42)
    n = len(sample_ids)

    treatments = ["Vehicle", "Drug A", "Drug B", "Drug A + Drug B"]
    genotypes = ["WT", "KO", "Het"]
    timepoints = ["Day 1", "Day 7", "Day 14", "Day 28"]
    sexes = ["M", "F"]

    return pd.DataFrame({
        "Sample ID": sample_ids,
        "Subject ID": [f"SUB-{rng.integers(100, 999)}" for _ in range(n)],
        "Treatment Group": [rng.choice(treatments) for _ in range(n)],
        "Genotype": [rng.choice(genotypes) for _ in range(n)],
        "Timepoint": [rng.choice(timepoints) for _ in range(n)],
        "Day": [rng.choice([1, 7, 14, 28]) for _ in range(n)],
        "Sex": [rng.choice(sexes) for _ in range(n)],
        "Age": [rng.integers(8, 24) for _ in range(n)],
        "Dose": [rng.choice([0, 1, 5, 10, 25]) for _ in range(n)],
        "Cohort": [rng.choice(["Cohort 1", "Cohort 2"]) for _ in range(n)],
    })
