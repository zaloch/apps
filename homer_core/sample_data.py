# Homer - Sample Data Generator
# Creates realistic HALO-like data for testing and demonstration
# Aligned with real HALO object export format (802_NeuN_SOX9_GFP_CD31_objectdata style)

import pandas as pd
import numpy as np
from itertools import product


def _generate_phenotype_combos(n_channels: int = 4) -> list[str]:
    """Generate all phenotype combination column names.

    Matches real HALO format: DAPI+ C1+ C2+ C3+ C4+, DAPI+ C1+ C2+ C3+ C4-, etc.
    Plus DAPI+ and DAPI- columns.
    """
    combos = []
    # All DAPI+ combinations of C1-C4 +/-
    for signs in product(["+", "-"], repeat=n_channels):
        channels = " ".join(f"C{i+1}{s}" for i, s in enumerate(signs))
        combos.append(f"DAPI+ {channels}")
    # All DAPI- combinations
    for signs in product(["+", "-"], repeat=n_channels):
        channels = " ".join(f"C{i+1}{s}" for i, s in enumerate(signs))
        combos.append(f"DAPI- {channels}")
    # Single DAPI+/DAPI- columns
    combos.extend(["DAPI+", "DAPI-"])
    return combos


def _per_channel_columns(channel: str) -> list[str]:
    """Generate per-channel column names matching real HALO format."""
    return [
        f"{channel} Positive Classification",
        f"{channel} Positive Nucleus Classification",
        f"{channel} Nucleus Intensity",
        f"{channel} % Nucleus Completeness",
        f"{channel} Cell Intensity",
    ]


def generate_object_data(n_cells: int = 5000, n_images: int = 3) -> pd.DataFrame:
    """Generate realistic HALO object-level data matching the real export format.

    Columns match 802_NeuN_SOX9_GFP_CD31_objectdata:
    Image Location, Analysis Region, Algorithm Name, Object Id,
    XMin, XMax, YMin, YMax,
    DAPI+ C1+ C2+ C3+ C4+, ... (all phenotype combos),
    DAPI+, DAPI-,
    DAPI Positive Classification, DAPI Nucleus Intensity, etc. per channel,
    Cell Area, Cytoplasm Area, Nucleus Area, Nucleus Perimeter, Nucleus Roundness

    Includes Treatment Group, Genotype, Subject ID, and Timepoint columns
    with biologically plausible treatment effects on marker expression.
    """
    rng = np.random.default_rng(42)

    # Fluorophore channels (matching the screenshot)
    channels = ["DAPI", "Cy5", "5-FAM", "Spectrum Aqua", "Rhodamine 6G"]

    # Treatment design: balanced across images
    treatments = ["Vehicle", "Drug A", "Drug B", "Drug A + Drug B"]
    genotypes = ["WT", "KO"]
    timepoints = ["Day 7", "Day 14"]

    # Build per-image assignments so each image belongs to one condition
    n_images = max(n_images, len(treatments))  # ensure at least 1 image per treatment
    image_meta = []
    for i in range(n_images):
        trt = treatments[i % len(treatments)]
        geno = genotypes[i % len(genotypes)]
        tp = timepoints[i % len(timepoints)]
        subj = f"SUB-{100 + i}"
        image_meta.append({
            "image_loc": f"\\\\server\\halo_data\\Study_001\\Slide_{i+1:03d}.scn",
            "treatment": trt,
            "genotype": geno,
            "timepoint": tp,
            "subject_id": subj,
        })

    # Treatment effect modifiers on channel positivity rates
    # Drug A increases C1/C2, Drug B increases C3/C4, combo increases all
    treatment_effects = {
        "Vehicle":        [0.25, 0.20, 0.20, 0.25],
        "Drug A":         [0.55, 0.50, 0.20, 0.25],
        "Drug B":         [0.25, 0.20, 0.50, 0.55],
        "Drug A + Drug B":[0.55, 0.50, 0.50, 0.55],
    }

    image_locs = [m["image_loc"] for m in image_meta]
    regions = ["Whole Brain", "Cortex", "Hippocampus", "Cerebellum"]
    algorithms = ["Multiplex FL v4.2.3"]

    # Phenotype combo columns
    phenotype_combos = _generate_phenotype_combos(4)

    records = []
    for obj_id in range(n_cells):
        img_idx = rng.integers(0, len(image_meta))
        meta = image_meta[img_idx]
        image_loc = meta["image_loc"]
        region = rng.choice(regions, p=[0.4, 0.25, 0.2, 0.15])
        algo = algorithms[0]

        # Spatial
        x_min = rng.uniform(0, 50000)
        y_min = rng.uniform(0, 50000)
        width = rng.uniform(10, 40)
        height = rng.uniform(10, 40)

        row = {
            "Image Location": image_loc,
            "Analysis Region": region,
            "Algorithm Name": algo,
            "Object Id": obj_id,
            "XMin": round(x_min, 1),
            "XMax": round(x_min + width, 1),
            "YMin": round(y_min, 1),
            "YMax": round(y_min + height, 1),
            "Treatment Group": meta["treatment"],
            "Genotype": meta["genotype"],
            "Timepoint": meta["timepoint"],
            "Subject ID": meta["subject_id"],
        }

        # Generate phenotype combo binary values with treatment effects
        dapi_positive = rng.random() > 0.05  # ~95% DAPI+
        base_rates = treatment_effects[meta["treatment"]]
        c_positives = [rng.random() < rate for rate in base_rates]

        for combo in phenotype_combos:
            parts = combo.strip().split()
            match = True
            for part in parts:
                if part == "DAPI+":
                    match = match and dapi_positive
                elif part == "DAPI-":
                    match = match and (not dapi_positive)
                elif part.startswith("C") and part.endswith("+"):
                    idx = int(part[1:-1]) - 1
                    match = match and c_positives[idx]
                elif part.startswith("C") and part.endswith("-"):
                    idx = int(part[1:-1]) - 1
                    match = match and (not c_positives[idx])
            row[combo] = 1 if match else 0

        # DAPI+/DAPI- single columns
        row["DAPI+"] = 1 if dapi_positive else 0
        row["DAPI-"] = 0 if dapi_positive else 1

        # Per-channel metrics (intensity shifted by treatment)
        for i, ch in enumerate(channels):
            is_positive = c_positives[i] if i < len(c_positives) else (rng.random() > 0.5)

            row[f"{ch} Positive Classification"] = 1 if is_positive else 0
            row[f"{ch} Positive Nucleus Classification"] = 1 if is_positive else 0
            row[f"{ch} Nucleus Intensity"] = round(rng.lognormal(7.5 if is_positive else 6.5, 0.5), 3)
            row[f"{ch} % Nucleus Completeness"] = round(rng.uniform(60, 100) if is_positive else rng.uniform(0, 100), 5)
            row[f"{ch} Cell Intensity"] = round(rng.lognormal(7.8 if is_positive else 6.8, 0.4), 3)

        # Morphology
        nucleus_area = rng.lognormal(4.5, 0.5)
        cytoplasm_area = nucleus_area * rng.uniform(0.3, 2.0)
        cell_area = nucleus_area + cytoplasm_area
        nucleus_perimeter = rng.uniform(15, 120)
        nucleus_roundness = rng.uniform(0.4, 1.0)

        row["Cell Area (\u03bcm\u00b2)"] = round(cell_area, 4)
        row["Cytoplasm Area (\u03bcm\u00b2)"] = round(cytoplasm_area, 4)
        row["Nucleus Area (\u03bcm\u00b2)"] = round(nucleus_area, 4)
        row["Nucleus Perimeter (\u03bcm)"] = round(nucleus_perimeter, 2)
        row["Nucleus Roundness"] = round(nucleus_roundness, 5)

        records.append(row)

    return pd.DataFrame(records)


def generate_summary_data(n_images: int = 12) -> pd.DataFrame:
    """Generate realistic HALO summary-level data.

    Column naming matches actual HALO analysis output:
    Image Tag, Algorithm Name, Job Id, Analysis Region,
    Total Cells, cell counts and fractions, H-Score, Intensity

    Includes Treatment Group, Genotype, Subject ID, and Timepoint columns
    with biologically plausible treatment effects on immune cell infiltration.
    """
    rng = np.random.default_rng(42)

    # Treatment design
    treatments = ["Vehicle", "Anti-PD1", "Anti-CTLA4", "Anti-PD1 + Anti-CTLA4"]
    genotypes = ["WT", "KO"]
    timepoints = ["Day 7", "Day 14"]

    # Ensure enough images for balanced design
    n_images = max(n_images, len(treatments) * 2)
    image_tags = [f"Patient_{i+1:03d}_Biopsy.scn" for i in range(n_images)]
    algorithms = ["Multiplex IHC v3.2.1", "Highplex FL v4.2.3"]
    regions = ["Tumor", "Stroma"]

    # Treatment effect multipliers on immune cell %: [CD3, CD8, PD-L1, CD68]
    treatment_effects = {
        "Vehicle":               {"cd3": 1.0, "cd8": 1.0, "pdl1": 1.0, "cd68": 1.0},
        "Anti-PD1":              {"cd3": 1.6, "cd8": 2.0, "pdl1": 0.5, "cd68": 1.2},
        "Anti-CTLA4":            {"cd3": 1.8, "cd8": 1.5, "pdl1": 0.8, "cd68": 1.4},
        "Anti-PD1 + Anti-CTLA4": {"cd3": 2.2, "cd8": 2.5, "pdl1": 0.3, "cd68": 1.6},
    }

    records = []
    for i, image_tag in enumerate(image_tags):
        trt = treatments[i % len(treatments)]
        geno = genotypes[i % len(genotypes)]
        tp = timepoints[i % len(timepoints)]
        subj_id = f"SUB-{100 + i}"
        effects = treatment_effects[trt]

        algo = rng.choice(algorithms)
        job_id = rng.integers(100000, 999999)

        for region in regions:
            total_cells = rng.integers(500, 15000)
            base_cd3 = rng.uniform(5, 45) if region == "Tumor" else rng.uniform(10, 60)
            cd3_pct = min(base_cd3 * effects["cd3"], 90)
            cd4_pct = cd3_pct * rng.uniform(0.3, 0.6)
            cd8_pct = min(cd3_pct * rng.uniform(0.2, 0.5) * effects["cd8"] / effects["cd3"], 60)
            cd20_pct = rng.uniform(2, 20)
            cd68_pct = min(rng.uniform(5, 35) * effects["cd68"], 60)
            pdl1_pct = max(rng.uniform(0, 40) * effects["pdl1"], 0)

            cd3_total = int(total_cells * cd3_pct / 100)
            cd3_weak_pct = rng.uniform(20, 50)
            cd3_mod_pct = rng.uniform(20, 50)
            cd3_strong_pct = 100 - cd3_weak_pct - cd3_mod_pct

            region_area = rng.uniform(50000, 500000)
            analyzed_area = rng.uniform(0.5, 5.0)

            records.append({
                "Image Tag": image_tag,
                "Algorithm Name": algo,
                "Job Id": int(job_id),
                "Analysis Region": region,
                "Treatment Group": trt,
                "Genotype": geno,
                "Timepoint": tp,
                "Subject ID": subj_id,
                "Total Cells": int(total_cells),
                "CD3+ Cells": int(cd3_total),
                "CD4+ Cells": int(total_cells * cd4_pct / 100),
                "CD8+ Cells": int(total_cells * cd8_pct / 100),
                "CD20+ Cells": int(total_cells * cd20_pct / 100),
                "CD68+ Cells": int(total_cells * cd68_pct / 100),
                "PD-L1+ Cells": int(total_cells * pdl1_pct / 100),
                "% CD3+ Cells": round(cd3_pct, 2),
                "% CD4+ Cells": round(cd4_pct, 2),
                "% CD8+ Cells": round(cd8_pct, 2),
                "% CD20+ Cells": round(cd20_pct, 2),
                "% CD68+ Cells": round(cd68_pct, 2),
                "% PD-L1+ Cells": round(pdl1_pct, 2),
                "% CD3 Weak Cells": round(cd3_weak_pct * cd3_pct / 100, 2),
                "% CD3 Moderate Cells": round(cd3_mod_pct * cd3_pct / 100, 2),
                "% CD3 Strong Cells": round(cd3_strong_pct * cd3_pct / 100, 2),
                "% Negative Cells": round(100 - cd3_pct - cd20_pct - cd68_pct, 2),
                "CD3 H-Score": round(rng.uniform(0, 300) * effects["cd3"], 1),
                "PD-L1 H-Score": round(rng.uniform(0, 300) * effects["pdl1"], 1),
                "DAPI Nucleus Intensity": round(rng.lognormal(5.0, 0.3), 2),
                "Region Area (\u03bcm\u00b2)": round(region_area, 2),
                "Analyzed Area (mm^2)": round(analyzed_area, 3),
                "Cell Density (cells/mm^2)": round(total_cells / analyzed_area, 1),
                "Spectrum FITC Cy5 Positive Cells": int(total_cells * rng.uniform(0.05, 0.3)),
                "% Spectrum FITC Cy5 Positive Cells": round(rng.uniform(5, 30), 2),
            })

    return pd.DataFrame(records)


def generate_cluster_data(n_clusters: int = 200, n_images: int = 4) -> pd.DataFrame:
    """Generate realistic HALO cluster/aggregated object data.

    Includes Treatment Group, Genotype, Subject ID, and Timepoint columns
    with biologically plausible treatment effects on cluster composition.
    """
    rng = np.random.default_rng(42)

    treatments = ["Vehicle", "Drug A", "Drug B", "Drug A + Drug B"]
    genotypes = ["WT", "KO"]
    timepoints = ["Day 7", "Day 14"]

    n_images = max(n_images, len(treatments))
    image_meta = []
    for i in range(n_images):
        trt = treatments[i % len(treatments)]
        geno = genotypes[i % len(genotypes)]
        tp = timepoints[i % len(timepoints)]
        image_meta.append({
            "tag": f"Sample_{i+1:03d}.scn",
            "sample_id": f"Sample_{i+1:03d}",
            "treatment": trt,
            "genotype": geno,
            "timepoint": tp,
            "subject_id": f"SUB-{100 + i}",
        })

    # Treatment effects on CD3/CD8 fractions
    treatment_mult = {
        "Vehicle": 1.0, "Drug A": 1.5, "Drug B": 1.4, "Drug A + Drug B": 2.0,
    }

    algorithms = ["Multiplex IHC v3.2.1"]

    records = []
    for _ in range(n_clusters):
        meta = image_meta[rng.integers(0, len(image_meta))]
        mult = treatment_mult[meta["treatment"]]
        total_cells = rng.integers(10, 500)
        region_area = rng.uniform(1000, 50000)

        records.append({
            "Sample ID": meta["sample_id"],
            "Algorithm Name": algorithms[0],
            "Job ID": int(rng.integers(100000, 999999)),
            "Treatment Group": meta["treatment"],
            "Genotype": meta["genotype"],
            "Timepoint": meta["timepoint"],
            "Subject ID": meta["subject_id"],
            "Total Cluster Count": rng.integers(1, 50),
            "Total Cell Count": int(total_cells),
            "Total Area Analyzed": round(region_area, 2),
            "Total Cells": int(total_cells),
            "Region Area (\u03bcm\u00b2)": round(region_area, 2),
            "% CD3+ Cells": round(min(rng.uniform(0, 60) * mult, 95), 2),
            "% CD4+ Cells": round(min(rng.uniform(0, 40) * mult, 70), 2),
            "% CD8+ Cells": round(min(rng.uniform(0, 40) * mult, 70), 2),
            "% CD20+ Cells": round(rng.uniform(0, 25), 2),
            "% CD68+ Cells": round(rng.uniform(0, 40), 2),
            "% PD-L1+ Cells": round(max(rng.uniform(0, 45) / mult, 0), 2),
            "CD3 H-Score": round(rng.uniform(0, 300) * mult, 1),
            "DAPI Nucleus Intensity": round(rng.lognormal(5.0, 0.3), 2),
        })

    return pd.DataFrame(records)
