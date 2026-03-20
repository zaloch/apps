# Homer - Sample Data Generator
# Creates realistic HALO-like data for testing and demonstration
# Aligned with anima HaloAnalysis column conventions

import pandas as pd
import numpy as np


def generate_object_data(n_cells: int = 5000, n_images: int = 3) -> pd.DataFrame:
    """Generate realistic HALO object-level (cell-by-cell) data.

    Column naming matches actual HALO exports as used in the anima module.
    """
    rng = np.random.default_rng(42)

    images = [f"Slide_{i+1}_Region_{j+1}" for i in range(n_images) for j in range(2)]
    annotations = ["Tumor", "Stroma", "Necrosis", "Normal"]
    phenotypes = [
        "CD3+,CD4+", "CD3+,CD8+", "CD3+,CD4-,CD8-",
        "CD20+", "CD68+", "PD-L1+", "Other",
    ]

    records = []
    for cell_id in range(n_cells):
        image = rng.choice(images)
        annotation = rng.choice(annotations, p=[0.4, 0.3, 0.1, 0.2])
        phenotype = rng.choice(phenotypes, p=[0.15, 0.15, 0.05, 0.1, 0.15, 0.1, 0.3])

        x_loc = rng.uniform(0, 50000)
        y_loc = rng.uniform(0, 50000)

        nucleus_area = rng.lognormal(4.5, 0.5)
        cytoplasm_area = nucleus_area * rng.uniform(1.5, 4.0)
        cell_area = nucleus_area + cytoplasm_area

        base_cd3 = 0.8 if "CD3+" in phenotype else 0.1
        base_cd4 = 0.7 if "CD4+" in phenotype else 0.05
        base_cd8 = 0.7 if "CD8+" in phenotype else 0.05
        base_cd20 = 0.8 if "CD20+" in phenotype else 0.05
        base_cd68 = 0.75 if "CD68+" in phenotype else 0.08
        base_pdl1 = 0.6 if "PD-L1+" in phenotype else 0.05

        records.append({
            "Image": image,
            "Analysis Region": annotation,
            "Cell ID": cell_id + 1,
            "XMin": x_loc - 10,
            "XMax": x_loc + 10,
            "YMin": y_loc - 10,
            "YMax": y_loc + 10,
            "X Location": x_loc,
            "Y Location": y_loc,
            "Cell Area (um^2)": round(cell_area, 2),
            "Nucleus Area (um^2)": round(nucleus_area, 2),
            "Cytoplasm Area (um^2)": round(cytoplasm_area, 2),
            "DAPI Nucleus Intensity": round(rng.lognormal(5.0, 0.3), 2),
            "CD3 Cytoplasm OD": round(max(0, rng.normal(base_cd3, 0.15)), 4),
            "CD4 Cytoplasm OD": round(max(0, rng.normal(base_cd4, 0.12)), 4),
            "CD8 Cytoplasm OD": round(max(0, rng.normal(base_cd8, 0.12)), 4),
            "CD20 Membrane OD": round(max(0, rng.normal(base_cd20, 0.15)), 4),
            "CD68 Cytoplasm OD": round(max(0, rng.normal(base_cd68, 0.12)), 4),
            "PD-L1 Membrane OD": round(max(0, rng.normal(base_pdl1, 0.15)), 4),
            "CD3 Positive Classification": 1 if "CD3+" in phenotype else 0,
            "CD4 Positive Classification": 1 if "CD4+" in phenotype else 0,
            "CD8 Positive Classification": 1 if "CD8+" in phenotype else 0,
            "CD20 Positive Classification": 1 if "CD20+" in phenotype else 0,
            "CD68 Positive Classification": 1 if "CD68+" in phenotype else 0,
            "PD-L1 Positive Classification": 1 if "PD-L1+" in phenotype else 0,
            "Cell Phenotype": phenotype,
            "Classifier Label": annotation,
        })

    return pd.DataFrame(records)


def generate_summary_data(n_images: int = 12) -> pd.DataFrame:
    """Generate realistic HALO summary-level data.

    Column naming matches actual HALO analysis output as used in the anima module:
    - Image Tag, Algorithm Name, Job Id (HALO metadata)
    - Sample ID (derived, .scn stripped)
    - Analysis Region, Total Cells, fraction columns with "%"
    - Region Area, H-Score, Intensity columns
    - Weak/Strong/Moderate classification columns
    """
    rng = np.random.default_rng(42)

    image_tags = [f"Patient_{i+1:03d}_Biopsy.scn" for i in range(n_images)]
    algorithms = ["Multiplex IHC v3.2.1", "Highplex FL v4.2.3"]
    regions = ["Tumor", "Stroma"]

    records = []
    for image_tag in image_tags:
        algo = rng.choice(algorithms)
        job_id = rng.integers(100000, 999999)

        for region in regions:
            total_cells = rng.integers(500, 15000)

            # Channel-specific cell counts and fractions
            cd3_pct = rng.uniform(5, 45) if region == "Tumor" else rng.uniform(10, 60)
            cd4_pct = cd3_pct * rng.uniform(0.3, 0.6)
            cd8_pct = cd3_pct * rng.uniform(0.2, 0.5)
            cd20_pct = rng.uniform(2, 20)
            cd68_pct = rng.uniform(5, 35)
            pdl1_pct = rng.uniform(0, 40)

            # Weak/Moderate/Strong for CD3
            cd3_total = int(total_cells * cd3_pct / 100)
            cd3_weak_pct = rng.uniform(20, 50)
            cd3_mod_pct = rng.uniform(20, 50)
            cd3_strong_pct = 100 - cd3_weak_pct - cd3_mod_pct

            region_area = rng.uniform(50000, 500000)  # um^2
            analyzed_area = rng.uniform(0.5, 5.0)     # mm^2

            records.append({
                "Image Tag": image_tag,
                "Algorithm Name": algo,
                "Job Id": int(job_id),
                "Analysis Region": region,
                # Derived by parse_halo_data:
                # "Sample ID" will be added by preprocess_halo_summary

                # Cell counts (total_data pattern: "Cells" without "%")
                "Total Cells": int(total_cells),
                "CD3+ Cells": int(cd3_total),
                "CD4+ Cells": int(total_cells * cd4_pct / 100),
                "CD8+ Cells": int(total_cells * cd8_pct / 100),
                "CD20+ Cells": int(total_cells * cd20_pct / 100),
                "CD68+ Cells": int(total_cells * cd68_pct / 100),
                "PD-L1+ Cells": int(total_cells * pdl1_pct / 100),

                # Fraction data (fraction_data pattern: "Cells" with "%")
                "% CD3+ Cells": round(cd3_pct, 2),
                "% CD4+ Cells": round(cd4_pct, 2),
                "% CD8+ Cells": round(cd8_pct, 2),
                "% CD20+ Cells": round(cd20_pct, 2),
                "% CD68+ Cells": round(cd68_pct, 2),
                "% PD-L1+ Cells": round(pdl1_pct, 2),

                # Weak/Moderate/Strong breakdown
                "% CD3 Weak Cells": round(cd3_weak_pct * cd3_pct / 100, 2),
                "% CD3 Moderate Cells": round(cd3_mod_pct * cd3_pct / 100, 2),
                "% CD3 Strong Cells": round(cd3_strong_pct * cd3_pct / 100, 2),

                # Negative
                "% Negative Cells": round(100 - cd3_pct - cd20_pct - cd68_pct, 2),

                # Intensity data (intensity_data pattern: "H-Score" or "Intensity")
                "CD3 H-Score": round(rng.uniform(0, 300), 1),
                "PD-L1 H-Score": round(rng.uniform(0, 300), 1),
                "DAPI Nucleus Intensity": round(rng.lognormal(5.0, 0.3), 2),

                # Spatial data
                "Region Area (μm²)": round(region_area, 2),
                "Analyzed Area (mm^2)": round(analyzed_area, 3),
                "Cell Density (cells/mm^2)": round(total_cells / analyzed_area, 1),

                # Channel-specific (Spectrum/Cy5 pattern from anima)
                "Spectrum FITC Cy5 Positive Cells": int(total_cells * rng.uniform(0.05, 0.3)),
                "% Spectrum FITC Cy5 Positive Cells": round(rng.uniform(5, 30), 2),
            })

    return pd.DataFrame(records)


def generate_cluster_data(n_clusters: int = 200, n_images: int = 4) -> pd.DataFrame:
    """Generate realistic HALO cluster/aggregated object data.

    Matches the output of HaloIngest.aggregate_data from anima.
    """
    rng = np.random.default_rng(42)

    image_tags = [f"Sample_{i+1:03d}.scn" for i in range(n_images)]
    algorithms = ["Multiplex IHC v3.2.1"]

    records = []
    for _ in range(n_clusters):
        image_tag = rng.choice(image_tags)
        algo = algorithms[0]
        job_id = rng.integers(100000, 999999)

        total_cells = rng.integers(10, 500)
        region_area = rng.uniform(1000, 50000)

        records.append({
            "Sample ID": image_tag.replace(".scn", ""),
            "Algorithm Name": algo,
            "Job ID": int(job_id),
            "Total Cluster Count": rng.integers(1, 50),
            "Total Cell Count": int(total_cells),
            "Total Area Analyzed": round(region_area, 2),
            "Total Cells": int(total_cells),
            "Region Area (μm²)": round(region_area, 2),
            "% CD3+ Cells": round(rng.uniform(0, 60), 2),
            "% CD4+ Cells": round(rng.uniform(0, 40), 2),
            "% CD8+ Cells": round(rng.uniform(0, 40), 2),
            "% CD20+ Cells": round(rng.uniform(0, 25), 2),
            "% CD68+ Cells": round(rng.uniform(0, 40), 2),
            "% PD-L1+ Cells": round(rng.uniform(0, 45), 2),
            "CD3 H-Score": round(rng.uniform(0, 300), 1),
            "DAPI Nucleus Intensity": round(rng.lognormal(5.0, 0.3), 2),
        })

    return pd.DataFrame(records)
