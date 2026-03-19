# Homer - Sample Data Generator
# Creates realistic HALO-like data for testing and demonstration

import pandas as pd
import numpy as np


def generate_object_data(n_cells: int = 5000, n_images: int = 3) -> pd.DataFrame:
    """Generate realistic HALO object-level (cell-by-cell) data."""
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

        # Spatial coordinates
        x_loc = rng.uniform(0, 50000)
        y_loc = rng.uniform(0, 50000)

        # Cell morphology
        nucleus_area = rng.lognormal(4.5, 0.5)
        cytoplasm_area = nucleus_area * rng.uniform(1.5, 4.0)
        cell_area = nucleus_area + cytoplasm_area

        # Marker intensities (vary by phenotype)
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
    """Generate realistic HALO summary-level data."""
    rng = np.random.default_rng(42)

    images = [f"Patient_{i+1:03d}_Slide_{j+1}" for i in range(n_images // 2) for j in range(2)]
    regions = ["Tumor", "Stroma"]

    records = []
    for image in images:
        for region in regions:
            total_cells = rng.integers(500, 15000)
            cd3_pct = rng.uniform(5, 45) if region == "Tumor" else rng.uniform(10, 60)
            cd4_pct = cd3_pct * rng.uniform(0.3, 0.6)
            cd8_pct = cd3_pct * rng.uniform(0.2, 0.5)
            cd20_pct = rng.uniform(2, 20)
            cd68_pct = rng.uniform(5, 35)
            pdl1_pct = rng.uniform(0, 40)

            analyzed_area = rng.uniform(0.5, 5.0)
            density = total_cells / analyzed_area

            records.append({
                "Image": image,
                "Analysis Region": region,
                "Total Cells": int(total_cells),
                "Total CD3+ Cells": int(total_cells * cd3_pct / 100),
                "% CD3 Positive": round(cd3_pct, 2),
                "Total CD4+ Cells": int(total_cells * cd4_pct / 100),
                "% CD4 Positive": round(cd4_pct, 2),
                "Total CD8+ Cells": int(total_cells * cd8_pct / 100),
                "% CD8 Positive": round(cd8_pct, 2),
                "Total CD20+ Cells": int(total_cells * cd20_pct / 100),
                "% CD20 Positive": round(cd20_pct, 2),
                "Total CD68+ Cells": int(total_cells * cd68_pct / 100),
                "% CD68 Positive": round(cd68_pct, 2),
                "Total PD-L1+ Cells": int(total_cells * pdl1_pct / 100),
                "% PD-L1 Positive": round(pdl1_pct, 2),
                "Analyzed Area (mm^2)": round(analyzed_area, 3),
                "Cell Density (cells/mm^2)": round(density, 1),
                "H-Score": round(rng.uniform(0, 300), 1),
            })

    return pd.DataFrame(records)
