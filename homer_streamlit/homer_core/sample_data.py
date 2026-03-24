# Homer - Sample Data Generator
# Creates realistic histology-like data for testing and demonstration
# Supports multiple organ/panel profiles with literature-based markers

import pandas as pd
import numpy as np
from itertools import product


# ── Organ / Panel Profiles ──────────────────────────────────────────────────
# Each profile defines channels (fluorophores mapped to markers), tissue
# regions, treatments with dose maps, and per-treatment positivity rates
# for each non-DAPI channel.  Marker names and cell-type annotations are
# sourced from published IHC/IF literature.

PROFILES: dict[str, dict] = {

    "Neuroscience (Brain)": {
        "channels": ["DAPI", "NeuN", "IBA1", "GFAP", "SOX9", "CD31", "MBP", "Olig2"],
        "channel_labels": {
            "NeuN": "Mature neurons",
            "IBA1": "Microglia",
            "GFAP": "Reactive astrocytes",
            "SOX9": "Astrocyte lineage",
            "CD31": "Endothelial / vasculature",
            "MBP": "Myelin / oligodendrocytes",
            "Olig2": "Oligodendrocyte lineage",
        },
        "regions": ["Whole Brain", "Cortex", "Hippocampus", "Cerebellum", "White Matter"],
        "region_weights": [0.30, 0.20, 0.20, 0.15, 0.15],
        "treatments": ["Vehicle", "Drug A", "Drug B", "Drug A + Drug B"],
        "dose_map": {"Vehicle": 0, "Drug A": 10, "Drug B": 25, "Drug A + Drug B": 35},
        "treatment_effects": {
            "Vehicle":         [0.25, 0.20, 0.15, 0.20, 0.25, 0.30, 0.18],
            "Drug A":          [0.55, 0.50, 0.35, 0.20, 0.25, 0.28, 0.16],
            "Drug B":          [0.25, 0.20, 0.40, 0.50, 0.55, 0.35, 0.22],
            "Drug A + Drug B": [0.55, 0.50, 0.45, 0.50, 0.55, 0.38, 0.25],
        },
        "summary_markers": ["NeuN", "IBA1", "GFAP", "SOX9", "CD31", "MBP", "Olig2", "MAP2", "Synaptophysin", "TH"],
        "summary_context": "neuroinflammation",
    },

    "Immuno-Oncology (TME)": {
        "channels": ["DAPI", "CD3", "CD8", "FOXP3", "PD-L1", "CD68", "CD20", "PanCK"],
        "channel_labels": {
            "CD3": "Pan T cells",
            "CD8": "Cytotoxic T cells",
            "FOXP3": "Regulatory T cells",
            "PD-L1": "Immune checkpoint ligand",
            "CD68": "Pan macrophages",
            "CD20": "B cells",
            "PanCK": "Tumor / epithelial cells",
        },
        "regions": ["Tumor Core", "Invasive Margin", "Stroma", "TLS"],
        "region_weights": [0.35, 0.25, 0.25, 0.15],
        "treatments": ["Vehicle", "Anti-PD1", "Anti-CTLA4", "Anti-PD1 + Anti-CTLA4"],
        "dose_map": {"Vehicle": 0, "Anti-PD1": 10, "Anti-CTLA4": 10, "Anti-PD1 + Anti-CTLA4": 20},
        "treatment_effects": {
            "Vehicle":                [0.20, 0.10, 0.08, 0.30, 0.25, 0.12, 0.55],
            "Anti-PD1":               [0.45, 0.35, 0.06, 0.12, 0.30, 0.18, 0.40],
            "Anti-CTLA4":             [0.50, 0.25, 0.04, 0.20, 0.35, 0.15, 0.42],
            "Anti-PD1 + Anti-CTLA4":  [0.60, 0.45, 0.03, 0.08, 0.40, 0.22, 0.30],
        },
        "summary_markers": ["CD3", "CD4", "CD8", "FOXP3", "CD20", "CD68", "PD-L1", "PanCK", "Granzyme B", "LAG3"],
        "summary_context": "checkpoint blockade",
    },

    "Breast Cancer": {
        "channels": ["DAPI", "ER", "PR", "HER2", "KI67", "CK5/6", "GATA3"],
        "channel_labels": {
            "ER": "Estrogen receptor",
            "PR": "Progesterone receptor",
            "HER2": "ERBB2-amplified tumor",
            "KI67": "Proliferating cells",
            "CK5/6": "Basal / myoepithelial",
            "GATA3": "Luminal differentiation",
        },
        "regions": ["Invasive Ductal", "DCIS", "Normal Epithelium", "Stroma"],
        "region_weights": [0.40, 0.20, 0.15, 0.25],
        "treatments": ["Vehicle", "Tamoxifen", "Trastuzumab", "Tamoxifen + Trastuzumab"],
        "dose_map": {"Vehicle": 0, "Tamoxifen": 20, "Trastuzumab": 8, "Tamoxifen + Trastuzumab": 28},
        "treatment_effects": {
            "Vehicle":                      [0.60, 0.55, 0.25, 0.40, 0.15, 0.50],
            "Tamoxifen":                    [0.20, 0.18, 0.25, 0.20, 0.15, 0.45],
            "Trastuzumab":                  [0.55, 0.50, 0.08, 0.30, 0.15, 0.48],
            "Tamoxifen + Trastuzumab":      [0.18, 0.15, 0.06, 0.15, 0.12, 0.40],
        },
        "summary_markers": ["ER", "PR", "HER2", "KI67", "CK5/6", "GATA3", "E-Cadherin", "P63", "AR", "CD44"],
        "summary_context": "breast oncology",
    },

    "Lung Cancer": {
        "channels": ["DAPI", "TTF1", "PD-L1", "CK7", "P40", "KI67", "Napsin A", "CD3"],
        "channel_labels": {
            "TTF1": "Adenocarcinoma / type II pneumocytes",
            "PD-L1": "Immune checkpoint ligand",
            "CK7": "Glandular epithelium",
            "P40": "Squamous cell carcinoma",
            "KI67": "Proliferating cells",
            "Napsin A": "Adenocarcinoma (cytoplasmic)",
            "CD3": "Tumor-infiltrating T cells",
        },
        "regions": ["Adenocarcinoma", "Squamous", "Normal Alveolar", "Stroma"],
        "region_weights": [0.35, 0.25, 0.20, 0.20],
        "treatments": ["Vehicle", "Pembrolizumab", "Chemotherapy", "Pembro + Chemo"],
        "dose_map": {"Vehicle": 0, "Pembrolizumab": 200, "Chemotherapy": 75, "Pembro + Chemo": 275},
        "treatment_effects": {
            "Vehicle":          [0.50, 0.35, 0.55, 0.20, 0.45, 0.45, 0.12],
            "Pembrolizumab":    [0.45, 0.12, 0.50, 0.18, 0.30, 0.40, 0.35],
            "Chemotherapy":     [0.30, 0.30, 0.35, 0.12, 0.20, 0.28, 0.15],
            "Pembro + Chemo":   [0.25, 0.08, 0.30, 0.10, 0.15, 0.22, 0.40],
        },
        "summary_markers": ["TTF1", "Napsin A", "P40", "CK7", "CK5/6", "PD-L1", "KI67", "ALK", "CD3", "CD68"],
        "summary_context": "lung oncology",
    },

    "Liver / Hepatology": {
        "channels": ["DAPI", "HepPar1", "Arginase1", "CK19", "CD68", "Alpha-SMA", "CD34", "Glypican3"],
        "channel_labels": {
            "HepPar1": "Hepatocytes",
            "Arginase1": "Hepatocyte (mitochondrial)",
            "CK19": "Bile duct epithelium",
            "CD68": "Kupffer cells",
            "Alpha-SMA": "Stellate cells / myofibroblasts",
            "CD34": "Sinusoidal capillarization",
            "Glypican3": "Hepatocellular carcinoma",
        },
        "regions": ["Hepatocytes", "Portal Tracts", "Sinusoids", "Bile Ducts"],
        "region_weights": [0.40, 0.25, 0.20, 0.15],
        "treatments": ["Vehicle", "CCl4", "Anti-fibrotic", "CCl4 + Anti-fibrotic"],
        "dose_map": {"Vehicle": 0, "CCl4": 1, "Anti-fibrotic": 50, "CCl4 + Anti-fibrotic": 51},
        "treatment_effects": {
            "Vehicle":                 [0.55, 0.50, 0.10, 0.15, 0.08, 0.12, 0.05],
            "CCl4":                    [0.30, 0.25, 0.25, 0.45, 0.50, 0.35, 0.20],
            "Anti-fibrotic":           [0.50, 0.45, 0.12, 0.18, 0.10, 0.14, 0.06],
            "CCl4 + Anti-fibrotic":    [0.38, 0.32, 0.20, 0.30, 0.25, 0.22, 0.12],
        },
        "summary_markers": ["HepPar1", "Arginase1", "CK19", "CK7", "CD34", "CD68", "Alpha-SMA", "Glypican3", "AFP", "Glutamine Synthetase"],
        "summary_context": "hepatology",
    },

    "Kidney / Nephrology": {
        "channels": ["DAPI", "WT1", "Nephrin", "AQP1", "AQP2", "CD31", "Alpha-SMA", "KIM1"],
        "channel_labels": {
            "WT1": "Podocytes",
            "Nephrin": "Slit diaphragm (podocyte health)",
            "AQP1": "Proximal tubule",
            "AQP2": "Collecting duct",
            "CD31": "Endothelial / vasculature",
            "Alpha-SMA": "Myofibroblasts (fibrosis)",
            "KIM1": "Proximal tubule injury",
        },
        "regions": ["Glomerulus", "Proximal Tubule", "Distal Tubule", "Interstitium"],
        "region_weights": [0.25, 0.30, 0.25, 0.20],
        "treatments": ["Vehicle", "UUO", "ACE Inhibitor", "UUO + ACEi"],
        "dose_map": {"Vehicle": 0, "UUO": 0, "ACE Inhibitor": 10, "UUO + ACEi": 10},
        "treatment_effects": {
            "Vehicle":        [0.35, 0.30, 0.50, 0.25, 0.20, 0.05, 0.03],
            "UUO":            [0.15, 0.08, 0.25, 0.12, 0.15, 0.45, 0.40],
            "ACE Inhibitor":  [0.33, 0.28, 0.48, 0.23, 0.22, 0.06, 0.04],
            "UUO + ACEi":     [0.22, 0.15, 0.35, 0.18, 0.18, 0.25, 0.20],
        },
        "summary_markers": ["WT1", "Nephrin", "AQP1", "AQP2", "CD31", "Alpha-SMA", "PAX8", "KIM1", "Tamm-Horsfall", "Podocalyxin"],
        "summary_context": "nephrology",
    },

    "GI / Colon": {
        "channels": ["DAPI", "CDX2", "CK20", "MUC2", "KI67", "CD3", "CD68", "Chromogranin A"],
        "channel_labels": {
            "CDX2": "Intestinal epithelium",
            "CK20": "Colonocytes",
            "MUC2": "Goblet cells",
            "KI67": "Proliferating cells",
            "CD3": "Intraepithelial T cells",
            "CD68": "Lamina propria macrophages",
            "Chromogranin A": "Neuroendocrine cells",
        },
        "regions": ["Epithelium", "Crypts", "Lamina Propria", "Muscularis"],
        "region_weights": [0.30, 0.30, 0.25, 0.15],
        "treatments": ["Vehicle", "DSS Colitis", "Anti-TNF", "DSS + Anti-TNF"],
        "dose_map": {"Vehicle": 0, "DSS Colitis": 3, "Anti-TNF": 5, "DSS + Anti-TNF": 8},
        "treatment_effects": {
            "Vehicle":          [0.55, 0.50, 0.40, 0.25, 0.15, 0.12, 0.08],
            "DSS Colitis":      [0.30, 0.25, 0.15, 0.50, 0.55, 0.45, 0.05],
            "Anti-TNF":         [0.52, 0.48, 0.38, 0.22, 0.12, 0.10, 0.07],
            "DSS + Anti-TNF":   [0.40, 0.38, 0.28, 0.30, 0.30, 0.25, 0.06],
        },
        "summary_markers": ["CDX2", "CK20", "MUC2", "KI67", "CD3", "CD68", "MLH1", "Chromogranin A", "Synaptophysin", "MSH2"],
        "summary_context": "GI pathology",
    },

    "Skin / Dermatology": {
        "channels": ["DAPI", "SOX10", "Melan-A", "CK14", "CD3", "KI67", "S100", "P63"],
        "channel_labels": {
            "SOX10": "Melanocytes / melanoma",
            "Melan-A": "Melanocyte differentiation",
            "CK14": "Basal keratinocytes",
            "CD3": "Dermal T cells",
            "KI67": "Proliferating cells",
            "S100": "Neural crest / melanoma (broad)",
            "P63": "Squamous / basal epithelium",
        },
        "regions": ["Epidermis", "Dermis", "Hair Follicle", "Melanoma"],
        "region_weights": [0.30, 0.30, 0.15, 0.25],
        "treatments": ["Vehicle", "Anti-PD1", "BRAF Inhibitor", "Anti-PD1 + BRAFi"],
        "dose_map": {"Vehicle": 0, "Anti-PD1": 10, "BRAF Inhibitor": 150, "Anti-PD1 + BRAFi": 160},
        "treatment_effects": {
            "Vehicle":              [0.35, 0.30, 0.45, 0.15, 0.40, 0.32, 0.42],
            "Anti-PD1":             [0.30, 0.25, 0.42, 0.45, 0.25, 0.28, 0.40],
            "BRAF Inhibitor":       [0.10, 0.08, 0.40, 0.18, 0.15, 0.10, 0.38],
            "Anti-PD1 + BRAFi":     [0.08, 0.06, 0.38, 0.50, 0.10, 0.08, 0.36],
        },
        "summary_markers": ["SOX10", "Melan-A", "HMB45", "S100", "CK14", "P63", "CD3", "KI67", "PD-L1", "CD68"],
        "summary_context": "dermatopathology",
    },

    "Lymph Node / Hematology": {
        "channels": ["DAPI", "CD20", "CD3", "CD8", "BCL6", "BCL2", "KI67", "CD138"],
        "channel_labels": {
            "CD20": "B cells",
            "CD3": "Pan T cells",
            "CD8": "Cytotoxic T cells",
            "BCL6": "Germinal center B cells",
            "BCL2": "Anti-apoptotic (follicular lymphoma)",
            "KI67": "Proliferating cells",
            "CD138": "Plasma cells",
        },
        "regions": ["Germinal Center", "Mantle Zone", "Paracortex", "Medullary Sinus"],
        "region_weights": [0.30, 0.25, 0.30, 0.15],
        "treatments": ["Vehicle", "R-CHOP", "Ibrutinib", "R-CHOP + Ibrutinib"],
        "dose_map": {"Vehicle": 0, "R-CHOP": 375, "Ibrutinib": 420, "R-CHOP + Ibrutinib": 795},
        "treatment_effects": {
            "Vehicle":               [0.50, 0.30, 0.15, 0.40, 0.35, 0.45, 0.10],
            "R-CHOP":                [0.15, 0.28, 0.12, 0.10, 0.20, 0.20, 0.08],
            "Ibrutinib":             [0.20, 0.25, 0.10, 0.12, 0.18, 0.22, 0.07],
            "R-CHOP + Ibrutinib":    [0.08, 0.22, 0.08, 0.05, 0.10, 0.12, 0.05],
        },
        "summary_markers": ["CD20", "CD3", "CD4", "CD8", "BCL6", "BCL2", "KI67", "CD138", "CD10", "MUM1"],
        "summary_context": "hematology",
    },

    "Pancreas": {
        "channels": ["DAPI", "Insulin", "Glucagon", "Somatostatin", "CK19", "PDX1", "Alpha-SMA", "KI67"],
        "channel_labels": {
            "Insulin": "Beta cells (islets)",
            "Glucagon": "Alpha cells (islets)",
            "Somatostatin": "Delta cells (islets)",
            "CK19": "Ductal epithelium",
            "PDX1": "Beta-cell transcription factor",
            "Alpha-SMA": "Stellate cells / desmoplasia",
            "KI67": "Proliferating cells",
        },
        "regions": ["Islets", "Exocrine Acinar", "Ductal", "Stroma"],
        "region_weights": [0.20, 0.35, 0.20, 0.25],
        "treatments": ["Vehicle", "STZ", "GLP-1 Agonist", "STZ + GLP-1"],
        "dose_map": {"Vehicle": 0, "STZ": 50, "GLP-1 Agonist": 1, "STZ + GLP-1": 51},
        "treatment_effects": {
            "Vehicle":       [0.35, 0.20, 0.12, 0.15, 0.30, 0.08, 0.10],
            "STZ":           [0.08, 0.18, 0.10, 0.20, 0.10, 0.35, 0.25],
            "GLP-1 Agonist": [0.42, 0.22, 0.14, 0.14, 0.38, 0.07, 0.12],
            "STZ + GLP-1":   [0.18, 0.20, 0.12, 0.18, 0.22, 0.20, 0.15],
        },
        "summary_markers": ["Insulin", "Glucagon", "Somatostatin", "CK19", "PDX1", "Amylase", "Alpha-SMA", "KI67", "NKX6.1", "Chromogranin A"],
        "summary_context": "pancreatic pathology",
    },

    "Prostate": {
        "channels": ["DAPI", "PSA", "PSMA", "P63", "AMACR", "AR", "ERG", "KI67"],
        "channel_labels": {
            "PSA": "Luminal epithelium",
            "PSMA": "Prostate membrane antigen",
            "P63": "Basal cells",
            "AMACR": "Adenocarcinoma / HGPIN",
            "AR": "Androgen receptor",
            "ERG": "TMPRSS2-ERG fusion",
            "KI67": "Proliferating cells",
        },
        "regions": ["Luminal Glands", "Basal Epithelium", "Stroma", "Adenocarcinoma"],
        "region_weights": [0.30, 0.20, 0.20, 0.30],
        "treatments": ["Vehicle", "Enzalutamide", "Docetaxel", "Enza + Docetaxel"],
        "dose_map": {"Vehicle": 0, "Enzalutamide": 160, "Docetaxel": 75, "Enza + Docetaxel": 235},
        "treatment_effects": {
            "Vehicle":             [0.55, 0.45, 0.30, 0.35, 0.60, 0.25, 0.40],
            "Enzalutamide":        [0.20, 0.18, 0.28, 0.15, 0.15, 0.22, 0.18],
            "Docetaxel":           [0.30, 0.25, 0.25, 0.20, 0.45, 0.20, 0.15],
            "Enza + Docetaxel":    [0.12, 0.10, 0.22, 0.08, 0.10, 0.18, 0.10],
        },
        "summary_markers": ["PSA", "PSMA", "AR", "P63", "AMACR", "ERG", "CK5/6", "KI67", "Synaptophysin", "NKX3.1"],
        "summary_context": "prostate oncology",
    },

    "Heart / Cardiology": {
        "channels": ["DAPI", "cTnT", "cTnI", "CD31", "Alpha-SMA", "Vimentin", "CD68", "Connexin43"],
        "channel_labels": {
            "cTnT": "Cardiomyocytes (structural)",
            "cTnI": "Cardiomyocytes (injury marker)",
            "CD31": "Endothelial / vasculature",
            "Alpha-SMA": "Smooth muscle / myofibroblasts",
            "Vimentin": "Mesenchymal / fibroblasts",
            "CD68": "Infiltrating macrophages",
            "Connexin43": "Gap junctions / conduction",
        },
        "regions": ["Myocardium", "Endocardium", "Epicardium", "Infarct Zone"],
        "region_weights": [0.35, 0.20, 0.20, 0.25],
        "treatments": ["Vehicle", "MI (Ligation)", "ACE Inhibitor", "MI + ACEi"],
        "dose_map": {"Vehicle": 0, "MI (Ligation)": 0, "ACE Inhibitor": 10, "MI + ACEi": 10},
        "treatment_effects": {
            "Vehicle":          [0.55, 0.50, 0.20, 0.12, 0.10, 0.08, 0.45],
            "MI (Ligation)":    [0.25, 0.15, 0.15, 0.45, 0.40, 0.50, 0.20],
            "ACE Inhibitor":    [0.52, 0.48, 0.22, 0.14, 0.12, 0.10, 0.42],
            "MI + ACEi":        [0.35, 0.30, 0.18, 0.28, 0.25, 0.25, 0.30],
        },
        "summary_markers": ["cTnT", "cTnI", "CD31", "Alpha-SMA", "Vimentin", "CD68", "Connexin43", "CD45", "Desmin", "N-Cadherin"],
        "summary_context": "cardiology",
    },
}

PROFILE_NAMES = list(PROFILES.keys())
DEFAULT_PROFILE = "Neuroscience (Brain)"


# ── Helpers ─────────────────────────────────────────────────────────────────

def _generate_phenotype_combos(n_channels: int = 4) -> list[str]:
    """Generate all phenotype combination column names.

    Matches real histology format: DAPI+ C1+ C2+ C3+ C4+, DAPI+ C1+ C2+ C3+ C4-, etc.
    Plus DAPI+ and DAPI- columns.
    """
    combos = []
    for signs in product(["+", "-"], repeat=n_channels):
        channels = " ".join(f"C{i+1}{s}" for i, s in enumerate(signs))
        combos.append(f"DAPI+ {channels}")
    for signs in product(["+", "-"], repeat=n_channels):
        channels = " ".join(f"C{i+1}{s}" for i, s in enumerate(signs))
        combos.append(f"DAPI- {channels}")
    combos.extend(["DAPI+", "DAPI-"])
    return combos


def _per_channel_columns(channel: str) -> list[str]:
    """Generate per-channel column names matching real histology format."""
    return [
        f"{channel} Positive Classification",
        f"{channel} Positive Nucleus Classification",
        f"{channel} Nucleus Intensity",
        f"{channel} % Nucleus Completeness",
        f"{channel} Cell Intensity",
    ]


def _get_profile(profile: str | None) -> dict:
    """Return profile dict, falling back to default."""
    if profile is None:
        profile = DEFAULT_PROFILE
    return PROFILES[profile]


def _build_image_meta(n_images: int, rng, profile: str | None = None) -> list[dict]:
    """Build per-image metadata assignments for a balanced experimental design.

    Every image gets Treatment Group, Genotype, Timepoint, Subject ID,
    Sex, Age, Dose, and Cohort — no empty fields.
    """
    p = _get_profile(profile)
    treatments = p["treatments"]
    dose_map = p["dose_map"]
    genotypes = ["WT", "KO"]
    timepoints = ["Day 7", "Day 14"]
    sexes = ["M", "F"]
    cohorts = ["Cohort 1", "Cohort 2"]

    n_images = max(n_images, len(treatments))
    meta = []
    for i in range(n_images):
        trt = treatments[i % len(treatments)]
        meta.append({
            "treatment": trt,
            "genotype": genotypes[i % len(genotypes)],
            "timepoint": timepoints[i % len(timepoints)],
            "subject_id": f"SUB-{100 + i}",
            "sex": sexes[i % len(sexes)],
            "age": int(rng.integers(8, 24)),
            "dose": dose_map[trt],
            "cohort": cohorts[i % len(cohorts)],
        })
    return meta


# ── Object Data Generator ───────────────────────────────────────────────────

def generate_object_data(
    n_cells: int = 5000,
    n_images: int = 4,
    profile: str | None = None,
) -> pd.DataFrame:
    """Generate realistic histology object-level data matching the real export format.

    Every row is fully populated — no empty fields. Includes experimental
    metadata columns: Treatment Group, Genotype, Subject ID, Timepoint,
    Sex, Age, Dose, Cohort.
    """
    rng = np.random.default_rng(42)
    p = _get_profile(profile)

    channels = p["channels"]
    regions = p["regions"]
    region_weights = p["region_weights"]
    treatment_effects = p["treatment_effects"]

    image_meta = _build_image_meta(n_images, rng, profile)
    for i, m in enumerate(image_meta):
        m["image_loc"] = f"\\\\server\\halo_data\\Study_001\\Slide_{i+1:03d}.scn"
        m["sample_id"] = f"Slide_{i+1:03d}"

    algorithms = ["Multiplex FL v4.2.3"]
    n_non_dapi = len(channels) - 1  # exclude DAPI
    phenotype_combos = _generate_phenotype_combos(n_non_dapi)

    records = []
    for obj_id in range(n_cells):
        meta = image_meta[rng.integers(0, len(image_meta))]
        region = rng.choice(regions, p=region_weights)

        x_min = rng.uniform(0, 50000)
        y_min = rng.uniform(0, 50000)
        width = rng.uniform(10, 40)
        height = rng.uniform(10, 40)

        row = {
            "Image Location": meta["image_loc"],
            "Sample ID": meta["sample_id"],
            "Analysis Region": region,
            "Algorithm Name": algorithms[0],
            "Object Id": obj_id,
            "XMin": round(x_min, 1),
            "XMax": round(x_min + width, 1),
            "YMin": round(y_min, 1),
            "YMax": round(y_min + height, 1),
            "Treatment Group": meta["treatment"],
            "Genotype": meta["genotype"],
            "Timepoint": meta["timepoint"],
            "Subject ID": meta["subject_id"],
            "Sex": meta["sex"],
            "Age": meta["age"],
            "Dose": meta["dose"],
            "Cohort": meta["cohort"],
        }

        # Phenotype combos with treatment effects
        dapi_positive = rng.random() > 0.05
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

        row["DAPI+"] = 1 if dapi_positive else 0
        row["DAPI-"] = 0 if dapi_positive else 1

        for i, ch in enumerate(channels):
            is_positive = c_positives[i] if i < len(c_positives) else (rng.random() > 0.5)
            row[f"{ch} Positive Classification"] = 1 if is_positive else 0
            row[f"{ch} Positive Nucleus Classification"] = 1 if is_positive else 0
            row[f"{ch} Nucleus Intensity"] = round(rng.lognormal(7.5 if is_positive else 6.5, 0.5), 3)
            row[f"{ch} % Nucleus Completeness"] = round(rng.uniform(60, 100) if is_positive else rng.uniform(0, 100), 5)
            row[f"{ch} Cell Intensity"] = round(rng.lognormal(7.8 if is_positive else 6.8, 0.4), 3)

        nucleus_area = rng.lognormal(4.5, 0.5)
        cytoplasm_area = nucleus_area * rng.uniform(0.3, 2.0)
        cell_area = nucleus_area + cytoplasm_area
        row["Cell Area (\u03bcm\u00b2)"] = round(cell_area, 4)
        row["Cytoplasm Area (\u03bcm\u00b2)"] = round(cytoplasm_area, 4)
        row["Nucleus Area (\u03bcm\u00b2)"] = round(nucleus_area, 4)
        row["Nucleus Perimeter (\u03bcm)"] = round(rng.uniform(15, 120), 2)
        row["Nucleus Roundness"] = round(rng.uniform(0.4, 1.0), 5)

        records.append(row)

    return pd.DataFrame(records)


# ── Summary Data Generator ──────────────────────────────────────────────────

def generate_summary_data(
    n_images: int = 12,
    profile: str | None = None,
) -> pd.DataFrame:
    """Generate realistic histology summary-level data.

    Every row is fully populated — no empty fields. Marker columns are
    derived from the selected profile's summary_markers list.
    """
    rng = np.random.default_rng(42)
    p = _get_profile(profile)

    treatments = p["treatments"]
    dose_map = p["dose_map"]
    regions = p["regions"][:2]  # Use first two regions for summary (e.g. Tumor/Stroma)
    markers = p["summary_markers"]
    algorithms = ["Multiplex IHC v3.2.1", "Highplex FL v4.2.3"]

    image_meta = _build_image_meta(n_images, rng, profile)
    for i, m in enumerate(image_meta):
        trt = treatments[i % len(treatments)]
        m["treatment"] = trt
        m["dose"] = dose_map[trt]
        m["image_tag"] = f"Patient_{i+1:03d}_Biopsy.scn"
        m["sample_id"] = f"Patient_{i+1:03d}_Biopsy"

    # Build per-treatment multipliers from the profile's treatment_effects
    # Map effect rates to multipliers relative to the first treatment (vehicle)
    base_effects = p["treatment_effects"]
    vehicle_key = treatments[0]
    vehicle_rates = base_effects[vehicle_key]
    avg_vehicle = sum(vehicle_rates) / len(vehicle_rates) if vehicle_rates else 0.2
    treatment_multipliers = {}
    for trt, rates in base_effects.items():
        avg_rate = sum(rates) / len(rates)
        treatment_multipliers[trt] = max(avg_rate / avg_vehicle, 0.3) if avg_vehicle > 0 else 1.0

    records = []
    for meta in image_meta:
        mult = treatment_multipliers[meta["treatment"]]
        algo = rng.choice(algorithms)
        job_id = int(rng.integers(100000, 999999))

        for region in regions:
            total_cells = int(rng.integers(500, 15000))
            region_area = rng.uniform(50000, 500000)
            analyzed_area = rng.uniform(0.5, 5.0)

            row = {
                "Image Tag": meta["image_tag"],
                "Sample ID": meta["sample_id"],
                "Algorithm Name": algo,
                "Job Id": job_id,
                "Analysis Region": region,
                "Treatment Group": meta["treatment"],
                "Genotype": meta["genotype"],
                "Timepoint": meta["timepoint"],
                "Subject ID": meta["subject_id"],
                "Sex": meta["sex"],
                "Age": meta["age"],
                "Dose": meta["dose"],
                "Cohort": meta["cohort"],
                "Total Cells": total_cells,
            }

            # Generate marker-specific columns
            for mk in markers:
                base_pct = rng.uniform(3, 50)
                # Apply treatment effect with region variation
                region_factor = 1.2 if region == regions[0] else 0.8
                pct = min(base_pct * mult * region_factor, 95)
                count = int(total_cells * pct / 100)
                row[f"{mk}+ Cells"] = count
                row[f"% {mk}+ Cells"] = round(pct, 2)

            # Add intensity / H-score / area columns using first two markers
            primary_mk = markers[0] if markers else "Marker"
            secondary_mk = markers[1] if len(markers) > 1 else primary_mk
            weak_pct = rng.uniform(20, 50)
            mod_pct = rng.uniform(20, 50)
            strong_pct = 100 - weak_pct - mod_pct
            primary_pct = row.get(f"% {primary_mk}+ Cells", 20)

            row[f"% {primary_mk} Weak Cells"] = round(weak_pct * primary_pct / 100, 2)
            row[f"% {primary_mk} Moderate Cells"] = round(mod_pct * primary_pct / 100, 2)
            row[f"% {primary_mk} Strong Cells"] = round(strong_pct * primary_pct / 100, 2)
            row["% Negative Cells"] = round(max(100 - sum(
                row[f"% {mk}+ Cells"] for mk in markers[:4]
            ), 0), 2)
            row[f"{primary_mk} H-Score"] = round(rng.uniform(0, 300) * mult, 1)
            row[f"{secondary_mk} H-Score"] = round(rng.uniform(0, 300) * mult, 1)
            row["DAPI Nucleus Intensity"] = round(rng.lognormal(5.0, 0.3), 2)
            row["Region Area (\u03bcm\u00b2)"] = round(region_area, 2)
            row["Analyzed Area (mm^2)"] = round(analyzed_area, 3)
            row["Cell Density (cells/mm^2)"] = round(total_cells / analyzed_area, 1)

            records.append(row)

    return pd.DataFrame(records)


# ── Cluster Data Generator ──────────────────────────────────────────────────

def generate_cluster_data(
    n_clusters: int = 200,
    n_images: int = 4,
    profile: str | None = None,
) -> pd.DataFrame:
    """Generate realistic histology cluster/aggregated object data.

    Every row is fully populated — no empty fields. Marker columns are
    derived from the selected profile's summary_markers list.
    """
    rng = np.random.default_rng(42)
    p = _get_profile(profile)

    markers = p["summary_markers"]
    treatments = p["treatments"]
    base_effects = p["treatment_effects"]
    vehicle_key = treatments[0]
    vehicle_rates = base_effects[vehicle_key]
    avg_vehicle = sum(vehicle_rates) / len(vehicle_rates) if vehicle_rates else 0.2
    treatment_multipliers = {}
    for trt, rates in base_effects.items():
        avg_rate = sum(rates) / len(rates)
        treatment_multipliers[trt] = max(avg_rate / avg_vehicle, 0.3) if avg_vehicle > 0 else 1.0

    algorithms = ["Multiplex IHC v3.2.1"]

    image_meta = _build_image_meta(n_images, rng, profile)
    for i, m in enumerate(image_meta):
        m["sample_id"] = f"Sample_{i+1:03d}"

    records = []
    for _ in range(n_clusters):
        meta = image_meta[rng.integers(0, len(image_meta))]
        mult = treatment_multipliers[meta["treatment"]]
        total_cells = int(rng.integers(10, 500))
        region_area = rng.uniform(1000, 50000)

        row = {
            "Sample ID": meta["sample_id"],
            "Algorithm Name": algorithms[0],
            "Job ID": int(rng.integers(100000, 999999)),
            "Treatment Group": meta["treatment"],
            "Genotype": meta["genotype"],
            "Timepoint": meta["timepoint"],
            "Subject ID": meta["subject_id"],
            "Sex": meta["sex"],
            "Age": meta["age"],
            "Dose": meta["dose"],
            "Cohort": meta["cohort"],
            "Total Cluster Count": int(rng.integers(1, 50)),
            "Total Cell Count": total_cells,
            "Total Area Analyzed": round(region_area, 2),
            "Total Cells": total_cells,
            "Region Area (\u03bcm\u00b2)": round(region_area, 2),
        }

        # Generate per-marker percentage columns
        for mk in markers:
            base_pct = rng.uniform(0, 50)
            row[f"% {mk}+ Cells"] = round(min(base_pct * mult, 95), 2)

        # Add primary marker H-score and DAPI intensity
        primary_mk = markers[0] if markers else "Marker"
        row[f"{primary_mk} H-Score"] = round(rng.uniform(0, 300) * mult, 1)
        row["DAPI Nucleus Intensity"] = round(rng.lognormal(5.0, 0.3), 2)

        records.append(row)

    return pd.DataFrame(records)
