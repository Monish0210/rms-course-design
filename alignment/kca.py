import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from loguru import logger


# ── Bloom's Level to KCA Score Mapping ───────────────────
# K = Knowledge depth, C = Cognitive level, A = Application
# Scores 1-3: 1=Low, 2=Medium, 3=High

BLOOMS_KCA = {
    "remember":   {"K": 1, "C": 1, "A": 1},
    "understand": {"K": 2, "C": 1, "A": 1},
    "apply":      {"K": 2, "C": 2, "A": 3},
    "analyze":    {"K": 3, "C": 3, "A": 2},
    "evaluate":   {"K": 3, "C": 3, "A": 3},
    "create":     {"K": 3, "C": 3, "A": 3},
    "unknown_verb": {"K": 1, "C": 1, "A": 1},
}

# ── PO Relevance Weights ──────────────────────────────────
# How much each KCA dimension contributes to each PO
# Scale 0-3: 0=None, 1=Low, 2=Medium, 3=High

PO_KCA_WEIGHTS = {
    "PO1":  {"K": 3, "C": 2, "A": 1},  # Engineering Knowledge
    "PO2":  {"K": 2, "C": 3, "A": 2},  # Problem Analysis
    "PO3":  {"K": 2, "C": 2, "A": 3},  # Design/Development
    "PO4":  {"K": 2, "C": 3, "A": 2},  # Investigations
    "PO5":  {"K": 1, "C": 2, "A": 3},  # Modern Tool Usage
    "PO6":  {"K": 1, "C": 2, "A": 1},  # Engineer and Society
    "PO7":  {"K": 1, "C": 1, "A": 1},  # Environment
    "PO8":  {"K": 1, "C": 1, "A": 1},  # Ethics
    "PO9":  {"K": 1, "C": 1, "A": 2},  # Team Work
    "PO10": {"K": 1, "C": 2, "A": 1},  # Communication
    "PO11": {"K": 1, "C": 2, "A": 2},  # Project Management
    "PO12": {"K": 2, "C": 2, "A": 1},  # Life Long Learning
}


def compute_kca_tensor(blooms_level: str) -> dict:
    """
    Compute KCA tensor T=[K,C,A] for a given Bloom's level.
    Returns dict with K, C, A scores (1-3).
    """
    return BLOOMS_KCA.get(blooms_level, BLOOMS_KCA["remember"])


def compute_po_attainment(kca: dict, po_key: str) -> int:
    """
    Compute attainment weight for one CLO against one PO.
    Uses dot product of KCA tensor and PO weight vector.
    Maps result to 0-3 scale.
    """
    weights = PO_KCA_WEIGHTS[po_key]

    # Weighted sum
    score = (
        kca["K"] * weights["K"] +
        kca["C"] * weights["C"] +
        kca["A"] * weights["A"]
    )

    # Normalize to 0-3 scale
    # Max possible score = 3*3 + 3*3 + 3*3 = 27
    # Thresholds: 0-6=0, 7-12=1, 13-18=2, 19-27=3
    if score <= 6:
        return 0
    elif score <= 12:
        return 1
    elif score <= 18:
        return 2
    else:
        return 3


def build_copo_matrix(clos: list[str],
                      blooms_levels: list[str],
                      po_keys: list[str] = None) -> dict:
    """
    Build the complete CO-PO attainment matrix.

    Args:
        clos: List of CLO strings
        blooms_levels: Bloom's level for each CLO
        po_keys: List of PO keys to include (default: all 12)

    Returns:
        Dict with matrix data and metadata
    """
    if po_keys is None:
        po_keys = list(PO_KCA_WEIGHTS.keys())

    matrix = []
    for i, (clo, level) in enumerate(zip(clos, blooms_levels)):
        kca = compute_kca_tensor(level)
        row = {
            "clo_index": i + 1,
            "clo": clo,
            "blooms_level": level,
            "kca": kca,
            "po_scores": {}
        }
        for po in po_keys:
            row["po_scores"][po] = compute_po_attainment(kca, po)
        matrix.append(row)

    return {
        "clos": clos,
        "po_keys": po_keys,
        "matrix": matrix
    }


def format_matrix_table(matrix_data: dict) -> str:
    """Format CO-PO matrix as a readable table string."""
    po_keys = matrix_data["po_keys"]
    matrix = matrix_data["matrix"]

    # Header
    header = f"{'CLO':<8} " + " ".join(f"{po:<5}" for po in po_keys)
    separator = "-" * len(header)

    rows = [header, separator]
    for row in matrix:
        scores = " ".join(
            f"{row['po_scores'][po]:<5}" for po in po_keys
        )
        rows.append(f"CLO{row['clo_index']:<5} {scores}")

    rows.append(separator)
    rows.append("Scale: 3=High  2=Medium  1=Low  0=No mapping")
    return "\n".join(rows)