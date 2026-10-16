import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from loguru import logger
from pydantic import BaseModel
from alignment.kca import build_copo_matrix, format_matrix_table
from agents.reviewer import run_reviewer_agent, VERB_TO_LEVEL


# ── Output Schema ─────────────────────────────────────────
class AlignmentResult(BaseModel):
    clos: list[str]
    blooms_levels: list[str]
    po_keys: list[str]
    matrix: list[dict]
    formatted_table: str


# ── Alignment Agent ───────────────────────────────────────
def run_alignment_agent(clos: list[str],
                        po_keys: list[str] = None) -> AlignmentResult:
    """
    Alignment Agent — computes CO-PO matrix using KCA tensor.

    Args:
        clos: Approved CLOs from Reviewer Agent
        po_keys: PO keys to map against (default: all 12)

    Returns:
        AlignmentResult with full CO-PO matrix
    """
    logger.info(f"Alignment Agent processing {len(clos)} CLOs")

    # Get Bloom's level for each CLO
    blooms_levels = []
    for clo in clos:
        first_word = clo.strip().split()[0].lower().rstrip(".,;:")
        level = VERB_TO_LEVEL.get(first_word, "remember")
        blooms_levels.append(level)
        logger.info(f"CLO: '{clo[:40]}...' → Bloom's: {level}")

    # Build CO-PO matrix
    matrix_data = build_copo_matrix(clos, blooms_levels, po_keys)

    # Format table
    table = format_matrix_table(matrix_data)

    logger.info("CO-PO matrix built successfully")

    return AlignmentResult(
        clos=clos,
        blooms_levels=blooms_levels,
        po_keys=matrix_data["po_keys"],
        matrix=matrix_data["matrix"],
        formatted_table=table
    )


# ── Main ──────────────────────────────────────────────────
if __name__ == "__main__":

    test_clos = [
        "Explain the principles of Software Defined Networking",
        "Design an SDN network topology using OpenFlow",
        "Implement a basic SDN controller using Python",
        "Analyze the performance of SDN-based networks",
        "Evaluate the security implications of SDN deployment"
    ]

    result = run_alignment_agent(test_clos)

    print("\n" + "="*60)
    print("ALIGNMENT AGENT OUTPUT — CO-PO MATRIX")
    print("="*60)
    print(f"\nCLOs processed: {len(result.clos)}")
    print(f"POs mapped: {len(result.po_keys)}")

    print("\nBloom's levels detected:")
    for clo, level in zip(result.clos, result.blooms_levels):
        print(f"  {level:<12} → {clo[:50]}")

    print("\nCO-PO ATTAINMENT MATRIX:")
    print(result.formatted_table)