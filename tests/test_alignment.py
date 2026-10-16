import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from alignment.kca import (
    compute_kca_tensor, compute_po_attainment,
    build_copo_matrix, format_matrix_table,
    BLOOMS_KCA, PO_KCA_WEIGHTS
)
from agents.alignment import run_alignment_agent


def test_kca_tensor_all_levels():
    """Test KCA tensor values for all Bloom's levels."""
    for level in ["remember", "understand", "apply",
                  "analyze", "evaluate", "create"]:
        kca = compute_kca_tensor(level)
        assert "K" in kca
        assert "C" in kca
        assert "A" in kca
        assert 1 <= kca["K"] <= 3
        assert 1 <= kca["C"] <= 3
        assert 1 <= kca["A"] <= 3
    print("\nKCA tensors valid for all Bloom's levels")


def test_higher_blooms_gives_higher_kca():
    """Test that higher Bloom's levels give higher KCA scores."""
    remember_kca = compute_kca_tensor("remember")
    evaluate_kca = compute_kca_tensor("evaluate")
    total_remember = sum(remember_kca.values())
    total_evaluate = sum(evaluate_kca.values())
    assert total_evaluate >= total_remember
    print(f"\nRemember KCA total: {total_remember}, "
          f"Evaluate KCA total: {total_evaluate}")


def test_po_attainment_range():
    """Test that PO attainment scores are in 0-3 range."""
    kca = {"K": 3, "C": 3, "A": 3}
    for po in PO_KCA_WEIGHTS.keys():
        score = compute_po_attainment(kca, po)
        assert 0 <= score <= 3, f"Score out of range for {po}: {score}"
    print("\nAll PO attainment scores in valid range 0-3")


def test_copo_matrix_dimensions():
    """Test CO-PO matrix has correct dimensions."""
    clos = [
        "Explain SDN principles",
        "Design SDN topology",
        "Implement SDN controller"
    ]
    levels = ["understand", "create", "apply"]
    matrix_data = build_copo_matrix(clos, levels)

    assert len(matrix_data["matrix"]) == 3
    for row in matrix_data["matrix"]:
        assert len(row["po_scores"]) == 12
    print(f"\nMatrix dimensions: {len(clos)} CLOs × 12 POs")


def test_format_matrix_table():
    """Test that matrix table formats correctly."""
    clos = ["Explain SDN", "Design SDN"]
    levels = ["understand", "create"]
    matrix_data = build_copo_matrix(clos, levels)
    table = format_matrix_table(matrix_data)

    assert "CLO" in table
    assert "PO1" in table
    assert "PO12" in table
    assert "3=High" in table
    print("\nMatrix table formatted correctly")


def test_alignment_agent_output():
    """Test full Alignment Agent output."""
    clos = [
        "Explain the principles of SDN",
        "Design an SDN network topology",
        "Implement a basic SDN controller",
        "Analyze SDN performance",
        "Evaluate SDN security"
    ]
    result = run_alignment_agent(clos)

    assert len(result.clos) == 5
    assert len(result.blooms_levels) == 5
    assert len(result.po_keys) == 12
    assert len(result.matrix) == 5
    assert len(result.formatted_table) > 0

    for row in result.matrix:
        for po, score in row["po_scores"].items():
            assert 0 <= score <= 3

    print(f"\nAlignment Agent output verified:")
    print(f"  CLOs: {len(result.clos)}")
    print(f"  POs: {len(result.po_keys)}")
    print(f"\n{result.formatted_table}")


def test_blooms_levels_detected_correctly():
    """Test Bloom's level detection from CLO verbs."""
    clos = [
        "Explain the SDN architecture",
        "Implement an SDN controller",
        "Evaluate SDN performance"
    ]
    result = run_alignment_agent(clos)

    assert result.blooms_levels[0] == "understand"
    assert result.blooms_levels[1] == "apply"
    assert result.blooms_levels[2] == "evaluate"
    print(f"\nBloom's levels: {result.blooms_levels}")