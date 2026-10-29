import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from agents.reviewer import (
    get_first_verb, check_blooms_level,
    review_clo, run_reviewer_agent,
    CLOReview, ReviewResult
)


def test_get_first_verb():
    """Test first verb extraction from CLO."""
    assert get_first_verb("Explain the SDN architecture") == "explain"
    assert get_first_verb("Design a network topology") == "design"
    assert get_first_verb("Implement OpenFlow controller") == "implement"
    assert get_first_verb("") == ""
    print("\nFirst verb extraction working")


def test_blooms_level_known_verbs():
    """Test that known Bloom's verbs are correctly identified."""
    assert check_blooms_level("explain") == "understand"
    assert check_blooms_level("design") == "create"
    assert check_blooms_level("implement") == "apply"
    assert check_blooms_level("analyze") == "analyze"
    assert check_blooms_level("evaluate") == "evaluate"
    assert check_blooms_level("create") == "create"
    assert check_blooms_level("define") == "remember"
    print("\nBloom's verb mapping verified")


def test_blooms_level_unknown_verb():
    """Test that non-Bloom's verbs return None or unknown_verb."""
    # These are verbs but not in Bloom's approved list
    # They may return None or unknown_verb depending on NLTK POS tagging
    result_study = check_blooms_level("study")
    result_know = check_blooms_level("know")
    result_random = check_blooms_level("xyzabc")

    # Must not return a valid Bloom's level
    valid_levels = {"remember", "understand", "apply",
                    "analyze", "evaluate", "create"}
    assert result_study not in valid_levels
    assert result_know not in valid_levels
    assert result_random not in valid_levels
    print(f"\nNon-Bloom's verbs correctly rejected: "
          f"study={result_study}, know={result_know}")


def test_review_clo_pass():
    """Test that good CLOs pass review."""
    good_clos = [
        "Explain the principles of SDN",
        "Design an SDN network topology",
        "Implement a basic SDN controller",
        "Analyze SDN network performance",
        "Evaluate SDN security implications"
    ]
    for clo in good_clos:
        review = review_clo(clo)
        assert review.passed is True, f"Should pass: {clo}"
        assert review.blooms_level is not None
    print(f"\nAll {len(good_clos)} good CLOs passed")


def test_review_clo_fail():
    """Test that bad CLOs fail review."""
    bad_clos = [
        "Study the basics of SDN",
        "Write a note on OpenFlow",
        "Know the SDN architecture",
        "Learn about SDN controllers"
    ]
    for clo in bad_clos:
        review = review_clo(clo)
        assert review.passed is False, f"Should fail: {clo}"
        assert review.feedback != ""
    print(f"\nAll {len(bad_clos)} bad CLOs correctly failed")


def test_review_clo_rejects_incomplete_statement():
    """Test that a Bloom verb alone is not accepted as a CLO."""
    review = review_clo("Explain")
    assert review.passed is False
    assert "incomplete" in review.feedback.lower()
    print("\nIncomplete CLO correctly rejected")


def test_reviewer_agent_all_pass():
    """Test reviewer with all valid CLOs."""
    clos = [
        "Explain the principles of Software Defined Networking",
        "Design an SDN network topology using OpenFlow",
        "Implement a basic SDN controller",
        "Analyze the performance of SDN networks",
        "Evaluate security implications of SDN"
    ]
    result = run_reviewer_agent(clos)
    assert result.all_passed is True
    assert len(result.passed_clos) == 5
    assert len(result.failed_clos) == 0
    print(f"\nAll 5 CLOs passed reviewer")


def test_reviewer_agent_some_fail():
    """Test reviewer with mixed CLOs."""
    clos = [
        "Explain the SDN architecture",
        "Study the OpenFlow protocol",
        "Implement SDN controllers",
        "Know the difference between SDN and traditional networking",
        "Evaluate SDN performance"
    ]
    result = run_reviewer_agent(clos)
    assert result.all_passed is False
    assert len(result.passed_clos) == 3
    assert len(result.failed_clos) == 2
    assert result.feedback_summary != ""
    print(f"\nMixed result: {len(result.passed_clos)} passed, "
          f"{len(result.failed_clos)} failed")


def test_feedback_summary_content():
    """Test that feedback summary contains useful information."""
    clos = ["Study the basics of SDN"]
    result = run_reviewer_agent(clos)
    assert result.all_passed is False
    assert "study" in result.feedback_summary.lower()
    print(f"\nFeedback: {result.feedback_summary}")