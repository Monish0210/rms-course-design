import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from workflow.graph import (
    WorkflowState, node_reviewer,
    should_revise, run_workflow
)
from agents.creator import CourseStructure
from agents.reviewer import ReviewResult, CLOReview


def test_workflow_state_initializes():
    """Test WorkflowState initializes correctly."""
    state = WorkflowState(
        subject="Software Defined Networking",
        level="B.Tech 3rd Year",
        num_modules=5
    )
    assert state.subject == "Software Defined Networking"
    assert state.revision_count == 0
    assert state.status == "initialized"
    assert state.feedback is None
    print("\nWorkflowState initialized correctly")


def test_should_revise_when_all_pass():
    """Test conditional edge returns False when all CLOs pass."""
    state = WorkflowState("SDN", "B.Tech")
    state.revision_count = 0
    state.review = ReviewResult(
        all_passed=True,
        reviews=[],
        passed_clos=["Explain SDN", "Design SDN"],
        failed_clos=[],
        feedback_summary="All passed"
    )
    result = should_revise(state)
    assert result is False
    print("\nNo revision needed when all CLOs pass")


def test_should_revise_when_clos_fail():
    """Test conditional edge returns True when CLOs fail."""
    state = WorkflowState("SDN", "B.Tech")
    state.revision_count = 0
    state.course = CourseStructure(
        modules=["M1"], clos=["Explain SDN", "Study SDN"],
        prerequisites=["Networking"]
    )
    state.review = ReviewResult(
        all_passed=False,
        reviews=[],
        passed_clos=["Explain SDN"],
        failed_clos=["Study SDN"],
        feedback_summary="CLO 'Study SDN': FAIL"
    )
    result = should_revise(state)
    assert result is True
    assert state.revision_count == 1
    assert state.feedback is not None
    print("\nRevision triggered correctly")


def test_should_not_revise_at_max_loops():
    """Test that revision stops at MAX_REVISION_LOOPS."""
    from config.settings import MAX_REVISION_LOOPS
    state = WorkflowState("SDN", "B.Tech")
    state.revision_count = MAX_REVISION_LOOPS
    state.course = CourseStructure(
        modules=["M1"], clos=["Explain SDN"],
        prerequisites=["Networking"]
    )
    state.review = ReviewResult(
        all_passed=False,
        reviews=[],
        passed_clos=["Explain SDN"],
        failed_clos=["Study SDN"],
        feedback_summary="Still failing"
    )
    result = should_revise(state)
    assert result is False
    print(f"\nRevision correctly stopped at max loops ({MAX_REVISION_LOOPS})")


def test_full_workflow_runs():
    """
    Integration test — runs the complete workflow end to end.
    Takes 2-4 minutes due to Ollama.
    """
    state = run_workflow(
        subject="Software Defined Networking",
        level="B.Tech 3rd Year",
        num_modules=5
    )

    assert state.status == "approved"
    assert state.course is not None
    assert len(state.course.modules) == 5
    assert len(state.course.clos) >= 1
    assert len(state.hallucination_log) >= 1

    print(f"\nWorkflow complete:")
    print(f"  Status: {state.status}")
    print(f"  Revisions: {state.revision_count}")
    print(f"  Final CLOs: {len(state.course.clos)}")
    print(f"  Hallucination log: {state.hallucination_log}")