import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from agents.creator import run_creator_agent, CourseStructure
from agents.reviewer import run_reviewer_agent, ReviewResult
from rag.retriever import get_retriever, retrieve, format_context
from config.settings import MAX_REVISION_LOOPS, QDRANT_COLLECTION


# ── State Definition ──────────────────────────────────────
class WorkflowState:
    def __init__(self, subject: str, level: str,
                 num_modules: int = 5):
        self.subject = subject
        self.level = level
        self.num_modules = num_modules
        self.context = ""
        self.course = None
        self.review = None
        self.revision_count = 0
        self.feedback = None
        self.status = "initialized"
        self.hallucination_log = []


# ── Node Functions ────────────────────────────────────────
def node_retrieve(state: WorkflowState,
                  model, client) -> WorkflowState:
    """Node 1 — Retrieve domain context from Qdrant."""
    logger.info(f"NODE: retrieve — subject: {state.subject}")

    query = f"{state.subject} architecture principles applications"
    chunks = retrieve(query, model, client,
                      collection_name=QDRANT_COLLECTION,
                      top_k=2)
    state.context = format_context(chunks)
    state.status = "retrieved"

    logger.info(f"Retrieved context: {len(state.context)} characters")
    return state


def node_creator(state: WorkflowState) -> WorkflowState:
    """Node 2 — Creator Agent generates course structure."""
    logger.info(f"NODE: creator — revision {state.revision_count}")

    state.course = run_creator_agent(
        subject=state.subject,
        level=state.level,
        context=state.context,
        num_modules=state.num_modules,
        feedback=state.feedback
    )
    state.status = "created"

    logger.info(f"Creator produced {len(state.course.clos)} CLOs")
    return state


def node_reviewer(state: WorkflowState) -> WorkflowState:
    """Node 3 — Reviewer Agent validates CLOs."""
    logger.info("NODE: reviewer")

    state.review = run_reviewer_agent(state.course.clos)

    # Log hallucination rate
    total = len(state.course.clos)
    failed = len(state.review.failed_clos)
    pass_rate = (total - failed) / total * 100
    state.hallucination_log.append({
        "revision": state.revision_count,
        "total_clos": total,
        "failed_clos": failed,
        "pass_rate": round(pass_rate, 1)
    })

    logger.info(
        f"Reviewer: {len(state.review.passed_clos)} passed, "
        f"{len(state.review.failed_clos)} failed"
    )
    return state


def should_revise(state: WorkflowState) -> bool:
    """
    Conditional edge — decide whether to loop back to Creator.
    Returns True if revision needed, False if done.
    """
    if state.review.all_passed:
        logger.info("All CLOs passed — moving to Alignment Agent")
        return False

    if state.revision_count >= MAX_REVISION_LOOPS:
        logger.warning(
            f"Max revision loops ({MAX_REVISION_LOOPS}) reached. "
            f"Proceeding with best available CLOs."
        )
        # Use only the passed CLOs if any, else keep all
        if state.review.passed_clos:
            state.course.clos = state.review.passed_clos
        return False

    # Revision needed
    state.revision_count += 1
    state.feedback = state.review.feedback_summary
    logger.info(
        f"Revision loop {state.revision_count}/{MAX_REVISION_LOOPS} — "
        f"sending feedback to Creator"
    )
    return True


# ── Main Workflow ─────────────────────────────────────────
def run_workflow(subject: str,
                 level: str,
                 num_modules: int = 5) -> WorkflowState:
    """
    Run the complete Creator-Reviewer workflow.

    This implements the LangGraph-style stateful pipeline:
    retrieve → creator → reviewer → [revise?] → alignment

    Args:
        subject: Course subject name
        level: Academic level
        num_modules: Number of modules to generate

    Returns:
        Final WorkflowState with approved course structure
    """
    logger.info(f"Starting workflow for: {subject}")

    # Initialize state
    state = WorkflowState(subject, level, num_modules)

    # Load retriever once
    model, client = get_retriever()

    # Node 1 — Retrieve
    state = node_retrieve(state, model, client)

    # Node 2 — Creator → Node 3 — Reviewer → Loop if needed
    while True:
        # Creator generates course
        state = node_creator(state)

        # Reviewer checks CLOs
        state = node_reviewer(state)

        # Conditional edge — revise or proceed
        if not should_revise(state):
            break

    state.status = "approved"
    logger.info(
        f"Workflow complete. "
        f"Revisions: {state.revision_count}. "
        f"Final CLOs: {len(state.course.clos)}"
    )
    return state


# ── Main ──────────────────────────────────────────────────
if __name__ == "__main__":

    state = run_workflow(
        subject="Software Defined Networking",
        level="B.Tech 3rd Year",
        num_modules=5
    )

    print("\n" + "="*60)
    print("WORKFLOW COMPLETE")
    print("="*60)

    print(f"\nStatus: {state.status}")
    print(f"Total revisions: {state.revision_count}")

    print("\nFINAL MODULES:")
    for i, m in enumerate(state.course.modules, 1):
        print(f"  {i}. {m}")

    print("\nAPPROVED CLOs:")
    for i, clo in enumerate(state.course.clos, 1):
        print(f"  CLO{i}: {clo}")

    print("\nPREREQUISITES:")
    for p in state.course.prerequisites:
        print(f"  - {p}")

    print("\nHALLUCINATION LOG:")
    for entry in state.hallucination_log:
        print(f"  Revision {entry['revision']}: "
              f"{entry['failed_clos']} failed / "
              f"{entry['total_clos']} total "
              f"({entry['pass_rate']}% pass rate)")