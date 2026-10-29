import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nltk
from loguru import logger
from pydantic import BaseModel


# ── Bloom's Taxonomy Verb Dictionary ─────────────────────
BLOOMS_VERBS = {
    "remember": [
        "define", "list", "recall", "recognize", "identify",
        "name", "state", "describe", "label", "match",
        "memorize", "repeat", "reproduce", "select"
    ],
    "understand": [
        "explain", "summarize", "interpret", "classify",
        "compare", "discuss", "distinguish", "estimate",
        "extend", "generalize", "give", "illustrate",
        "infer", "outline", "paraphrase", "predict",
        "rewrite", "translate"
    ],
    "apply": [
        "implement", "use", "execute", "solve", "apply",
        "demonstrate", "calculate", "complete", "examine",
        "modify", "operate", "show", "sketch",
        "practice", "construct", "produce", "develop"
    ],
    "analyze": [
        "analyze", "differentiate", "organize", "attribute",
        "deconstruct", "distinguish", "examine", "experiment",
        "question", "test", "compare", "contrast",
        "criticize", "inspect", "investigate", "separate"
    ],
    "evaluate": [
        "evaluate", "judge", "justify", "assess", "argue",
        "defend", "select", "support", "value", "appraise",
        "critique", "prioritize", "recommend", "validate"
    ],
    "create": [
        "create", "design", "construct", "develop", "formulate",
        "build", "generate", "plan", "produce", "propose",
        "assemble", "compose", "devise", "invent", "make"
    ]
}

# Flat lookup: verb → bloom's level
VERB_TO_LEVEL = {}
for level, verbs in BLOOMS_VERBS.items():
    for verb in verbs:
        VERB_TO_LEVEL[verb.lower()] = level


# ── Output Schema ─────────────────────────────────────────
class CLOReview(BaseModel):
    clo: str
    passed: bool
    blooms_level: str | None
    first_verb: str
    feedback: str


class ReviewResult(BaseModel):
    all_passed: bool
    reviews: list[CLOReview]
    passed_clos: list[str]
    failed_clos: list[str]
    feedback_summary: str


# ── Bloom's Checker ───────────────────────────────────────
def get_first_verb(clo: str) -> str:
    """Extract the first word of a CLO as the action verb."""
    words = clo.strip().split()
    if not words:
        return ""
    return words[0].lower().rstrip(".,;:")


def check_blooms_level(verb: str) -> str | None:
    """
    Return the Bloom's level for a verb, or None if not found.
    Uses direct dictionary lookup as primary method.
    NLTK POS tag used only as secondary confirmation.
    """
    verb_clean = verb.lower().rstrip(".,;:")

    # Primary check — direct Bloom's verb list lookup
    if verb_clean in VERB_TO_LEVEL:
        return VERB_TO_LEVEL[verb_clean]

    # Secondary check — NLTK POS tag to confirm it is a verb
    try:
        tokens = nltk.word_tokenize(f"{verb_clean} the concept")
        tags = nltk.pos_tag(tokens)
        pos = tags[0][1] if tags else ""
        if pos.startswith("VB"):
            return "unknown_verb"
    except Exception:
        pass

    return None


def review_clo(clo: str) -> CLOReview:
    """
    Review a single CLO against Bloom's Taxonomy.
    Returns a CLOReview with pass/fail verdict and feedback.
    """
    first_verb = get_first_verb(clo)
    blooms_level = check_blooms_level(first_verb)
    word_count = len(clo.strip().split())

    if word_count < 3:
        return CLOReview(
            clo=clo,
            passed=False,
            blooms_level=blooms_level if blooms_level != "unknown_verb" else None,
            first_verb=first_verb,
            feedback=(
                "FAIL — the CLO is incomplete. It must be a specific, "
                "assessable statement of at least 3 words describing what "
                "the student will be able to do."
            )
        )

    if blooms_level and blooms_level != "unknown_verb":
        return CLOReview(
            clo=clo,
            passed=True,
            blooms_level=blooms_level,
            first_verb=first_verb,
            feedback=f"PASS — '{first_verb}' maps to Bloom's '{blooms_level}' level"
        )
    elif blooms_level == "unknown_verb":
        return CLOReview(
            clo=clo,
            passed=False,
            blooms_level=None,
            first_verb=first_verb,
            feedback=(
                f"FAIL — '{first_verb}' is a verb but not in Bloom's approved list. "
                f"Replace with one of: explain, design, implement, analyze, evaluate, create"
            )
        )
    else:
        return CLOReview(
            clo=clo,
            passed=False,
            blooms_level=None,
            first_verb=first_verb,
            feedback=(
                f"FAIL — '{first_verb}' is not a recognized Bloom's Taxonomy verb. "
                f"CLO must start with: explain, design, implement, analyze, evaluate, "
                f"create, describe, compare, assess, demonstrate, or similar action verb"
            )
        )


# ── Reviewer Agent ────────────────────────────────────────
def run_reviewer_agent(clos: list[str]) -> ReviewResult:
    """
    Main Reviewer Agent function.
    Reviews all CLOs and returns pass/fail results with feedback.

    Args:
        clos: List of CLO strings from Creator Agent

    Returns:
        ReviewResult with verdicts for each CLO
    """
    logger.info(f"Reviewer Agent checking {len(clos)} CLOs")

    reviews = []
    passed_clos = []
    failed_clos = []

    for clo in clos:
        review = review_clo(clo)
        reviews.append(review)

        if review.passed:
            passed_clos.append(clo)
            logger.info(f"PASS: '{clo[:50]}...' → {review.blooms_level}")
        else:
            failed_clos.append(clo)
            logger.warning(f"FAIL: '{clo[:50]}...' → {review.feedback}")

    all_passed = len(failed_clos) == 0

    # Build feedback summary for Creator Agent revision
    if failed_clos:
        feedback_lines = []
        for review in reviews:
            if not review.passed:
                feedback_lines.append(
                    f"- CLO '{review.clo}': {review.feedback}"
                )
        feedback_summary = "\n".join(feedback_lines)
    else:
        feedback_summary = "All CLOs passed Bloom's Taxonomy validation."

    logger.info(
        f"Review complete: {len(passed_clos)} passed, "
        f"{len(failed_clos)} failed"
    )

    return ReviewResult(
        all_passed=all_passed,
        reviews=reviews,
        passed_clos=passed_clos,
        failed_clos=failed_clos,
        feedback_summary=feedback_summary
    )


# ── Main ──────────────────────────────────────────────────
if __name__ == "__main__":

    # Test 1 — Good CLOs that should all pass
    good_clos = [
        "Explain the principles of Software Defined Networking",
        "Design an SDN network topology using OpenFlow",
        "Implement a basic SDN controller using Python",
        "Analyze the performance of SDN-based networks",
        "Evaluate the security implications of SDN deployment"
    ]

    print("="*60)
    print("TEST 1 — Good CLOs (should all pass)")
    print("="*60)
    result = run_reviewer_agent(good_clos)
    for review in result.reviews:
        status = "PASS" if review.passed else "FAIL"
        print(f"[{status}] {review.clo[:55]}")
        print(f"       Verb: '{review.first_verb}' → {review.blooms_level}")

    print(f"\nAll passed: {result.all_passed}")

    # Test 2 — Bad CLOs that should fail
    bad_clos = [
        "Study the basics of SDN",
        "Write a note on OpenFlow protocol",
        "Understand SDN architecture",
        "Know the difference between SDN and traditional networking",
        "Learn about SDN controllers"
    ]

    print("\n" + "="*60)
    print("TEST 2 — Bad CLOs (should all fail)")
    print("="*60)
    result2 = run_reviewer_agent(bad_clos)
    for review in result2.reviews:
        status = "PASS" if review.passed else "FAIL"
        print(f"[{status}] {review.clo[:55]}")
        print(f"       Feedback: {review.feedback}")

    print(f"\nAll passed: {result2.all_passed}")
    print(f"\nFeedback summary for Creator Agent:")
    print(result2.feedback_summary)