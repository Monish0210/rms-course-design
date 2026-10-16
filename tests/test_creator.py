import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from agents.creator import (
    build_creator_prompt, parse_course_json,
    CourseStructure, run_creator_agent
)
from rag.retriever import get_retriever, retrieve, format_context


# ── Unit Tests (no LLM needed) ────────────────────────────

def test_prompt_contains_subject():
    """Test that prompt includes subject and level."""
    prompt = build_creator_prompt(
        subject="Software Defined Networking",
        level="B.Tech 3rd Year",
        context="SDN separates control and data plane",
        num_modules=5
    )
    assert "Software Defined Networking" in prompt
    assert "B.Tech 3rd Year" in prompt
    assert "SDN separates control and data plane" in prompt
    print("\nPrompt contains all required fields")


def test_parse_valid_json():
    """Test that valid JSON is parsed correctly."""
    valid_json = '''
    {
        "modules": [
            "Module 1: Introduction to SDN",
            "Module 2: SDN Architecture",
            "Module 3: OpenFlow Protocol",
            "Module 4: SDN Controllers",
            "Module 5: SDN Applications"
        ],
        "clos": [
            "Explain the principles of Software Defined Networking",
            "Design SDN network topologies using OpenFlow",
            "Implement SDN controllers for network management",
            "Analyze performance of SDN-based networks",
            "Evaluate SDN solutions for enterprise environments"
        ],
        "prerequisites": [
            "Computer Networks",
            "Network Protocols",
            "Linux Fundamentals"
        ]
    }
    '''
    course = parse_course_json(valid_json)
    assert isinstance(course, CourseStructure)
    assert len(course.modules) == 5
    assert len(course.clos) == 5
    assert len(course.prerequisites) == 3
    print(f"\nParsed {len(course.modules)} modules successfully")


def test_parse_json_with_markdown():
    """Test that JSON wrapped in markdown is handled."""
    markdown_json = '''```json
    {
        "modules": ["Module 1: SDN Basics"],
        "clos": ["Explain SDN architecture"],
        "prerequisites": ["Networking"]
    }
```'''
    course = parse_course_json(markdown_json)
    assert course.modules[0] == "Module 1: SDN Basics"
    print("\nMarkdown-wrapped JSON parsed correctly")


def test_parse_invalid_json_raises():
    """Test that invalid JSON raises ValueError."""
    with pytest.raises(ValueError):
        parse_course_json("This is not JSON at all")
    print("\nInvalid JSON correctly raises ValueError")


def test_course_structure_schema():
    """Test CourseStructure Pydantic model."""
    course = CourseStructure(
        modules=["Module 1: Test"],
        clos=["Explain the test concept"],
        prerequisites=["Math"]
    )
    assert course.modules == ["Module 1: Test"]
    assert course.clos == ["Explain the test concept"]
    assert course.prerequisites == ["Math"]
    print("\nCourseStructure schema validated")


# ── Integration Test (calls Ollama) ───────────────────────

def test_creator_agent_with_real_context():
    """
    Integration test — calls Ollama with real RAG context.
    This test takes 1-2 minutes.
    """
    # Get real context from Qdrant
    model, client = get_retriever()
    chunks = retrieve(
        "Software Defined Networking OpenFlow",
        model, client, top_k=3
    )
    context = format_context(chunks)

    # Run creator agent
    course = run_creator_agent(
        subject="Software Defined Networking",
        level="B.Tech 3rd Year",
        context=context,
        num_modules=5
    )

    # Validate output
    assert isinstance(course, CourseStructure)
    assert len(course.modules) == 5
    assert len(course.clos) >= 5
    assert len(course.prerequisites) >= 1

    # Every CLO must start with a capital letter word
    for clo in course.clos:
        assert len(clo) > 10, f"CLO too short: {clo}"

    print(f"\nCreator Agent produced:")
    print(f"  Modules: {len(course.modules)}")
    print(f"  CLOs: {len(course.clos)}")
    print(f"  Prerequisites: {len(course.prerequisites)}")
    for clo in course.clos:
        print(f"  - {clo}")