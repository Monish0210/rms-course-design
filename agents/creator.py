import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import requests
from loguru import logger
from pydantic import BaseModel, ValidationError
from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL


# ── Output Schema ─────────────────────────────────────────
class CourseStructure(BaseModel):
    modules: list[str]
    clos: list[str]
    prerequisites: list[str]


# ── Prompt Template ───────────────────────────────────────
def build_creator_prompt(subject: str, level: str,
                         context: str, num_modules: int = 5) -> str:
    # Trim context to max 1500 characters to fit in model context window
    trimmed_context = context[:1500] if len(context) > 1500 else context

    return f"""You are an academic course designer. Design a course using ONLY the context below.

SUBJECT: {subject}
LEVEL: {level}
MODULES NEEDED: {num_modules}

CONTEXT:
{trimmed_context}

Return ONLY a JSON object with exactly these fields:
- "modules": list of {num_modules} module name strings
- "clos": list of 5 CLOs each starting with a Bloom's verb
- "prerequisites": list of 2-3 prerequisite subjects

Rules:
1. CLOs must start with: Explain, Design, Implement, Analyze, Evaluate, or Create
2. Base content on the context above only
3. Return ONLY JSON, no extra text

JSON:"""


# ── LLM Call ──────────────────────────────────────────────
def call_ollama(prompt: str, model: str = OLLAMA_MODEL,
                temperature: float = 0.3) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 512,
            "num_ctx": 2048
        }
    }

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=300
        )
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()

    except requests.exceptions.Timeout:
        logger.error("Ollama request timed out after 120 seconds")
        raise
    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to Ollama. Is it running?")
        raise
    except Exception as e:
        logger.error(f"Ollama call failed: {e}")
        raise


# ── JSON Parser ───────────────────────────────────────────
def parse_course_json(raw_response: str) -> CourseStructure:
    """
    Parse and validate the LLM response as a CourseStructure.
    Handles common LLM formatting issues.
    """
    text = raw_response.strip()

    # Remove markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        text = "\n".join(lines).strip()

    # Find JSON object boundaries
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in response: {text[:200]}")

    json_str = text[start:end]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from LLM: {e}\nRaw: {json_str[:300]}")

    # Clamp CLOs to exactly 5 — LLM sometimes generates more
    if "clos" in data and len(data["clos"]) > 5:
        data["clos"] = data["clos"][:5]

    try:
        course = CourseStructure(**data)
    except ValidationError as e:
        raise ValueError(f"JSON does not match schema: {e}")

    return course


# ── Creator Agent ─────────────────────────────────────────
def run_creator_agent(subject: str,
                      level: str,
                      context: str,
                      num_modules: int = 5,
                      feedback: str = None) -> CourseStructure:
    """
    Main Creator Agent function.

    Args:
        subject: The course subject name
        level: Academic level e.g. B.Tech 3rd Year
        context: Retrieved domain passages from RAG
        num_modules: Number of modules to generate
        feedback: Optional Reviewer feedback for revision loop

    Returns:
        Validated CourseStructure object
    """
    logger.info(f"Creator Agent starting for subject: {subject}")

    # Add reviewer feedback to prompt if revision loop
    prompt = build_creator_prompt(subject, level, context, num_modules)
    if feedback:
        prompt += f"\n\nREVIEWER FEEDBACK — Fix these CLOs:\n{feedback}\n\nReturn corrected JSON:"
        logger.info("Revision mode: incorporating Reviewer feedback")

    # Call Ollama
    logger.info(f"Calling Ollama model: {OLLAMA_MODEL}")
    raw_response = call_ollama(prompt)
    logger.info(f"Raw response length: {len(raw_response)} characters")

    # Parse and validate
    course = parse_course_json(raw_response)
    logger.info(f"Generated {len(course.modules)} modules, "
                f"{len(course.clos)} CLOs, "
                f"{len(course.prerequisites)} prerequisites")

    return course


# ── Main ──────────────────────────────────────────────────
if __name__ == "__main__":
    from rag.retriever import get_retriever, retrieve, format_context

    # Step 1 — Retrieve domain context
    print("Loading retriever...")
    model, client = get_retriever()

    query = "Software Defined Networking SDN control plane OpenFlow"
    chunks = retrieve(query, model, client, top_k=2)
    context = format_context(chunks)

    print(f"Retrieved {len(chunks)} chunks for context")
    print(f"Context length: {len(context)} characters")

    # Step 2 — Run Creator Agent
    print("\nRunning Creator Agent (this may take 1-2 minutes)...")
    course = run_creator_agent(
        subject="Software Defined Networking",
        level="B.Tech 3rd Year",
        context=context,
        num_modules=5
    )

    # Step 3 — Display output
    print("\n" + "="*60)
    print("CREATOR AGENT OUTPUT")
    print("="*60)

    print("\nMODULES:")
    for i, module in enumerate(course.modules, 1):
        print(f"  {i}. {module}")

    print("\nCOURSE LEARNING OUTCOMES (CLOs):")
    for i, clo in enumerate(course.clos, 1):
        print(f"  CLO{i}: {clo}")

    print("\nPREREQUISITES:")
    for prereq in course.prerequisites:
        print(f"  - {prereq}")

    print("\n" + "="*60)
    print("Raw JSON output:")
    print(course.model_dump_json(indent=2))