import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger
from workflow.graph import run_workflow

app = FastAPI(
    title="RMS Course Design API",
    description="Intelligent End-to-End Course Design Framework",
    version="1.0.0"
)


# ── Request Schema ────────────────────────────────────────
class GenerateRequest(BaseModel):
    subject: str
    level: str
    num_modules: int = 5


# ── Response Schema ───────────────────────────────────────
class GenerateResponse(BaseModel):
    status: str
    subject: str
    level: str
    modules: list[str]
    prerequisites: list[str]
    clos: list[dict]
    copo_matrix: list[dict]
    formatted_table: str
    revisions: int
    hallucination_log: list[dict]
    average_similarity: float | None = None


# ── Health Check ──────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "running", "project": "RMS Course Design Framework"}


@app.get("/health")
def health():
    return {"status": "healthy"}


# ── Generate Endpoint ─────────────────────────────────────
@app.post("/generate", response_model=GenerateResponse)
def generate_course(request: GenerateRequest):
    """
    Generate a complete course package for a given subject.
    Input: subject name, academic level, number of modules
    Output: syllabus, CLOs, CO-PO matrix
    """
    logger.info(
        f"Generate request: subject={request.subject}, "
        f"level={request.level}, modules={request.num_modules}"
    )

    if not request.subject.strip():
        raise HTTPException(
            status_code=400,
            detail="Subject name cannot be empty"
        )

    if not request.level.strip():
        raise HTTPException(
            status_code=400,
            detail="Academic level cannot be empty"
        )

    if not 1 <= request.num_modules <= 10:
        raise HTTPException(
            status_code=400,
            detail="Number of modules must be between 1 and 10"
        )

    try:
        # Run the complete pipeline
        state = run_workflow(
            subject=request.subject,
            level=request.level,
            num_modules=request.num_modules
        )

        # Build CLO list with Bloom's levels
        clos_with_levels = []
        for i, (clo, level) in enumerate(
                zip(state.course.clos,
                    state.alignment.blooms_levels), 1):
            clos_with_levels.append({
                "index": i,
                "clo": clo,
                "blooms_level": level
            })

        return GenerateResponse(
            status=state.status,
            subject=request.subject,
            level=request.level,
            modules=state.course.modules,
            prerequisites=state.course.prerequisites,
            clos=clos_with_levels,
            copo_matrix=state.alignment.matrix,
            formatted_table=state.alignment.formatted_table,
            revisions=state.revision_count,
            hallucination_log=state.hallucination_log
        )

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Course generation failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)