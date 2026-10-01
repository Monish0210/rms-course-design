# Intelligent End-to-End Course Design Framework
## For Emerging Subjects using Domain-Constrained RAG

**Students:** Patel Monish (23BCE239) | Geria Tejas (23BCE082)
**Guide:** Dr. Gaurang Raval
**Institute:** Dept. of CSE, Institute of Technology, Nirma University

---

## Setup

1. Install Ollama from https://ollama.com
2. Pull model: `ollama pull mistral:7b-instruct-q4_K_M`
3. Install Qdrant from https://github.com/qdrant/qdrant/releases/latest
4. Run qdrant.exe before starting the project
5. Create virtual environment: `python -m venv .venv`
6. Activate: `.venv\Scripts\activate`
7. Install packages: `pip install -r requirements.txt`
8. Copy `.env.example` to `.env`
9. Run verification tests: `pytest tests/`

## Project Structure

- `agents/`     — Creator, Reviewer, Alignment agents
- `rag/`        — PDF ingestion, embedding, retrieval
- `workflow/`   — LangGraph state graph
- `validators/` — Bloom's Taxonomy POS rules
- `alignment/`  — KCA tensor CO-PO computation
- `data/raw/`   — Input PDFs (not tracked in git)
- `ui/`         — Streamlit faculty interface
- `api/`        — FastAPI backend
- `evaluation/` — Cosine similarity scoring
- `tests/`      — One test file per milestone
- `config/`     — Settings and environment loader

## Vector Store

Qdrant runs locally at http://localhost:6333
Dashboard: http://localhost:6333/dashboard
Collection name: rms_corpus

## Tech Stack

- LLM: Mistral 7B via Ollama (local, no internet)
- Vector Store: Qdrant (local, visual dashboard)
- Embeddings: all-MiniLM-L6-v2 (384-dim, offline)
- Orchestration: LangGraph
- RAG: LangChain
- UI: Streamlit
- API: FastAPI