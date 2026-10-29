import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import requests
import pandas as pd
from config.settings import (
    OLLAMA_MODEL, QDRANT_COLLECTION, EMBEDDING_MODEL
)

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="RMS Course Design Framework",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://localhost:8000"


# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.title("RMS Course Design")
    st.markdown("**Intelligent End-to-End Course Design Framework**")
    st.markdown("---")

    st.markdown("### System Info")
    st.markdown(f"- **LLM:** `{OLLAMA_MODEL}`")
    st.markdown(f"- **Embeddings:** `{EMBEDDING_MODEL}`")
    st.markdown(f"- **Vector Store:** Qdrant `{QDRANT_COLLECTION}`")
    st.markdown(f"- **Mode:** 100% Local / Air-gapped")

    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
    1. Enter subject and level
    2. System retrieves domain context from uploaded PDFs
    3. Creator Agent generates syllabus and CLOs
    4. Reviewer Agent validates Bloom's Taxonomy
    5. Alignment Agent computes CO-PO matrix
    6. Complete course package displayed below
    """)

    st.markdown("---")
    st.markdown(
        "**Patel Monish (23BCE239)**  \n"
        "**Geria Tejas (23BCE082)**  \n"
        "Guide: Dr. Gaurang Raval  \n"
        "Nirma University"
    )


# ── Main Page ─────────────────────────────────────────────
st.title("Intelligent Course Design Framework")
st.markdown(
    "Generate a complete NBA-compliant course package — "
    "Syllabus, Bloom's-aligned CLOs, and CO-PO Matrix — "
    "automatically from your domain documents."
)

st.markdown("---")

# ── Input Form ────────────────────────────────────────────
st.subheader("Course Details")

col1, col2, col3 = st.columns(3)

with col1:
    subject = st.text_input(
        "Subject Name",
        placeholder="e.g. Software Defined Networking",
        help="Enter the name of the emerging subject"
    )

with col2:
    level = st.selectbox(
        "Academic Level",
        options=[
            "B.Tech 1st Year",
            "B.Tech 2nd Year",
            "B.Tech 3rd Year",
            "B.Tech 4th Year",
            "M.Tech 1st Year",
            "M.Tech 2nd Year"
        ],
        index=2
    )

with col3:
    num_modules = st.slider(
        "Number of Modules",
        min_value=3,
        max_value=8,
        value=5,
        help="How many modules should the course have"
    )

st.markdown("")

generate_btn = st.button(
    "Generate Course Package",
    type="primary",
    use_container_width=True
)


# ── Generation ────────────────────────────────────────────
if generate_btn:

    if not subject.strip():
        st.error("Please enter a subject name.")
        st.stop()

    # Check API is running
    try:
        requests.get(f"{API_URL}/health", timeout=3)
    except Exception:
        st.error(
            "FastAPI backend is not running. "
            "Start it with: `python api/main.py`"
        )
        st.stop()

    # Run pipeline
    with st.spinner(
        f"Generating course for **{subject}**... "
        "This takes 2-4 minutes on CPU."
    ):
        try:
            response = requests.post(
                f"{API_URL}/generate",
                json={
                    "subject": subject,
                    "level": level,
                    "num_modules": num_modules
                },
                timeout=600
            )
            response.raise_for_status()
            data = response.json()

        except requests.exceptions.Timeout:
            st.error(
                "Request timed out. "
                "Ollama is still generating — try again."
            )
            st.stop()
        except requests.exceptions.HTTPError as e:
            st.error(f"API error: {e.response.json().get('detail', str(e))}")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.stop()

    st.success(
        f"Course generated successfully! "
        f"Revisions: {data['revisions']} | "
        f"Status: {data['status']}"
    )

    st.markdown("---")

    # ── Tab Layout ─────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "Syllabus", "CLOs", "CO-PO Matrix", "Pipeline Log"
    ])

    # ── Tab 1: Syllabus ────────────────────────────────────
    with tab1:
        st.subheader(f"Syllabus — {data['subject']}")
        st.caption(f"Level: {data['level']}")

        st.markdown("#### Modules")
        for i, module in enumerate(data["modules"], 1):
            st.markdown(f"**{i}.** {module}")

        st.markdown("#### Prerequisites")
        for prereq in data["prerequisites"]:
            st.markdown(f"- {prereq}")

    # ── Tab 2: CLOs ────────────────────────────────────────
    with tab2:
        st.subheader("Course Learning Outcomes")
        st.caption(
            "All CLOs validated against Bloom's Taxonomy "
            "using POS-based rule engine (More et al. 2025)"
        )

        for clo_item in data["clos"]:
            level_colors = {
                "remember": "gray",
                "understand": "blue",
                "apply": "green",
                "analyze": "orange",
                "evaluate": "red",
                "create": "violet"
            }
            bloom_level = clo_item["blooms_level"]
            color = level_colors.get(bloom_level, "gray")

            col_clo, col_badge = st.columns([5, 1])
            with col_clo:
                st.markdown(
                    f"**CLO{clo_item['index']}:** {clo_item['clo']}"
                )
            with col_badge:
                st.markdown(
                    f":{color}[{bloom_level.upper()}]"
                )

    # ── Tab 3: CO-PO Matrix ────────────────────────────────
    with tab3:
        st.subheader("CO-PO Attainment Matrix")
        st.caption(
            "Computed using KCA tensor T=[K,C,A] "
            "(Li et al. 2025) | Scale: 3=High 2=Medium 1=Low 0=None"
        )

        # Build DataFrame for display
        matrix_rows = []
        po_keys = data["copo_matrix"][0]["po_scores"].keys() \
            if data["copo_matrix"] else []

        for row in data["copo_matrix"]:
            matrix_row = {
                "CLO": f"CLO{row['clo_index']}",
                "Bloom's Level": row["blooms_level"],
            }
            matrix_row.update(row["po_scores"])
            matrix_rows.append(matrix_row)

        if matrix_rows:
            df = pd.DataFrame(matrix_rows)
            df = df.set_index("CLO")

            # Color the numeric cells
            po_cols = [c for c in df.columns if c.startswith("PO")]

            def color_cell(val):
                if val == 3:
                    return "background-color: #1a7a1a; color: white"
                elif val == 2:
                    return "background-color: #2e74b5; color: white"
                elif val == 1:
                    return "background-color: #f0a500; color: white"
                else:
                    return "background-color: #f0f0f0; color: #999"

            styled_df = df.style.applymap(
                color_cell, subset=po_cols
            )
            st.dataframe(styled_df, use_container_width=True)

            st.markdown("""
            **Legend:**
            🟢 **3** = High attainment &nbsp;&nbsp;
            🔵 **2** = Medium attainment &nbsp;&nbsp;
            🟡 **1** = Low attainment &nbsp;&nbsp;
            ⬜ **0** = No mapping
            """)

    # ── Tab 4: Pipeline Log ────────────────────────────────
    with tab4:
        st.subheader("Pipeline Execution Log")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Status", data["status"].upper())
        with col_b:
            st.metric("Revision Loops", data["revisions"])
        with col_c:
            log = data["hallucination_log"]
            if log:
                rate = log[-1]["pass_rate"]
                st.metric("Final Pass Rate", f"{rate}%")

        st.markdown("#### Hallucination Log")
        if data["hallucination_log"]:
            log_df = pd.DataFrame(data["hallucination_log"])
            st.dataframe(log_df, use_container_width=True)

        st.markdown("#### System Configuration")
        config_data = {
            "Parameter": [
                "LLM Model", "Embedding Model",
                "Vector Store", "Max Revision Loops",
                "Chunk Size", "Mode"
            ],
            "Value": [
                OLLAMA_MODEL, EMBEDDING_MODEL,
                f"Qdrant ({QDRANT_COLLECTION})",
                os.getenv("MAX_REVISION_LOOPS", "3"),
                os.getenv("CHUNK_SIZE", "512"),
                "100% Local / Air-gapped"
            ]
        }
        st.dataframe(
            pd.DataFrame(config_data),
            use_container_width=True,
            hide_index=True
        )