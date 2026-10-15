import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from rag.embedder import (
    get_embedding_model, get_qdrant_client,
    create_collection, embed_and_store,
    get_collection_info
)
from config.settings import QDRANT_COLLECTION


def test_embedding_model_loads():
    """Test embedding model loads and produces 384-dim vectors."""
    model = get_embedding_model()
    vector = model.encode("Software Defined Networking test")
    assert vector.shape == (384,)
    print(f"\nEmbedding shape: {vector.shape}")


def test_qdrant_client_connects():
    """Test Qdrant client connects successfully."""
    client = get_qdrant_client()
    collections = client.get_collections()
    assert collections is not None
    print(f"\nQdrant connected. Collections: {len(collections.collections)}")


def test_collection_created():
    """Test collection exists in Qdrant."""
    client = get_qdrant_client()
    create_collection(client, QDRANT_COLLECTION)
    existing = [c.name for c in client.get_collections().collections]
    assert QDRANT_COLLECTION in existing
    print(f"\nCollection '{QDRANT_COLLECTION}' exists")


def test_vectors_stored():
    """Test that vectors are stored in Qdrant collection."""
    client = get_qdrant_client()
    info = get_collection_info(client, QDRANT_COLLECTION)
    assert info["total_vectors"] is not None
    assert info["total_vectors"] > 0
    print(f"\nTotal vectors in Qdrant: {info['total_vectors']}")


def test_small_batch_embed_and_store():
    """Test embedding and storing a small batch of 3 chunks."""
    model = get_embedding_model()
    client = get_qdrant_client()

    test_collection = "rms_test"
    create_collection(client, test_collection)

    test_chunks = [
        {
            "text": "SDN separates control plane from data plane",
            "source": "test.pdf",
            "page": 1,
            "chunk_index": 9000
        },
        {
            "text": "OpenFlow protocol enables SDN controller communication",
            "source": "test.pdf",
            "page": 2,
            "chunk_index": 9001
        },
        {
            "text": "ChromaDB stores document embeddings locally",
            "source": "test.pdf",
            "page": 3,
            "chunk_index": 9002
        }
    ]

    stored = embed_and_store(
        test_chunks, model, client,
        test_collection, batch_size=3
    )
    assert stored == 3

    # Clean up test collection
    client.delete_collection(test_collection)
    print(f"\nSmall batch test: {stored} chunks embedded and stored")

# ── Milestone 3 — RAG Retrieval Tests ──────────────────────

from rag.retriever import retrieve, format_context, get_retriever

@pytest.fixture(scope="module")
def retriever():
    """Shared retriever for all tests."""
    model, client = get_retriever()
    return model, client


def test_retriever_initializes(retriever):
    """Test retriever loads model and connects to Qdrant."""
    model, client = retriever
    assert model is not None
    assert client is not None
    print("\nRetriever initialized successfully")


def test_retrieve_returns_results(retriever):
    """Test that a query returns results."""
    model, client = retriever
    results = retrieve("Software Defined Networking", model, client, top_k=5)
    assert isinstance(results, list)
    assert len(results) > 0
    assert len(results) <= 5
    print(f"\nRetrieved {len(results)} results")


def test_results_have_required_fields(retriever):
    """Test that every result has text, source, page, score."""
    model, client = retriever
    results = retrieve("OpenFlow protocol", model, client, top_k=3)
    for result in results:
        assert "text" in result
        assert "source" in result
        assert "page" in result
        assert "score" in result
        assert len(result["text"]) > 0
        assert result["score"] > 0
    print("\nAll result fields verified")


def test_scores_are_sorted(retriever):
    """Test that results are in descending score order."""
    model, client = retriever
    results = retrieve("SDN control plane", model, client, top_k=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    print(f"\nScores in order: {scores}")


def test_relevant_content_retrieved(retriever):
    """Test that SDN query returns SDN-relevant content."""
    model, client = retriever
    results = retrieve(
        "Software Defined Networking OpenFlow controller",
        model, client, top_k=5
    )
    combined_text = " ".join([r["text"].lower() for r in results])
    assert any(
        keyword in combined_text
        for keyword in ["sdn", "openflow", "controller",
                        "software defined", "network"]
    )
    print("\nRelevant SDN content confirmed in results")


def test_format_context(retriever):
    """Test that format_context produces a valid string."""
    model, client = retriever
    chunks = retrieve("SDN architecture", model, client, top_k=3)
    context = format_context(chunks)
    assert isinstance(context, str)
    assert len(context) > 0
    assert "[Source 1:" in context
    print(f"\nFormatted context length: {len(context)} characters")