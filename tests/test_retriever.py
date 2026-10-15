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