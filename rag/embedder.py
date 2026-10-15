import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from loguru import logger
from config.settings import (
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, EMBEDDING_MODEL
)


def get_embedding_model() -> SentenceTransformer:
    """Load the embedding model."""
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("Embedding model loaded successfully")
    return model


def get_qdrant_client() -> QdrantClient:
    """Connect to local Qdrant instance."""
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    logger.info(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    return client


def create_collection(client: QdrantClient, collection_name: str,
                      vector_size: int = 384) -> None:
    """
    Create a Qdrant collection if it does not already exist.
    Vector size 384 matches all-MiniLM-L6-v2 output dimensions.
    """
    existing = [c.name for c in client.get_collections().collections]

    if collection_name in existing:
        logger.info(f"Collection '{collection_name}' already exists — skipping creation")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )
    logger.info(f"Created collection '{collection_name}' with vector size {vector_size}")


def embed_and_store(chunks: list[dict],
                    model: SentenceTransformer,
                    client: QdrantClient,
                    collection_name: str,
                    batch_size: int = 32) -> int:
    """
    Embed all chunks and store them in Qdrant.
    Processes in batches to avoid memory issues.
    Returns total number of points stored.
    """
    total_chunks = len(chunks)
    total_stored = 0

    logger.info(f"Embedding and storing {total_chunks} chunks in batches of {batch_size}")

    for batch_start in range(0, total_chunks, batch_size):
        batch = chunks[batch_start: batch_start + batch_size]
        texts = [chunk["text"] for chunk in batch]

        # Generate embeddings for this batch
        embeddings = model.encode(texts, show_progress_bar=False)

        # Build Qdrant points
        points = []
        for i, (chunk, embedding) in enumerate(zip(batch, embeddings)):
            point_id = batch_start + i
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        "text": chunk["text"],
                        "source": chunk["source"],
                        "page": chunk["page"],
                        "chunk_index": chunk["chunk_index"]
                    }
                )
            )

        # Upload batch to Qdrant
        client.upsert(
            collection_name=collection_name,
            points=points
        )

        total_stored += len(batch)
        logger.info(f"Stored batch {batch_start // batch_size + 1} "
                    f"({total_stored}/{total_chunks} chunks)")

    logger.info(f"All {total_stored} chunks stored in Qdrant "
                f"collection '{collection_name}'")
    return total_stored


def get_collection_info(client: QdrantClient,
                        collection_name: str) -> dict:
    """Return basic info about the collection."""
    info = client.get_collection(collection_name)
    # Qdrant 1.9 uses points_count instead of vectors_count
    total = (info.vectors_count or
             info.points_count or
             client.count(collection_name).count)
    return {
        "name": collection_name,
        "total_vectors": total,
        "status": str(info.status)
    }


if __name__ == "__main__":
    from rag.ingestion import ingest_folder

    # Step 1 — Load and chunk all PDFs
    chunks = ingest_folder("data/raw")
    print(f"Chunks to embed: {len(chunks)}")

    # Step 2 — Load embedding model
    model = get_embedding_model()

    # Step 3 — Connect to Qdrant
    client = get_qdrant_client()

    # Step 4 — Create collection
    create_collection(client, QDRANT_COLLECTION)

    # Step 5 — Embed and store
    total = embed_and_store(chunks, model, client, QDRANT_COLLECTION)

    # Step 6 — Verify
    info = get_collection_info(client, QDRANT_COLLECTION)
    print(f"\nCollection: {info['name']}")
    print(f"Total vectors stored: {info['total_vectors']}")
    print(f"Status: {info['status']}")