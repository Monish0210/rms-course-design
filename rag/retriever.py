import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from loguru import logger
from config.settings import (
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, EMBEDDING_MODEL
)


def get_retriever():
    """
    Return a ready-to-use retriever with model and Qdrant client.
    """
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    logger.info("Retriever initialized")
    return model, client


def retrieve(query: str,
             model: SentenceTransformer,
             client: QdrantClient,
             collection_name: str = QDRANT_COLLECTION,
             top_k: int = 5) -> list[dict]:
    """
    Search Qdrant for the top-K most relevant chunks for a query.

    Args:
        query: The subject or topic to search for
        model: The embedding model
        client: Qdrant client
        collection_name: Which collection to search
        top_k: Number of results to return

    Returns:
        List of dicts with text, source, page, score
    """
    # Convert query to vector
    query_vector = model.encode(query).tolist()

    # Search Qdrant
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True
    )

    # Format results
    retrieved = []
    for result in results:
        retrieved.append({
            "text": result.payload["text"],
            "source": result.payload["source"],
            "page": result.payload["page"],
            "score": round(result.score, 4)
        })

    logger.info(f"Query: '{query}' → retrieved {len(retrieved)} chunks")
    return retrieved


def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a single context string
    to pass to the LLM prompt.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}: {chunk['source']}, Page {chunk['page']}]\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(context_parts)


if __name__ == "__main__":
    model, client = get_retriever()

    # Test queries
    test_queries = [
        "Software Defined Networking architecture",
        "OpenFlow protocol controller",
        "SDN control plane data plane separation",
        "network virtualization SDN applications",
        "NBA program outcomes engineering"
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        chunks = retrieve(query, model, client, top_k=3)
        for i, chunk in enumerate(chunks, 1):
            print(f"\nResult {i} (score: {chunk['score']})")
            print(f"  Source: {chunk['source']}, Page {chunk['page']}")
            print(f"  Text: {chunk['text'][:150]}...")

    # Show formatted context for first query
    print(f"\n{'='*60}")
    print("Formatted context for LLM (first query):")
    print(f"{'='*60}")
    chunks = retrieve(test_queries[0], model, client, top_k=3)
    print(format_context(chunks))