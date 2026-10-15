import pytest
from rag.ingestion import load_pdf, chunk_text, ingest_folder
import os


def test_load_pdf_returns_pages():
    """Test that load_pdf returns a non-empty list of pages."""
    pdf_files = [f for f in os.listdir("data/raw") if f.endswith(".pdf")]
    assert len(pdf_files) > 0, "No PDFs in data/raw"

    first_pdf = os.path.join("data/raw", pdf_files[0])
    pages = load_pdf(first_pdf)

    assert isinstance(pages, list)
    assert len(pages) > 0
    assert "text" in pages[0]
    assert "source" in pages[0]
    assert "page" in pages[0]
    print(f"\nLoaded {len(pages)} pages from {pdf_files[0]}")


def test_chunk_text_produces_chunks():
    """Test that chunk_text splits pages into correct chunk sizes."""
    sample_pages = [
        {
            "text": " ".join([f"word{i}" for i in range(1200)]),
            "source": "test.pdf",
            "page": 1
        }
    ]
    chunks = chunk_text(sample_pages, chunk_size=512, overlap=64)

    assert len(chunks) > 1
    for chunk in chunks:
        word_count = len(chunk["text"].split())
        assert word_count <= 512
        assert "source" in chunk
        assert "page" in chunk
        assert "chunk_index" in chunk
    print(f"\nProduced {len(chunks)} chunks from 1200 words")


def test_chunk_overlap_works():
    """Test that consecutive chunks share overlapping words."""
    sample_pages = [
        {
            "text": " ".join([f"word{i}" for i in range(700)]),
            "source": "test.pdf",
            "page": 1
        }
    ]
    chunks = chunk_text(sample_pages, chunk_size=512, overlap=64)

    assert len(chunks) >= 2
    first_chunk_words = chunks[0]["text"].split()
    second_chunk_words = chunks[1]["text"].split()

    # Last 64 words of chunk 1 should appear at start of chunk 2
    overlap_from_first = first_chunk_words[-64:]
    start_of_second = second_chunk_words[:64]
    assert overlap_from_first == start_of_second
    print(f"\nOverlap verified: {overlap_from_first[:3]}...")


def test_ingest_folder_processes_all_pdfs():
    """Test that ingest_folder processes all PDFs in data/raw."""
    chunks = ingest_folder("data/raw")

    assert isinstance(chunks, list)
    assert len(chunks) > 0

    sources = set(chunk["source"] for chunk in chunks)
    print(f"\nProcessed {len(sources)} PDF files")
    print(f"Total chunks: {len(chunks)}")
    for source in sources:
        source_chunks = [c for c in chunks if c["source"] == source]
        print(f"  {source}: {len(source_chunks)} chunks")


def test_all_chunks_have_required_fields():
    """Test that every chunk has text, source, page, chunk_index."""
    chunks = ingest_folder("data/raw")

    for i, chunk in enumerate(chunks):
        assert "text" in chunk, f"Chunk {i} missing text"
        assert "source" in chunk, f"Chunk {i} missing source"
        assert "page" in chunk, f"Chunk {i} missing page"
        assert "chunk_index" in chunk, f"Chunk {i} missing chunk_index"
        assert len(chunk["text"]) > 0, f"Chunk {i} has empty text"

    print(f"\nAll {len(chunks)} chunks have required fields")