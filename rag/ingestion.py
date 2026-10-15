import fitz  # PyMuPDF
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loguru import logger
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP


def load_pdf(file_path: str) -> list[dict]:
    """
    Read a single PDF and return a list of page-level text blocks.
    Each block has: text, source filename, page number.
    """
    doc = fitz.open(file_path)
    pages = []
    filename = os.path.basename(file_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()
        if text:  # skip empty pages
            pages.append({
                "text": text,
                "source": filename,
                "page": page_num + 1
            })

    doc.close()
    logger.info(f"Loaded {len(pages)} pages from {filename}")
    return pages


def chunk_text(pages: list[dict], chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Split page text into fixed-size token chunks with overlap.
    Each chunk carries source filename and page number as metadata.
    """
    chunks = []

    for page in pages:
        text = page["text"]
        words = text.split()
        total_words = len(words)

        if total_words == 0:
            continue

        start = 0
        while start < total_words:
            end = min(start + chunk_size, total_words)
            chunk_words = words[start:end]
            chunk_text_str = " ".join(chunk_words)

            chunks.append({
                "text": chunk_text_str,
                "source": page["source"],
                "page": page["page"],
                "chunk_index": len(chunks)
            })

            if end == total_words:
                break
            start += chunk_size - overlap

    return chunks


def ingest_folder(folder_path: str = "data/raw") -> list[dict]:
    """
    Load and chunk all PDFs in a folder.
    Returns a flat list of all chunks across all documents.
    """
    all_chunks = []
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

    if not pdf_files:
        logger.warning(f"No PDF files found in {folder_path}")
        return []

    logger.info(f"Found {len(pdf_files)} PDF files in {folder_path}")

    for pdf_file in pdf_files:
        file_path = os.path.join(folder_path, pdf_file)
        try:
            pages = load_pdf(file_path)
            chunks = chunk_text(pages)
            all_chunks.extend(chunks)
            logger.info(f"{pdf_file} -> {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Failed to process {pdf_file}: {e}")

    logger.info(f"Total chunks from all documents: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    chunks = ingest_folder("data/raw")
    print(f"\nTotal chunks produced: {len(chunks)}")
    print(f"\nSample chunk 1:")
    print(f"  Source : {chunks[0]['source']}")
    print(f"  Page   : {chunks[0]['page']}")
    print(f"  Words  : {len(chunks[0]['text'].split())}")
    print(f"  Text   : {chunks[0]['text'][:200]}...")
    print(f"\nSample chunk 2:")
    print(f"  Source : {chunks[1]['source']}")
    print(f"  Page   : {chunks[1]['page']}")
    print(f"  Words  : {len(chunks[1]['text'].split())}")
    print(f"  Text   : {chunks[1]['text'][:200]}...")