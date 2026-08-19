import os
import yaml
from src.ingestion.chunker import process_document_to_chunks
from src.indexing.bm25_store import BM25Store
from src.indexing.vector_store import VectorStore


def run_full_ingestion(config_path: str = "config/sources.yaml"):
    """Reads all PDF sources from config, chunks them, and indexes them in ChromaDB and BM25."""
    if not os.path.exists(config_path):
        print(f"Error: Config file '{config_path}' not found.")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    all_chunks = []
    sources = config.get("sources", [])

    print(f"Found {len(sources)} sources to ingest.\n")

    for source in sources:
        file_path = source["file_path"]
        if not os.path.exists(file_path):
            print(f"Skipping missing file: {file_path}")
            continue

        print(f"Processing: {source['title']} ({file_path})...")
        chunks = process_document_to_chunks(
            file_path=file_path,
            source_id=source["source_id"],
            document_title=source["title"],
            document_type=source["document_type"],
        )
        print(f"  -> Generated {len(chunks)} chunks.")
        all_chunks.extend(chunks)

    print(f"\nTotal Chunks collected from all documents: {len(all_chunks)}")

    # Indexing into Vector Store (ChromaDB)
    print("Indexing all chunks into Vector Store (ChromaDB)...")
    v_store = VectorStore()
    v_store.add_chunks(all_chunks)

    # Indexing into BM25 Store
    print("Indexing all chunks into BM25 Keyword Store...")
    b_store = BM25Store()
    b_store.build_index(all_chunks)

    print("\nIngestion and Indexing completed successfully for all files!")


if __name__ == "__main__":
    run_full_ingestion()