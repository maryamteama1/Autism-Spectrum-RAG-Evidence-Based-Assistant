import json
import pandas as pd
from src.indexing.vector_store import VectorStore


def export_all_chunks():
    """Fetches all stored chunks from VectorStore (ChromaDB) for Ragas evaluation dataset creation."""
    print("Connecting to Vector Store (ChromaDB)...")
    v_store = VectorStore()

    # Get all documents/chunks from ChromaDB collection
    results = v_store.collection.get(include=["documents", "metadatas"])

    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])
    ids = results.get("ids", [])

    print(f"Total Chunks found in VectorStore: {len(documents)}")

    chunks_data = []
    for chunk_id, text, meta in zip(ids, documents, metadatas):
        chunks_data.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "document_title": meta.get("document_title", ""),
                "page_number": meta.get("page_number", None),
                "source_id": meta.get("source_id", ""),
            }
        )

    # 1. Save as JSON format (Recommended for Ragas pipelines)
    json_path = "data/all_chunks_for_ragas.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)
    print(f"✅ Exported {len(chunks_data)} chunks to '{json_path}'")

    # 2. Save as CSV format (Easier for inspection/Excel)
    csv_path = "data/all_chunks_for_ragas.csv"
    df = pd.DataFrame(chunks_data)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ Exported {len(chunks_data)} chunks to '{csv_path}'")


if __name__ == "__main__":
    export_all_chunks()