import os
import pickle
from rank_bm25 import BM25Okapi
from src.schemas.chunk import TextChunk


class BM25Store:

    def __init__(self, persist_directory: str = "storage/bm25"):
        """Initializes BM25 keyword index storage."""
        self.persist_directory = persist_directory
        self.index_file = os.path.join(persist_directory, "bm25_index.pkl")
        self.chunks_file = os.path.join(persist_directory, "chunks.pkl")
        os.makedirs(persist_directory, exist_ok=True)

        self.bm25 = None
        self.chunks: list[TextChunk] = []

    def build_index(self, chunks: list[TextChunk]):
        """Tokenizes text chunks and builds the BM25 index."""
        self.chunks = chunks
        tokenized_corpus = [
            chunk.text.lower().split() for chunk in self.chunks
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.save()

    def save(self):
        """Saves BM25 index and chunks to disk using pickle."""
        with open(self.index_file, "wb") as f:
            pickle.dump(self.bm25, f)
        with open(self.chunks_file, "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self):
        """Loads existing BM25 index and chunks from disk."""
        if os.path.exists(self.index_file) and os.path.exists(self.chunks_file):
            with open(self.index_file, "rb") as f:
                self.bm25 = pickle.load(f)
            with open(self.chunks_file, "rb") as f:
                self.chunks = pickle.load(f)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Performs BM25 keyword search for a query."""
        if not self.bm25:
            self.load()

        if not self.bm25:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices based on BM25 scores
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                chunk = self.chunks[idx]
                results.append(
                    {
                        "text": chunk.text,
                        "metadata": {
                            "source_id": chunk.source_id,
                            "document_title": chunk.document_title,
                            "page_number": chunk.page_number,
                            "document_type": chunk.document_type,
                        },
                        "bm25_score": float(scores[idx]),
                    }
                )
        return results


if __name__ == "__main__":
    from src.ingestion.chunker import process_document_to_chunks

    # Load test chunks
    chunks = process_document_to_chunks(
        file_path="data/raw/document.pdf",
        source_id="nice_cg128",
        document_title="NICE CG128",
        document_type="clinical_guideline",
    )

    # Build BM25 index
    bm25_store = BM25Store()
    bm25_store.build_index(chunks)
    print("BM25 Index successfully built and saved!")

    # Test keyword search
    query = "diagnostic assessment"
    results = bm25_store.search(query, top_k=2)
    print(f"\nKeyword Search results for query: '{query}'")
    for r in results:
        print(
            f"- Page {r['metadata']['page_number']} (Score: {r['bm25_score']:.2f}): {r['text'][:100]}..."
        )