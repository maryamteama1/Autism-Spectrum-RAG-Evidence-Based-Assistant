from src.indexing.vector_store import VectorStore
from src.indexing.bm25_store import BM25Store


class HybridRetriever:

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        rrf_k: int = 60,
    ):
        """Initializes hybrid retrieval combining Semantic and Keyword search using RRF."""
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.rrf_k = rrf_k

    def search(
        self, query: str, top_k: int = 5, fetch_k: int = 10
    ) -> list[dict]:
        """Executes vector and BM25 search, then fuses and reranks results using RRF."""
        # 1. Fetch results from both search engines
        vector_results = self.vector_store.search(query, top_k=fetch_k)
        bm25_results = self.bm25_store.search(query, top_k=fetch_k)

        # 2. Apply Reciprocal Rank Fusion (RRF)
        scores = {}
        doc_map = {}

        # Process Vector search results
        for rank, res in enumerate(vector_results):
            text = res["text"]
            doc_map[text] = res["metadata"]
            if text not in scores:
                scores[text] = 0.0
            scores[text] += 1.0 / (self.rrf_k + rank + 1)

        # Process BM25 search results
        for rank, res in enumerate(bm25_results):
            text = res["text"]
            doc_map[text] = res["metadata"]
            if text not in scores:
                scores[text] = 0.0
            scores[text] += 1.0 / (self.rrf_k + rank + 1)

        # 3. Sort docs by fused RRF score
        sorted_texts = sorted(
            scores.keys(), key=lambda t: scores[t], reverse=True
        )[:top_k]

        fused_results = []
        for text in sorted_texts:
            fused_results.append(
                {
                    "text": text,
                    "metadata": doc_map[text],
                    "rrf_score": scores[text],
                }
            )

        return fused_results


if __name__ == "__main__":
    # Test hybrid retrieval
    v_store = VectorStore()
    b_store = BM25Store()

    retriever = HybridRetriever(vector_store=v_store, bm25_store=b_store)

    query = "diagnostic assessment steps for autism"
    results = retriever.search(query, top_k=3)

    print(f"Hybrid Search Results for query: '{query}'\n")
    for idx, res in enumerate(results, start=1):
        print(f"[{idx}] Page {res['metadata']['page_number']} (RRF Score: {res['rrf_score']:.4f})")
        print(f"    Text: {res['text'][:120]}...\n")