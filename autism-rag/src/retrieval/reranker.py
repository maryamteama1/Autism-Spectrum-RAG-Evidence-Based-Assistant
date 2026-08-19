from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(
        self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """Initializes a local CrossEncoder reranker model."""
        self.model = CrossEncoder(model_name)

    def rerank(
        self, query: str, candidates: list[dict], top_n: int = 3
    ) -> list[dict]:
        """Reranks candidate chunks based on pair relevance score with the query."""
        if not candidates:
            return []

        # Form pairs of (query, chunk_text)
        pairs = [[query, doc["text"]] for doc in candidates]

        # Predict relevance scores
        scores = self.model.predict(pairs)

        # Attach scores to documents
        for idx, score in enumerate(scores):
            candidates[idx]["rerank_score"] = float(score)

        # Sort documents by rerank score descending
        reranked = sorted(
            candidates, key=lambda x: x["rerank_score"], reverse=True
        )[:top_n]

        return reranked


if __name__ == "__main__":
    from src.indexing.bm25_store import BM25Store
    from src.indexing.vector_store import VectorStore
    from src.retrieval.hybrid import HybridRetriever

    # Load search components
    v_store = VectorStore()
    b_store = BM25Store()
    hybrid_retriever = HybridRetriever(vector_store=v_store, bm25_store=b_store)
    reranker = Reranker()

    query = "diagnostic assessment steps for autism"

    # Step 1: Hybrid Retrieval
    hybrid_candidates = hybrid_retriever.search(query, top_k=6)

    # Step 2: Rerank Candidates
    final_evidence = reranker.rerank(query, hybrid_candidates, top_n=3)

    print(f"--- Final Reranked Evidence for: '{query}' ---\n")
    for idx, item in enumerate(final_evidence, start=1):
        print(
            f"[{idx}] Page {item['metadata']['page_number']} | Rerank Score: {item['rerank_score']:.4f}"
        )
        print(f"    Text: {item['text'][:150]}...\n")