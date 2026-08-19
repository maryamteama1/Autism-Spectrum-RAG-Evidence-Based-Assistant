import requests


class LocalLLM:

    def __init__(
        self,
        model_name: str = "qwen2.5:1.5b",
        ollama_url: str = "http://localhost:11434/api/generate",
    ):
        """Initializes connection to local Ollama instance."""
        self.model_name = model_name
        self.ollama_url = ollama_url

    def build_prompt(self, query: str, context_chunks: list[dict]) -> str:
        """Constructs a grounded RAG prompt with strict citation constraints."""
        context_str = ""
        for idx, item in enumerate(context_chunks, start=1):
            meta = item.get("metadata", {})
            title = meta.get("document_title", "Unknown Source")
            page = meta.get("page_number", "N/A")
            context_str += (
                f"\n--- Document {idx}: {title} (Page {page}) ---\n"
                f"{item['text']}\n"
            )

        prompt = f"""You are a specialized clinical & evidence-based assistant for Autism Spectrum Disorders.
Your job is to provide accurate, grounded answers based STRICTLY on the provided Context Documents below.

RULES:
1. Base your answer ONLY on the provided Context. Do NOT use outside knowledge or extrapolate beyond the text.
2. If the context does not contain enough evidence to answer, state clearly: "I cannot find sufficient evidence in the provided documents."
3. Always cite your sources in the text using page numbers or document titles (e.g., [NICE CG128, Page 15]).

Context Documents:
{context_str}

User Question: {query}

Grounded Answer:"""
        return prompt

    def generate(self, prompt: str) -> str:
        """Sends prompt to local Ollama model and returns the generated text response."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=60)
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            else:
                return f"Error from Ollama API: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Failed to connect to local Ollama server: {str(e)}"


if __name__ == "__main__":
    from src.indexing.bm25_store import BM25Store
    from src.indexing.vector_store import VectorStore
    from src.retrieval.hybrid import HybridRetriever
    from src.retrieval.reranker import Reranker

    # 1. Initialize Pipeline Components
    v_store = VectorStore()
    b_store = BM25Store()
    hybrid_retriever = HybridRetriever(vector_store=v_store, bm25_store=b_store)
    reranker = Reranker()
    llm = LocalLLM(model_name="qwen2.5:1.5b")

    # 2. End-to-End Query Test
    query = "What are the common signs of autism in school-aged children?"

    print(f"User Query: '{query}'\nSearching for evidence...")
    candidates = hybrid_retriever.search(query, top_k=8)
    top_evidence = reranker.rerank(query, candidates, top_n=3)

    # 3. Build Prompt & Generate Response
    prompt = llm.build_prompt(query, top_evidence)
    print("\nGenerating grounded response from LLM...\n")
    response = llm.generate(prompt)

    print("--- LLM Response ---")
    print(response)