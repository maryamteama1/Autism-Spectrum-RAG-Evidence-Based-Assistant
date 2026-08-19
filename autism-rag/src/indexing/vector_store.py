import os
import chromadb
from sentence_transformers import SentenceTransformer
from src.schemas.chunk import TextChunk


class VectorStore:

    def __init__(
        self,
        persist_directory: str = "storage/chroma",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        """Initializes ChromaDB vector database with a local embedding model."""
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_model = SentenceTransformer(model_name)
        self.collection = self.client.get_or_create_collection(
            name="autism_knowledge_base"
        )

    def add_chunks(self, chunks: list[TextChunk]):
        """Embeds text chunks and stores them in ChromaDB with metadata."""
        if not chunks:
            return

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.encode(texts).tolist()

        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [
            {
                "source_id": chunk.source_id,
                "document_title": chunk.document_title,
                "page_number": chunk.page_number,
                "document_type": chunk.document_type,
            }
            for chunk in chunks
        ]

        self.collection.add(
            ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas
        )

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Performs semantic vector search for a user query."""
        query_embedding = self.embedding_model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding, n_results=top_k
        )

        formatted_results = []
        if results and results["documents"]:
            for i in range(len(results["documents"][0])):
                formatted_results.append(
                    {
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i]
                        if "distances" in results
                        else None,
                    }
                )
        return formatted_results


if __name__ == "__main__":
    from src.ingestion.chunker import process_document_to_chunks

    # Load and process chunks
    chunks = process_document_to_chunks(
        file_path="data/raw/document.pdf",
        source_id="nice_cg128",
        document_title="NICE CG128",
        document_type="clinical_guideline",
    )

    # Store in ChromaDB
    store = VectorStore()
    store.add_chunks(chunks)
    print("Chunks successfully indexed in ChromaDB!")

    # Quick search test
    query = "What are the signs of autism in children?"
    search_results = store.search(query, top_k=2)
    print(f"\nSearch test for query: '{query}'")
    for r in search_results:
        print(
            f"- Page {r['metadata']['page_number']}: {r['text'][:100]}..."
        )