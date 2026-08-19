import re
import streamlit as st
from src.generation.llm import LocalLLM
from src.indexing.bm25_store import BM25Store
from src.indexing.vector_store import VectorStore
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker

# Page Configuration
st.set_page_config(
    page_title="Autism Spectrum RAG Assistant",
    page_icon="🧩",
    layout="wide",
)


# Strict & Smart Cleaning Function
def clean_retrieved_sources(sources):
    cleaned = []
    for src in sources:
        text = src.get("text", "").strip()

        # 1. Skip very short text chunks (less than 100 characters)
        if len(text) < 100:
            continue

        # 2. Skip table of contents (dots followed by page numbers)
        if "...." in text or "............" in text:
            continue

        # 3. Skip lines with URLs / web links
        if re.search(r"https?://\S+|www\.\S+", text):
            continue

        # 4. Skip copyright & terms of rights text
        lower_text = text.lower()
        if (
            "notice of rights" in lower_text
            or "all rights reserved" in lower_text
            or "terms-and-conditions" in lower_text
            or "page of 43" in lower_text
        ):
            continue

        cleaned.append(src)

    return cleaned


# Initialize Components (Cached for speed)
@st.cache_resource
def load_rag_pipeline():
    v_store = VectorStore()
    b_store = BM25Store()
    hybrid_retriever = HybridRetriever(vector_store=v_store, bm25_store=b_store)
    reranker = Reranker()
    llm = LocalLLM(model_name="qwen2.5:1.5b")
    return hybrid_retriever, reranker, llm


hybrid_retriever, reranker, llm = load_rag_pipeline()

# Title and Description
st.title("🧩 Autism Spectrum RAG Evidence-Based Assistant")
st.caption(
    "Grounded clinical & guide assistance powered by Local RAG with strict citations."
)

# Chat History Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 View Retrieved Sources & Evidence"):
                for src in message["sources"]:
                    st.write(
                        f"**Doc:** {src['metadata'].get('document_title', 'N/A')} | **Page:** {src['metadata'].get('page_number', 'N/A')}"
                    )
                    st.caption(f'"{src["text"][:300]}..."')

# Chat Input
if prompt := st.chat_input("Ask a clinical or family-related autism question..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner(
            "Searching evidence and generating grounded response..."
        ):
            # 1. Retrieval & Reranking (Get top 12 to filter out noise safely)
            candidates = hybrid_retriever.search(prompt, top_k=12)
            top_evidence = reranker.rerank(prompt, candidates, top_n=6)

            # 2. Strict Noise Filtering
            clean_evidence = clean_retrieved_sources(top_evidence)

            # Keep top 3 high quality chunks
            final_evidence = (
                clean_evidence[:3] if clean_evidence else top_evidence[:3]
            )

            # 3. Build Prompt & Generate
            full_prompt = llm.build_prompt(prompt, final_evidence)
            response_text = llm.generate(full_prompt)

            # Display Response
            st.markdown(response_text)

            # Display Sources Expander
            with st.expander("📚 View Retrieved Sources & Evidence"):
                for src in final_evidence:
                    st.write(
                        f"**Doc:** {src['metadata'].get('document_title', 'N/A')} | **Page:** {src['metadata'].get('page_number', 'N/A')}"
                    )
                    st.caption(f'"{src["text"][:300]}..."')

    # Save Assistant Response to History
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_text,
            "sources": final_evidence,
        }
    )