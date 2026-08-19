import pandas as pd
from datasets import Dataset
from langchain_ollama import ChatOllama
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from src.generation.llm import LocalLLM
from src.indexing.bm25_store import BM25Store
from src.indexing.vector_store import VectorStore
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker

# 1. Load Evaluation Ground Truth Dataset
eval_df = pd.read_csv("ragas_eval_dataset.csv")

# 2. Initialize RAG Components
v_store = VectorStore()
b_store = BM25Store()
retriever = HybridRetriever(vector_store=v_store, bm25_store=b_store)
reranker = Reranker()
llm = LocalLLM(model_name="qwen2.5:1.5b")

# 3. Collect RAG Pipeline Outputs
questions = []
answers = []
contexts_list = []
ground_truths = []

print("Running RAG Pipeline over Evaluation Dataset...")
for idx, row in eval_df.iterrows():
    q = row["question"]
    gt = row["ground_truth"]

    # Retrieval & Reranking
    candidates = retriever.search(q, top_k=8)
    top_chunks = reranker.rerank(q, candidates, top_n=3)

    retrieved_texts = [c["text"] for c in top_chunks]

    # LLM Generation
    prompt = llm.build_prompt(q, top_chunks)
    generated_ans = llm.generate(prompt)

    questions.append(q)
    answers.append(generated_ans)
    contexts_list.append(retrieved_texts)
    ground_truths.append([gt])

# 4. Prepare Dataset for Ragas
ragas_data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts_list,
    "ground_truth": ground_truths,
}
ragas_dataset = Dataset.from_dict(ragas_data)

# 5. Initialize Local Evaluator LLM (Ollama)
eval_llm = ChatOllama(model="qwen2.5:1.5b")

# 6. Compute Ragas Metrics
print("\nComputing Ragas Evaluation Metrics locally...")
results = evaluate(
    dataset=ragas_dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
    llm=eval_llm,
)

print("\n=== Ragas Evaluation Results ===")
print(results)

# Export Results to CSV
results_df = results.to_pandas()
results_df.to_csv("ragas_evaluation_results.csv", index=False)
print("\nSaved evaluation scores to 'ragas_evaluation_results.csv'")