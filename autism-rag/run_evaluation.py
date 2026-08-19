import json
import pandas as pd
from src.generation.llm import LocalLLM
from src.indexing.bm25_store import BM25Store
from src.indexing.vector_store import VectorStore
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker

# Ground Truth Dataset
eval_data = [
    {
        "id": 1,
        "question": "What are the common signs of autism in school-aged children regarding social interaction?",
        "ground_truth_answer": "Social interaction signs in school-aged children include difficulty interpreting subtle social cues, non-verbal communication challenges, and conversational flow issues.",
        "core_keywords": ["social", "cues", "interaction", "communication"],
    },
    {
        "id": 2,
        "question": "What are the core diagnostic assessment steps recommended by NICE guidelines CG128 for children under 19?",
        "ground_truth_answer": "Diagnostic steps include physical examination, differential diagnosis, coexisting conditions assessment, and direct consultation.",
        "core_keywords": [
            "physical",
            "examination",
            "assessment",
            "coexisting",
            "consultation",
        ],
    },
    {
        "id": 3,
        "question": "How is eye-tracking technology utilized as a biomarker in autism diagnosis?",
        "ground_truth_answer": "Eye-tracking quantifies visual fixation patterns, gaze duration, and social preferences as an objective biomarker.",
        "core_keywords": [
            "eye",
            "tracking",
            "gaze",
            "visual",
            "biomarker",
            "fixation",
        ],
    },
    {
        "id": 4,
        "question": "What actions should be taken regarding reports from pre-schools or schools during an autism evaluation?",
        "ground_truth_answer": "Clinicians should obtain pre-school or school reports while avoiding repeated unnecessary assessments.",
        "core_keywords": [
            "school",
            "report",
            "pre-school",
            "assessment",
            "information",
        ],
    },
    {
        "id": 5,
        "question": "What strategies can families implement at home to support daily routines for children with autism?",
        "ground_truth_answer": "Families can use visual schedules, maintain predictable daily routines, and provide sensory-friendly environments.",
        "core_keywords": [
            "visual",
            "schedules",
            "routine",
            "structure",
            "home",
            "family",
        ],
    },
]

v_store = VectorStore()
b_store = BM25Store()
retriever = HybridRetriever(vector_store=v_store, bm25_store=b_store)
reranker = Reranker()

K = 3  # Evaluating Precision@3
results = []

print(f"🚀 Computing Precision@{K} & Recall Metrics...\n" + "=" * 55)

for idx, item in enumerate(eval_data, start=1):
    query = item["question"]
    core_keywords = item["core_keywords"]

    # 1. Retrieve top-k chunks after Reranking
    candidates = retriever.search(query, top_k=8)
    top_chunks = reranker.rerank(query, candidates, top_n=K)

    # 2. Precision@K calculation: How many returned chunks contain core keywords?
    relevant_chunks = 0
    for chunk in top_chunks:
        text = chunk["text"].lower()
        if any(kw.lower() in text for kw in core_keywords):
            relevant_chunks += 1

    precision_at_k = round(relevant_chunks / K, 2)

    # 3. Recall calculation
    all_text = " ".join([c["text"].lower() for c in top_chunks])
    found_tokens = [w for w in core_keywords if w.lower() in all_text]
    recall_score = round(len(found_tokens) / len(core_keywords), 2)

    results.append(
        {
            "ID": item["id"],
            "Question": query,
            f"Precision@{K}": precision_at_k,
            "Recall": recall_score,
        }
    )

    print(
        f"[{idx}/{len(eval_data)}] Question: {query[:40]}... | Precision@{K}: {precision_at_k} | Recall: {recall_score}"
    )

df = pd.DataFrame(results)
df.to_csv("evaluation_report.csv", index=False)

print("\n" + "=" * 55)
print(f"📊 Mean Precision@{K}: {df[f'Precision@{K}'].mean():.2f}")
print(f"📊 Mean Recall: {df['Recall'].mean():.2f}")
print("📁 Saved results to 'evaluation_report.csv'")