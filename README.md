🧩 Autism Spectrum RAG — Evidence-Based Assistant

<p align="center">
  <img src="docs/images/ui.png" alt="Autism Spectrum RAG Assistant UI" width="900">
</p>

<h3 align="center">
Evidence-Based Retrieval-Augmented Generation Assistant for Autism-Related Questions
</h3>

<p align="center">
  A local, evidence-grounded RAG system that retrieves relevant knowledge, reranks the evidence, generates a grounded answer, and exposes the supporting sources and page numbers.
</p>

🎯 Overview

Autism Spectrum RAG — Evidence-Based Assistant is a Retrieval-Augmented Generation (RAG) system built to answer autism-related clinical, educational, and family-support questions using a curated knowledge base.

Instead of asking the language model to answer only from its internal training knowledge, the system first retrieves relevant evidence from the project's documents and then provides that evidence to a local LLM.

The result is a pipeline designed around:

🔎 Evidence retrieval

🧠 Semantic search

🔤 Keyword search

🔀 Hybrid retrieval

🏆 Reciprocal Rank Fusion (RRF)

🎯 CrossEncoder reranking

🛡️ Grounded generation

📚 Source and page citations

⚡ Local caching

🖥️ Interactive Streamlit interface

🧪 Retrieval and RAG evaluation

🔒 Local inference with Ollama

✨ Key Features

1. 📚 Evidence-Based Answers

The assistant is designed to answer from retrieved evidence instead of freely generating an answer from the LLM's internal knowledge.

The prompt instructs the model to:

use the supplied evidence

avoid unsupported outside knowledge

state when the available evidence is insufficient

provide source/page information

2. 🔎 Semantic Vector Search

The system uses:

sentence-transformers/all-MiniLM-L6-v2

to convert documents and user questions into vector representations.

This allows the retriever to find text that is semantically related to the question, even when the exact words are different.

Example:

Question:
"What difficulties may autistic children have with social situations?"


can retrieve evidence discussing social interaction difficulties even when the wording is not identical.

3. 🔤 BM25 Keyword Search

The system also uses:

BM25Okapi

for keyword-based retrieval.

BM25 is especially useful when exact terminology matters, such as:

DSM-5

ADHD

echolalia

epilepsy

genetic conditions

named recommendations

clinical terminology

This complements semantic search.

4. 🔀 Hybrid Retrieval

Instead of depending on only one retrieval strategy, the project combines:

Semantic Vector Search
          +
       BM25 Search
          ↓
      RRF Fusion

This gives the system the advantages of both:

Retrieval Method

Strength

Vector Search

Understands semantic meaning

BM25

Finds exact / important keywords

Hybrid Search

Combines both signals

5. 🧮 Reciprocal Rank Fusion (RRF)

The results from vector search and BM25 are combined using Reciprocal Rank Fusion.

Current configuration:

rrf_k = 60

Conceptually:

Vector Ranking ──┐
                 ├──→ RRF ──→ Combined Ranking
BM25 Ranking ────┘

A chunk that appears near the top of multiple retrieval lists receives a stronger combined ranking.

6. 🎯 CrossEncoder Reranking

After hybrid retrieval, the candidates are reranked using:

cross-encoder/ms-marco-MiniLM-L-6-v2

The CrossEncoder evaluates:

(question, retrieved_chunk)

and assigns a relevance score.

Current flow:

Hybrid Retrieval
      ↓
Up to 12 candidates
      ↓
CrossEncoder Reranker
      ↓
6 candidates
      ↓
Evidence filtering
      ↓
Up to 3 final evidence chunks

This helps reduce irrelevant context before it reaches the LLM.

7. 🧹 Evidence Filtering

Before generation, retrieved evidence is cleaned to reduce obvious noise.

The current filtering removes or ignores things such as:

very short chunks

table-of-contents-like text

URLs/web links

common copyright/rights text

This helps the LLM receive cleaner evidence.

8. 🤖 Local LLM Generation

The current generation model is:

Qwen2.5 1.5B

running locally through:

Ollama

The project communicates with the local Ollama service rather than requiring a paid cloud LLM API.

This makes the system:

local

lightweight

privacy-friendly

suitable for offline/local experimentation

easier to reproduce without cloud API keys

9. 📚 Source & Page Citations

The system preserves source metadata throughout the pipeline.

Retrieved chunks contain information such as:

Document Title
Page Number
Source ID
Chunk ID
Document Type

The UI exposes the retrieved evidence through:

View Retrieved Sources & Evidence

Users can inspect the supporting document/page information instead of receiving only a black-box answer.

Example citation style:

[NICE CG128, Page 15]

Citation Accuracy

The current project supports citation display and source traceability, but it does not yet have a separate automated citation validator.

Current status:

Capability

Status

Source metadata

✅

Page metadata

✅

Evidence snippets

✅

Citation instructions to LLM

✅

Source/page display in UI

✅

Automated citation validation

🚧 Future

Citation Accuracy metric

🚧 Future

A future citation validator can check whether a generated claim is actually supported by the cited retrieved evidence.

⚡ Performance & Caching

The Streamlit application uses:

@st.cache_resource

to cache expensive initialized resources such as:

VectorStore

BM25Store

HybridRetriever

CrossEncoder Reranker

Local LLM

Instead of rebuilding/loading these components on every Streamlit rerun:

First run
   ↓
Load resources
   ↓
Cache resources
   ↓
Reuse them on later reruns

Important distinction

The current project has resource/model caching, not a full answer cache.

It does not currently do:

Question
   ↓
Check previous answer
   ↓
Return cached answer

A semantic/exact query cache can be added later.

🌊 Streaming

The current generation configuration returns the complete response rather than token-by-token streaming.

Current behavior:

User Question
      ↓
LLM generates
      ↓
Complete Answer
      ↓
Display

A future streaming implementation could display the answer progressively:

"The..."
"The diagnostic..."
"The diagnostic process..."

Streaming mainly improves perceived response time and user experience.

🧠 Complete RAG Architecture

                         ┌─────────────────────┐
                         │   Autism Documents  │
                         │       / PDFs        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                           PDF Text Extraction
                                    │
                                    ▼
                               Cleaning
                                    │
                                    ▼
                                Chunking
                                    │
                                    ▼
                          Metadata Enrichment
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
             Sentence Embeddings              BM25 Index
                    │                               │
                    ▼                               ▼
                ChromaDB                     BM25 Store
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                           Hybrid Retrieval
                                    │
                                    ▼
                            RRF Score Fusion
                                    │
                                    ▼
                         Candidate Documents
                                    │
                                    ▼
                         CrossEncoder Reranker
                                    │
                                    ▼
                           Evidence Filtering
                                    │
                                    ▼
                         Top Evidence Chunks
                                    │
                                    ▼
                         Grounded LLM Prompt
                                    │
                                    ▼
                         Qwen2.5 1.5B / Ollama
                                    │
                         ┌──────────┴──────────┐
                         ▼                     ▼
                     Answer              Sources/Evidence
                         │                     │
                         └──────────┬──────────┘
                                    ▼
                              Streamlit UI

🔄 Query-Time Pipeline

When the user enters a question:

1. User asks a question
          ↓
2. Query is embedded
          ↓
3. Vector search retrieves semantic matches
          ↓
4. BM25 retrieves keyword matches
          ↓
5. RRF combines both rankings
          ↓
6. CrossEncoder reranks candidates
          ↓
7. Evidence filtering removes obvious noise
          ↓
8. Top evidence chunks are selected
          ↓
9. Evidence is inserted into the LLM prompt
          ↓
10. Qwen2.5 1.5B generates a grounded answer
          ↓
11. Answer + retrieved evidence are shown in Streamlit

📥 Ingestion Pipeline

Documents go through:

PDF
 ↓
PyMuPDF Extraction
 ↓
Text Cleaning
 ↓
Page-aware Chunking
 ↓
Metadata
 ↓
Embeddings
 ↓
ChromaDB

The current custom chunking strategy:

processes content page by page

primarily splits on paragraph boundaries

targets around 500 characters

uses around 50 characters of overlap

handles oversized paragraphs by splitting them by words

assigns a UUID to each chunk

preserves the original page number

🗄️ Vector Database

The project uses:

ChromaDB

Persistent local storage:

storage/chroma/

Collection:

autism_knowledge_base

The vector store keeps the chunk embeddings together with metadata needed for retrieval and evidence display.

Why ChromaDB?

local

open-source

simple to integrate

persistent

suitable for an MVP

no external database server required

🔤 BM25 Store

The project maintains a separate BM25 index under:

storage/bm25/

The index is persisted locally so it does not need to be rebuilt every time the application starts.

🧪 Evaluation

The project evaluates both retrieval quality and RAG answer quality.

Retrieval Evaluation

Current retrieval metrics include:

Precision@3

How many of the top 3 retrieved chunks are relevant?

Relevant retrieved chunks
──────────────────────────
Top 3 retrieved chunks

Recall@3

How much of the relevant evidence was successfully retrieved in the top 3?

The current checked-in evaluation reports:

Mean Precision@3 = 0.93
Mean Recall@3    = 0.67

These results come from the current evaluation set and should be interpreted as an MVP/internal benchmark rather than a final large-scale benchmark.

📊 RAGAS Evaluation

The project also contains a RAGAS evaluation pipeline.

Configured metrics include:

Faithfulness

Answer Relevancy

Context Precision

Context Recall

The evaluation flow is:

Evaluation Question
        ↓
Your Actual RAG Pipeline
        ↓
Retrieved Context
        ↓
Generated Answer
        ↓
RAGAS
        ↓
Evaluation Metrics

Results are exported for analysis.

🧾 Evaluation Dataset

The project contains a ground-truth evaluation set with fields such as:

question

ground-truth answer

core keywords

It also contains exported chunks for evaluation/testset work.

The evaluation set can be expanded with:

expected source

expected page

reference answer

relevant chunks

unanswerable questions

multi-hop questions

different difficulty levels

🖥️ User Interface

The project uses Streamlit.

The current interface provides:

💬 Chat-style interaction

🔍 Question input

🤖 Generated answer

📚 Retrieved sources

📄 Page numbers

🧾 Evidence snippets

🌓 Clean dark interface

🧠 Local RAG backend

Example UI:

<p align="center">
  <img src="docs/images/ui.png" alt="Streamlit UI" width="900">
</p>

🧰 Technology Stack

Component

Technology

Programming Language

Python

UI

Streamlit

PDF Extraction

PyMuPDF

Embeddings

Sentence Transformers

Embedding Model

all-MiniLM-L6-v2

Vector Database

ChromaDB

Keyword Retrieval

BM25Okapi

Hybrid Retrieval

Vector + BM25

Rank Fusion

RRF

Reranking

CrossEncoder

Reranker Model

ms-marco-MiniLM-L-6-v2

LLM

Qwen2.5 1.5B

Local LLM Runtime

Ollama

Evaluation

RAGAS + custom retrieval evaluation

Storage

Local persistent storage

Version Control

Git/GitHub

📁 Project Structure

autism-rag/
│
├── config/
│   └── sources.yaml
│
├── data/
│   ├── raw/
│   ├── all_chunks_for_ragas.csv
│   └── all_chunks_for_ragas.json
│
├── src/
│   ├── generation/
│   │   └── llm.py
│   │
│   ├── indexing/
│   │   ├── bm25_store.py
│   │   └── vector_store.py
│   │
│   ├── ingestion/
│   │   ├── chunker.py
│   │   └── pdf_extractor.py
│   │
│   ├── retrieval/
│   │   ├── hybrid.py
│   │   └── reranker.py
│   │
│   └── schemas/
│
├── storage/
│   ├── bm25/
│   └── chroma/
│
├── app.py
├── evaluate_ragas.py
├── run_evaluation.py
├── export_chunks.py
├── rag_evaluation_set.json
├── evaluation_report.csv
├── test_questions.txt
└── Autism_RAG_Architecture.md

🚀 Installation & Setup

1. Clone the repository

git clone https://github.com/maryamteama1/Autism-Spectrum-RAG-Evidence-Based-Assistant.git
cd Autism-Spectrum-RAG-Evidence-Based-Assistant/autism-rag

2. Create a virtual environment

Windows

python -m venv .venv
.venv\Scripts\activate

Linux/macOS

python -m venv .venv
source .venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

4. Install Ollama

Install Ollama and pull the local model:

ollama pull qwen2.5:1.5b

Make sure Ollama is running locally.

5. Run the application

streamlit run app.py

The application will open in the browser.

🧪 Running Evaluation

For the custom retrieval evaluation:

python run_evaluation.py

For RAGAS evaluation:

python evaluate_ragas.py

The resulting reports can be exported to CSV for analysis.

🎬 Live Demo

The recommended hackathon demo flow is:

1. Ask a real question

For example:

What are the common signs of autism in school-aged children regarding social interaction?

2. Show the answer

Explain that the answer is generated from retrieved evidence.

3. Open:

View Retrieved Sources & Evidence

Show:

source/document

page

evidence snippet

4. Explain retrieval

Vector Search
     +
BM25
     ↓
RRF
     ↓
CrossEncoder

5. Explain grounding

The selected evidence is passed to the local Qwen model, which is instructed not to invent information outside the provided context.

6. Show evaluation

Demonstrate the retrieval metrics and RAGAS evaluation results.

🛡️ Safety & Scope

This system is an evidence-based information assistant, not a medical diagnostic system.

It should not replace:

professional clinical assessment

diagnosis

treatment decisions

emergency medical care

For medical decisions, users should consult qualified healthcare professionals.

⚠️ Current Limitations

The current implementation is an MVP and has several areas that can be improved.

Retrieval

Evaluation set is still relatively small.

Retrieval performance can be improved with larger/diverse test sets.

Embedding and reranker models can be benchmarked against alternatives.

Generation

Qwen2.5 1.5B is intentionally lightweight.

A stronger local model may improve answer quality at the cost of latency/resources.

Citations

Source/page metadata is preserved.

Sources are displayed in the UI.

The LLM is instructed to cite evidence.

Automated citation correctness validation is not implemented yet.

Performance

Resource caching is implemented.

Exact/semantic answer caching is not implemented.

Token-by-token response streaming is not implemented yet.

🚧 Future Improvements

🔗 Citation Verification

Build a dedicated citation validator:

Generated Claim
      ↓
Extract Citation
      ↓
Locate Cited Evidence
      ↓
Check Claim ↔ Evidence
      ↓
Citation Correctness

Potential metrics:

Citation Accuracy

Citation Coverage

Unsupported Claim Rate

⚡ Semantic / Query Caching

Add:

User Question
      ↓
Cache Lookup
   ↙       ↘
Found      Not Found
 ↓            ↓
Answer     Run RAG
              ↓
          Save Answer

This can reduce repeated retrieval/generation latency.

🌊 Streaming Generation

Enable Ollama streaming to progressively display generated tokens.

This improves the perceived responsiveness of the application.

🧪 Stronger Evaluation

Build a larger evaluation dataset with:

easy questions

difficult questions

multi-hop questions

source-specific questions

unanswerable questions

adversarial questions

citation-specific questions

🔬 Retrieval Ablation

Compare:

Vector only
    vs
BM25 only
    vs
Vector + BM25
    vs
Vector + BM25 + RRF
    vs
Vector + BM25 + RRF + Reranker

This shows exactly which part of the pipeline improves performance.

📌 Why This Architecture?

The project intentionally combines several retrieval techniques rather than relying on a single search method.

Vector Search

Good at semantic similarity.

BM25

Good at exact terminology.

RRF

Combines different rankings without requiring complicated score normalization.

CrossEncoder

Provides a stronger relevance judgment after the initial fast retrieval.

Local LLM

Generates the final answer from the selected evidence.

Together:

Fast Broad Retrieval
        ↓
Hybrid Search
        ↓
Ranking Fusion
        ↓
Precise Reranking
        ↓
Small High-Quality Context
        ↓
Grounded Generation

🏆 Project Highlights

✅ Fully local RAG pipeline

✅ Evidence-grounded generation

✅ Semantic vector retrieval

✅ BM25 keyword retrieval

✅ Hybrid retrieval

✅ RRF rank fusion

✅ CrossEncoder reranking

✅ Persistent ChromaDB

✅ Persistent BM25 index

✅ Local Qwen2.5 LLM

✅ Ollama integration

✅ Streamlit interface

✅ Source/page evidence display

✅ Resource caching

✅ Retrieval evaluation

✅ RAGAS evaluation

