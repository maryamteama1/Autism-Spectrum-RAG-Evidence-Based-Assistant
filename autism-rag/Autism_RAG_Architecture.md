# Autism Knowledge Intelligence System
## Final Architecture & Build Blueprint

### 1. Project Goal

We are building a **multi-source, citation-aware RAG system about Autism**.

The goal is NOT just:

> PDF → chatbot

The goal is:

> **Authoritative autism documents → structured knowledge base → hybrid retrieval → evidence-aware local LLM → answer with source/page citations**

The first sources are:

1. **100 Day Kit for Families of School Age Children Newly Diagnosed with Autism**
   - Practical/family-oriented content
   - Symptoms, communication, behavior, education, interventions, family support

2. **NICE CG128 — Autism spectrum disorder in under 19s: recognition, referral and diagnosis**
   - Recognition
   - Referral
   - Diagnostic assessment
   - Differential diagnosis
   - Coexisting conditions
   - Medical investigations
   - Age-specific features

More sources can be added later without rewriting the core pipeline.

---

# 2. Design Principles

### Free + Local First

The base system should not require paid APIs or cloud infrastructure.

| Layer | Technology |
|---|---|
| Language | Python |
| PDF extraction | PyMuPDF |
| Validation/config | Pydantic 2.x + YAML |
| Embeddings | sentence-transformers |
| Vector DB | ChromaDB |
| Keyword retrieval | BM25 |
| Reranking | sentence-transformers CrossEncoder |
| LLM | Ollama + local open-source model |
| UI | Streamlit |
| Evaluation | Local Python evaluation pipeline |
| Version control | Git/GitHub |

Models can be replaced later, but the application should work locally.

---

# 3. High-Level Architecture

```text
                    AUTISM SOURCE DOCUMENTS
                              |
                +-------------+-------------+
                |                           |
          100 Day Kit                   NICE CG128
                |                           |
                +-------------+-------------+
                              |
                              v
                     DOCUMENT REGISTRY
                        sources.yaml
                              |
                              v
                     INGESTION PIPELINE
                              |
          +-------------------+-------------------+
          |                   |                   |
          v                   v                   v
    PDF Extraction        Cleaning            Metadata
    page-preserving                            enrichment
          |                   |                   |
          +-------------------+-------------------+
                              |
                              v
                         CHUNKING
                              |
                              v
                    VALIDATED CHUNKS
                              |
                  +-----------+-----------+
                  |                       |
                  v                       v
             BM25 Index             Embeddings
                                        |
                                        v
                                  ChromaDB
                                        |
                  +---------------------+
                  |
                  v
                     QUERY PIPELINE
                              |
                        User Question
                              |
                              v
                       Query Analysis
                              |
                              v
                  Hybrid Retrieval
                  /              \
             BM25 Search      Vector Search
                  \              /
                   +------v------+
                          |
                          v
                       Fusion
                          |
                          v
                       Reranker
                          |
                          v
                   Evidence Selector
                          |
                          v
                    Context Builder
                          |
                          v
                    Local LLM
                          |
                          v
              Answer + Evidence + Citations
                          |
                          v
                     Streamlit UI
```

---

# 4. Project Folder Structure

```text
autism-rag/
│
├── data/
│   ├── raw/
│   │   ├── 100_day_kit.pdf
│   │   └── NICE_CG128.pdf
│   │
│   └── processed/
│       ├── extracted/
│       └── chunks/
│
├── config/
│   ├── sources.yaml
│   └── vocabularies.yaml
│
├── src/
│   ├── config/
│   │   ├── loader.py
│   │   └── settings.py
│   │
│   ├── schemas/
│   │   ├── document.py
│   │   ├── chunk.py
│   │   ├── retrieval.py
│   │   └── answer.py
│   │
│   ├── ingestion/
│   │   ├── pdf_extractor.py
│   │   ├── cleaner.py
│   │   ├── chunker.py
│   │   ├── metadata.py
│   │   └── pipeline.py
│   │
│   ├── indexing/
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   └── bm25_store.py
│   │
│   ├── retrieval/
│   │   ├── semantic.py
│   │   ├── keyword.py
│   │   ├── hybrid.py
│   │   ├── fusion.py
│   │   └── reranker.py
│   │
│   ├── generation/
│   │   ├── ollama_client.py
│   │   ├── prompt_builder.py
│   │   ├── context_builder.py
│   │   └── answer_generator.py
│   │
│   ├── evidence/
│   │   ├── citation_builder.py
│   │   └── evidence_validator.py
│   │
│   ├── evaluation/
│   │   ├── dataset.py
│   │   ├── retrieval_metrics.py
│   │   ├── answer_metrics.py
│   │   └── run_evaluation.py
│   │
│   └── utils/
│       ├── logging.py
│       └── text.py
│
├── app/
│   └── streamlit_app.py
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   ├── test_citations.py
│   └── test_end_to_end.py
│
├── storage/
│   ├── chroma/
│   └── bm25/
│
├── scripts/
│   ├── ingest.py
│   ├── index.py
│   └── evaluate.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 5. Data Flow: Ingestion

```text
PDF
 |
 v
PyMuPDF
 |
 +--> page text
 +--> page number
 |
 v
Cleaner
 |
 v
Chunker
 |
 v
Metadata Enricher
 |
 v
Pydantic Validation
 |
 v
Processed Chunks
```

## Critical rule

**Page numbers must never be lost.**

Every chunk must know exactly where it came from.

---

# 6. Metadata Design

## Document-level metadata

```yaml
source_id: nice_cg128
title: Autism spectrum disorder in under 19s
document_type: clinical_guideline
organization: NICE
age_group: under_19
topics:
  - diagnosis
  - recognition
  - referral
```

## Chunk-level metadata

```python
{
    "chunk_id": "...",
    "source_id": "nice_cg128",
    "document_title": "Autism spectrum disorder in under 19s",
    "page_number": 17,
    "section": "1.5 Autism diagnostic assessment",
    "topic": "diagnosis",
    "age_group": "under_19",
    "document_type": "clinical_guideline"
}
```

## Why this matters

The same concept can appear in multiple documents.

For example, "diagnosis" in NICE is a clinical guideline concept, while diagnosis in the 100 Day Kit may be explained for families.

Metadata allows us to preserve that distinction.

---

# 7. Controlled Vocabularies

Topics should be controlled rather than arbitrary strings.

Example:

```text
diagnosis
recognition
referral
symptoms
communication
behavior
treatment
education
family_support
comorbidities
genetics
safety
sensory
sleep
feeding
```

Age groups:

```text
preschool
primary_school
secondary_school
school_age
under_19
all_ages
```

These lists should live in configuration and be validated by Pydantic.

---

# 8. Chunking Strategy

Do NOT blindly split every N characters.

Preferred strategy:

```text
Page
  ↓
Section-aware splitting
  ↓
Paragraph-aware splitting
  ↓
Target chunk size
  ↓
Small overlap
```

The chunk should preserve enough context to answer a question without becoming unnecessarily large.

For guidelines, preserve recommendation boundaries whenever possible.

Example:

```text
Recommendation 1.5.5
    ↓
one coherent chunk or small group of coherent chunks
```

This is especially important for NICE recommendations.

---

# 9. Indexing Layer

We use TWO retrieval systems.

## A. Semantic retrieval

```text
Question
   ↓
Embedding model
   ↓
Vector
   ↓
ChromaDB
   ↓
Top-K semantic chunks
```

Useful for conceptual similarity.

## B. Keyword retrieval

```text
Question
   ↓
BM25
   ↓
Top-K keyword chunks
```

Useful for exact terms such as:

- DSM-5
- ADHD
- echolalia
- epilepsy
- fragile X
- recommendation 1.5.5

---

# 10. Hybrid Retrieval

Do not rely on only one retriever.

```text
             User Question
                  |
        +---------+---------+
        |                   |
        v                   v
     BM25              Vector Search
        |                   |
      Top-K                Top-K
        \                   /
         \                 /
          +------v--------+
                 |
              Fusion
                 |
                 v
             Candidates
                 |
                 v
              Reranker
                 |
                 v
             Final Evidence
```

The fusion layer should remove duplicates and retain source/page metadata.

---

# 11. Reranking

The first retrieval stage should be broad.

The reranker then asks:

> Which retrieved chunks are actually most relevant to THIS question?

Use a local CrossEncoder.

Input:

```text
(question, chunk)
```

Output:

```text
relevance score
```

Then select the strongest evidence for the LLM.

---

# 12. Query Analysis

For a stronger system, extract lightweight query attributes.

Example:

Question:

> What are the signs of autism in a 7-year-old?

Possible analysis:

```json
{
  "topic": "symptoms",
  "age_group": "primary_school",
  "intent": "informational"
}
```

These attributes can optionally influence retrieval filters.

Do NOT build a complicated autonomous agent for this.

Keep it deterministic and understandable.

---

# 13. Context Builder

The LLM should NOT receive random chunks.

The Context Builder should:

1. Sort evidence by relevance.
2. Remove duplicates.
3. Preserve source/page.
4. Preserve enough surrounding context.
5. Keep the context within the model's context window.
6. Separate evidence from different documents clearly.

Example:

```text
SOURCE: NICE CG128
PAGE: 17
SECTION: 1.5.5

[Evidence...]

SOURCE: 100 Day Kit
PAGE: XX
SECTION: ...

[Evidence...]
```

---

# 14. Local LLM

Use:

```text
Ollama
   ↓
Local open-source model
```

The exact model can be selected based on laptop RAM/CPU and tested for quality.

The architecture should NOT hard-code one model.

Configuration should allow:

```yaml
llm:
  provider: ollama
  model: <local-model>
```

---

# 15. Answer Policy

The LLM prompt should enforce:

### Grounding

Answer from retrieved evidence.

### No fabrication

Never invent:
- facts
- citations
- page numbers
- sources

### Insufficient evidence

If the retrieved evidence is insufficient:

> "I couldn't find enough evidence in the available sources to answer this reliably."

### Source distinction

If two sources provide different information, keep them attributed separately.

---

# 16. Evidence & Citation Layer

This is one of the most important parts of the project.

The answer should map back to evidence:

```text
Claim
  ↓
Evidence chunk
  ↓
Page
  ↓
Document
```

Example UI:

```text
Answer

Autism diagnostic assessment should include developmental
history, observation of social/communication skills, medical
history, physical examination, and consideration of differential
diagnoses.

Sources:
[1] NICE CG128 — p.17
[2] NICE CG128 — p.18
```

The citation system must use stored metadata, not generated text.

---

# 17. Evaluation

Do NOT evaluate the project only by saying:

> "The chatbot gives good answers."

Create an evaluation set.

Example:

```json
{
  "question": "What should be included in an autism diagnostic assessment?",
  "expected_sources": ["nice_cg128"],
  "expected_pages": [17, 18],
  "reference_answer": "..."
}
```

Create questions covering:

- diagnosis
- symptoms
- age-specific signs
- comorbidities
- treatment
- family support
- education
- cross-source questions
- unanswerable questions

## Retrieval metrics

Measure:

- Recall@K
- Precision@K
- MRR
- source hit rate

## Answer metrics

Measure:

- groundedness
- citation correctness
- answer relevance
- unsupported claim rate

The most important demo metric is:

> **Can the system retrieve the correct evidence and cite it correctly?**

---

# 18. UI

Use Streamlit.

Main screen:

```text
================================================
       Autism Knowledge Intelligence System
================================================

Ask a question about autism...

[ What are the diagnostic steps for autism? ]

                    [Ask]

------------------------------------------------

ANSWER

...

------------------------------------------------

EVIDENCE

NICE CG128
Page 17
Section 1.5.5

[relevant passage]

------------------------------------------------

SOURCES

• NICE CG128
• 100 Day Kit

------------------------------------------------

RETRIEVAL DETAILS
Topic: diagnosis
Age group: under_19
Evidence chunks: 4
------------------------------------------------
```

Optional advanced panel:

```text
Retrieved Sources
Retrieval scores
BM25 score
Semantic score
Reranker score
```

This can be hidden by default for a cleaner demo.

---

# 19. Safety / Medical Scope

This is an educational knowledge system, not a diagnostic tool.

The UI should clearly state:

> This system provides information from its indexed sources and is not a substitute for professional medical diagnosis or advice.

The system must never present itself as diagnosing a person.

---

# 20. Extensibility

Adding a new source should look like:

```text
1. Put PDF in data/raw/
2. Add metadata to sources.yaml
3. Run ingestion
4. Rebuild/update indexes
```

No core pipeline modification.

Example:

```yaml
source_id: autism_guideline_03
title: ...
document_type: clinical_guideline
organization: ...
age_group: under_19
topics:
  - treatment
  - management
file_path: data/raw/new_guideline.pdf
```

---

# 21. End-to-End Runtime

When the user asks a question:

```text
USER QUESTION
     |
     v
Query Analyzer
     |
     +--> topic
     +--> age group
     +--> intent
     |
     v
BM25 + Vector Search
     |
     v
Candidate Fusion
     |
     v
CrossEncoder Reranker
     |
     v
Evidence Validator
     |
     v
Context Builder
     |
     v
Local LLM
     |
     v
Citation Validator
     |
     v
FINAL ANSWER
     +
SOURCE/PAGE CITATIONS
```

---

# 22. Error Handling

The system should handle:

### Bad PDF

Log the document and continue where possible.

### Empty page

Do not create empty chunks.

### Missing metadata

Fail validation rather than silently inventing metadata.

### No retrieval results

Return insufficient-evidence response.

### Weak retrieval

Do not force an answer.

### LLM failure

Show a clear local-model error.

### Citation mismatch

Reject or flag the answer rather than displaying fabricated citations.

---

# 23. Testing Strategy

Minimum tests:

```text
PDF extraction
    ↓
page numbers preserved

Cleaning
    ↓
headers/footers handled

Chunking
    ↓
chunks non-empty + reasonable size

Metadata
    ↓
controlled vocabulary validation

Retrieval
    ↓
known questions retrieve expected source

Citation
    ↓
source/page exists in retrieved evidence

End-to-end
    ↓
question → answer → valid evidence
```

---

# 24. Final MVP Definition

The MVP is DONE when:

- Multiple PDFs can be ingested.
- Page numbers are preserved.
- Metadata is validated.
- Embeddings are generated locally.
- ChromaDB works locally.
- BM25 works locally.
- Hybrid retrieval works.
- Reranking works.
- A local LLM generates answers.
- Answers are grounded in retrieved evidence.
- Sources and pages are displayed.
- The system can say "not enough evidence."
- Streamlit UI works.
- Evaluation dataset exists.
- Retrieval and answer metrics can be reported.
- A new PDF can be added without rewriting the pipeline.
- README explains setup and execution.

---

# 25. What Makes This Project Strong

The project is NOT impressive because it uses many libraries.

It is strong because it combines:

1. **Multi-source knowledge**
2. **Structured metadata**
3. **Hybrid retrieval**
4. **Local embeddings**
5. **Reranking**
6. **Evidence-aware generation**
7. **Citation validation**
8. **Hallucination control**
9. **Evaluation**
10. **Extensible source registry**
11. **A usable interface**

The core story for the hackathon is:

> "We built a local, multi-source, evidence-grounded Autism Knowledge Intelligence System that retrieves information from authoritative documents, reranks the evidence, generates an answer using a local LLM, and traces the answer back to the original source and page."

---

# 26. Build Order

Do not try to build everything simultaneously.

```text
1. Ingestion
        ↓
2. Chunking + metadata validation
        ↓
3. Embeddings
        ↓
4. ChromaDB
        ↓
5. BM25
        ↓
6. Hybrid retrieval
        ↓
7. Reranking
        ↓
8. Context + evidence layer
        ↓
9. Local LLM
        ↓
10. Citation validation
        ↓
11. Evaluation
        ↓
12. Streamlit UI
        ↓
13. End-to-end testing
        ↓
14. README + demo
```

The architecture is fixed, but implementation can be adjusted when real test results reveal problems.
