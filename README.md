# compliance-rag-portfolio

Production-grade RAG system: structure-aware chunking, hybrid BM25+vector retrieval, cross-encoder reranking, and RAGAS eval — built on EU compliance documents (GDPR, DORA, EU AI Act).

## What this demonstrates

| Component | Technique |
|-----------|-----------|
| Chunking | Structure-aware splitting on article/section boundaries |
| Retrieval | Hybrid BM25 + dense vector search |
| Reranking | Cross-encoder reranker (Cohere / local) |
| Grounding | Inline citation injection + faithfulness check |
| Eval | RAGAS harness — faithfulness, answer relevancy, context precision, context recall |
| Observability | Langfuse tracing per query |

## Architecture

```
ingestion → chunking → embedding → hybrid retrieval → reranking → generation → eval
```

## Eval results

> Baseline vs tuned retrieval on 75-question eval set (GDPR + DORA)

| Metric | Baseline | Tuned |
|--------|----------|-------|
| Faithfulness | - | - |
| Answer Relevancy | - | - |
| Context Precision | - | - |
| Context Recall | - | - |

*(results added as eval harness matures)*

## Setup

```bash
cp .env.example .env
pip install -r requirements.txt
```
